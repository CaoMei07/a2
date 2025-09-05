# 暂时单机 - 手动设置环境变量
import os  
os.environ['RANK'] = '0'  
os.environ['WORLD_SIZE'] = '1'  
os.environ['MASTER_ADDR'] = 'localhost'  
os.environ['MASTER_PORT'] = '12355'  
os.environ['LOCAL_RANK'] = '0' 
os.environ['DS_SKIP_CUDA_CHECK'] = '1'
os.environ['DS_BUILD_OPS'] = '0'  # 禁用所有 ops 构建   
import numpy as np 
import deepspeed  
import torch  
from transformers import AutoModelForCausalLM, AutoTokenizer  
from deepspeed.runtime.utils import DummyOptim  
from deepspeed.runtime.zero.partition_parameters import Init  
import deepspeed.comm as dist  
from deepspeed.runtime.zero.partition_parameters import GatheredParameters
from deepspeed.runtime.zero.partitioned_param_coordinator import PartitionedParameterCoordinator  
from deepspeed.profiling.flops_profiler import get_model_profile  
from deepspeed.profiling.flops_profiler import FlopsProfiler  
import time  
from deepspeed.utils import set_z3_leaf_modules
from deepspeed.runtime.zero.stage3 import unwrap_model_for_generation  
  

# 创建 MoE 模型 
# model = AutoModelForCausalLM.from_pretrained(r"C:\Users\DELL\Desktop\PProject\ZipMoE\models\phi-mini-moe")

# 确保 CUDA 可用  
print(f"CUDA available: {torch.cuda.is_available()}")  
print(f"CUDA device count: {torch.cuda.device_count()}")  
  
# 预先初始化分布式环境  
dist.init_distributed(dist_backend='gloo')  

# 先加载模型，再应用 ZeRO-3 转换  
print("Loading model without ZeRO-3 context...")  
model = AutoModelForCausalLM.from_pretrained(  
    r"/home/mint/Projects/ZipMoE/models/deepseek-moe-16b-base",  
    torch_dtype=torch.half,  
    trust_remote_code=True,
    device_map=None  # 避免自动设备映射  
)  
tokenizer = AutoTokenizer.from_pretrained(r"/home/mint/Projects/ZipMoE/models/deepseek-moe-16b-base")

# 设置专家级别的叶子节点  
set_z3_leaf_modules(model, ['DeepseekMLP'])

print("Applying ZeRO-3 conversion to loaded model...")  
# 创建完整的DeepSpeed配置，确保Init和推理引擎使用相同配置  
ds_config = {  
    "zero_optimization": {  
        "stage": 3,  
        "offload_param": {  
            "device": "nvme",  
            "nvme_path": "/home/mint/Projects/ZipMoE/framework/nvme_offload",  
            "pin_memory": False,  
            "buffer_count": 12,  # 适中的缓冲区数量  
            "buffer_size": int(2.1e8),  # 稍小的缓冲区大小  
            "max_in_cpu": int(8e9)  # 增加 CPU 缓存  
        },  
        "zero_hpz_partition_size": 1,  
        "stage3_prefetch_bucket_size": int(8e6),  # 增加预取桶大小  
        "stage3_max_live_parameters": int(5e7),  # 增加活跃参数  
        "stage3_max_reuse_distance": int(5e7),   # 增加重用距离  
        "stage3_param_persistence_threshold": int(1e7),  # 降低持久化阈值  
        "overlap_comm": False  
    },  
    "fp16": {  
        "enabled": True  
    },  
    "aio": {  
        "block_size": 1048576,  
        "queue_depth": 8,  
        "thread_count": 1,  
        "single_submit": False,  
        "overlap_events": True  
    }  
}
# 使用 Init 的 module 参数来转换已加载的模型  
with Init(remote_device="nvme", pin_memory=False, module=model, config_dict_or_path=ds_config, dtype=torch.half):  
    pass  # 转换在 __init__ 中自动完成  
print("Model conversion completed.")

# # 在模型转换后添加验证  
# print("Checking leaf modules:")  
# for name, module in model.named_modules():  
#     if hasattr(module, '_z3_leaf'):  
#         print(f"Leaf module: {name} -> {module.__class__.__name__}")

if tokenizer.pad_token is None:  
    tokenizer.pad_token = tokenizer.eos_token
    
# 单机版本 - 直接设置参数  
expert_parallel_size = 1  
#tensor_slicing_size = 1  

 
# 使用 DummyOptim 表示这是推理场景  
optimizer = DummyOptim(model.parameters())  


# 初始化 DeepSpeed 推理引擎 
ds_engine = deepspeed.init_inference(  
    model=model,  
    dtype=torch.half,  
    # keep_module_on_host=True,  # 防止大模型在初始化时被移动到GPU设备上
    #tensor_parallel={"tp_size": 1},  
    moe={"enabled": True, "ep_size": 1, "moe_experts": [64]},  
    zero=ds_config["zero_optimization"],
    replace_with_kernel_inject=False,
    enable_cuda_graph=False
)

# 在引擎创建后立即强制同步配置  
if hasattr(ds_engine, 'module') and hasattr(ds_engine.module, 'parameter_offload'):  
    param_offload = ds_engine.module.parameter_offload  
    if hasattr(param_offload, 'param_coordinator'):  
        coordinator = param_offload.param_coordinator  
          
        # 从引擎配置强制更新协调器参数  
        engine_config = ds_engine._config.zero  
        coordinator._PartitionedParameterCoordinator__max_n_available_params = engine_config.max_live_parameters  
        coordinator._PartitionedParameterCoordinator__max_reuse_dist_in_numel = engine_config.max_reuse_distance  
        coordinator._PartitionedParameterCoordinator__prefetch_bucket_sz = engine_config.prefetch_bucket_size  
          
        print("=== Updated Parameter Coordinator Settings ===")  
        print(f"Max available parameters: {coordinator._PartitionedParameterCoordinator__max_n_available_params:,}")  
        print(f"Max reuse distance: {coordinator._PartitionedParameterCoordinator__max_reuse_dist_in_numel:,}")  
        print(f"Prefetch bucket size: {coordinator._PartitionedParameterCoordinator__prefetch_bucket_sz:,}")
        
# 验证参数协调器是否正确工作    
if hasattr(ds_engine, 'parameter_offload'):    
    print("Offload manager exists")    
    coordinator = ds_engine.parameter_offload.param_coordinator    
    print(f"Coordinator type: {type(coordinator)}")    
    print(f"Coordinator class name: {coordinator.__class__.__name__}")  
      
    # 检查是否是您的专家感知调度器  
    if coordinator.__class__.__name__ == 'ExpertAwareParameterCoordinator':  
        print("✓ Expert-aware coordinator detected!")  
    else:  
        print("✗ Standard coordinator detected")  
else:    
    print("No parameter_offload found")

# # 添加调试代码验证参数卸载管理器状态  
# print(f"Engine attributes: {[attr for attr in dir(ds_engine) if 'offload' in attr.lower() or 'parameter' in attr.lower()]}")  
  
# print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")  
# print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")  

model = ds_engine.module

# 批量输入示例  
input_texts = [  
    '讲讲numpy的用法',  
    '介绍强化学习'
]    

# 批量tokenization  
inputs = tokenizer(  
    input_texts,   
    return_tensors='pt',   
    padding=True,  # 重要：批量处理需要padding  
    truncation=True,  
    max_length=512  # 设置最大长度  
)  
  
# 批量输入预处理  
inputs = {k: v.to('cuda') if k in ['input_ids', 'attention_mask']     
          else v.to('cuda').to(torch.half)     
          for k, v in inputs.items()}  
  
print(f"Batch size: {inputs['input_ids'].shape[0]}")  
print(f"Sequence length: {inputs['input_ids'].shape[1]}")

model.eval()  
  
# ========== 阶段1: 批量Prompt处理阶段时间测量 ==========    
print("=== Batch Prompt Processing Stage Profiling ===")    
    
torch.cuda.synchronize()    
prompt_start_time = time.perf_counter()    
    
# 执行批量prompt处理    
with torch.no_grad():    
    prompt_outputs = model(**inputs, use_cache=True)    
    past_key_values = prompt_outputs.past_key_values    
    
torch.cuda.synchronize()    
prompt_end_time = time.perf_counter()    
prompt_duration = prompt_end_time - prompt_start_time    
    
batch_size = inputs['input_ids'].shape[0]  
print(f"Batch Prompt Processing:")  
print(f"  - Batch size: {batch_size}")  
print(f"  - Wall Clock Duration: {prompt_duration:.2f} seconds ({prompt_duration*1000:.2f} ms)")    
print(f"  - Per sample: {(prompt_duration/batch_size)*1000:.2f} ms")  
# ========== 阶段2: 批量Token生成阶段时间测量 ==========    
print("\n=== Batch Token Generation Stage Profiling ===")    
    
current_input_ids = inputs['input_ids']    
max_new_tokens = 15   
batch_size = current_input_ids.shape[0]  
generated_tokens = [[] for _ in range(batch_size)]  # 每个样本的生成token  
active_sequences = set(range(batch_size))  # 跟踪活跃序列  
    
generation_step_times = []    
total_generation_start = time.perf_counter()    
    
for step in range(max_new_tokens):    
    if not active_sequences:  # 所有序列都已完成  
        break  
          
    torch.cuda.synchronize()    
    step_start = time.perf_counter()    
        
    with torch.no_grad():    
        outputs = model(    
            input_ids=current_input_ids[:, -1:],    
            past_key_values=past_key_values,    
            use_cache=True    
        )    
            
        next_tokens = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)    
          
        # 检查每个序列是否结束  
        sequences_to_remove = set()  
        for i in range(batch_size):  
            if i in active_sequences:  
                token = next_tokens[i].item()  
                if token == tokenizer.eos_token_id:  
                    sequences_to_remove.add(i)  
                else:  
                    generated_tokens[i].append(token)  
          
        # 移除已完成的序列  
        active_sequences -= sequences_to_remove  
                
        current_input_ids = torch.cat([current_input_ids, next_tokens], dim=-1)    
        past_key_values = outputs.past_key_values    
        
    torch.cuda.synchronize()    
    step_end = time.perf_counter()    
    step_duration = step_end - step_start    
    generation_step_times.append(step_duration)    
        
    print(f"Generation Step {step+1}: {step_duration*1000:.2f} ms (Active: {len(active_sequences)})")
  
torch.cuda.synchronize()    
total_generation_end = time.perf_counter()    
total_generation_duration = total_generation_end - total_generation_start 

# ========== 批量性能总结分析 ==========    
print("\n=== Batch Performance Summary ===")    
    
avg_generation_step = sum(generation_step_times) / len(generation_step_times) if generation_step_times else 0    
total_inference_time = prompt_duration + total_generation_duration    
  
print(f"Batch Processing Results:")  
print(f"  - Batch size: {batch_size}")  
print(f"  - Prompt Processing: {prompt_duration:.2f} seconds ({prompt_duration*1000:.2f} ms)")    
print(f"  - Token Generation: {total_generation_duration:.2f} seconds ({total_generation_duration*1000:.2f} ms)")  
print(f"  - Total time: {total_inference_time:.2f} seconds")  
  
# 计算批量吞吐量  
total_generated_tokens = sum(len(tokens) for tokens in generated_tokens)  
if total_generation_duration > 0:    
    batch_tokens_per_second = total_generated_tokens / total_generation_duration  
    per_sample_tokens_per_second = batch_tokens_per_second / batch_size  
    print(f"  - Batch throughput: {batch_tokens_per_second:.2f} tokens/second")  
    print(f"  - Per sample throughput: {per_sample_tokens_per_second:.2f} tokens/second")  
  
# 解码批量结果  
print(f"\n=== Generated Results ===")  
for i, (input_text, tokens) in enumerate(zip(input_texts, generated_tokens)):  
    final_output = tokenizer.decode(current_input_ids[i], skip_special_tokens=True)    
    new_part = final_output[len(input_text):].strip()    
    print(f"Sample {i+1}: '{new_part}' ({len(tokens)} tokens)")
             
# ========== 系统清理 ==========  
# 1. 专家感知调度器清理  
if hasattr(ds_engine, 'parameter_offload'):  
    coordinator = ds_engine.parameter_offload.param_coordinator
    print("\n=== Expert Activation Analysis ===")  
    if hasattr(coordinator, 'expert_activation_history'):
        first_two_history = list(coordinator.expert_activation_history.items())[:2]
        print(f"[EXPERT_FETCH_HISTORY]:{first_two_history}")
        #coordinator.expert_activation_history.clear()  
    if hasattr(coordinator, 'expert_fetch_count'): 
        first_five_count = list(coordinator.expert_fetch_count.items())[:5]
        print(f"[EXPERT_FETCH_COUNT]:{first_five_count}")  
        coordinator.expert_fetch_count.clear()  
    if hasattr(coordinator, 'expert_fetch_latency'):  
        first_five_latency = list(coordinator.expert_fetch_latency.items())[:5]
        print(f"[EXPERT_FETCH_LATENCY]:{first_five_latency}")
        total_fetch_time = 0  
        fetch_count = 0  
      
        for expert_key, latencies in coordinator.expert_fetch_latency.items():  
            total_fetch_time += sum(latencies)  
            fetch_count += len(latencies)  
        
        avg_fetch_latency = total_fetch_time / fetch_count if fetch_count > 0 else 0       
        print(f"Expert Parameter Transfer Analysis:")  
        print(f"  - Total fetch operations: {fetch_count}")  
        print(f"  - Total fetch time: {total_fetch_time*1000:.2f} ms")  
        print(f"  - Average fetch latency: {avg_fetch_latency*1000:.2f} ms")  
        coordinator.expert_fetch_latency.clear()  
    # if hasattr(coordinator, 'expert_truefetch_latency'):  
    #     print("---------------------------------------")
    #     avg_truefetch_latency = np.mean(coordinator.expert_truefetch_latency)
    #     first_10_truefetch_latency = coordinator.expert_truefetch_latency[:10]
    #     print(f"[EXPERT_TRUEFETCH_LATENCY]:{first_10_truefetch_latency}") 
    #     print(f"  - Total true fetch operations: {len(coordinator.expert_truefetch_latency)}")  
    #     print(f"  - Total true fetch time: {sum(coordinator.expert_truefetch_latency)*1000:.2f} ms")  
    #     print(f"  - Average fetch latency: {avg_truefetch_latency*1000:.2f} ms")  
        coordinator.expert_truefetch_latency.clear() 
    if hasattr(coordinator, 'get_expert_cache_stats'):
        print("---------------------------------------")
        stats = coordinator.get_expert_cache_stats()
    if hasattr(coordinator, 'expert_release_latency'):  
        # first_five_latency = list(coordinator.expert_fetch_latency.items())[:5]
        # print(f"[EXPERT_FETCH_LATENCY]:{first_five_latency}")
        total_release_time = 0  
        release_count = 0  
      
        for expert_key, latencies in coordinator.expert_release_latency.items():  
            total_release_time += sum(latencies)  
            release_count += len(latencies)  
        
        avg_release_latency = total_release_time / release_count if release_count > 0 else 0       
        print(f"Expert Parameter Transfer Analysis:")  
        print(f"  - Total release operations: {release_count}")  
        print(f"  - Total release time: {total_release_time*1000:.2f} ms")  
        print(f"  - Average release latency: {avg_release_latency*1000:.2f} ms")  
        coordinator.expert_release_latency.clear()  
        
  
# # 2. DeepSpeed状态卸载  
# if hasattr(ds_engine.optimizer, 'offload_states'):  
#     ds_engine.optimizer.offload_states()  
  
# 3. 分区缓存清理  
if hasattr(ds_engine, 'empty_partition_cache'):  
    ds_engine.empty_partition_cache()  
  
# 4. 推理容器内存释放  
if hasattr(ds_engine, '_inference_containers'):  
    for container in ds_engine._inference_containers:  
        if hasattr(container, 'release_memory'):  
            container.release_memory()  
  
# 5. 彻底的内存清理  
import gc  
gc.collect()  
from deepspeed.runtime.utils import empty_cache  
empty_cache()

