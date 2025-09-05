# import sys    
# sys.path.append('C:/Users/DELL/Desktop/PProject/ZipMoE') 

# from deepspeed.moe.sharded_moe import TopKGate  
  
# # 创建 DeepSpeed TopKGate（使用默认模式）  
# deepspeed_gate = TopKGate(  
#     model_dim=test_input.shape[-1],  
#     num_experts=16,  # 根据您的模型调整  
#     k=2,  
#     capacity_factor=1.0,  
#     eval_capacity_factor=1.0  
# )  
  
# with torch.no_grad():  
#     ds_gate_output = deepspeed_gate(test_input)  
      
#     print(f"\nDeepSpeed TopKGate 输出类型: {type(ds_gate_output)}")  
#     if isinstance(ds_gate_output, tuple):  
#         print(f"DeepSpeed 输出是元组，包含 {len(ds_gate_output)} 个元素:")  
#         for i, element in enumerate(ds_gate_output):  
#             print(f"  元素 {i}: 类型={type(element)}, 形状={element.shape if hasattr(element, 'shape') else 'N/A'}")


import torch  
from deepspeed.profiling.flops_profiler import get_model_profile  
from transformers import AutoModelForCausalLM, AutoTokenizer  
  
model = AutoModelForCausalLM.from_pretrained(  
    r"/home/mint/Desktop/zipmoe/models/deepseek-moe-16b-base",  
    torch_dtype=torch.half,  
    device_map=None  
)  
  
def test_phimoe_expert_size(expert_module, input_shape):  
    """测试单个deepseek-moe-16b-base专家模块的大小"""  
      
    flops, macs, params = get_model_profile(  
        model=expert_module,  
        input_shape=input_shape,  
        print_profile=True,  
        detailed=True,  
        warm_up=10,  
        as_string=True  
    )  
      
    print(f"xxxx Expert Analysis:")  
    print(f"  - Total Parameters: {params}")  
    print(f"  - Total FLOPs: {flops}")  
    print(f"  - Total MACs: {macs}")  
      
    return flops, macs, params  
  
def measure_expert_from_model(model, expert_id=0, layer_id=0):  
    """从完整模型中提取并测量特定专家"""  
      
    expert_module = None  
    for name, module in model.named_modules():  
        if f'layers.{layer_id}.block_sparse_moe.experts.{expert_id}' in name and module.__class__.__name__ == 'PhimoeBlockSparseTop2MLP':  
            expert_module = module  
            break  
      
    if expert_module is None:  
        print(f"Could not find expert {expert_id} in layer {layer_id}")  
        return None  
      
    total_params = sum(p.numel() for p in expert_module.parameters())  
    param_memory_mb = total_params * 4 / (1024 * 1024)  
      
    print(f"Expert {expert_id} in Layer {layer_id} Analysis:")  
    print(f"  - Module: {expert_module.__class__.__name__}")  
    print(f"  - Total Parameters: {total_params:,}")  
    print(f"  - Memory (FP32): {param_memory_mb:.2f} MB")  
      
    print(f"  - Submodule breakdown:")  
    for name, child in expert_module.named_children():  
        child_params = sum(p.numel() for p in child.parameters())  
        print(f"    - {name}: {child_params:,} parameters")  
      
    return total_params, param_memory_mb, expert_module  
  
# 主测试逻辑  
result = measure_expert_from_model(model, expert_id=0, layer_id=31)  
if result:  
    total_params, memory_mb, expert_module = result  
      
    # FLOP分析  
    input_shape = (1, 2048, 4096)  
    flops, macs, params = test_phimoe_expert_size(expert_module, input_shape)