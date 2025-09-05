
import sys
import torch
import time
import os
import deepspeed
from deepspeed import comm as dist
from deepspeed.utils.logging import log_dist 
from deepspeed import _SINGLE_MACHINE_INFERENCE_MODE  # zyr 导入全局变量单机判断
from torch.nn.modules import Module
from packaging import version as pkg_version
from deepspeed.runtime.checkpoint_engine.torch_checkpoint_engine import TorchCheckpointEngine
from ..comm.comm import init_distributed
from ..moe.utils import has_moe_layers
from ..moe.sharded_moe import TopKGate
from deepspeed.accelerator import get_accelerator

# --------------------------zyr-------------------------
#from ..module_inject.policy import TransformerPolicy
try:  
    from ..module_inject.policy import TransformerPolicy  
except ImportError:  
    class TransformerPolicy:  
        hf_model_config = None
# -------------------------------------------------------

from ..model_implementations.transformers.ds_transformer import DeepSpeedTransformerInference

DS_INFERENCE_ENABLED = False
from torch import nn

INFERENCE_MODEL_TIMER = "model-forward-inference"


class InferenceEngine(Module):
    inference_mp_group=None
    inference_ep_group=None
    expert_mp_group=None
    
    def __init__(self,model,config):
        """
        Args:
            model: torch.nn.Module
            config: DeepSpeedInferenceConfig
        """
        global DS_INFERENCE_ENABLED  # 设置全局推理标志
        DS_INFERENCE_ENABLED = True
        
        super().__init__()
        if DeepSpeedTransformerInference.workspace is not None:
            self.destroy()  # 用于清理工作空间（包括内存空间）
        
        self.module=model
        self._config=config
        
        self._get_model_config_generate(config)  # 获取模型配置用于生成（向后兼容）
        
        # 模型配置和兼容性处理
        if hasattr(self.module, "generate"):  # 处理生成方法补丁
            self.generate = self._generate   
        # -----------------------zyr----------------------------
        if hasattr(self.module,"config") and config.replace_with_kernel_inject:
            TransformerPolicy.hf_model_config = self.module.config
        # -----------------------------------------------------
        
        # 验证配置的数据类型是否被当前加速器支持
        if config.dtype not in get_accelerator().supported_dtypes():
            raise ValueError(
                f"Data type {config.dtype} is not supported by {get_accelerator().device_name()} accelerator")
        
        # 设置注入字典
        self.injection_dict = config.injection_policy

        self.quantize_merge_count = 1                   # 量化
        self.quantization_scales = None
        # 专家并行
        self.ep_group = None  # config.moe.ep_group
        self.expert_mp_group = None  # config.moe.ep_mp_group
        
        # 性能优化和工具初始化
        self.cuda_graph_created = False     # 初始化cuda图
        self.checkpoint_engine = TorchCheckpointEngine()  # 检查点引擎
        quantization_setting = None         # 量化设置
        self._init_quantization_setting(
            quantization_setting)  
        self.model_profile_enabled = False  # 性能分析工具
        self._model_times = []
        
        # 模型特定优化
        if not self.injection_dict and config.replace_with_kernel_inject:
            # This is a hack to remove the prepare_mask function on HF side for BLOOM architecture
            self.remove_mask_prepare_for_bloom() # mash准备函数的移除
        
        # cuda图的版本检查
        if get_accelerator().device_name() == 'cuda' and config.enable_cuda_graph:
            assert pkg_version.parse(torch.__version__) >= pkg_version.parse("1.10"), \
                "If you want to use cuda graph, please upgrade torch to at least v1.10"

        # 将模型转换为预期的数据类型
        if config.dtype:
            self._convert_to_dtype(config)

        if isinstance(self.module, torch.nn.Module): # 检测MoE层
            moe, _ = has_moe_layers(self.module)
        else:
            moe = False
# ---------------------------zyr更改--------------------------------
        if moe:  
            # 如果是单机模式，直接跳过专家并行组创建  
            if not _SINGLE_MACHINE_INFERENCE_MODE:  
                world_size = dist.get_world_size()  
                if world_size > 1:  
                    self._create_ep_parallel_group(config.moe.moe_experts)
        
        if hasattr(config, 'zero') and config.zero and config.zero.stage == 3:  
            self.parameter_offload = self.initialize_ds_offload(  
                module=self.module,  
                timers=getattr(self, 'timers', None),  
                ds_config=config,  
                overlap_comm=True,  
                prefetch_bucket_size=getattr(config.zero, 'stage3_prefetch_bucket_size', 50000000),  
                max_reuse_distance=getattr(config.zero, 'stage3_max_reuse_distance', 1000000000),  
                max_live_parameters=getattr(config.zero, 'stage3_max_live_parameters', 1000000000),  
                param_persistence_threshold=100000,  
                model_persistence_threshold=sys.maxsize,  
                dp_process_group=None,  
                offload_param_config=config.zero.offload_param,  
                #mpu=self.mpu
                zero_param_parallel_group=None,  
                zero_quantized_weights=False,  
                zero_quantized_nontrainable_weights=False,  
                zero_module_granularity_threshold=0,  
                log_trace_cache_warnings=False  
            )  
            # 确保模型也能访问到卸载管理器  
            self.module.parameter_offload = self.parameter_offload
            
            # 获取专家感知参数协调器  
            if hasattr(self.parameter_offload, 'param_coordinator'):  
                expert_coordinator = self.parameter_offload.param_coordinator  
                
                # 为所有 MoE 层设置协调器  
                layer_id = 0  
                for name, module in self.module.named_modules():  
                    if hasattr(module, '_set_expert_coordinator'):  
                        module._set_expert_coordinator(expert_coordinator, layer_id)  
                        print(f"[EXPERT_COORDINATOR] Set coordinator for {name} at layer {layer_id}")  
                        layer_id += 1  
        else:  
            self.parameter_offload = None
            
        # 添加门控钩子
        #self._setup_gate_adapters()
        
# ------------------------------------------------------------------       
        
        # 支持三种推理模式
        if self.injection_dict: # 用户自己提供的dict
            # 1. 用户指定张量并行
            assert not config.replace_with_kernel_inject, "Cannot use both user specified injection policy and kernel injection"
            for client_module, injection_policy in self.injection_dict.items(): # 对应模块名称或注入策略
                
                assert issubclass(client_module,
                                  torch.nn.Module), f"{client_module} is not a subclass of torch.nn.Module"

                # construct the tuple and pass that instead of a string or dict.
                if isinstance(injection_policy, str):
                    config.injection_policy_tuple = (injection_policy, )
                else:
                    config.injection_policy_tuple = injection_policy

                layer_names = [name for name, _ in self.module.named_modules()]
                for policy in config.injection_policy_tuple:
                    if not any(name.endswith(policy) for name in layer_names):
                        raise ValueError(f"Injection policy layer'{policy}' not valid.")

                self._apply_injection_policy(config, client_module)
        else:
            if config.replace_with_kernel_inject:
                # 2. DeepSpeed内核注入（自动识别并替换常见模块...）
                self._apply_injection_policy(config)
        
        # 设备配置和最终设置
        device = get_accelerator().current_device_name()
        # NOTE: This check assumes a Hugging Face hierarchy for the device type i.e. module.device.type
        is_meta_device = hasattr(self.module, "device") and self.module.device.type == 'meta'
        if is_meta_device:  # meta设备处理
            self.module.to_empty(device=device)
        elif not config.keep_module_on_host:
            self.module.to(device)  # 模块移动到目标设备

        # 检查是否可以在替代模块中创建本地cuda图
        self.local_cuda_graph = self._local_cuda_graph_used(self.module)
        self._is_compiled = False   # 编译支持的检查
        
    # ----------------------zyr添加-------------------------
    def _init_distributed_if_needed(self, config):  
        """根据配置自动初始化分布式通信"""  
        # from deepspeed import comm as dist  
        
        # 检查是否为真正的单机配置  
        is_single_machine = (  
            config.tensor_parallel.tp_size <= 1 and  
            (not hasattr(config, 'moe') or config.moe.ep_size <= 1)  
        )  
        
        # 如果是单机，完全跳过通信初始化  
        if is_single_machine:  
            return False  # 返回 False 表示未初始化通信  
        
        # 只有在真正需要分布式时才初始化  
        if not (hasattr(dist, 'is_initialized') and dist.is_initialized()):  
            dist.init_distributed()  
        
        return True  # 返回 True 表示已初始化通信
    
    # ------------------------------------------------------
    
    # -------------------------zyr-----------------------
    def initialize_ds_offload(
        self,
        module,
        timers,
        ds_config,
        overlap_comm,
        prefetch_bucket_size,
        max_reuse_distance,
        max_live_parameters,
        param_persistence_threshold,
        model_persistence_threshold,
        dp_process_group,
        offload_param_config,
        #mpu,
        zero_param_parallel_group,
        zero_quantized_weights,
        zero_quantized_nontrainable_weights,
        zero_module_granularity_threshold,
        log_trace_cache_warnings,
    ):
        from deepspeed.runtime.zero.parameter_offload import DeepSpeedZeRoOffload  
        return DeepSpeedZeRoOffload(module=module,
                                    timers=timers,
                                    ds_config=ds_config,
                                    overlap_comm=overlap_comm,
                                    prefetch_bucket_size=prefetch_bucket_size,
                                    max_reuse_distance=max_reuse_distance,
                                    max_live_parameters=max_live_parameters,
                                    param_persistence_threshold=param_persistence_threshold,
                                    model_persistence_threshold=model_persistence_threshold,
                                    dp_process_group=dp_process_group,
                                    offload_param_config=offload_param_config,
                                    #mpu=mpu,
                                    zero_param_parallel_group=zero_param_parallel_group,
                                    zero_quantized_weights=zero_quantized_weights,
                                    zero_quantized_nontrainable_weights=zero_quantized_nontrainable_weights,
                                    zero_module_granularity_threshold=zero_module_granularity_threshold,
                                    log_trace_cache_warnings=log_trace_cache_warnings)
    # --------------------------------------------------------------
    
    # ----------------------------zyr添加钩子获取门控结果的函数-------------------
    def _setup_gate_adapters(self):  
        """为不同类型的门控层设置适配器 - 集成到 DeepSpeedEngine"""  
        if not hasattr(self, 'gate_hooks'):  
            self.gate_hooks = []  
        if not hasattr(self, 'expert_selections'):  
            self.expert_selections = {}  
            
        for name, module in self.module.named_modules():  
            is_gate_module = False  
            
            # 1. 标准 DeepSpeed MoE 门控  
            if isinstance(module, TopKGate):  
                is_gate_module = True  
                
            # 2. DeepSeek MoE 门控 - 精确匹配  
            elif (name.endswith('.mlp.gate') and   
                not name.endswith('.mlp.gate_proj') and  
                hasattr(module, '__class__') and   
                'Gate' in module.__class__.__name__):  
                is_gate_module = True  
                
            # 3. 其他 MoE 门控模式  
            elif (name.endswith('.gate') and   
                not name.endswith('.gate_proj') and   
                not name.endswith('.mlp.gate') and  
                ('moe' in name.lower() or 'expert' in name.lower())):  
                is_gate_module = True  
                
            if is_gate_module:  
                if isinstance(module, TopKGate):  
                    self._add_topk_gate_adapter(module, name)  
                else:  
                    self._add_custom_gate_adapter(module, name)  
                    
    def _add_custom_gate_adapter(self, module, name):  
        """为自定义门控层（如 DeepSeek MoEGate）添加专家选择钩子"""  
        def gate_forward_hook(module, input, output):  
            try:  
                layer_id = self._extract_layer_id_from_name(name)  
                if layer_id is None:  
                    return  
                    
                # 处理 DeepSeek MoEGate 输出格式: (topk_idx, topk_weight, aux_loss)  
                if isinstance(output, tuple) and len(output) >= 2:  
                    topk_idx = output[0]  # 专家索引  
                    topk_weight = output[1]  # 专家权重  
                    
                    if torch.is_tensor(topk_idx) and topk_idx.dtype in [torch.int32, torch.int64, torch.long]:  
                        # 转换为 CPU 并提取专家索引  
                        expert_indices = topk_idx.cpu().tolist()  
                        
                        # 展平二维列表并去重  
                        if len(expert_indices) > 0 and isinstance(expert_indices[0], list):  
                            flattened_indices = []  
                            for token_experts in expert_indices:  
                                if isinstance(token_experts, list):  
                                    flattened_indices.extend(token_experts)  
                                else:  
                                    flattened_indices.append(token_experts)  
                            unique_experts = sorted(list(set(flattened_indices)))  
                        else:  
                            unique_experts = sorted(list(set(expert_indices)))  
                        
                        # 验证专家索引范围  
                        if hasattr(module, 'n_routed_experts'):  
                            n_experts = module.n_routed_experts  
                            if unique_experts and (max(unique_experts) >= n_experts or min(unique_experts) < 0):  
                                unique_experts = [idx for idx in unique_experts if 0 <= idx < n_experts]  
                        
                        # 存储到 Engine 的专家选择记录中  
                        self.expert_selections[layer_id] = unique_experts  
                        
                        # 如果有参数协调器，通知它  
                        if (hasattr(self, 'parameter_offload') and   
                            hasattr(self.parameter_offload, 'param_coordinator')):  
                            coordinator = self.parameter_offload.param_coordinator  
                            if hasattr(coordinator, '_update_expert_activation_history'):  
                                coordinator._update_expert_activation_history(unique_experts, layer_id)  
                            setattr(coordinator, 'expert_indices', unique_experts)  
                            setattr(coordinator, 'layer_id', layer_id)  
                            
            except Exception as e:  
                # 使用兼容的方式获取 rank 信息  
                rank = getattr(self, 'global_rank', 0) if hasattr(self, 'global_rank') else 0  
                if rank == 0:  
                    print(f"Gate hook error for {name}: {e}")  
                    
        # 注册钩子  
        hook_handle = module.register_forward_hook(gate_forward_hook)  
        self.gate_hooks.append(hook_handle)  
    
    def _add_topk_gate_adapter(self, module, name):  
        """为标准 TopKGate 添加协调器"""  
        if (hasattr(self, 'parameter_offload') and   
            hasattr(self.parameter_offload, 'param_coordinator')):  
            try:  
                layer_id = self._extract_layer_id_from_name(name)  
                if layer_id is not None and hasattr(module, '_set_expert_coordinator'):  
                    module._set_expert_coordinator(self.parameter_offload.param_coordinator, layer_id)  
                    if self.global_rank == 0:  
                        logger.info(f"Set coordinator for TopKGate layer {layer_id}")  
            except Exception as e:  
                if self.global_rank == 0:  
                    logger.warning(f"Failed to set coordinator for {name}: {e}")  
    
    def _extract_layer_id_from_name(self, name):  
        """从模块名称中提取层ID"""  
        if 'layers.' in name:  
            try:  
                layer_id = int(name.split('layers.')[1].split('.')[0])  
                return layer_id  
            except (ValueError, IndexError):  
                pass  
        return None  
    
    def _cleanup_gate_hooks(self):  
        """清理门控钩子 - 在 Engine 销毁时调用"""  
        if hasattr(self, 'gate_hooks'):  
            for hook in self.gate_hooks:  
                hook.remove()  
            self.gate_hooks.clear()
    # -------------------------------------------------
    
    # 清理推理引擎的资源
    def destroy(self):
        DeepSpeedTransformerInference.layer_id = 0
        DeepSpeedSelfAttention.num_layers = 0
        if DeepSpeedTransformerInference.workspace.is_allocated(): # 检查工作空间是否分配
            DeepSpeedTransformerInference.workspace.release_workspace() # 释放工作空间内存
        DeepSpeedTransformerInference.workspace = None # 将工作空间置空
    # 性能分析
    def profile_model_time(self, use_cuda_events=True):
        if not self.model_profile_enabled and not self._config.enable_cuda_graph: # 未启用模型分析且未使用cuda图
            self.module.register_forward_pre_hook(self._pre_forward_hook)  # 注册前向传播的钩子函数
            self.module.register_forward_hook(self._post_forward_hook)
        self.model_profile_enabled = True
        self.use_cuda_events = use_cuda_events  # 决定是否使用cuda事件进行计时
        if self.use_cuda_events:                # 如果使用cuda事件进行计时
            self.timers = SynchronizedWallClockTimer() # 创建...实例         
    # 配置获取
    def _get_model_config_generate(self, config): # 若没有指定config则从模块中获取
        # this is being passed to replace_transformer_layer(config=self.user_model_config_dict)
        self.config = getattr(self.module, 'config', None) if config.config is None else config.config
    # BLOOM架构优化
    def remove_mask_prepare_for_bloom(self):
        if hasattr(self.module, 'transformer'): # 检查模块有无transformer
            if hasattr(self.module.transformer, '_prepare_attn_mask'): # 跳过HF的mask准备逻辑，改用lambda直接返回attention_mask
                self.module.transformer._prepare_attn_mask = lambda attention_mask, *args, **kwargs: attention_mask
    # 注意力偏置构建
    def build_attn_bias(self):
        if hasattr(self.module, 'transformer'): # 替换原方法
            if hasattr(self.module.transformer, '_attn_bias'):
                self.module.transformer._attn_bias_orig = self.module.transformer._attn_bias
                self.module.transformer.__class__._attn_bias = build_mpt_atten_bias_tensor
    # 前向传播前钩子（在模型前向传播开始前被调用）
    def _pre_forward_hook(self, module, *inputs, **kwargs):
        if self.use_cuda_events:
            self.timers(INFERENCE_MODEL_TIMER).start() # 启动计时器
        else:
            get_accelerator().synchronize()  # 同步加速器设备并记录开始时间
            self._start = time.time()
    # 前向传播后钩子
    def _post_forward_hook(self, module, input, output):
        if self.use_cuda_events:
            self.timers(INFERENCE_MODEL_TIMER).stop() # 停止计时器
            elapsed_time = self.timers(INFERENCE_MODEL_TIMER).elapsed(reset=True) # 计算经过时间
        else:
            get_accelerator().synchronize()
            self._end = time.time()
            elapsed_time = (self._end - self._start) * 1e3  # convert seconds to ms
        self._model_times.append(elapsed_time)  # 用于性能分析

    # 创建专家并行组  
    def _create_ep_parallel_group(self, moe_experts):
        # Call the init process
        self.ep_group = {}
        self.expert_mp_group = {}
        moe_experts = moe_experts if type(moe_experts) is list else [moe_experts]
        for e in moe_experts:
            self.ep_group.update({e: None})         # 向字典添加专家
            self.expert_mp_group.update({e: None})
        for moe_ep_size in self.ep_group.keys():  # 每个专家并行组中的进程数量
            num_ep_groups = dist.get_world_size() // moe_ep_size # 计算专家并行组数量
            for i in range(num_ep_groups):
                ep_cnt = i * moe_ep_size
                size = dist.get_world_size() if moe_ep_size > dist.get_world_size() else moe_ep_size
                ranks = list(range(ep_cnt, ep_cnt + size)) # [0,1,2],[3,4,5] moe_ep_size = 3
                _ep_group = dist.new_group(ranks) # 为每个组分配连续的进程
                if dist.get_rank() in ranks:  # 如果当前进程在该组中，存入
                    self.ep_group.update({moe_ep_size: _ep_group})
            
            if dist.get_world_size() > moe_ep_size: # 将单个专家的参数分割到多个GPU上如[0,3]负责专家0，[1,4]专家1
                num_expert_mp_groups = dist.get_world_size() // num_ep_groups
                expert_mp_size = dist.get_world_size() // moe_ep_size  
                for i in range(num_expert_mp_groups): # 下面采用交错分布模式
                    expert_mp_comm_ranks = [i + nr * moe_ep_size for nr in range(expert_mp_size)]
                    _expert_mp_group = dist.new_group(expert_mp_comm_ranks)
                    if dist.get_rank() in expert_mp_comm_ranks:
                        self.expert_mp_group.update({moe_ep_size: _expert_mp_group})
    # 初始化量化
    def _init_quantization_setting(self, quantization_setting):
        self.quantize_bits = 8    # 量化的比特位是 8 位，即使用 INT8 量化
        self.mlp_extra_grouping = False   # MLP 中是否启用“额外分组量化”
        self.quantize_groups = 1   # 将一个张量划分为多少个组进行分组量化 1统一量化
        if type(quantization_setting) is tuple:
            self.mlp_extra_grouping, \
            self.quantize_groups = quantization_setting
        elif quantization_setting is not None:
            self.quantize_groups = quantization_setting
        log_dist(
            f"quantize_bits = {self.quantize_bits} "
            f"mlp_extra_grouping = {self.mlp_extra_grouping}, "
            f"quantize_groups = {self.quantize_groups}", [0])  # 记录日志只在rank0上打印
    # 模型检查点加载
    def load_model_with_checkpoint(self, r_module):
        self.mp_replace = ReplaceWithTensorSlicing(
            mp_group=self.mp_group, mp_size=self._config.tensor_parallel.tp_size)  #, out_dim=0, in_dim=1)
        error_msgs = []

        def load(module, state_dict, prefix):
            args = (state_dict, prefix, {}, True, [], [], error_msgs)
            if hasattr(module, 'weight'): # 权重参数处理
                if module.weight.data.is_meta:
                    # meta tensor cannot be casted or copied to, so we need to replace it with a normal tensor here
                    module.weight = torch.nn.parameter.Parameter(data=torch.empty_like(module.weight.data,
                                                                                       device="cpu"),
                                                                 requires_grad=module.weight.data.requires_grad)
                if 'query_key_value' in prefix:
                    module.weight = self.mp_replace.strided_copy(module.weight.data,
                                                                 state_dict[prefix + 'weight'],
                                                                 num_splits=3)
                else:
                    module.weight = self.mp_replace.copy(module.weight.data, state_dict[prefix + 'weight'])
            else: # 归一化层处理
                if module.norm.weight.data.is_meta:
                    # meta tensor cannot be casted or copied to, so we need to replace it with a normal tensor here
                    module.norm.weight = torch.nn.parameter.Parameter(
                        data=torch.empty_like(module.norm.weight.data, device="cpu"),
                        requires_grad=module.norm.weight.data.requires_grad)
                module.norm.weight = self.mp_replace.copy(module.norm.weight.data, state_dict[prefix + 'weight'])
            if prefix + 'bias' in self.key_list: # 偏置参数处理
                if hasattr(module, 'norm'): 
                    if module.norm.bias.data.is_meta:
                        # meta tensor cannot be casted or copied to, so we need to replace it with a normal tensor here
                        module.norm.bias = torch.nn.parameter.Parameter(
                            data=torch.empty_like(module.norm.bias.data, device="cpu"),
                            requires_grad=module.norm.bias.data.requires_grad)
                    module.norm.bias = self.mp_replace.copy(module.norm.bias, state_dict[prefix + 'bias'])
                else:
                    if module.bias.data.is_meta:
                        # meta tensor cannot be casted or copied to, so we need to replace it with a normal tensor here
                        module.bias = torch.nn.parameter.Parameter(data=torch.empty_like(module.bias.data,
                                                                                         device="cpu"),
                                                                   requires_grad=module.bias.data.requires_grad)
                    data = state_dict[prefix + 'bias']
                    data = data.to(get_accelerator().current_device_name()) # 把数据移动到当前加速器设备上
                    module.bias = self.mp_replace.copy(module.bias, data)
        # 支持的层类型
        layer_policies = {
            nn.Linear: load,
            nn.Embedding: load,
            nn.LayerNorm: load,
            LinearLayer: load,     # Deepspeed自定义层
            LinearAllreduce: load
        }
        # 递归加载模块
        def load_module_recursive(module, prefix='', level=0):
            for name, child in module.named_children(): # 逐层遍历模块的子模块
                if child.__class__ in layer_policies:
                    checking_key = prefix + name + '.'
                    if not any(checking_key in item for item in self.key_list):
                        continue # 当前模块不在 key_list 中，不需要加载
                    if len(list(child.parameters())) > 0 and list(child.parameters())[0].numel() == 0:
                        if len(child.weight.ds_shape) == 1:  # ds_shape 维度为1表示是LayerNorm
                            child = Normalize(dim=child.weight.ds_shape[-1], dtype=child.weight.dtype, eps=child.eps)
                            setattr(module, name, child)  # 把新模块替换进去
                    load(child, self.sd, prefix + name + '.')
                else:  # 如果不是支持的层类型，递归进入子模块
                    load_module_recursive(child, prefix if level == 0 else prefix + name + '.', level + 1)

        load_module_recursive(r_module)

        embedding_weight = None

        for n, p in r_module.named_parameters():
            if "word_embeddings." in n or "embed_tokens." in n or "wte." in n: # 1：BERT,2:HF名称,3:GPT
                embedding_weight = p
        if embedding_weight is not None and hasattr(r_module, "lm_head") and hasattr(
                r_module.lm_head, "weight") and r_module.lm_head.weight.is_meta:
            r_module.lm_head.weight = embedding_weight  # 如果lm_head的权重是meta类型(未初始化),共享embedding
    # 应用注入策略(模型优化)
    def _apply_injection_policy(self, config, client_module=None):
        # client_module is only passed when using the injection_dict method.
        checkpoint_dir = config.checkpoint # 加载检查点
        checkpoint = SDLoaderFactory.get_sd_loader_json(checkpoint_dir,
                                                        self.checkpoint_engine) if checkpoint_dir is not None else None
        # 通用注入
        generic_injection(self.module, dtype=config.dtype, enable_cuda_graph=config.enable_cuda_graph)

        if isinstance(self.module, torch.nn.Module): # 是Pytorch模块，进行替换
            # config is our DeepSpeedInferenceConfig and self.config is the HF model config
            replace_transformer_layer(client_module, self.module, checkpoint, config, self.config)
    # 获取检查点文件名
    def _get_all_ckpt_names(self, checkpoints_path, tag):
        ckpt_file_pattern = self._get_ckpt_name(checkpoints_path, tag, mp_placeholder="*")
        import glob  # 用来查找目录和文件

        ckpt_files = glob.glob(ckpt_file_pattern)  # 匹配符合模式的文件
        ckpt_files.sort()
        return ckpt_files
    # 生成检查点文件名
    def _get_ckpt_name(self, checkpoints_path, tag, mp_placeholder=None):  
        if mp_placeholder is not None:  
            mp_rank_str = mp_placeholder  
        else:  
            mp_rank = 0  # 单GPU环境下始终为0  
            mp_rank_str = "{:02d}".format(mp_rank)  # eg:00  
    
        ckpt_name = os.path.join(  
            checkpoints_path,  
            "mp_rank_" + mp_rank_str + "_model_states.pt",  
        )  
        return ckpt_name
        
    def _load_checkpoint(self, load_dir, load_module_strict=True, tag=None):
        is_pipe_parallel = isinstance(self.module, PipelineModule)
        if is_pipe_parallel:  # 管道并行
            raise RuntimeError('pipeline parallelism is currently not supported in inference.')
        if not isinstance(load_dir, dict) and os.path.isdir(load_dir):
            if tag is None: # 如果没有指定tag，会尝试从latest文件中读取最新的标签
                latest_path = os.path.join(load_dir, "latest")
                if os.path.isfile(latest_path):
                    with open(latest_path, "r") as fd:
                        tag = fd.read().strip()

            ckpt_list = self._get_all_ckpt_names(load_dir, tag)
            sd_loader = SDLoaderFactory.get_sd_loader(ckpt_list, self.checkpoint_engine)
        else:  # 处理JSON格式的检查点
            sd_loader = SDLoaderFactory.get_sd_loader_json(load_dir, self.checkpoint_engine)

        checkpoint = sd_loader['checkpoints']

        if type(checkpoint) is list:  # 列表形式检查点加载
            self.sd = torch.load(checkpoint[0], map_location='cpu', weights_only=False)
            self.key_list = list(self.sd.keys())

            self.load_model_with_checkpoint(self.module)  # 加载参数

            for i in range(1, len(checkpoint)):
                if not dist.is_initialized() or dist.get_rank() == 0:
                    print(f"loading checkpoint ({i})")
                self.sd = torch.load(checkpoint[i], map_location=get_accelerator().device_name(), weights_only=False)
                self.key_list = list(self.sd.keys()) # 循环加载剩余检查点 每次加载到GPU设备并更新参数
                self.load_model_with_checkpoint(self.module) 
        else: # 单一检查点加载
            mp_rank = 0  #if self.mpu is None else self.mpu.get_model_parallel_rank()
            # 加载检查点
            load_path, checkpoint, quantize_config = sd_loader.load(1,
                                                                    mp_rank,
                                                                    is_pipe_parallel=is_pipe_parallel,
                                                                    quantize=(self._config.dtype is torch.int8),
                                                                    quantize_groups=self.quantize_groups,
                                                                    mlp_extra_grouping=self.mlp_extra_grouping)
            # 处理量化相关配置
            self.quantization_scales, self.quantize_merge_count = quantize_config

            moe, _ = has_moe_layers(self.module)
            if moe:  # 如果模型包含moe，加载moe状态字典
                from deepspeed.runtime.engine import DeepSpeedEngine
                old_moe_load = False
                if not isinstance(checkpoint['num_experts'], list):# 检查是否为旧版的MOE加载格式
                    old_moe_load = True # 调用专门的moe加载方法
                DeepSpeedEngine.load_moe_state_dict(load_dir,
                                                    tag,
                                                    state_dict=checkpoint[self._choose_module_key(checkpoint)],
                                                    old_moe_load=old_moe_load,
                                                    model=self.module,
                                                    #mpu=self.mpu,
                                                    checkpoint_engine=self.checkpoint_engine)

            self.module.load_state_dict(state_dict=checkpoint[self._choose_module_key(checkpoint)],
                                        strict=load_module_strict)
    # 确定模块键是model还是module
    def _choose_module_key(self, sd):
        assert not ('module' in sd
                    and 'model' in sd), "checkpoint has both 'model' and 'module' keys, not sure how to proceed"
        assert 'module' in sd or 'model' in sd, "checkpoint contains neither 'model' or 'module' keys, not sure how to proceed"
        if 'module' in sd:
            return 'module'
        elif 'model' in sd:
            return 'model'
    # 数据类型转换
    def _convert_to_dtype(self, config):
        if not isinstance(self.module, torch.nn.Module): # 只对pytorch模块处理
            return
        # 暂时禁用（之前的条件在注释里）这是本来就这样写的不是我改的
        if False:  #config.dtype is torch.int8 and self.quantization_scales is None:
            quantizer = WeightQuantization(mlp_extra_grouping=self.mlp_extra_grouping)
            model, self.quantization_scales = quantizer.model_quantize(self.module, self.injection_dict,self.quantize_bits, self.quantize_groups)
        # 数据类型转换                                                          
        elif config.dtype == torch.half:
            self.module.half()
        elif config.dtype == torch.bfloat16:
            self.module.bfloat16()
        elif config.dtype == torch.float:
            self.module.float()
    # cuda图创建
    def _create_cuda_graph(self, *inputs, **kwargs):
        # warmup to create the workspace and cublas handle
        cuda_stream = get_accelerator().Stream()  # 创建一个新的 CUDA stream（异步流）
        cuda_stream.wait_stream(get_accelerator().current_stream()) # 等待当前 stream 的操作完成，避免乱序执行
        with get_accelerator().stream(cuda_stream): # 用这个stream运行模型几次（warmup）
            for i in range(3):
                ret = self.module(*inputs, **kwargs)
        get_accelerator().current_stream().wait_stream(cuda_stream) # 主线程的CUDA stream等待warmup stream完成，确保图录制前万事俱备
        # 正式录制 CUDA Graph
        # create cuda_graph and assign static_inputs and static_outputs
        self._cuda_graphs = get_accelerator().create_graph() 
        self.static_inputs = inputs # 保存下“静态输入”变量（这要求输入 shape 不变）
        self.static_kwargs = kwargs
        # 进入“图捕获模式” 运行一次 forward，所有操作将记录到 CUDA graph 中
        with get_accelerator().capture_to_graph(self._cuda_graphs):
            self.static_output = self.module(*self.static_inputs, **self.static_kwargs)

        self.cuda_graph_created = True  # 构建完毕
    # cuda图重放 # 和_create_cuda_graph()配合避免重复的内核启动开销
    def _graph_replay(self, *inputs, **kwargs): 
        for i in range(len(inputs)):
            if torch.is_tensor(inputs[i]):
                self.static_inputs[i].copy_(inputs[i])  # 如果是张量复制到静态输入缓冲区
        for k in kwargs:
            if torch.is_tensor(kwargs[k]):
                self.static_kwargs[k].copy_(kwargs[k])
        get_accelerator().replay_graph(self._cuda_graphs)
        return self.static_output
    # 获取模型执行时间
    def model_times(self):
        assert self.model_profile_enabled, "model profiling is not enabled"
        model_times = self._model_times
        if self._config.enable_cuda_graph and len(self._model_times) == 0: # 兼容性检查
            raise ValueError("Model times are empty and cuda graph is enabled. If "
                             "this is a GPT-style model this combo is not supported. If this is a "
                             "BERT-style model this is a bug, please report it. "
                             f"Model type is: {type(self.module)}")
        self._model_times = []
        return model_times # 返回并清空
    # 检查给定模块是否匹配任何同意策略
    def _module_match(self, module):
        for policy in generic_policies:
            policy = policy()
            if policy.match_replaced(module):
                return True
        return False
    # 检查本地cuda图使用
    def _local_cuda_graph_used(self, module):
        if isinstance(module, torch.nn.Module):
            return False
        else:
            sub_module_cuda_graph = False
            for name in module.__dict__.keys():
                sub_module = getattr(module, name)
                # 对每个子模块检查是否匹配策略且可以使用cuda图
                if self._module_match(sub_module) and hasattr(sub_module, "enable_cuda_graph"):
                    sub_module_cuda_graph = True
            # 找到任何使用cuda图的子模块返回true
            return sub_module_cuda_graph
    
    # 前向传播
    def forward(self, *inputs, **kwargs):
        """Execute forward propagation

        Arguments:
            *inputs: Variable length input list
            **kwargs: variable length keyword arguments
        """
        start = None # 性能监控初始化
        if self.model_profile_enabled and get_accelerator().device_name() == 'cuda' and self._config.enable_cuda_graph:
            get_accelerator().synchronize()  # 同步设备
            start = time.time()
        # cuda图执行路径
        if get_accelerator().device_name() == 'cuda' and self._config.enable_cuda_graph and not self.local_cuda_graph:
            if self.cuda_graph_created: # 若已被创建
                outputs = self._graph_replay(*inputs, **kwargs) # 重放
            else: # 没创建创建
                self._create_cuda_graph(*inputs, **kwargs) 
                outputs = self._graph_replay(*inputs, **kwargs)
        # 标准执行路径
        else:
            outputs = self.module(*inputs, **kwargs)
        # 性能监控结束（同步设备记录时间）
        if self.model_profile_enabled and self._config.enable_cuda_graph:
            get_accelerator().synchronize() 
            duration = (time.time() - start) * 1e3  # convert seconds to ms
            self._model_times.append(duration)

        return outputs
    # 文本生成
    def _generate(self, *inputs, **kwargs):
        # Reset KV-cache at the beginning of generate
        if hasattr(self.module, 'reset_cache'):
            self.module.reset_cache()  # 重置KV-cache
        num_beams = 1
        if "generation_config" in kwargs:
            gen_config = kwargs["generation_config"]
            num_beams = getattr(gen_config, "num_beams", 1)
        if "num_beams" in kwargs:
            num_beams = kwargs["num_beams"]

        if num_beams > 1: # beam search参数检查 目前不支持num_beams > 1
            raise NotImplementedError("DeepSpeed does not support `num_beams` > 1, if this is important to you please "
                                      "add your request to: https://github.com/deepspeedai/DeepSpeed/issues/2506")
        # 输入长度验证
        if ("input_ids" in kwargs) and (kwargs["input_ids"].dim() == 2):
            for input_tensor in kwargs["input_ids"]:
                tensor_length = input_tensor.shape[-1]
                if tensor_length > self._config.max_out_tokens: # 检查每个输入序列的长度是否超过配置的最大输出token数
                    raise RuntimeError(
                        f"Input with size {tensor_length} exceeds maximum length of {self._config.max_out_tokens}. Please increase max_tokens in the DeepSpeed Inference Config."
                    )
        # 调用底层生成方法
        return self.module.generate(*inputs, **kwargs)
    # 使用指定后端的参数 编译模块
    def compile(self, backend=get_accelerator().get_compile_backend(), compile_kwargs={}) -> None:
        """
        Compile the module using the specified backend and kwargs.
        """
        if not is_compile_supported():  # 检查当前pytorch版本是否支持编译
            raise RuntimeError("compile is not supported in your version of PyTorch.")

        if self._is_compiled: # 避免重复编译
            return

        # Avoid graph breaks
        deepspeed.utils.nvtx.enable_nvtx = False # 禁用NVTX功能
        self.module.compile(backend=backend, **compile_kwargs) # 调用底层模块的编译方法
        self._is_compiled = True

    @property
    def is_compiled(self) -> bool:  # 编译状态属性
        return self._is_compiled
