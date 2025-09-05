import torch 
from torch import Tensor
import time
from dataclasses import dataclass
import collections
from collections import UserDict
from typing import Deque, Set, Dict

from deepspeed import comm as dist
from deepspeed.runtime.zero.partitioned_param_coordinator import (  
    debug_rank0,
    PartitionedParameterCoordinator,  
    ZeRoTraceMode,  
    get_all_parameters,  
    InflightParamRegistry,
    iter_params
)

from deepspeed.moe.utils import has_moe_layers  
from deepspeed.moe.layer import MoE  
from deepspeed.moe.experts import Experts  
from deepspeed.moe.sharded_moe import get_selected_experts  
from deepspeed.utils import z3_leaf_module

import logging
from deepspeed.utils.logging import logger
from deepspeed.runtime.zero.offload_config import OffloadDeviceEnum
from deepspeed.runtime.zero.partition_parameters import *

from deepspeed.accelerator import get_accelerator

# 仅在启用 NVMe 卸载功能时才是必需的（已粘一份）
from deepspeed.runtime.swap_tensor.partitioned_param_swapper import PartitionedParamStatus

from deepspeed.runtime.zero.partitioned_param_profiler import PartitionedParameterProfiler
# 调试 没写暂时可禁
# from deepspeed.utils.debug import debug_param2name_id_shape
import deepspeed.runtime.compiler as compiler
from deepspeed.runtime.compiler import is_compiling


# # 设置日志级别为 DEBUG  
# logger.setLevel(logging.DEBUG)  
# logging.basicConfig(level=logging.DEBUG)  

class ExpertAwareParameterCoordinator(PartitionedParameterCoordinator):  
    def __init__(  
        self,  
        prefetch_bucket_sz: int,  
        max_reuse_distance_in_numel: int,  
        max_available_parameters_in_numel: int,  
        allgather_stream,  
        inflight_param_registry,  
         root_module=None,
        prefetch_nvme: bool = False,  
        timers=None, 
        zero_quantized_weights=False,  
        zero_quantized_nontrainable_weights=False,  
        fast_sharding_for_leaf_module=False,  
        log_trace_cache_warnings=False,  
    ) -> None:  
        # 调用父类初始化  
        super().__init__(  
            prefetch_bucket_sz=prefetch_bucket_sz,  
            max_reuse_distance_in_numel=max_reuse_distance_in_numel,  
            max_available_parameters_in_numel=max_available_parameters_in_numel,  
            allgather_stream=allgather_stream,  
            inflight_param_registry=inflight_param_registry,  
            prefetch_nvme=prefetch_nvme,  
            timers=timers,  
            zero_quantized_weights=zero_quantized_weights,  
            zero_quantized_nontrainable_weights=zero_quantized_nontrainable_weights,  
            fast_sharding_for_leaf_module=fast_sharding_for_leaf_module,  
            log_trace_cache_warnings=log_trace_cache_warnings,  
        )  
          
        # === 专家感知的调度组件 (MoE-Infinity 启发) === 
        self._root_module = root_module  
        
        # 模块发现和调度映射  
        # self.module_param_map = {}  # 模块名到参数的映射  
        # self.gate_modules = set()  # 门控模块集合  
        # self.expert_modules = {}  # 专家模块映射 {expert_id: module}  
        # self.moe_layers = []  # MoE 层列表  
        
        # 缓存模块名称映射，避免重复查找  
        self._module_name_cache = {}  
        self._expert_id_cache = {}  
        self._module_name_cache_built = False  # 标记缓存是否已构建  
        
        # 预计算的专家ID映射，避免运行时查找
        self._expert_id_map = {}  # 预计算的专家ID映射  
        self._layer_id_map = {}   # 预计算的层ID映射  
        self._precompute_expert_mappings()  
          
        # 动态专家调度策略 (MoE-Infinity 核心)  
        # self.current_expert_indices = []  # 当前激活的专家索引  
        self.expert_activation_history = {}  # 专家激活历史，用于预测  
        # self.max_activation_history_size = 100  # 可配置的历史记录最大长度
        # self.routing_predictions = {}  # 基于历史的路由预测  
        # self.cached_experts = set()   # 跟踪GPU中缓存的专家参数
          
        # 智能预取调度 (替代传统预取桶)  
        # self.module_prefetch_enabled = True  
        # self.predictive_prefetch_enabled = True  # 基于历史模式的预测性预取  
        # self.prefetch_window_size = 4  # 预取窗口大小  
        # self.expert_prefetch_queue = []  # 专家参数预取队列  
          
        # 调度优化参数  
        # self.expert_scheduling_policy = "demand_driven"  # 按需调度策略  
        # self.gate_priority_boost = True  # 门控模块优先级提升  
        # self.expert_batch_scheduling = False  # 专家批量调度  
          
        # 调度统计和监控  
        self.expert_fetch_count = {}  # 每个专家的获取次数  
        self.gate_fetch_latency = []  # 门控获取延迟统计  
        self.expert_fetch_latency = {}  # 专家获取延迟统计  
        # self.scheduling_decisions = []  # 调度决策历史 
        
        
    @compiler.disable
    @instrument_w_nvtx
    @torch.no_grad()
    def fetch_sub_module(self, current_submodule: Module, forward: bool) -> None: # 负责为子模块获取所需的参数
        # print(f"[DEBUG] fetch_sub_module called for: {current_submodule.__class__.__name__}") 
        if current_submodule.__class__.__name__ == 'PhimoeSparseMoeBlock':  
            return 
        """负责为子模块获取所需的参数"""  
        if current_submodule.__class__.__name__ == 'PhimoeBlockSparseTop2MLP':
            # 使用MoE感知的智能调度  
            self._fetch_expert_module_intelligently(current_submodule, forward)   
        else:  
            # 使用父类的传统预取桶机制  
            super().fetch_sub_module(current_submodule, forward)
    
    
    def _fetch_expert_module_intelligently(self, current_submodule: Module, forward: bool) -> None:
        """基于门控输出智能预取专家模块参数"""  
        #print(f"[DEBUG] Intelligent expert fetch for: {current_submodule.__class__.__name__}")  
        layer_id = self._get_layer_id(current_submodule) 
        # 获取当前激活的专家索引  
        expert_indices = self._get_expert_indices_from_gate(current_submodule)  
        
        if not expert_indices:  
            #print(f"[DEBUG] No expert indices found, skipping fetch for {current_submodule.__class__.__name__}")  
            return  
        
        # 展平expert_indices，处理二维和一维情况  
        flattened_indices = self._flatten_expert_indices(expert_indices)  
        # print(f"[DEBUG] Flattened expert indices: {flattened_indices}")  
        
        # 找到当前专家模块在父容器中的索引  
        expert_id = self._get_expert_id_from_module(current_submodule)  
        
        if expert_id is None:  
            #print(f"[DEBUG] Could not determine expert ID for {current_submodule.__class__.__name__}, falling back to traditional fetch")   
            return  
        
        # 只有当前专家在激活列表中时才预取参数  
        if expert_id in flattened_indices:  
            # print(f"[DEBUG] Expert {expert_id} is active, fetching parameters using super().fetch_sub_module()")  
            start_time = time.perf_counter()
            # 直接使用父类的fetch_sub_module方法  
            super().fetch_sub_module(current_submodule, forward)  
            end_time = time.perf_counter()
            
            # expert_key = f"layer_{layer_id}_expert_{expert_id}"  
            expert_key = (layer_id, expert_id)      
            # 统计信息  
            self.expert_fetch_count[expert_key] = self.expert_fetch_count.get(expert_key, 0) + 1  
            if expert_key not in self.expert_fetch_latency:  
                self.expert_fetch_latency[expert_key] = []  
            self.expert_fetch_latency[expert_key].append(end_time - start_time)
       
        else:  
            pass 
            #print(f"[DEBUG] Expert {expert_id} is not active, skipping parameter fetch")  
       
    
    def _precompute_expert_mappings(self):  
        """预计算所有专家模块的ID映射"""  
        if not self._root_module:  
            return  
        
        for name, module in self._root_module.named_modules():  
            if module.__class__.__name__ == 'PhimoeBlockSparseTop2MLP' and 'experts.' in name:  
                try:  
                    # 提取专家ID  
                    expert_id = int(name.split('experts.')[-1].split('.')[0])  
                    self._expert_id_map[module] = expert_id  
                    
                    # 同时预计算层ID  
                    if 'layers.' in name:  
                        layer_id = int(name.split('layers.')[1].split('.')[0])  
                        self._layer_id_map[module] = layer_id  
                        
                except (ValueError, IndexError):  
                    continue  
        
        print(f"[EXPERT_COORDINATOR] Precomputed mappings for {len(self._expert_id_map)} expert modules")  
    
    def _get_expert_id_from_module(self, expert_module: Module) -> int:  
        """从专家模块获取其在父容器中的索引 - 预计算优化版本"""  
        # 直接从预计算的映射中获取  
        return self._expert_id_map.get(expert_module, None)  

    def _is_expert_module(self, current_submodule):    
        """判断当前子模块是否为专家模块 - 简化版本"""    
        # 1. DeepSpeed MoE 类    
        if isinstance(current_submodule, (Experts, MoE)):    
            return True    
          
        # 2. 通过模块路径直接判断    
        module_path = self._get_module_path(current_submodule)    
        if module_path and 'experts' in module_path:    
            return True    
          
        # 3. 基于类名的简单检测    
        class_name = type(current_submodule).__name__.lower()    
        if 'experts' in class_name:    
            return True    
          
        return False    
    

    
    def _build_module_name_cache(self) -> None:  
        """一次性构建模块名称缓存"""  
        if self._module_name_cache_built or not self._root_module:  
            return  
        
        for name, mod in self._root_module.named_modules():  
            self._module_name_cache[mod] = name  
        
        self._module_name_cache_built = True  
            

    # def _flatten_expert_indices(self, expert_indices) -> set:  
    #     """展平专家索引，处理二维和一维情况"""  
    #     if not expert_indices:  
    #         return set()  
        
    #     # 使用更高效的展平方式  
    #     if isinstance(expert_indices[0], (list, tuple)):  
    #         return set().union(*expert_indices)  # 更高效的展平  
    #     else:  
    #         return set(expert_indices)
    def _flatten_expert_indices(self, expert_indices) -> set:  
        """展平专家索引 - 优化版本"""  
        if not expert_indices:  
            return set()  
        
        # 预分配集合大小以减少重新分配  
        if isinstance(expert_indices[0], (list, tuple)):  
            # 估算总大小  
            estimated_size = sum(len(sublist) for sublist in expert_indices if sublist)  
            if estimated_size == 0:  
                return set()  
            # 使用 itertools.chain 更高效  
            import itertools  
            return set(itertools.chain.from_iterable(expert_indices))  
        else:  
            return set(expert_indices)
        
    
    def _get_expert_indices_from_gate(self, expert_module: Module) -> List[int]:  
        """从门控网络获取激活的专家索引""" 
        #print(f"[GATE_DEBUG] Processing module: {current_submodule.__class__.__name__}")
         
        # 直接从回调设置的属性中获取  
        if hasattr(self, 'expert_indices') and self.expert_indices:  
            return self.expert_indices  
        
        # 回退方案：返回空列表  
        return []
    
    def _get_layer_id(self, submodule: Module) -> int:  
        """从子模块中提取层ID - 预计算优化版本"""  
        # 优先使用预计算的映射  
        if submodule in self._layer_id_map:  
            return self._layer_id_map[submodule]  
        
        # 回退到原有的查找方式（用于非专家模块）  
        if hasattr(self, '_root_module') and self._root_module:  
            for name, module in self._root_module.named_modules():  
                if module is submodule and 'layers.' in name:  
                    try:  
                        layer_id = int(name.split('layers.')[1].split('.')[0])  
                        # 缓存结果  
                        self._layer_id_map[submodule] = layer_id  
                        return layer_id  
                    except (ValueError, IndexError):  
                        pass  
        return None
    
    
    def _update_expert_activation_history(self, expert_indices: List[int], layer_id: int) -> None:  
        """更新专家激活历史"""   
        # 展平嵌套列表（prompt阶段：嵌套的每个token[,]）
        if expert_indices and isinstance(expert_indices[0], (list, tuple)):  
            flat_indices = [idx for sublist in expert_indices for idx in sublist]  
        else:  
            flat_indices = expert_indices   
             
        layer_key = f'layer_{layer_id}'  
        if layer_key not in self.expert_activation_history:  
            self.expert_activation_history[layer_key] = {}  
        
        # 记录每个激活的专家
        for expert_idx in flat_indices:
            expert_key = f'expert_{expert_idx}'  
            if expert_key not in self.expert_activation_history[layer_key]:  
                self.expert_activation_history[layer_key][expert_key] = 0  
            self.expert_activation_history[layer_key][expert_key] += 1
     
        
    def release_sub_module(self, submodule: Module, forward=False) -> None:  
        """释放子模块参数，对专家模块进行特殊处理"""  
        if submodule.__class__.__name__ == 'PhimoeSparseMoeBlock':  
            return 
        
        if submodule.__class__.__name__ == 'PhimoeBlockSparseTop2MLP':  
            # 获取专家ID和索引  
            expert_id = self._get_expert_id_from_module(submodule)  
            expert_indices = self._get_expert_indices_from_gate(submodule)  
            flattened_indices = self._flatten_expert_indices(expert_indices)  
            layer_id = self._get_layer_id(submodule)  # 需要实现这个函数  
            
            # 只有激活的专家才释放参数  
            if expert_id in flattened_indices:  
                # print(f"[DEBUG] Expert {expert_id} is active, releasing parameters using super().release_sub_module()")  
                
                start_time = time.time()  
                # 直接使用父类的release_sub_module方法  
                super().release_sub_module(submodule, forward)  
                end_time = time.time()  
                
                expert_key = (layer_id, expert_id)
                
                # 统计信息  
                if not hasattr(self, 'expert_release_count'):  
                    self.expert_release_count = {}  
                if not hasattr(self, 'expert_release_latency'):  
                    self.expert_release_latency = {}  
                    
                self.expert_release_count[expert_key] = self.expert_release_count.get(expert_key, 0) + 1  
                if expert_key not in self.expert_release_latency:  
                    self.expert_release_latency[expert_key] = []  
                self.expert_release_latency[expert_key].append(end_time - start_time)  
            else:
                pass   
                #print(f"[DEBUG] Expert {expert_id} is not active, skipping parameter release")  
        else:  
            # 常规模块使用传统释放策略  
            super().release_sub_module(submodule, forward)
            
        # 调试信息  
        #print(f"[EXPERT_COORDINATOR] Released {released_count}/{total_params} parameters for experts {expert_indices} in layer {layer_id}")
        

