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


class ExpertAwareParameterCoordinator_hook(PartitionedParameterCoordinator):  
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
        moe_block_class_name: str = 'DeepseekMoE',    #'PhimoeSparseMoeBlock',  
        expert_class_name: str = 'DeepseekMLP'    #'PhimoeBlockSparseTop2MLP'
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

        # 存储可配置的模块类名  
        self._moe_block_class_name = moe_block_class_name  
        self._expert_class_name = expert_class_name    
        
        # 缓存模块名称映射，避免重复查找  
        self._module_name_cache = {}  
        self._expert_id_cache = {}  
        self._module_name_cache_built = False  # 标记缓存是否已构建  
        
        # 预计算的专家ID映射，避免运行时查找
        self._expert_id_map = {}  # 预计算的专家ID映射  
        self._layer_id_map = {}   # 预计算的层ID映射  
        self._expert_layer_to_module = {}   # (layer_id, expert_id) -> module  
        self._precompute_expert_mappings()  
          
        # 动态专家调度策略 (MoE-Infinity 核心)  
        # self.current_expert_indices = []  # 当前激活的专家索引  
        self.expert_activation_history = {}  # 专家激活历史，用于预测  
        # self.max_activation_history_size = 100  # 可配置的历史记录最大长度
        # self.routing_predictions = {}  # 基于历史的路由预测  
        # self.cached_experts = set()   # 跟踪GPU中缓存的专家参数
          
        # 冷热专家统计属性


        # 调度统计和监控  
        self.expert_fetch_count = {}  # 每个专家的获取次数  
        #self.gate_fetch_latency = []  # 门控获取延迟统计  
        self.expert_truefetch_latency = []  # 非GPU数据真实传输时间
        self.expert_fetch_latency = {}  # 专家获取延迟统计  
        # self.scheduling_decisions = []  # 调度决策历史 
        
        # === 专家参数持久化缓存系统 ===
        self._expert_cache = {}  # {expert_key: cached_params}  
        self._expert_cache_status = {}  # {expert_key: cache_status}  
        self._expert_cache_hits = 0  
        self._expert_cache_misses = 0  
        self._max_cached_experts = 30  # 可配置的最大缓存专家数量  
        self._expert_lru_order = collections.OrderedDict()  # LRU 缓存顺序  
        
        # 专家参数强制持久化标记  
        self._persistent_expert_params = set()  # 强制持久化的专家参数  
        self._expert_persistence_enabled = True  # 是否启用专家持久化

        # === 轻量级内存池组件 ===  
        # 只对高频使用的对象使用内存池  
        self._set_pool = []  
        self._list_pool = []  
        self._dict_pool = []
        self._max_pool_size = 10  # 较小的池大小，避免内存浪费  
        
        # 预分配一些对象到池中  
        for _ in range(3):  # 预分配少量对象  
            self._set_pool.append(set())  
            self._list_pool.append([]) 
            self._dict_pool.append({})  
        
        self._pool_stats = {  
            'set_hits': 0,  
            'set_misses': 0,  
            'list_hits': 0,  
            'list_misses': 0,  
            'dict_hits': 0,  
            'dict_misses': 0  
        } 
        
    @compiler.disable
    @instrument_w_nvtx
    @torch.no_grad()
    def fetch_sub_module(self, current_submodule: Module, forward: bool) -> None: # 负责为子模块获取所需的参数
        
        #print(f"[DEBUG] Executing: {current_submodule.__class__.__name__} - {getattr(current_submodule, 'ds_id', 'no_id')}")  
        # if current_submodule.__class__.__name__ == self._moe_block_class_name:  
        #     return 
        
        """负责为子模块获取所需的参数"""  
        if self._is_expert_module(current_submodule):  # Deepseek16b里的第0层没有moe而是一个DeepseekMLP同名的线性层 
            self._fetch_expert_module_intelligently(current_submodule, forward)   
        else:  
            # 使用父类的传统预取桶机制  
            super().fetch_sub_module(current_submodule, forward)

    
    def _fetch_expert_module_intelligently(self, current_submodule: Module, forward: bool) -> None:  
        """基于门控输出智能预取专家模块参数 - 缓存优化版本"""  
        layer_id = self._get_layer_id(current_submodule)  
        #if current_submodule.__class__.__name__ == 'DeepseekMLP':
        # if layer_id == 0:
        #     super().fetch_sub_module(current_submodule, forward)
        #     return
        expert_indices = self._get_expert_indices_from_gate(current_submodule)  
        
        if not expert_indices:  
            return  
        
        flattened_indices = self._flatten_expert_indices(expert_indices)  
        expert_id = self._get_expert_id_from_module(current_submodule)  
        
        if expert_id is None or expert_id not in flattened_indices:  
            return  
        
        expert_key = (layer_id, expert_id)  
        
        # 检查专家是否已缓存  
        if self._is_expert_cached(expert_key):  
            self._expert_cache_hits += 1  
            self._update_expert_lru(expert_key)  
            #print(f"[EXPERT_CACHE] Cache hit for expert {expert_key}")  
            
            # 即使缓存命中，仍需要确保参数可用  
            self._ensure_cached_expert_available(current_submodule)  
        else:  
            self._expert_cache_misses += 1  
            #print(f"[EXPERT_CACHE] Cache miss for expert {expert_key}")  
            
            # 正常获取参数  
            start_time = time.perf_counter()  
            super().fetch_sub_module(current_submodule, forward)  
            end_time = time.perf_counter()  
            
            # 缓存该专家参数  
            self._cache_expert_parameters(current_submodule, expert_key)  
            
            # 更新统计  
            self.expert_fetch_count[expert_key] = self.expert_fetch_count.get(expert_key, 0) + 1  
            if expert_key not in self.expert_fetch_latency:  
                self.expert_fetch_latency[expert_key] = []  
            self.expert_fetch_latency[expert_key].append(end_time - start_time)  
    
    # ---------------------------Klotski batch 热冷专家调整---------------
    


    # ----------------------------缓存 + 持久化--------------------------
    def _cache_expert_parameters(self, expert_module: Module, expert_key: tuple) -> None:  
        """缓存专家参数到 GPU 内存"""  
        if not self._expert_persistence_enabled:  
            return  
        
        # 检查缓存容量  
        if len(self._expert_cache) >= self._max_cached_experts:  
            # self._evict_least_recently_used_expert() 
            self._evict_least_frequently_used_expert() 
        
        # 强制参数持久化  
        cached_params = []  
        for param in iter_params(expert_module, recurse=z3_leaf_module(expert_module)):  
            # 设置持久化标志  
            param.ds_persist = True  
            param.ds_active_sub_modules.add(expert_module.ds_id)  
            self._persistent_expert_params.add(param)  
            cached_params.append(param)  
        
        # 更新缓存  
        self._expert_cache[expert_key] = cached_params  
        self._expert_cache_status[expert_key] = 'cached'  
        self._expert_lru_order[expert_key] = time.time()  
        
        #print(f"[EXPERT_CACHE] Cached expert {expert_key} with {len(cached_params)} parameters")  

    def _evict_least_recently_used_expert(self) -> None:  
        """驱逐最近最少使用的专家参数"""  
        if not self._expert_lru_order:  
            return  
        
        # 获取最旧的专家  
        oldest_expert_key = next(iter(self._expert_lru_order))  
        self._evict_expert_from_cache(oldest_expert_key)  

    def _evict_least_frequently_used_expert(self) -> None:  
        """驱逐最近最少使用的专家参数 - 改进版本"""  
        if not self._expert_lru_order:  
            return  
        
        # 找到使用频率最低的专家而不是最旧的  
        min_usage_expert = None  
        min_usage_count = float('inf')  
        
        for expert_key in self._expert_lru_order:  
            usage_count = self.expert_fetch_count.get(expert_key, 0)  
            if usage_count < min_usage_count:  
                min_usage_count = usage_count  
                min_usage_expert = expert_key  
        
        if min_usage_expert:  
            self._evict_expert_from_cache(min_usage_expert)

    def _evict_expert_from_cache(self, expert_key: tuple) -> None:  
        """从缓存中驱逐指定专家"""  
        if expert_key not in self._expert_cache:  
            return  
        
        cached_params = self._expert_cache[expert_key]  
        
        # 恢复参数的正常释放行为  
        for param in cached_params:  
            param.ds_persist = False  
            param.ds_active_sub_modules.clear()  
            self._persistent_expert_params.discard(param)  
            
            # 强制释放参数  
            if param.ds_status == ZeroParamStatus.AVAILABLE:  
                param.partition()  
        
        # 清理缓存记录  
        del self._expert_cache[expert_key]  
        del self._expert_cache_status[expert_key]  
        self._expert_lru_order.pop(expert_key, None)  
        
        #print(f"[EXPERT_CACHE] Evicted expert {expert_key} from cache")


    def _is_expert_cached(self, expert_key: tuple) -> bool:  
        """检查专家是否已缓存"""  
        return expert_key in self._expert_cache and self._expert_cache_status.get(expert_key) == 'cached'  
    
    def _update_expert_lru(self, expert_key: tuple) -> None:  
        """更新专家的 LRU 顺序"""  
        self._expert_lru_order[expert_key] = time.time()  
    
    def _ensure_cached_expert_available(self, expert_module: Module) -> None:  
        """确保缓存的专家参数可用"""  
        for param in iter_params(expert_module, recurse=z3_leaf_module(expert_module)):  
            param.ds_active_sub_modules.add(expert_module.ds_id)  
            # 如果参数不可用，强制获取  
            if param.ds_status != ZeroParamStatus.AVAILABLE:  
                super().fetch_sub_module(expert_module, True)  
                break

    def get_expert_cache_stats(self) -> dict:    
        """获取专家缓存统计信息"""    
        total_requests = self._expert_cache_hits + self._expert_cache_misses    
        hit_rate = self._expert_cache_hits / total_requests if total_requests > 0 else 0.0    
        
        stats = {    
            'cache_hits': self._expert_cache_hits,    
            'cache_misses': self._expert_cache_misses,    
            'hit_rate': hit_rate,    
            'cached_experts': len(self._expert_cache),    
            'max_cache_size': self._max_cached_experts,    
            'persistent_params': len(self._persistent_expert_params)    
        }  
        
        # 打印缓存统计信息  
        print(f"[EXPERT_CACHE_STATS] Cache hits: {self._expert_cache_hits}")  
        print(f"[EXPERT_CACHE_STATS] Cache misses: {self._expert_cache_misses}")  
        print(f"[EXPERT_CACHE_STATS] Hit rate: {hit_rate:.2%}")  
        print(f"[EXPERT_CACHE_STATS] Cached experts: {len(self._expert_cache)}/{self._max_cached_experts}")  
        print(f"[EXPERT_CACHE_STATS] Persistent params: {len(self._persistent_expert_params)}")  
        
        return stats
    
    def clear_expert_cache(self) -> None:  
        """清空专家缓存"""  
        for expert_key in list(self._expert_cache.keys()):  
            self._evict_expert_from_cache(expert_key)  
        
        #print("[EXPERT_CACHE] Cleared all cached experts")
    # ------------------------------------------------------------------



    # ----------------------------预测  +  预取------------------------------
    def _prefetch_next_layer_experts(self, current_layer_id: int, current_flattened_indices: set):  
        """预取下一层专家 - 使用已展平的索引"""  
        if current_layer_id is None:  
            return  
        
        next_layer_id = current_layer_id + 1  
        
        # 基于当前激活的专家预测下一层  
        predicted_experts = self._predict_next_layer_experts(next_layer_id, current_flattened_indices)  
        
        if predicted_experts:  
            # 限制预取数量，避免内存压力  
            for expert_id in predicted_experts[:2]:  # 只预取前2个最可能的专家  
                self._prefetch_expert_if_available(next_layer_id, expert_id)  
        #pass
    
    def _predict_next_layer_experts(self, layer_id: int, current_experts: set) -> list:  
        """基于历史模式预测下一层专家"""  
        layer_key = f'layer_{layer_id}'  
        
        # 优先使用历史激活数据  
        if hasattr(self, 'expert_activation_history') and layer_key in self.expert_activation_history:  
            # 按激活频率排序，返回最常用的专家ID  
            expert_freq = sorted(  
                self.expert_activation_history[layer_key].items(),  
                key=lambda x: x[1], reverse=True  
            )  
            return [int(expert_key.split('_')[1]) for expert_key, _ in expert_freq[:3]]  
        else:  
            # 回退策略：基于当前层的专家模式  
            # 假设相邻层可能使用相似的专家  
            return list(current_experts)[:2]  
    
    def _find_expert_module(self, layer_id: int, expert_id: int):  
        """查找指定层和专家ID的模块 - O(1)查找"""  
        return self._expert_layer_to_module.get((layer_id, expert_id), None)  
    
    def _prefetch_expert_if_available(self, layer_id: int, expert_id: int):  
        """如果专家模块存在且参数未加载，则进行预取"""  
        if not self._root_module:  
            return  
        
        # 使用 O(1) 查找方法  
        target_module = self._find_expert_module(layer_id, expert_id)  
        if target_module is None:  
            return  
        
        try:  
            # 直接调用父类的参数获取机制进行预取  
            # 父类会自动检查参数状态并只获取需要的参数  
            super().fetch_sub_module(target_module, forward=True)  
            #print(f"[PREFETCH] Attempted prefetch for expert ({layer_id}, {expert_id})")  
        except Exception as e:  
            print(f"[PREFETCH] Failed to prefetch expert ({layer_id}, {expert_id}): {e}")
    # ----------------------------------------------------------------------
    
    def _precompute_expert_mappings(self):    
        """预计算所有专家模块的ID映射 - 增强版本"""    
        if not self._root_module:    
            return    
        
        for name, module in self._root_module.named_modules():    
            if module.__class__.__name__ == self._expert_class_name and 'experts.' in name:    
                try:    
                    # 提取专家ID    
                    expert_id = int(name.split('experts.')[-1].split('.')[0])    
                    self._expert_id_map[module] = expert_id    
                    
                    # 同时预计算层ID    
                    if 'layers.' in name:    
                        layer_id = int(name.split('layers.')[1].split('.')[0])    
                        self._layer_id_map[module] = layer_id    
                        
                        # 创建反向映射，实现 O(1) 查找  
                        self._expert_layer_to_module[(layer_id, expert_id)] = module  
                        
                except (ValueError, IndexError):    
                    continue    
        
        #print(f"[EXPERT_COORDINATOR] Precomputed mappings for {len(self._expert_id_map)} expert modules")
    
    def _get_expert_id_from_module(self, expert_module: Module) -> int:  
        """从专家模块获取其在父容器中的索引 - 预计算优化版本"""  
        # 直接从预计算的映射中获取  
        return self._expert_id_map.get(expert_module, None)  

    def _is_expert_module(self, current_submodule: Module) -> bool:  
        """判断当前子模块是否为专家模块 - 简化版本"""   
        # 1. DeepSpeed MoE 类检测  
        if isinstance(current_submodule, (Experts, MoE)):  
            return True  
        
        # 2. 通过模块路径精确判断 - 只有 experts.数字 路径才是专家  
        module_path = self._get_module_path(current_submodule)  
        if module_path:  
            import re  
            # 匹配 experts.数字 模式  
            if re.search(r'\.experts\.\d+', module_path):  
                return True  
        
        return False
    
    def _get_module_path(self, module: Module) -> str:  
        """获取模块的完整路径名称"""  
        # 使用缓存优化查找  
        if not self._module_name_cache_built:  
            self._build_module_name_cache()  
        
        return self._module_name_cache.get(module, "")

    
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
        """展平专家索引 - 内存池优化版本，支持深度嵌套"""    
        if not expert_indices:    
            return set()    
        
        # 使用内存池获取可重用集合    
        result_set = self._get_reusable_set()    
        try:    
            # 使用栈来避免递归  
            stack = list(expert_indices)  
            while stack:  
                item = stack.pop()  
                if isinstance(item, (list, tuple)):  
                    stack.extend(item)  
                else:  
                    result_set.add(item)  
            
            # 创建返回集合的副本    
            return set(result_set)    
        finally:    
            # 返回到池中    
            self._return_reusable_set(result_set)
        
    
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
        """释放子模块参数，对缓存的专家模块进行特殊处理"""  
        if submodule.__class__.__name__ == self._moe_block_class_name:  
            return  
        
        if submodule.__class__.__name__ == self._expert_class_name:  
            expert_id = self._get_expert_id_from_module(submodule)  
            layer_id = self._get_layer_id(submodule)  
            expert_key = (layer_id, expert_id)  
            
            # 如果专家已缓存，不释放其参数  
            if self._is_expert_cached(expert_key):  
                #print(f"[EXPERT_CACHE] Protecting cached expert {expert_key} from release")  
                # 只移除活跃模块标记，但不释放参数  
                for param in iter_params(submodule, recurse=z3_leaf_module(submodule)):  
                    param.ds_active_sub_modules.discard(submodule.ds_id)  
                return  
            
            # 非缓存专家使用正常释放逻辑  
            expert_indices = self._get_expert_indices_from_gate(submodule)  
            if not expert_indices:  
                return  
            
            flattened_indices = self._flatten_expert_indices(expert_indices)  
            if expert_id is None or expert_id not in flattened_indices:  
                return  
            
            start_time = time.perf_counter()  
            super().release_sub_module(submodule, forward)  
            end_time = time.perf_counter()  
            
            # 更新释放统计  
            if not hasattr(self, 'expert_release_count'):  
                self.expert_release_count = {}  
            if not hasattr(self, 'expert_release_latency'):  
                self.expert_release_latency = {}  
            
            self.expert_release_count[expert_key] = self.expert_release_count.get(expert_key, 0) + 1  
            if expert_key not in self.expert_release_latency:  
                self.expert_release_latency[expert_key] = []  
            self.expert_release_latency[expert_key].append(end_time - start_time)  
        else:  
            super().release_sub_module(submodule, forward)
    
    # 内存池
    def _get_reusable_set(self) -> set:  
        """获取可重用的集合对象 - 带统计"""  
        if self._set_pool:  
            self._pool_stats['set_hits'] += 1  
            result = self._set_pool.pop()  
            result.clear()  
            return result  
        else:  
            self._pool_stats['set_misses'] += 1  
            return set()   
    
    def _return_reusable_set(self, obj: set):  
        """返回集合对象到池中"""  
        if obj is not None and len(self._set_pool) < self._max_pool_size:  
            obj.clear()  # 清空内容  
            self._set_pool.append(obj)  
        # 如果池已满或对象为空，让对象被垃圾回收
    
    def _get_reusable_list(self) -> list:  
        """获取可重用的列表对象"""  
        if self._list_pool:  
            result = self._list_pool.pop()  
            result.clear()  
            return result  
        return []  
    
    def _return_reusable_list(self, obj: list):  
        """返回列表对象到池中"""  
        if obj is not None and len(self._list_pool) < self._max_pool_size:  
            obj.clear()  
            self._list_pool.append(obj)
            
    def _get_reusable_dict(self) -> dict:  
        """获取可重用的字典对象"""  
        if self._dict_pool:  
            result = self._dict_pool.pop()  
            result.clear()  
            return result  
        return {}  
    
    def _return_reusable_dict(self, obj: dict):  
        """返回字典对象到池中"""  
        if obj is not None and len(self._dict_pool) < self._max_pool_size:  
            obj.clear()  
            self._dict_pool.append(obj)
    
    def get_pool_efficiency(self) -> dict:  
        """获取内存池使用效率统计"""  
        stats = {}  
        for obj_type in ['set', 'list', 'dict']:  
            hits = self._pool_stats[f'{obj_type}_hits']  
            misses = self._pool_stats[f'{obj_type}_misses']  
            total = hits + misses  
            if total > 0:  
                stats[f'{obj_type}_hit_rate'] = hits / total  
            else:  
                stats[f'{obj_type}_hit_rate'] = 0.0  
        return stats
    
    def _update_stats_with_pool(self, expert_key: tuple, duration: float):  
        """使用内存池优化的统计信息更新"""  
        if expert_key not in self.expert_fetch_count:  
            self.expert_fetch_count[expert_key] = 0  
            # 使用列表池而不是直接创建  
            self.expert_fetch_latency[expert_key] = self._get_reusable_list()  
        
        self.expert_fetch_count[expert_key] += 1  
        self.expert_fetch_latency[expert_key].append(duration)
        
        
    def _adjust_pool_sizes(self):  
        """智能动态调整池大小"""  
        for pool_name in ['set', 'list', 'dict']:  
            hits = self._pool_stats[f'{pool_name}_hits']  
            misses = self._pool_stats[f'{pool_name}_misses']  
            total = hits + misses  
            
            if total > 100:  # 有足够的样本数据  
                hit_rate = hits / total  
                pool = getattr(self, f'_{pool_name}_pool')  
                
                if hit_rate < 0.8 and len(pool) < self._max_pool_size:  
                    # 命中率低，增加池大小  
                    for _ in range(min(3, self._max_pool_size - len(pool))):  
                        if pool_name == 'set':  
                            pool.append(set())  
                        elif pool_name == 'list':  
                            pool.append([])  
                        elif pool_name == 'dict':  
                            pool.append({})  
                elif hit_rate > 0.95 and len(pool) > 3:  
                    # 命中率很高但池可能过大，适当减少  
                    pool.pop()