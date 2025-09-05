# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from .abstract_accelerator import DeepSpeedAccelerator

# During setup stage torch may not be installed, pass on no torch will
# allow op builder related API to be executed.
try:
    import torch
except ImportError as e:
    pass

try:
    import oneccl_bindings_for_pytorch  # noqa: F401 # type: ignore
    oneccl_imported_p = True
except ImportError as e:
    oneccl_imported_p = False

import os


# accelerator for Intel CPU
class CPU_Accelerator(DeepSpeedAccelerator):

    def __init__(self):
        self._name = 'cpu'
        self._compile_backend = "inductor"
        if oneccl_imported_p:
            self._communication_backend_name = 'ccl'
        else:
            # fallback to gloo if oneccl_binding_for_pytorch is not installed
            self._communication_backend_name = 'gloo'
        try:
            import psutil   # 内存信息获取
            mem = psutil.Process().memory_info().rss  # 尝试使用 psutil 获取当前进程内存
            self.max_mem = mem
        except ImportError as e:
            self.max_mem = 0

    def is_synchronized_device(self):   # CPU 是同步设备
        return True

    def use_host_timers(self):          # 使用主机计时器
        return self.is_synchronized_device()

    def resolves_data_dependency(self): # 解决数据依赖
        return self.is_synchronized_device()

    def handles_memory_backpressure(self):  # 处理内存反压
        return self.is_synchronized_device()

    # Device APIs
    def device_name(self, device_index=None):
        return 'cpu'

    def device(self, device_index=None):
        return None

    def set_device(self, device_index):
        return

    def current_device(self):
        return os.environ.get('LOCAL_RANK', 0)

    def current_device_name(self):
        return 'cpu'

    # 因为我们的设备是Jetson系列，没有NUMA系统，所以改为：既兼容NUMA架构，也支持Jetson
    def device_count():
        # 手动指定（环境变量优先）
        device_count = int(os.environ.get('LOCAL_SIZE', 0))
        if device_count > 0:
            return device_count
        else:
            # 尝试使用 NUMA 节点数
            from zipmoe.utils.numa import get_numa_cores   
            numa_core_lists = get_numa_cores()
            if numa_core_lists:
                # 过滤掉重复或空的 NUMA core list
                numa_count = 0
                prev_core_list = []
                for core_list in numa_core_lists:
                    if len(core_list) > 0 and core_list != prev_core_list:
                        numa_count += 1
                        prev_core_list = core_list
                if numa_count > 0:
                    return numa_count

            # fallback: 返回 Python 当前可用 CPU 数（可能是容器限制），若要使用全部cpu查询psutil.cpu_count(logical=True)
            return os.cpu_count() or 1
    # ---------------------------------以下是zyr注释掉的------------
    # def device_count(self):
    #     device_count = int(os.environ.get('LOCAL_SIZE', 0))
    #     if device_count > 0:
    #         return device_count
    #     else:
    #         from deepspeed.utils.numa import get_numa_cores
    #         # Count NUMA node for number of cpu accelerators. On machine with HBM
    #         # In flat mode, HBM is in separate NUMA node with no cores on this node.
    #         # Ignore these NUMA nodes with no cores.
    #         numa_core_lists = get_numa_cores()
    #         if not numa_core_lists:
    #             return 1
    #         numa_count = 0
    #         prev_core_list = []
    #         for core_list in numa_core_lists: # 统计有效的 NUMA 节点数量
    #             if len(core_list) > 0 and core_list != prev_core_list:
    #                 numa_count += 1
    #                 prev_core_list = core_list
    #         return numa_count
    # ------------------------------------------------------------------

    def synchronize(self, device_index=None):  # CPU不需要同步操作，直接返回
        return

    # RNG APIs   # 主要是对 PyTorch 随机数 API 的封装
    def random(self):   
        return torch.random

    def set_rng_state(self, new_state, device_index=None): # 设置全局随机数状态
        if device_index is None:
            return torch.set_rng_state(new_state)
        return torch.set_rng_state(new_state, device_index)

    def get_rng_state(self, device_index=None): # 获取当前的随机数生成器状态
        return torch.get_rng_state()

    def manual_seed(self, seed):  # 手动设置随机数种子
        return torch.manual_seed(seed)

    def manual_seed_all(self, seed):
        return torch.manual_seed(seed)

    def initial_seed(self):  # 返回初始的随机数种子
        return torch.initial_seed()

    def default_generator(self, device_index):  # 返回默认的随机数生成器 
        return torch.default_generator

    # Streams/Events  在CPU上无意义
    @property
    def Stream(self):
        return None

    def stream(self, stream):  # 空上下文
        from deepspeed.runtime.utils import noop_context
        return noop_context()

    def current_stream(self, device_index=None):
        return None

    def default_stream(self, device_index=None):
        return None

    @property
    def Event(self):
        return None

    # Memory management
    def empty_cache(self):  # CPU不像GPU那样有显存缓存需要清理，操作系统会自动管理CPU内存
        return

    def get_rss(self):
        import psutil
        mem = psutil.Process().memory_info().rss  # 获取当前进程的RSS（Resident Set Size，常驻内存大小）
        if mem > self.max_mem:  # 更新最大值
            self.max_mem = mem
        return mem

    def reset_rss(self):   # 重置最大内存记录为当前内存使用量
        import psutil
        mem = psutil.Process().memory_info().rss
        self.max_mem = mem
        return mem

    def memory_allocated(self, device_index=None):  # 返回当前分配的内存
        return self.get_rss()

    def max_memory_allocated(self, device_index=None):
        self.get_rss()
        return self.max_mem

    def reset_max_memory_allocated(self, device_index=None):
        self.reset_rss()
        return

    def memory_cached(self, device_index=None):
        return self.get_rss()

    def max_memory_cached(self, device_index=None):
        self.get_rss()
        return self.max_mem

    def reset_max_memory_cached(self, device_index=None):
        self.reset_rss()
        return

    def memory_stats(self, device_index=None):  # 返回一个字典格式的内存统计信息
        mem = self.get_rss()
        mem_stat = {}
        mem_stat['allocated_bytes.all.current'] = mem
        mem_stat['allocated_bytes.all.peak'] = self.max_mem
        return mem_stat

    def reset_peak_memory_stats(self, device_index=None):
        self.reset_rss()
        return

    def memory_reserved(self, device_index=None):
        return self.get_rss()

    def max_memory_reserved(self, device_index=None):
        self.get_rss()
        return self.max_mem

    def total_memory(self, device_index=None):   # 返回系统总虚拟内存大小
        import psutil
        return psutil.virtual_memory().total   

    def available_memory(self, device_index=None):  # 返回系统可用内存大小
        import psutil
        return psutil.virtual_memory().available

    # Misc
    def amp(self):   # 返回PyTorch的CPU自动混合精度模块
        return torch.cpu.amp

    def is_available(self):
        return True

    def range_push(self, msg):
        # TODO itt is currently not supported yet
        # return torch.profiler.itt.range_push(msg)
        return

    def range_pop(self):
        # TODO itt is currently not supported yet
        # return torch.profiler.itt.range_pop()
        return

    def lazy_call(self, callback):
        return callback()

    def communication_backend_name(self):
        return self._communication_backend_name

    def is_triton_supported(self):  # CPU 不支持 Triton 内核
        return False

    # Data types
    def is_bf16_supported(self):
        return True

    def is_fp16_supported(self):
        try:
            if torch.ops.mkldnn._is_mkldnn_fp16_supported():
                return True
        except:
            return False

    def supported_dtypes(self):   # 支持的数据类型列表
        supported_dtypes = [torch.float, torch.bfloat16]
        if self.is_fp16_supported():
            supported_dtypes.append(torch.float16)
        return supported_dtypes

    # Graph operations
    def create_graph(self):   # CPU 加速器不支持CUDA图（CUDA Graph）功能
        return None

    def capture_to_graph(self, graph, pool=None, stream=None):
        from deepspeed.runtime.utils import noop_context
        return noop_context()

    def replay_graph(self, graph):
        return

    # Tensor operations  每个属性都直接返回对应的 PyTorch 张量类型
    @property
    def BFloat16Tensor(self):
        return torch.BFloat16Tensor

    @property
    def ByteTensor(self):    # 8位无符号整数
        return torch.ByteTensor

    @property
    def DoubleTensor(self):   # 64位浮点数
        return torch.DoubleTensor

    @property
    def FloatTensor(self):    # 32位浮点数
        return torch.FloatTensor

    @property
    def HalfTensor(self):     # 16位浮点数
        return torch.HalfTensor

    @property
    def IntTensor(self):      # 32位整数
        return torch.IntTensor

    @property
    def LongTensor(self):     # 64位整数
        return torch.LongTensor

    def pin_memory(self, tensor, align_bytes=1):  # 内存固定
        return tensor

    def is_pinned(self, tensor):   # 检查是否已固定到内存
        return tensor.is_pinned()

    def op_builder_dir(self):  # 返回操作构建器目录
        try:
            # is op_builder from deepspeed or a 3p version? this should only succeed if it's deepspeed
            # if successful this also means we're doing a local install and not JIT compile path
            from op_builder import __deepspeed__  # noqa: F401 # type: ignore
            return "op_builder.cpu"
        except ImportError:
            return "deepspeed.ops.op_builder.cpu"

    def on_accelerator(self, tensor):  # 检查张量是否在当前加速器（CPU）上
        device_str = str(tensor.device)
        if device_str.startswith('cpu'):  # 以cpu开头
            return True
        else:
            return False

    # create an instance of op builder and return, name specified by class_name
    def create_op_builder(self, op_name):
        builder_class = self.get_op_builder(op_name)   # 获取构建器类
        if builder_class is not None:
            return builder_class()   # 创建并返回构建器实例
        return None

    # return an op builder class, name specified by class_name
    def get_op_builder(self, class_name):
        try:
            # is op_builder from deepspeed or a 3p version? this should only succeed if it's deepspeed
            # if successful this also means we're doing a local install and not JIT compile path
            from op_builder import __deepspeed__  # noqa: F401 # type: ignore
            from op_builder.cpu import AsyncIOBuilder, CCLCommBuilder, ShareMemCommBuilder, FusedAdamBuilder, CPUAdamBuilder, NotImplementedBuilder
        except ImportError:
            from deepspeed.ops.op_builder.cpu import AsyncIOBuilder, CCLCommBuilder, ShareMemCommBuilder, FusedAdamBuilder, CPUAdamBuilder, NotImplementedBuilder

        if class_name == "CCLCommBuilder":
            return CCLCommBuilder
        elif class_name == "ShareMemCommBuilder":
            return ShareMemCommBuilder
        elif class_name == "FusedAdamBuilder":
            return FusedAdamBuilder
        elif class_name == "CPUAdamBuilder":
            return CPUAdamBuilder
        elif class_name == "AsyncIOBuilder":
            return AsyncIOBuilder
        else:
            # return a NotImplementedBuilder to avoid get NoneType[Name] in unit tests
            return NotImplementedBuilder

    def build_extension(self): # 返回PyTorch的C++扩展构建器，用于编译 CPU 操作的C++代码
        from torch.utils.cpp_extension import BuildExtension
        return BuildExtension

    def export_envs(self):  # 导出的环境变量列表
        return []

    # TODO: cpu's visible envs is confirmed, keep as CUDA_VISIBLE_DEVICES
    def visible_devices_envs(self):
        return ['CUDA_VISIBLE_DEVICES']
    # 设置设备可见性环境变量
    def set_visible_devices_envs(self, current_env, local_accelerator_ids):
        for env in self.visible_devices_envs():
            current_env[env] = ",".join(map(str, local_accelerator_ids))

    def get_compile_backend(self):  # 返回当前的编译后端（默认为 "inductor"）
        return self._compile_backend

    def set_compile_backend(self, backend):
        supported_backends = torch._dynamo.list_backends(exclude_tags=())
        if backend in supported_backends:
            self._compile_backend = backend
        else:
            raise ValueError(
                f"{backend} not supported by {self.device_name()}. Supported Backends are {supported_backends}")
