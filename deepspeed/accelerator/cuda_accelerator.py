# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import functools
import os
import pkgutil
import importlib
import sys

from .abstract_accelerator import DeepSpeedAccelerator
# During setup stage torch may not be installed, pass on no torch will
# allow op builder related API to be executed.
try:
    import torch.cuda
except ImportError:
    pass

# Delay import pynvml to avoid import error when CUDA is not available
pynvml = None


class CUDA_Accelerator(DeepSpeedAccelerator):

    def __init__(self):
        self._name = 'cuda'
        self._communication_backend_name = 'nccl' if sys.platform != 'win32' else 'gloo'
        self._compile_backend = "inductor"
        if pynvml is None:
            self._init_pynvml()
    # pynvml是NVIDIA提供的NVML（NVIDIA Management Library）的Python封装接口
    def _init_pynvml(self): 
        global pynvml
        try:
            import pynvml
        except ImportError:
            return
        try:
            pynvml.nvmlInit()
        except pynvml.NVMLError:
            pynvml = None
            return

    def is_synchronized_device(self):   # CUDA GPU 是异步设备
        return False

    def use_host_timers(self):          # 不使用主机计时器（而是使用 GPU 事件）
        return self.is_synchronized_device()

    def resolves_data_dependency(self): # 不会自动解决数据依赖关系
        return self.is_synchronized_device()

    def handles_memory_backpressure(self):  # 不会同步处理内存反压
        return self.is_synchronized_device()

    # Device APIs
    def device_name(self, device_index=None): # 返回设备名称，如'cuda'或'cuda:0'
        if device_index is None:
            return 'cuda'
        return 'cuda:{}'.format(device_index)

    def communication_backend_version(self):  # 返回NCCL通信后端版本
        return torch.cuda.nccl.version()

    def device(self, device_index=None):      # 创建CUDA设备上下文
        return torch.cuda.device(device_index)

    def set_device(self, device_index):       # 设置当前CUDA设备
        torch.cuda.set_device(device_index)

    def current_device(self):           # 获取当前设备索引
        return torch.cuda.current_device()

    def current_device_name(self):      # 获取当前设备名称字符串
        return 'cuda:{}'.format(torch.cuda.current_device())

    def device_count(self):         # 返回可用 CUDA 设备数量
        return torch.cuda.device_count()

    def synchronize(self, device_index=None):      # 同步指定设备上的操作
        return torch.cuda.synchronize(device_index)

    # RNG APIs
    def random(self):
        return torch.random

    def set_rng_state(self, new_state, device_index=None):
        if device_index is None:
            return torch.cuda.set_rng_state(new_state)

        return torch.cuda.set_rng_state(new_state, device_index)

    def get_rng_state(self, device_index=None):
        if device_index is None:
            return torch.cuda.get_rng_state()

        return torch.cuda.get_rng_state(device_index)

    def manual_seed(self, seed):
        return torch.cuda.manual_seed(seed)

    def manual_seed_all(self, seed):
        return torch.cuda.manual_seed_all(seed)

    def initial_seed(self):
        return torch.cuda.initial_seed()

    def default_generator(self, device_index):
        return torch.cuda.default_generators[device_index]

    # Streams/Events
    @property
    def Stream(self):
        return torch.cuda.Stream

    def stream(self, stream):    # 在指定的 CUDA 流中执行操作
        return torch.cuda.stream(stream)

    def current_stream(self, device_index=None):
        return torch.cuda.current_stream(device_index)

    def default_stream(self, device_index=None):
        return torch.cuda.default_stream(device_index)

    @property
    def Event(self):
        return torch.cuda.Event

    # Memory management
    def empty_cache(self):      # 清空 CUDA 内存缓存，释放未使用的缓存内存
        return torch.cuda.empty_cache()

    def memory_allocated(self, device_index=None):  # 返回当前设备已分配的内存量
        return torch.cuda.memory_allocated(device_index)

    def max_memory_allocated(self, device_index=None):  # 返回历史最大内存分配量
        return torch.cuda.max_memory_allocated(device_index)

    def reset_max_memory_allocated(self, device_index=None):  # 重置最大内存分配统计
        return torch.cuda.reset_max_memory_allocated(device_index)

    def memory_cached(self, device_index=None):    # 返回当前缓存的内存量
        return torch.cuda.memory_cached(device_index)

    def max_memory_cached(self, device_index=None):   # 返回历史最大缓存内存量
        return torch.cuda.max_memory_cached(device_index)

    def reset_max_memory_cached(self, device_index=None):   # 重置最大缓存内存统计
        return torch.cuda.reset_max_memory_cached(device_index)

    def memory_stats(self, device_index=None):      # 返回详细的内存统计信息
        if hasattr(torch.cuda, 'memory_stats'):
            return torch.cuda.memory_stats(device_index)

    def reset_peak_memory_stats(self, device_index=None):  # 重置峰值内存统计
        if hasattr(torch.cuda, 'reset_peak_memory_stats'):
            return torch.cuda.reset_peak_memory_stats(device_index)

    def memory_reserved(self, device_index=None):   # 跟踪保留内存
        if hasattr(torch.cuda, 'memory_reserved'):
            return torch.cuda.memory_reserved(device_index)

    def max_memory_reserved(self, device_index=None):
        if hasattr(torch.cuda, 'max_memory_reserved'):
            return torch.cuda.max_memory_reserved(device_index)

    def total_memory(self, device_index=None):   # 获取设备的总内存容量
        return torch.cuda.get_device_properties(device_index).total_memory

    def _get_nvml_gpu_id(self, torch_gpu_id):
        """
        credit: https://discuss.pytorch.org/t/making-pynvml-match-torch-device-ids-cuda-visible-devices/103020

        Remap torch device id to nvml device id, respecting CUDA_VISIBLE_DEVICES.

        If the latter isn't set return the same id
        """
        # if CUDA_VISIBLE_DEVICES is used automagically remap the id since pynvml ignores this env var
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            ids = list(map(int, os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")))
            return ids[torch_gpu_id]  # remap 将PyTorch设备ID映射到实际的NVML设备ID
        else:
            return torch_gpu_id

    def available_memory(self, device_index=None):
        if pynvml: # 如果pynvml可用，直接从NVIDIA驱动获取准确的可用内存
            if device_index is None:
                device_index = self.current_device()
            handle = pynvml.nvmlDeviceGetHandleByIndex(self._get_nvml_gpu_id(device_index))
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return info.free
        else:
            return self.total_memory(device_index) - self.memory_allocated(device_index)

    # Data types
    def is_bf16_supported(self):
        if not torch.cuda.is_available():
            return True
        return torch.cuda.is_bf16_supported()

    def is_fp16_supported(self):
        if not torch.cuda.is_available():
            return True
        # See https://docs.nvidia.com/deeplearning/tensorrt/support-matrix/index.html#hardware-precision-matrix
        # FP16 on compute capability 6.x is deprecated
        allow_deprecated_fp16 = os.environ.get('DS_ALLOW_DEPRECATED_FP16', '0') == '1'
        major, _ = torch.cuda.get_device_capability()   # 获取 GPU 的计算能力版本
        if major >= 7:
            return True
        elif major == 6 and allow_deprecated_fp16:  # 环境变量DS_ALLOW_DEPRECATED_FP16允许用户强制启用弃用的FP16支持
            return True
        else:
            return False

    def supported_dtypes(self):    # 返回当前设备支持的所有数据类型列表
        supported_dtypes = [torch.float]
        if self.is_fp16_supported():
            supported_dtypes.append(torch.half)
        if self.is_bf16_supported():
            supported_dtypes.append(torch.bfloat16)
        return supported_dtypes

    # Misc  杂项
    def amp(self):   # 自动混合精度模块
        if hasattr(torch.cuda, 'amp'):
            return torch.cuda.amp
        return None

    def is_available(self):   # 检查 CUDA 是否可用
        return torch.cuda.is_available()

    def range_push(self, msg):  # NVIDIA NVTX 性能分析的范围标记
        if hasattr(torch.cuda.nvtx, 'range_push'):
            return torch.cuda.nvtx.range_push(msg)

    def range_pop(self):
        if hasattr(torch.cuda.nvtx, 'range_pop'):
            return torch.cuda.nvtx.range_pop()

    def lazy_call(self, callback):   # 延迟执行回调函数
        return torch.cuda._lazy_call(callback)

    def communication_backend_name(self):
        return self._communication_backend_name

    def is_triton_supported(self):
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            return True
        else:
            return False

    # Graph operations
    def create_graph(self):   # 创建一个新的 CUDA 图对象
        return torch.cuda.CUDAGraph()

    def capture_to_graph(self, graph, pool=None, stream=None): # 捕获CUDA操作到指定的图中
        return torch.cuda.graph(graph, pool, stream)   # 创建捕获上下文

    def replay_graph(self, graph):  # 执行之前捕获的操作序列
        graph.replay()
        return

    # Tensor operations
    # 每个属性都返回一个偏函数，预设了数据类型和设备为'cuda'
    @property
    def BFloat16Tensor(self):  
        return functools.partial(torch.tensor, dtype=torch.bfloat16, device='cuda')

    @property
    def ByteTensor(self):
        return functools.partial(torch.tensor, dtype=torch.uint8, device='cuda')

    @property
    def DoubleTensor(self):
        return functools.partial(torch.tensor, dtype=torch.double, device='cuda')

    @property
    def FloatTensor(self):
        return functools.partial(torch.tensor, dtype=torch.float, device='cuda')

    @property
    def HalfTensor(self):
        return functools.partial(torch.tensor, dtype=torch.half, device='cuda')

    @property
    def IntTensor(self):
        return functools.partial(torch.tensor, dtype=torch.int, device='cuda')

    @property
    def LongTensor(self):
        return functools.partial(torch.tensor, dtype=torch.long, device='cuda')

    def pin_memory(self, tensor, align_bytes=1):  # 将张量固定到内存
        return tensor.pin_memory()

    def is_pinned(self, tensor):
        return tensor.is_pinned()

    def on_accelerator(self, tensor):
        device_str = str(tensor.device)
        if device_str.startswith('cuda:'):
            return True
        else:
            return False

    def op_builder_dir(self):
        try:
            # is op_builder from deepspeed or a 3p version? this should only succeed if it's deepspeed
            # if successful this also means we're doing a local install and not JIT compile path
            from op_builder import __deepspeed__  # noqa: F401 # type: ignore
            return "op_builder"
        except ImportError:
            return "deepspeed.ops.op_builder"

    # dict that holds class name <--> class type mapping i.e.
    # 'AsyncIOBuilder': <class 'op_builder.async_io.AsyncIOBuilder'>
    # this dict will be filled at init stage
    class_dict = None

    def _lazy_init_class_dict(self):
        if self.class_dict is not None:
            return
        else:
            self.class_dict = {}
            # begin initialize for create_op_builder()
            # put all valid class name <--> class type mapping into class_dict
            op_builder_dir = self.op_builder_dir()
            op_builder_module = importlib.import_module(op_builder_dir)
            op_builder_absolute_path = os.path.dirname(op_builder_module.__file__)
            for _, module_name, _ in pkgutil.iter_modules([op_builder_absolute_path]):
                # avoid self references,
                # skip sub_directories which contains ops for other backend(cpu, npu, etc.).
                if module_name != 'all_ops' and module_name != 'builder' and not os.path.isdir(
                        os.path.join(op_builder_absolute_path, module_name)):
                    module = importlib.import_module("{}.{}".format(op_builder_dir, module_name))
                    for member_name in module.__dir__():
                        if member_name.endswith(
                                'Builder'
                        ) and member_name != "OpBuilder" and member_name != "CUDAOpBuilder" and member_name != "TorchCPUOpBuilder":  # avoid abstract classes
                            if not member_name in self.class_dict:
                                self.class_dict[member_name] = getattr(module, member_name)
            # end initialize for create_op_builder()

    # create an instance of op builder and return, name specified by class_name
    def create_op_builder(self, class_name):
        self._lazy_init_class_dict()
        if class_name in self.class_dict:
            return self.class_dict[class_name]()  # 请求的类名存在，创建并返回实例
        else:
            return None

    # return an op builder class, name specified by class_name
    def get_op_builder(self, class_name):  # 返回对应的类对象
        self._lazy_init_class_dict()
        if class_name in self.class_dict:
            return self.class_dict[class_name]
        else:
            return None

    def build_extension(self):
        from torch.utils.cpp_extension import BuildExtension
        return BuildExtension

    def export_envs(self):
        return ['NCCL']

    def visible_devices_envs(self):
        return ['CUDA_VISIBLE_DEVICES']

    def set_visible_devices_envs(self, current_env, local_accelerator_ids):
        for env in self.visible_devices_envs():
            current_env[env] = ",".join(map(str, local_accelerator_ids))

    def get_compile_backend(self):
        return self._compile_backend

    def set_compile_backend(self, backend):
        supported_backends = torch._dynamo.list_backends(exclude_tags=())
        if backend in supported_backends:
            self._compile_backend = backend
        else:
            raise ValueError(
                f"{backend} not supported by {self.device_name()}. Supported Backends are {supported_backends}")
