from torch.utils.cpp_extension import load  
import torch  
import os  
  
sources = [  
    'deepspeed/ops/csrc/aio/py_lib/deepspeed_py_io_handle.cpp',  
    'deepspeed/ops/csrc/aio/py_lib/deepspeed_py_aio.cpp',  
    'deepspeed/ops/csrc/aio/py_lib/deepspeed_py_aio_handle.cpp',  
    'deepspeed/ops/csrc/aio/py_lib/deepspeed_aio_thread.cpp',  
    'deepspeed/ops/csrc/aio/common/deepspeed_aio_utils.cpp',  
    'deepspeed/ops/csrc/aio/common/deepspeed_aio_common.cpp',  
    'deepspeed/ops/csrc/aio/common/deepspeed_aio_types.cpp',  
    'deepspeed/ops/csrc/aio/py_lib/deepspeed_cpu_op.cpp',  
    'deepspeed/ops/csrc/aio/py_lib/deepspeed_aio_op_desc.cpp',  
    'deepspeed/ops/csrc/aio/py_lib/deepspeed_py_copy.cpp',  
    'deepspeed/ops/csrc/aio/py_lib/deepspeed_pin_tensor.cpp',  
    'deepspeed/ops/csrc/aio/py_lib/py_ds_aio.cpp'  
]  
  
include_dirs = [  
    'deepspeed/ops/csrc/aio/py_lib',  
    'deepspeed/ops/csrc/aio/common'  
]  
  
module = load(  
    name='deepspeed_aio',  
    sources=sources,  
    extra_include_paths=include_dirs,  
    extra_cflags=['-Wall', '-O0', '-std=c++17', '-shared', '-fPIC', '-Wno-reorder', '-fopenmp'],  
    extra_ldflags=['-laio', '-fopenmp'],  
    verbose=True  
)  
print('编译成功！')