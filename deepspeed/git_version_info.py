# # Copyright (c) Microsoft Corporation.
# # SPDX-License-Identifier: Apache-2.0

# # DeepSpeed Team


# try:
#     #  This is populated by setup.py
#     from .git_version_info_installed import *  # noqa: F401 # type: ignore
# except ModuleNotFoundError:
#     import os
#     if os.path.isfile('version.txt'):
#         # Will be missing from checkouts that haven't been installed (e.g., readthedocs)
#         version = open('version.txt', 'r').read().strip()
#     else:
#         version = "0.0.0"
#     git_hash = '[none]'
#     git_branch = '[none]'

#     from .ops.op_builder.all_ops import ALL_OPS
#     installed_ops = dict.fromkeys(ALL_OPS.keys(), False)
#     accelerator_name = ""
#     torch_info = {'version': "0.0", "cuda_version": "0.0", "hip_version": "0.0"}

# # # compatible_ops list is recreated for each launch
# # from .ops.op_builder.all_ops import ALL_OPS

# # compatible_ops = dict.fromkeys(ALL_OPS.keys(), False)
# # for op_name, builder in ALL_OPS.items():
# #     op_compatible = builder.is_compatible()
# #     compatible_ops[op_name] = op_compatible
# #     compatible_ops["deepspeed_not_implemented"] = False

# # -----------------zyr修改跳过兼容性检查-------------------
# # compatible_ops list is recreated for each launch  
# from .ops.op_builder.all_ops import ALL_OPS  
  
# import os  
# compatible_ops = dict.fromkeys(ALL_OPS.keys(), False)  
  
# if os.getenv("DS_BUILD_OPS", "1") == "0" or os.getenv("DS_SKIP_CUDA_CHECK", "0") == "1":  
#     # 跳过所有兼容性检查，设置所有操作为不兼容  
#     for op_name in ALL_OPS.keys():  
#         compatible_ops[op_name] = False  
# else:  
#     # 正常进行兼容性检查  
#     for op_name, builder in ALL_OPS.items():  
#         op_compatible = builder.is_compatible()  
#         compatible_ops[op_name] = op_compatible  
  
# compatible_ops["deepspeed_not_implemented"] = False


# Copyright (c) Microsoft Corporation.  
# SPDX-License-Identifier: Apache-2.0  
  
# DeepSpeed Team  
  
try:  
    #  This is populated by setup.py  
    from .git_version_info_installed import *  # noqa: F401 # type: ignore  
except ModuleNotFoundError:  
    import os  
    if os.path.isfile('version.txt'):  
        # Will be missing from checkouts that haven't been installed (e.g., readthedocs)  
        version = open('version.txt', 'r').read().strip()  
    else:  
        version = "0.0.0"  
    git_hash = '[none]'  
    git_branch = '[none]'  
  
    # 先检查环境变量，再决定是否导入ALL_OPS  
    import os  
    if os.getenv("DS_BUILD_OPS", "1") == "0" or os.getenv("DS_SKIP_CUDA_CHECK", "0") == "1":  
        # 跳过导入，直接设置空字典  
        installed_ops = {}  
        compatible_ops = {}  
    else:  
        from .ops.op_builder.all_ops import ALL_OPS  
        installed_ops = dict.fromkeys(ALL_OPS.keys(), False)  
        # compatible_ops 在后面设置  
      
    accelerator_name = ""  
    torch_info = {'version': "0.0", "cuda_version": "0.0", "hip_version": "0.0"}  
  
# compatible_ops list is recreated for each launch    
import os    
if os.getenv("DS_BUILD_OPS", "1") == "0" or os.getenv("DS_SKIP_CUDA_CHECK", "0") == "1":    
    # 跳过所有兼容性检查，设置所有操作为不兼容    
    compatible_ops = {}  
else:    
    # 只有在需要时才导入ALL_OPS  
    if 'ALL_OPS' not in locals():  
        from .ops.op_builder.all_ops import ALL_OPS  
    compatible_ops = dict.fromkeys(ALL_OPS.keys(), False)    
    for op_name, builder in ALL_OPS.items():    
        op_compatible = builder.is_compatible()    
        compatible_ops[op_name] = op_compatible    
    
compatible_ops["deepspeed_not_implemented"] = False