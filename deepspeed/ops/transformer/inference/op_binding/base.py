# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from ..config import DeepSpeedInferenceConfig

# from deepspeed.ops.op_builder import InferenceBuilder  # TODO：CUDA 内核构建器


class BaseOp(torch.nn.Module):
    inference_module = None

    def __init__(self, config: DeepSpeedInferenceConfig):
        super(BaseOp, self).__init__()
        self.config = config
        if BaseOp.inference_module is None:
            builder = InferenceBuilder()  # CUDA 内核构建器
            BaseOp.inference_module = builder.load()  # 延迟加载（lazy loading）模式