# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from deepspeed.runtime.config_utils import DeepSpeedConfigModel


#########################################
# BFLOAT16 support
#########################################
# BFLOAT16 feature. By default, this feature is not enabled.
# Users can configure in ds_config.json as below example:
BFLOAT16_FORMAT = '''
BFLOAT16 parameters should be of the format:
"bf16": {
  "enabled": true,
}
'''
BFLOAT16 = "bf16"
BFLOAT16_OLD = "bfloat16"  # keeping for backwards compatibility


def get_bfloat16_config(param_dict):
    bf16_config_dict = param_dict.get(BFLOAT16, None)
    if bf16_config_dict is None:
        bf16_config_dict = param_dict.get(BFLOAT16_OLD, {})
    return DeepSpeedBF16Config(**bf16_config_dict)


class DeepSpeedBF16Config(DeepSpeedConfigModel):
    """
    For bfloat16 configuration
    """

    enabled: bool = False
    """
    Enable bfloat16 mixed-precision training/inference
    """


#########################################
# FP16 support
#########################################
# FP16 feature. By default, this feature is not enabled.
# Users can configure in ds_config.json as below example:
FP16_FORMAT = '''
FP16 parameters should be of the format:
"fp16": {
  "enabled": true,
  "auto_cast": false,
}
'''
FP16 = "fp16"


def get_float16_config(param_dict):
    fp16_config_dict = param_dict.get(FP16, {})
    return DeepSpeedFP16Config(**fp16_config_dict)


class DeepSpeedFP16Config(DeepSpeedConfigModel):
    """
    For float16 configuration
    """

    enabled: bool = False
    """
    Enable fp16 mixed-precision training/inference
    """

    auto_cast: bool = False
    """
    Automatically cast inputs to fp16
    """

   