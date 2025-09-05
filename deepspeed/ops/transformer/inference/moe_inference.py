# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import json
import math
import torch
from torch.autograd import Function
import torch.nn as nn
from .ds_attention import DeepSpeedSelfAttention  
from .config import DeepSpeedInferenceConfig
# from .op_binding import SoftmaxOp, VectorMatMulOp, GELUGemmOp   
# from .op_binding.bias_residual import BiasResidualOp
# from .op_binding.einsum_sec_sm_ecm import EinsumSecSmEcmOp
# from .op_binding.layer_norm import LayerNormOp
# from .op_binding.moe_res_matmul import MoEResMatmulOp
# from deepspeed import comm as dist
from ....moe.sharded_moe import TopKGate    




class DeepSpeedMoEInferenceConfig(DeepSpeedInferenceConfig):
    """Initialize the DeepSpeed Transformer Config.
        Arguments:
            hidden_size: The hidden size of the transformer layer
            intermediate_size: The intermediate size of the feed-forward part of transformer layer
            heads: The number of heads in the self-attention of the transformer layer
            num_hidden_layers: The number of transformer layers
            layer_norm_eps: The epsilon value for the layer norm
            local_rank: Optional: The rank of GPU running the transformer kernel, it is not required
                to use if the model already set the current device, otherwise need to set it
                so that the transformer kernel can work on the right device
            mp_size (optional): This argument is mainly used to create the parameters on the kernel side
                using model-parallel architecture. If the client model already takes care of this, there is no
                need to pass this argument.
            fp16: Enable half-precision computation
            bf16: Enable bf16 floating point computation
            pre_layer_norm: Select between Pre-LN or Post-LN transformer architecture
            stochastic_mode:  Enable for high performance, please note that this flag has some level of
                non-determinism and can produce different results on different runs.  However, we have seen
                that by enabling it, the pretraining tasks such as BERT are not affected and can obtain
                a high accuracy level. On the other hand, for the downstream tasks, such as fine-tuning, we recommend
                to turn it off in order to be able to reproduce the same result through the regular kernel execution.

            scale_attention: If true, both q and k are scaled by 1/sqrt(attention_heads) before attention computation.
            return_tuple: if True, returns the transformer output as a tuple, otherwise returns as a tensor
    """

    def __init__(self,
                 hidden_size=-1,
                 intermediate_size=-1,
                 heads=-1,
                 num_hidden_layers=-1,
                 layer_norm_eps=1e-12,
                 local_rank=-1,
                 mp_size=1,
                 fp16=False,
                 bf16=False,
                 q_int8=False,
                 pre_layer_norm=True,
                 stochastic_mode=False,
                 scale_attention=True,
                 triangular_masking=True,
                 local_attention=False,
                 window_size=256,
                 return_tuple=True,
                 moe_experts=1,
                 global_experts=1,
                 k=1,
                 capacity_factor=1.,
                 eval_capacity_factor=1.,
                 min_capacity=1,
                 noisy_gate_policy=None,
                 drop_tokens=True,
                 use_rts=False,
                 mlp_type='standard',
                 scale_attn_by_inverse_layer_idx=False):
        super(DeepSpeedMoEInferenceConfig,
              self).__init__(hidden_size, (intermediate_size if intermediate_size > 0 else 4 * hidden_size), heads,
                             num_hidden_layers, layer_norm_eps, local_rank, mp_size, fp16, bf16, q_int8,
                             pre_layer_norm, stochastic_mode, scale_attention, triangular_masking, local_attention,
                             window_size, return_tuple)
        self.moe_experts = moe_experts  # 本地专家数量
        self.k = k                      # top-k
        self.capacity_factor = capacity_factor  # 容量因子，控制专家的处理能力
        self.eval_capacity_factor = eval_capacity_factor
        self.min_capacity = min_capacity             # 最小容量设置
        self.noisy_gate_policy = noisy_gate_policy   # 噪声门控策略
        self.drop_tokens = drop_tokens               # 是否丢弃超出容量的token
        self.use_rts = use_rts                       # 是否使用RTS
        self.global_experts = global_experts  # 全局专家总数
        self.mlp_type = mlp_type              # mlp类型(标准和残差)
        self.scale_attn_by_inverse_layer_idx = scale_attn_by_inverse_layer_idx
    # 配置加载方法
    @classmethod
    def from_dict(cls, json_object):  # 从字典加载配置
        config = DeepSpeedInferenceConfig()
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):  # 从JSON文件加载配置
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))


class DeepSpeedMLPFunction(Function):  # 自定义的 torch.autograd.Function
    # 重载了前向和后向
    @staticmethod
    def forward(ctx, input, inter_w, inter_b, config, output_b, output_w, q_scales, q_groups, merge_count, mp_group,
                async_op, gelu_gemm_func, vector_matmul_func):
        if config.q_int8:  # 使用INT8量化，执行Linear1 + GELU，再Linear2，也是量化GEMM
            intermediate = gelu_gemm_func(input, inter_w, inter_b, config.epsilon, q_scales[2],
                                          (q_groups * (2**merge_count)), config.pre_layer_norm)
            output = vector_matmul_func(intermediate, output_w, q_scales[3], q_groups, (merge_count))
        else: # 普通推理（无量化）
            output = gelu_gemm_func(input, inter_w, inter_b, output_w, config.epsilon, config.pre_layer_norm, async_op)
        if mp_group is not None and dist.get_world_size(group=mp_group) > 1:
            dist.all_reduce(output, group=mp_group, async_op=async_op) # 聚合多个GPU的输出

        return output + output_b  # 加上偏置

    @staticmethod
    def backward(ctx, grad_output):  # 显式禁止反向传播
        raise RuntimeError('You are running with DeepSpeed Inference mode. \
                            Please switch to Training mode for running backward!')


class DeepSpeedMoEMLP(nn.Module):

    def __init__(self, config, q_scales=None, q_groups=1, merge_count=1, mlp_extra_grouping=False, mp_group=None):
        super(DeepSpeedMoEMLP, self).__init__()

        self.config = config
        self.attn_nw = nn.Parameter(torch.Tensor(self.config.hidden_size))
        self.attn_nb = nn.Parameter(torch.Tensor(self.config.hidden_size))
        interm_size = self.config.intermediate_size // (1 if mp_group is None else dist.get_world_size(group=mp_group))
        self.inter_w = nn.Parameter(torch.Tensor(self.config.hidden_size, interm_size))
        self.inter_b = nn.Parameter(torch.Tensor(interm_size))
        self.output_w = nn.Parameter(torch.Tensor((interm_size), self.config.hidden_size))
        self.output_b = nn.Parameter(torch.Tensor(self.config.hidden_size))

        # used for quantization
        self.q_scales = q_scales
        self.q_groups = q_groups * 2 if mlp_extra_grouping else q_groups
        self.merge_count = int(math.log2(merge_count))
        self.mp_group = mp_group
        # 两个函数通常是用 Triton / CUDA / C++ 编写的 fused kernel
        # self.gelu_gemm_func = GELUGemmOp(self.config) # GELU(Linear1(x))
        # self.vector_matmul_func = VectorMatMulOp(self.config)  # Linear2(intermediate)

    def forward(self, input, async_op=False):
        return DeepSpeedMLPFunction.apply(input, self.inter_w, self.inter_b, self.config, self.output_b, self.output_w,
                                          self.q_scales, self.q_groups, self.merge_count, self.mp_group, async_op,
                                          self.gelu_gemm_func, self.vector_matmul_func)


class DeepSpeedMoEInference(nn.Module):
    """Initialize the DeepSpeed MoE Transformer Layer.
        Arguments:
            layer_id: The layer index starting from 0, e.g. if model has 24 transformer layers,
                layer_id will be 0,1,2...23 when each layer object is instantiated
            config: An object of DeepSpeedInferenceConfig
            mp_group: Model parallelism group initialized on the modeling side.
            quantize_scales: This argument groups all the layers' scales used for quantization
            quantize_groups: Number of groups used for quantizing the model
            merge_count: Shows the number of model-parallel checkpoints merged before running inference.
                We use this argument to control the quantization scale for the model parameters if a bigger
                quantize-grouping than 1 is used.
            mlp_extra_grouping: This flag is used to show a 2x higher number of groups used for the MLP part
                of a Transformer layer. We use this feature for quantization to reduce the convergence impact
                for specific downstream tasks.
    """
    layer_id = 0

    def __init__(self,
                 config,
                 mp_group=None,
                 ep_group=None,
                 expert_mp_group=None,
                 quantize_scales=None,
                 quantize_groups=1,
                 merge_count=1,
                 mlp_extra_grouping=False):
        super(DeepSpeedMoEInference, self).__init__()

        self.config = config
        self.config.layer_id = DeepSpeedMoEInference.layer_id
        # 断言不支持bf16
        assert self.config.dtype != torch.bfloat16, "DeepSpeed MoE Transformer Inference not yet tested for bfloat support"
        # 递增全局层ID计数器
        DeepSpeedMoEInference.layer_id += 1
        # 参数配置
        self.attention = DeepSpeedSelfAttention(self.config, mp_group, quantize_scales, quantize_groups, merge_count)
        self.attn_nw = nn.Parameter(torch.Tensor(self.config.hidden_size))
        self.attn_nb = nn.Parameter(torch.Tensor(self.config.hidden_size))

        self.norm_w = nn.Parameter(torch.Tensor(self.config.hidden_size))
        self.norm_b = nn.Parameter(torch.Tensor(self.config.hidden_size))
        # 残差MLP处理
        if config.mlp_type == 'residual':
            self.res_mlp = DeepSpeedMoEMLP(config, quantize_scales, quantize_groups, merge_count, mlp_extra_grouping,
                                           mp_group)
            self.res_coef = nn.Parameter(torch.Tensor(self.config.hidden_size, 2))
            self.coef_func = SoftmaxOp(self.config)
            self.vector_matmul_func = VectorMatMulOp(self.config)

        config.mp_size = 1  # 创建多个专家MLP，数量由moe.experts决定
        self.mlp = nn.ModuleList(
            DeepSpeedMoEMLP(config, quantize_scales, quantize_groups, merge_count, mlp_extra_grouping, expert_mp_group)
            for i in range(self.config.moe_experts))
        # 门控机制初始化
        self.moe_gate = TopKGate(self.config.hidden_size, self.config.global_experts, self.config.k,
                                 self.config.capacity_factor, self.config.eval_capacity_factor,
                                 self.config.min_capacity, self.config.noisy_gate_policy, self.config.drop_tokens,
                                 self.config.use_rts, self.ep_group)
        # 进程组和操作符设置
        self.ep_group = ep_group
        self.mp_group = mp_group
        self.expert_mp_group = expert_mp_group

        print("DeepSpeed MoE Transformer Inference config is ", self.config.__dict__)

        # self.bias_residual_func = BiasResidualOp(self.config)
        # self.ds_layernorm = LayerNormOp(self.config)
        # self.einsum_sec_sm_ecm = EinsumSecSmEcmOp(self.config)
        # self.moe_res_matmul = MoEResMatmulOp(self.config)
    # 残差系数函数 计算残差MLP的混合系数
    def res_coef_func(self, inp, async_op):
        inp = self.vector_matmul_func(inp, self.res_coef, async_op)
        return self.coef_func(inp, torch.empty(1), False, False, False, 256, async_op)
    # MOE门控函数  执行门控计算
    def moe_gate_einsum(self, attention_output):
        _, combined_weights, dispatch_mask, _ = self.moe_gate(
            attention_output.view(-1, self.config.hidden_size),
            None,
        )
        # dispatched_attention = self.einsum_sec_sm_ecm(dispatch_mask.type_as(attention_output),
        #                                               attention_output.view(-1, self.config.hidden_size))
        dispatched_attention = torch.einsum('sec,sm->ecm', dispatch_mask.type_as(attention_output),   
                                   attention_output.view(-1, self.config.hidden_size))
        return dispatched_attention, combined_weights
    # 将输入分别分发给不同的专家MLP模块执行 并将它们的输出合并
    def expert_exec(self, dispatched_input):
        dispatched_input = dispatched_input.reshape(self.config.global_experts // self.config.moe_experts,
                                                    self.config.moe_experts, -1, self.config.hidden_size)

        chunks = dispatched_input.chunk(self.config.moe_experts, dim=1) # 按moe_expert维度分块，每个专家一块
        expert_outputs = torch.empty((
            self.config.moe_experts,
            chunks[0].shape[0],
        ) + chunks[0].shape[2:],
                                     dtype=dispatched_input.dtype,
                                     device=dispatched_input.device)
        for chunk, expert in zip(chunks, range(len(self.mlp))):
            expert_outputs[expert] = self.mlp[expert](chunk.view(-1, dispatched_input.shape[-2],
                                                                 dispatched_input.shape[-1]))
        return expert_outputs
    # 处理专家并行组之间的all-to-all通信 每个进程都把自己的一部分数据发给所有人，同时也从所有人那里收到自己的一份数据
    def _alltoall(self, dispatched_attention):
        if dist.get_world_size(group=self.ep_group) > 1:
            dispatched_input = torch.empty_like(dispatched_attention)
            dist.all_to_all_single(dispatched_input, dispatched_attention, group=self.ep_group)
            return dispatched_input
        else:
            return dispatched_attention
    # 输出缩放，根据门控权重对专家输出进行加权组合
    def scale_expert_output(self, attention_output, expert_output, combined_weights):
        combined_output = torch.matmul(
            combined_weights.type_as(attention_output).reshape(combined_weights.shape[0], -1),
            expert_output.reshape(-1, expert_output.shape[-1]))
        return combined_output.reshape(attention_output.shape)
    # 前向传播主函数
    def forward(self,
                input,
                input_mask=None,
                attention_mask=None,
                head_mask=None,
                layer_past=None,
                get_key_value=False,
                get_present=False,
                encoder_output=None,
                enc_dec_attn_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                use_cache=False,
                output_attentions=False):
        get_present = (get_present or get_key_value or use_cache)
        input_mask = input_mask if attention_mask is None else attention_mask
        input_type = input.dtype
        # 数据类型转换
        if (self.config.dtype in [torch.float16, torch.int8]) and input_type == torch.float:
            input = input.half()

        with torch.no_grad():
            attention_output = self.attention(input, input_mask, head_mask, layer_past, get_present,
                                              encoder_hidden_states, encoder_attention_mask, output_attentions,
                                              self.norm_w, self.norm_b)

            if get_present:  # 根据不同的输出需求（缓存、注意力权重）处理返回
                attention_output, p_key, p_value = attention_output[0:3]
                presents = (p_key, p_value)
            elif output_attentions:
                attention_output, _, _, context_output = attention_output[0:4]
            else:
                attention_output = attention_output[0]
            # 标准 Transformer 架构中注意力层的残差连接
            residual_add = attention_output + self.attention.attn_ob
            #attention_output = self.ds_layernorm(residual_add, self.attn_nw, self.attn_nb, self.config.epsilon)
            attention_output = torch.nn.functional.layer_norm(residual_add, (self.config.hidden_size,),   
                                                 self.attn_nw, self.attn_nb, self.config.epsilon)
            #  MoE残差连接 mlp_type 配置控制
            if self.config.mlp_type == 'residual':
                res_mlp_out = self.res_mlp(attention_output, async_op=True)
                res_coef_out = self.res_coef_func(attention_output, async_op=True)
            # 专家模型并行处理
            if self.expert_mp_group is not None: # 收集所有设备上的注意力输出
                world_size = dist.get_world_size(group=self.expert_mp_group)
                gather_buffer = torch.empty(world_size * attention_output.numel(),
                                            dtype=attention_output.dtype,
                                            device=attention_output.device)
                dist.all_gather_into_tensor(gather_buffer, attention_output, group=self.expert_mp_group)
                attention_output = gather_buffer.view(-1, *attention_output.size()[1:])

            ############## MoE Gating + Experts ###############
            dispatched_attention, combined_weights = self.moe_gate_einsum(attention_output)  # 门控计算
            dispatched_input = self._alltoall(dispatched_attention)  # token 分发 专家并行通信，将 tokens 路由到对应专家
            expert_outputs = self.expert_exec(dispatched_input)  # 执行专家计算
            expert_output = self._alltoall(expert_outputs)       # 收集专家输出
            output = self.scale_expert_output(attention_output, expert_output, combined_weights)  # 根据门控权重缩放和组合专家输出
            ################################################
            # 处理专家并行的输出分割
            if self.expert_mp_group is not None:
                output = output.split(output.shape[0] // dist.get_world_size(group=self.expert_mp_group),
                                      dim=0)[dist.get_rank(group=self.expert_mp_group)]
            # 残差 MLP 混合
            if self.config.mlp_type == 'residual':
                # self.moe_res_matmul(res_mlp_out, res_coef_out, output)
                coef_t = res_coef_out.transpose(1, 2).contiguous()  
                coef1, coef2 = torch.split(coef_t, split_size_or_sections=coef_t.shape[-1] // 2, dim=-1)  
                output = res_mlp_out * coef1 + output * coef2
            # 偏置残差连接
            #output = self.bias_residual_func(output, residual_add, torch.empty(1))
            output = output + residual_add
            # 层归一化
            if not self.config.pre_layer_norm:
                output = self.ds_layernorm(output, self.norm_w, self.norm_b, self.config.epsilon)
            # 数据类型恢复
            if input_type != output.dtype:
                output = output.to(input_type)

        if get_present:
            output = (output, presents)

        if self.config.return_tuple:
            return output if type(output) is tuple else (output, )
        else:
            return output
