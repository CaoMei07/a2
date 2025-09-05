import torch  
from transformers import AutoModelForCausalLM, AutoTokenizer  
  
# 加载模型和分词器  
model = AutoModelForCausalLM.from_pretrained(  
    "/home/mint/Projects/ZipMoE/framework/models/deepseek-moe-16b-base",  
    trust_remote_code=True,  
    torch_dtype=torch.float16  
)  
tokenizer = AutoTokenizer.from_pretrained(  
    "/home/mint/Projects/ZipMoE/framework/models/deepseek-moe-16b-base",  
    trust_remote_code=True  
)  
  
# 准备测试输入  
test_text = "用中文讲一下python里logging的用法"  
inputs = tokenizer(test_text, return_tensors="pt")  
  
# 查找真正的 MoE 门控层（从第1层开始）  
gate_layer = None  
for name, module in model.named_modules():  
    # 查找 MoEGate 类型的模块，而不是 Linear 的 gate_proj  
    if ('mlp.gate' in name and   
        type(module).__name__ == 'MoEGate' and  # 确保是 MoEGate 类型  
        'layers.1' in name):  # 从第1层开始查找  
        gate_layer = module  
        print(f"找到真正的门控层: {name}, 类型: {type(module).__name__}")  
        if hasattr(module, 'weight'):  
            print(f"门控权重形状: {module.weight.shape}")  
        print(f"专家数量: {module.n_routed_experts}")  
        print(f"Top-K: {module.top_k}")  
        break  
  
if gate_layer is None:  
    print("未找到 MoEGate，打印所有包含 'gate' 的模块：")  
    for name, module in model.named_modules():  
        if 'gate' in name.lower():  
            print(f"模块: {name} -> 类型: {type(module).__name__}")  
            if hasattr(module, 'weight'):  
                print(f"  权重形状: {module.weight.shape}")  
else:  
    # 测试门控输出  
    with torch.no_grad():  
        # 获取模型的嵌入层输出作为测试输入  
        hidden_states = model.model.embed_tokens(inputs.input_ids)  
        print(f"输入隐藏状态形状: {hidden_states.shape}")  
          
        # 直接调用门控层  
        gate_output = gate_layer(hidden_states)  
          
        print(f"\n=== 门控输出分析 ===")  
        print(f"输出类型: {type(gate_output)}")  
          
        if isinstance(gate_output, tuple):  
            print(f"元组长度: {len(gate_output)}")  
              
            # 根据 DeepSeek MoEGate 源码，返回 (topk_idx, topk_weight, aux_loss)  
            topk_idx = gate_output[0]  # 专家索引  
            topk_weight = gate_output[1]  # 专家权重  
            aux_loss = gate_output[2] if len(gate_output) > 2 else None  # 辅助损失  
              
            print(f"\n--- topk_idx (专家索引) ---")  
            print(f"类型: {type(topk_idx)}")  
            print(f"形状: {topk_idx.shape}")  
            print(f"数据类型: {topk_idx.dtype}")  
            print(f"内容样本: {topk_idx[:3]}")  # 显示前3个token的专家选择  
              
            # 验证专家索引范围  
            min_idx = topk_idx.min().item()  
            max_idx = topk_idx.max().item()  
            print(f"专家索引范围: [{min_idx}, {max_idx}]")  
              
            # 统计每个专家被选中的次数  
            unique_experts, counts = torch.unique(topk_idx, return_counts=True)  
            print(f"被激活的专家: {unique_experts.tolist()}")  
            print(f"激活次数: {counts.tolist()}")  
              
            print(f"\n--- topk_weight (专家权重) ---")  
            print(f"类型: {type(topk_weight)}")  
            print(f"形状: {topk_weight.shape}")  
            print(f"数据类型: {topk_weight.dtype}")  
            print(f"权重范围: [{topk_weight.min():.4f}, {topk_weight.max():.4f}]")  
            print(f"权重样本: {topk_weight[:3]}")  
              
            print(f"\n--- aux_loss (辅助损失) ---")  
            if aux_loss is not None:  
                print(f"类型: {type(aux_loss)}")  
                if hasattr(aux_loss, 'shape'):  
                    print(f"形状: {aux_loss.shape}")  
                print(f"值: {aux_loss}")  
            else:  
                print("辅助损失为 None (可能在推理模式)")  
          
        # 展平专家索引用于你的协调器  
        print(f"\n=== 为协调器准备的专家索引 ===")  
        if isinstance(gate_output, tuple) and len(gate_output) >= 1:  
            topk_idx = gate_output[0]  
            expert_indices = topk_idx.cpu().tolist()  
              
            # 展平二维列表  
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
              
            print(f"展平后的唯一专家索引: {unique_experts}")  
            print(f"激活的专家数量: {len(unique_experts)}")  
              
            # 验证范围  
            if unique_experts:  
                if max(unique_experts) >= gate_layer.n_routed_experts or min(unique_experts) < 0:  
                    print(f"⚠️  警告: 专家索引超出预期范围 [0, {gate_layer.n_routed_experts-1}]")  
                else:  
                    print(f"✅ 专家索引在有效范围内 [0, {gate_layer.n_routed_experts-1}]")  
  
print("\n=== 测试完成 ===")