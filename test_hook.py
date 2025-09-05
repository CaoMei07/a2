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
  
class DeepSeekMoEHookTester:  
    """专门用于测试 DeepSeek MoE 门控钩子的类"""  
      
    def __init__(self, model):  
        self.model = model  
        self.hooks = []  
        self.expert_selections = {}  
        self.gate_outputs = {}  
          
    def setup_gate_hooks(self):  
        """为 DeepSeek MoE 门控层设置钩子"""  
        for name, module in self.model.named_modules():  
            # 精确匹配 DeepSeek MoE 门控  
            if (name.endswith('.mlp.gate') and   
                not name.endswith('.mlp.gate_proj') and  
                hasattr(module, '__class__') and   
                'Gate' in module.__class__.__name__):  
                  
                print(f"[SETUP] 找到 DeepSeek MoE 门控: {name} ({type(module).__name__})")  
                self._add_gate_hook(module, name)  
                  
    def _add_gate_hook(self, module, name):  
        """为门控层添加钩子"""  
        def gate_forward_hook(module, input, output):  
            print(f"[HOOK] 🔥 门控钩子触发: {name}")  
              
            try:  
                # 处理 DeepSeek MoEGate 输出格式: (topk_idx, topk_weight, aux_loss)  
                if isinstance(output, tuple) and len(output) >= 2:  
                    topk_idx = output[0]  # 专家索引  
                    topk_weight = output[1]  # 专家权重  
                    aux_loss = output[2] if len(output) > 2 else None  
                      
                    print(f"[HOOK] 输出格式: tuple, 长度={len(output)}")  
                    print(f"[HOOK] topk_idx 形状: {topk_idx.shape}, 类型: {topk_idx.dtype}")  
                    print(f"[HOOK] topk_weight 形状: {topk_weight.shape}")  
                      
                    if torch.is_tensor(topk_idx) and topk_idx.dtype in [torch.int32, torch.int64, torch.long]:  
                        # 转换为 CPU 并提取专家索引  
                        expert_indices = topk_idx.cpu().tolist()  
                          
                        # 展平二维列表并去重  
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
                          
                        # 存储结果  
                        self.expert_selections[name] = unique_experts  
                        self.gate_outputs[name] = {  
                            'topk_idx': topk_idx.cpu(),  
                            'topk_weight': topk_weight.cpu(),  
                            'aux_loss': aux_loss.cpu() if aux_loss is not None else None,  
                            'unique_experts': unique_experts  
                        }  
                          
                        print(f"[HOOK] {name}: 激活专家 {unique_experts}")  
                        print(f"[HOOK] 激活专家数量: {len(unique_experts)}")  
                          
                        # 验证专家索引范围  
                        if hasattr(module, 'n_routed_experts'):  
                            n_experts = module.n_routed_experts  
                            if unique_experts and (max(unique_experts) >= n_experts or min(unique_experts) < 0):  
                                print(f"[HOOK] ⚠️ 警告: 专家索引超出范围 [0, {n_experts-1}]")  
                          
                    else:  
                        print(f"[HOOK] ⚠️ topk_idx 不是整数张量: {type(topk_idx)}")  
                          
                else:  
                    print(f"[HOOK] ⚠️ 意外的输出格式: {type(output)}")  
                    if isinstance(output, tuple):  
                        print(f"[HOOK] 元组长度: {len(output)}")  
                        for i, item in enumerate(output):  
                            print(f"[HOOK] 元素 {i}: {type(item)}")  
                              
            except Exception as e:  
                print(f"[HOOK] ❌ 处理门控输出时出错: {e}")  
                import traceback  
                traceback.print_exc()  
                  
        # 注册钩子  
        hook_handle = module.register_forward_hook(gate_forward_hook)  
        self.hooks.append(hook_handle)  
        print(f"[SETUP] ✅ 为 {name} 注册钩子成功")  
          
    def cleanup_hooks(self):  
        """清理所有钩子"""  
        for hook in self.hooks:  
            hook.remove()  
        print(f"[CLEANUP] 清理了 {len(self.hooks)} 个钩子")  
        self.hooks.clear()  
          
    def get_results(self):  
        """获取钩子捕获的结果"""  
        return {  
            'expert_selections': self.expert_selections,  
            'gate_outputs': self.gate_outputs  
        }  
  
# 测试代码  
def test_deepseek_moe_hooks():  
    print("=== DeepSeek MoE 门控钩子测试 ===")  
      
    # 创建钩子测试器  
    hook_tester = DeepSeekMoEHookTester(model)  
      
    # 设置钩子  
    print("\n=== 设置门控钩子 ===")  
    hook_tester.setup_gate_hooks()  
      
    if not hook_tester.hooks:  
        print("❌ 未找到任何门控层，打印所有包含 'gate' 的模块：")  
        for name, module in model.named_modules():  
            if 'gate' in name.lower():  
                print(f"  {name} -> {type(module).__name__}")  
        return  
      
    # 准备测试输入  
    test_text = "用中文讲一下python里logging的用法"  
    inputs = tokenizer(test_text, return_tensors="pt")  
      
    print(f"\n=== 执行前向传播 ===")  
    print(f"输入文本: {test_text}")  
    print(f"输入 token 数量: {inputs.input_ids.shape[1]}")  
      
    try:  
        with torch.no_grad():  
            # 执行前向传播，触发钩子  
            outputs = model(**inputs)  
              
        print(f"\n=== 钩子捕获结果 ===")  
        results = hook_tester.get_results()  
          
        for gate_name, experts in results['expert_selections'].items():  
            print(f"\n门控层: {gate_name}")  
            print(f"  激活的专家: {experts}")  
            print(f"  专家数量: {len(experts)}")  
              
            if gate_name in results['gate_outputs']:  
                gate_info = results['gate_outputs'][gate_name]  
                print(f"  topk_idx 形状: {gate_info['topk_idx'].shape}")  
                print(f"  topk_weight 形状: {gate_info['topk_weight'].shape}")  
                if gate_info['aux_loss'] is not None:  
                    print(f"  aux_loss: {gate_info['aux_loss']}")  
                      
    except Exception as e:  
        print(f"❌ 前向传播出错: {e}")  
        import traceback  
        traceback.print_exc()  
          
    finally:  
        # 清理钩子  
        print(f"\n=== 清理钩子 ===")  
        hook_tester.cleanup_hooks()  
  
if __name__ == "__main__":  
    test_deepseek_moe_hooks()