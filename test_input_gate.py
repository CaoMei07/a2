import torch  
import torch.nn.functional as F  
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM  
  
def test_gate_input_format():  
    """测试 DeepSeek MoE 门控层的输入格式"""  
      
    # 加载模型和分词器  
    # 加载模型和分词器  
    model = AutoModelForCausalLM.from_pretrained(  
        "/home/mint/Desktop/zipmoe/models/deepseek-moe-16b-base",  
        trust_remote_code=True,  
        torch_dtype=torch.float16  
    )  
    tokenizer = AutoTokenizer.from_pretrained(  
        "/home/mint/Desktop/zipmoe/models/deepseek-moe-16b-base",  
        trust_remote_code=True  
    )    
      
    # 准备测试输入  
    test_text = "Hello world, this is a test."  
    inputs = tokenizer(test_text, return_tensors="pt")  
      
    print(f"输入 token 数量: {inputs['input_ids'].shape}")  
      
    # 获取第一个 MoE 层的门控模块  
    moe_layer = None  
    gate_module = None  
      
    for name, module in model.named_modules():  
        if name.endswith('.mlp.gate') and 'MoEGate' in str(type(module)):  
            gate_module = module  
            moe_layer_name = name  
            print(f"找到门控模块: {name}, 类型: {type(module)}")  
            break  
      
    if gate_module is None:  
        print("未找到 MoEGate 模块")  
        return  
      
    # Hook 来捕获门控层的输入  
    input_shapes = []  
      
    def capture_input_hook(module, input, output):  
        if len(input) > 0:  
            input_tensor = input[0]  
            input_shapes.append(input_tensor.shape)  
            print(f"门控层输入形状: {input_tensor.shape}")  
            print(f"输入数据类型: {input_tensor.dtype}")  
            print(f"输入维度数: {input_tensor.dim()}")  
              
            # 检查是否是预期的二维格式  
            if input_tensor.dim() == 1:  
                print("⚠️  警告: 输入是一维张量，这可能导致维度不匹配错误")  
            elif input_tensor.dim() == 2:  
                print("✅ 输入是二维张量，格式正确")  
            else:  
                print(f"❓ 输入是 {input_tensor.dim()} 维张量")  
      
    # 注册 hook  
    hook = gate_module.register_forward_hook(capture_input_hook)  
      
    try:  
        # 执行前向传播  
        print("\n开始前向传播...")  
        with torch.no_grad():  
            outputs = model(**inputs)  
        print("前向传播完成")  
          
    except Exception as e:  
        print(f"前向传播出错: {e}")  
        print(f"错误类型: {type(e)}")  
          
    finally:  
        # 移除 hook  
        hook.remove()  
      
    # 分析结果  
    if input_shapes:  
        print(f"\n捕获到的输入形状: {input_shapes}")  
        for i, shape in enumerate(input_shapes):  
            print(f"调用 {i+1}: {shape}")  
    else:  
        print("\n未捕获到任何输入（可能门控层未被调用）")  
  
def test_manual_gate_call():  
    """手动测试门控层调用"""  
    print("\n=== 手动门控层测试 ===")  
      
    # 创建测试数据  
    batch_size = 1  
    seq_len = 5  
    hidden_dim = 2048  
      
    # 测试不同的输入格式  
    test_cases = [  
        ("一维张量", torch.randn(5)),  
        ("二维张量 [seq_len, hidden_dim]", torch.randn(seq_len, hidden_dim)),  
        ("三维张量 [batch, seq_len, hidden_dim]", torch.randn(batch_size, seq_len, hidden_dim)),  
        ("二维张量 [batch*seq_len, hidden_dim]", torch.randn(batch_size * seq_len, hidden_dim))  
    ]  
      
    for case_name, test_input in test_cases:  
        print(f"\n测试 {case_name}: {test_input.shape}")  
          
        # 模拟线性层调用  
        try:  
            weight = torch.randn(64, hidden_dim if test_input.dim() > 1 else test_input.shape[0])  
            result = F.linear(test_input, weight)  
            print(f"✅ 成功: 输出形状 {result.shape}")  
        except Exception as e:  
            print(f"❌ 失败: {e}")  
  
if __name__ == "__main__":  
    print("=== DeepSeek MoE 门控层输入格式测试 ===")  
    test_gate_input_format()  
    test_manual_gate_call()