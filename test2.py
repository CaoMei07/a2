#!/usr/bin/env python3  
"""  
DeepSeek MoE 16B 门控输出格式测试文件  
测试目标：分析 DeepSeek MoE 模型中门控层的输出格式和专家选择机制  
"""  
  
import os  
import torch  
import torch.nn as nn  
from transformers import AutoModelForCausalLM, AutoTokenizer  
import deepspeed.comm as dist  
from deepspeed.runtime.zero.partition_parameters import Init  
from deepspeed.utils import set_z3_leaf_modules  
  
# 环境设置  
os.environ['RANK'] = '0'  
os.environ['WORLD_SIZE'] = '1'  
os.environ['MASTER_ADDR'] = 'localhost'  
os.environ['MASTER_PORT'] = '12356'  
os.environ['LOCAL_RANK'] = '0'  
os.environ['DS_SKIP_CUDA_CHECK'] = '1'  
os.environ['DS_BUILD_OPS'] = '0'  
  
class MoEGateAnalyzer:  
    """DeepSeek MoE 门控分析器"""  
      
    def __init__(self):  
        self.gate_outputs = {}  
        self.gate_inputs = {}  
        self.layer_info = {}  
          
    def register_hooks(self, model):  
        """为所有门控层注册钩子"""  
        for name, module in model.named_modules():  
            # 检测 DeepSeek MoE 门控层  
            if 'mlp.gate' in name and hasattr(module, 'forward'):  
                layer_id = self._extract_layer_id(name)  
                if layer_id is not None:  
                    self._register_gate_hook(module, name, layer_id)  
                    print(f"[ANALYZER] Registered hook for {name} (layer {layer_id})")  
      
    def _extract_layer_id(self, name):  
        """从模块名称提取层ID"""  
        if 'layers.' in name:  
            try:  
                return int(name.split('layers.')[1].split('.')[0])  
            except (ValueError, IndexError):  
                return None  
        return None  
      
    def _register_gate_hook(self, module, name, layer_id):  
        """注册门控钩子"""  
        def gate_hook(module, input, output):  
            # 保存输入信息  
            if isinstance(input, tuple) and len(input) > 0:  
                input_tensor = input[0]  
                self.gate_inputs[layer_id] = {  
                    'shape': input_tensor.shape,  
                    'dtype': input_tensor.dtype,  
                    'device': input_tensor.device,  
                    'sample_values': input_tensor.flatten()[:10].detach().cpu().tolist()  
                }  
              
            # 分析输出格式  
            self.gate_outputs[layer_id] = self._analyze_output(output, layer_id)  
              
            # 保存层信息  
            self.layer_info[layer_id] = {  
                'module_name': name,  
                'module_type': type(module).__name__,  
                'module_params': {k: v for k, v in module.named_parameters()}  
            }  
          
        module.register_forward_hook(gate_hook)  
      
    def _analyze_output(self, output, layer_id):  
        """详细分析门控输出"""  
        analysis = {  
            'output_type': type(output).__name__,  
            'is_tuple': isinstance(output, tuple),  
        }  
          
        if isinstance(output, tuple):  
            analysis['tuple_length'] = len(output)  
            analysis['elements'] = []  
              
            for i, element in enumerate(output):  
                element_info = {  
                    'index': i,  
                    'type': type(element).__name__,  
                    'shape': getattr(element, 'shape', 'N/A'),  
                    'dtype': getattr(element, 'dtype', 'N/A'),  
                }  
                  
                # 如果是张量，提取更多信息  
                if torch.is_tensor(element):  
                    element_info.update({  
                        'min_val': element.min().item() if element.numel() > 0 else 'N/A',  
                        'max_val': element.max().item() if element.numel() > 0 else 'N/A',  
                        'mean_val': element.mean().item() if element.numel() > 0 else 'N/A',  
                        'sample_values': element.flatten()[:5].detach().cpu().tolist() if element.numel() > 0 else []  
                    })  
                      
                    # 尝试提取专家选择信息  
                    if len(element.shape) >= 2:  
                        try:  
                            # 假设这可能是 logits 或 gates  
                            top_values, top_indices = torch.topk(element, k=min(8, element.shape[-1]), dim=-1)  
                            element_info['top_experts'] = {  
                                'indices': top_indices.cpu().tolist()[:3],  # 只显示前3个token的结果  
                                'values': top_values.cpu().tolist()[:3]  
                            }  
                        except Exception as e:  
                            element_info['topk_error'] = str(e)  
                  
                analysis['elements'].append(element_info)  
          
        elif torch.is_tensor(output):  
            # 单个张量输出  
            analysis.update({  
                'shape': output.shape,  
                'dtype': output.dtype,  
                'min_val': output.min().item() if output.numel() > 0 else 'N/A',  
                'max_val': output.max().item() if output.numel() > 0 else 'N/A',  
                'mean_val': output.mean().item() if output.numel() > 0 else 'N/A',  
                'sample_values': output.flatten()[:10].detach().cpu().tolist() if output.numel() > 0 else []  
            })  
              
            # 尝试专家选择  
            if len(output.shape) >= 2:  
                try:  
                    top_values, top_indices = torch.topk(output, k=min(8, output.shape[-1]), dim=-1)  
                    analysis['top_experts'] = {  
                        'indices': top_indices.cpu().tolist()[:3],  
                        'values': top_values.cpu().tolist()[:3]  
                    }  
                except Exception as e:  
                    analysis['topk_error'] = str(e)  
          
        return analysis  
      
    def print_analysis(self):  
        """打印分析结果"""  
        print("\n" + "="*80)  
        print("DeepSeek MoE 门控输出格式分析报告")  
        print("="*80)  
          
        for layer_id in sorted(self.gate_outputs.keys()):  
            print(f"\n--- Layer {layer_id} ---")  
              
            # 输入信息  
            if layer_id in self.gate_inputs:  
                input_info = self.gate_inputs[layer_id]  
                print(f"输入: shape={input_info['shape']}, dtype={input_info['dtype']}")  
                print(f"输入样本值: {input_info['sample_values']}")  
              
            # 输出信息  
            output_info = self.gate_outputs[layer_id]  
            print(f"输出类型: {output_info['output_type']}")  
            print(f"是否为元组: {output_info['is_tuple']}")  
              
            if output_info['is_tuple']:  
                print(f"元组长度: {output_info['tuple_length']}")  
                for element in output_info['elements']:  
                    print(f"  元素 {element['index']}: {element['type']}, shape={element['shape']}")  
                    if 'top_experts' in element:  
                        print(f"    Top专家索引: {element['top_experts']['indices']}")  
                        print(f"    Top专家值: {element['top_experts']['values']}")  
            else:  
                print(f"张量形状: {output_info.get('shape', 'N/A')}")  
                if 'top_experts' in output_info:  
                    print(f"Top专家索引: {output_info['top_experts']['indices']}")  
                    print(f"Top专家值: {output_info['top_experts']['values']}")  
              
            # 模块信息  
            if layer_id in self.layer_info:  
                layer_info = self.layer_info[layer_id]  
                print(f"模块: {layer_info['module_name']} ({layer_info['module_type']})")  
                param_count = sum(p.numel() for p in layer_info['module_params'].values())  
                print(f"参数数量: {param_count:,}")  
  
def main():  
    """主测试函数"""  
    print("开始 DeepSeek MoE 16B 门控测试...")  
      
    # 初始化分布式环境  
    dist.init_distributed(dist_backend='gloo')  
      
    # DeepSpeed 配置  
    ds_config = {  
        "zero_optimization": {  
            "stage": 3,  
            "offload_param": {  
                "device": "cpu",  
                "pin_memory": False  
            }  
        },  
        "fp16": {"enabled": True}  
    }  
      
    # 加载模型  
    print("加载 DeepSeek MoE 模型...")  
    with Init(remote_device="cpu", pin_memory=False, config_dict_or_path=ds_config, dtype=torch.half):  
        model = AutoModelForCausalLM.from_pretrained(  
            r"/home/mint/Desktop/zipmoe/models/deepseek-moe-16b-base",  
            torch_dtype=torch.half,  
            device_map=None  # 避免自动设备映射  
        )  

      
    # 设置叶子模块  
    set_z3_leaf_modules(model, ['DeepseekMLP'])  
      
    # 加载分词器  
    tokenizer = AutoTokenizer.from_pretrained(r"/home/mint/Desktop/zipmoe/models/deepseek-moe-16b-base")  
    if tokenizer.pad_token is None:  
        tokenizer.pad_token = tokenizer.eos_token  
      
    # 创建分析器并注册钩子  
    analyzer = MoEGateAnalyzer()  
    analyzer.register_hooks(model)  
      
    # 准备测试输入  
    test_texts = [  
        "The mixture of experts architecture",  
        "DeepSeek MoE is a large language model",  
        "Expert routing in neural networks"  
    ]  
      
    model.eval()  
      
    print("\n开始门控分析...")  
    for i, text in enumerate(test_texts):  
        print(f"\n测试输入 {i+1}: '{text}'")  
          
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)  
        inputs = {k: v.to('cuda') if torch.cuda.is_available() else v for k, v in inputs.items()}  
          
        with torch.no_grad():  
            # 只进行前向传播，不生成  
            outputs = model(**inputs, output_hidden_states=True)  
          
        print(f"完成测试 {i+1}")  
      
    # 打印分析结果  
    analyzer.print_analysis()  
      
    print("\n门控测试完成！")  
  
if __name__ == "__main__":  
    main()