
# '''
# @cached_property
# 将类的方法转换为一个属性，该属性的值只计算一次，然后缓存为普通属性。
# 因此，只要实例持续存在，缓存结果就会可用，我们可以将该方法用作类的属性。
# '''

# # # 创建基础模型和分词器  
# # model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-tiny-MoE-instruct")  
# # tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-tiny-MoE-instruct")  

# # # 保存到本地
# # model.save_pretrained(r"C:\Users\DELL\Desktop\PProject\ZipMoE\models\phi-tiny-moe")
# # tokenizer.save_pretrained(r"C:\Users\DELL\Desktop\PProject\ZipMoE\models\phi-tiny-moe")
  
# import sys    
# sys.path.append('/home/mint/Desktop/zipmoe')    
    
from deepspeed.moe.layer import MoE    
from deepspeed.moe.experts import Experts    
from transformers import AutoModelForCausalLM, AutoTokenizer    
import torch.nn as nn   
import torch 
  
class ExpertAwareC:  
    def __init__(self, model):  
        self.module = model  # 添加模型引用  
      
    def _is_expert_module(self, current_submodule) -> bool:  
        """判断当前子模块是否为专家模块 - 简化版本"""  
        
        # 1. DeepSpeed MoE 类检测  
        if isinstance(current_submodule, (Experts, MoE)):  
            return True  
        
        # 2. 通过模块路径精确判断 - 只有 experts.数字 路径才是专家  
        module_path = self._get_module_path(current_submodule)  
        if module_path:  
            import re  
            # 匹配 experts.数字 模式  
            if re.search(r'\.experts\.\d+', module_path):  
                return True  
        
        return False
      
    def _get_module_path(self, module) -> str:    
        """获取模块的完整路径名称"""    
        # 通过遍历根模块找到路径    
        if hasattr(self, 'module'):    
            for name, submodule in self.module.named_modules():    
                if submodule is module:    
                    return name    
        return ""  
  
model = AutoModelForCausalLM.from_pretrained(r"/home/mint/Projects/ZipMoE/models/deepseek-moe-16b-base",
                                             trust_remote_code=True,
                                             torch_dtype=torch.float16)  
  
# 创建一个实例并传入模型  
ex = ExpertAwareC(model)  
  
for name, module in model.named_modules():    
    is_expert = ex._is_expert_module(module)    
    print(f"Module {name}: {type(module).__name__} -> Expert: {is_expert}")


