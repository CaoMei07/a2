# zyr简化版ccl
from .torch import TorchBackend  
  
def build_ccl_op():  
    return None  
  
class CCLBackend(TorchBackend):  
    def __init__(self, *args, **kwargs):  
        super().__init__(*args, **kwargs)  
        self.initialized = False  
      
    def is_initialized(self):  
        return False