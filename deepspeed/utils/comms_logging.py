def get_caller_func(frame=3):  
    return "unknown_caller"  
  
class CommsLogger:  
    def __init__(self):  
        self.enabled = False  
      
    def configure(self, config):  
        pass  
      
    def append(self, *args, **kwargs):  
        pass