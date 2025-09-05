from typing import Callable, Dict, Union, Iterable, Container, List  
from torch.nn.parameter import Parameter  
from torch.optim import Optimizer  
from torch.optim.lr_scheduler import _LRScheduler  
  
DeepSpeedOptimizerCallable = \
    Callable[[Union[Iterable[Parameter], Dict[str, Iterable]]], Optimizer]  
DeepSpeedSchedulerCallable = Callable[[Optimizer], _LRScheduler]