import os
import torch

def clone_tensors_for_torch_save(item, device=torch.device('cpu')):
    """
    Returns a copy of ``item`` with all enclosed tensors replaced by clones on a specified device.
    Works on individual tensors, and tensors contained/nested in lists, tuples, and dicts.

    Parameters:
        - ``item``: tensor to clone or (possibly nested) container of tensors to clone.
        - ``device``: target device (defaults to 'cpu')

    Returns:
        - copy of ``item`` with cloned tensors on target device
    """
    if torch.is_tensor(item):
        if type(device) is str:
            device = torch.device(device)
        if device == item.device:
            return item.detach().clone()
        else:
            return item.detach().to(device)
    elif isinstance(item, list):
        return [clone_tensors_for_torch_save(v, device) for v in item]
    elif isinstance(item, tuple):
        return tuple([clone_tensors_for_torch_save(v, device) for v in item])
    elif isinstance(item, dict):
        return type(item)({k: clone_tensors_for_torch_save(v, device) for k, v in item.items()})
    else:
        return item
