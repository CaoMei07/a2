from contextlib import contextmanager  
from deepspeed.runtime.zero.partition_parameters import GatheredParameters  
from deepspeed.runtime.zero.offload_config import OffloadDeviceEnum, OffloadStateTypeEnum

@contextmanager  
def unwrap_model_for_generation(model):  
    """For ZeRO-3 models, we gather the weights once to speed up generation."""  
    with GatheredParameters(model.parameters()):  
        yield model  
  
def move_to_cpu(tensor_list):  
    for tensor in tensor_list:  
        tensor.data = tensor.data.cpu()
        

def estimate_zero3_model_states_mem_needs(total_params,
                                          largest_layer_params,
                                          num_gpus_per_node=1,
                                          num_nodes=1,
                                          cpu_offload=True,
                                          cpu_offload_params=True,
                                          zero_init=True,
                                          additional_buffer_factor=1.5):

    total_gpus = num_nodes * num_gpus_per_node
    gpus_factor = 1 / num_nodes
    largest_layer_memory = (4 * largest_layer_params)

    if cpu_offload:
        if cpu_offload_params:
            gpu_mem = largest_layer_memory

            if zero_init:
                cpu_mem = total_params * 18 * gpus_factor * additional_buffer_factor
            else:
                cpu_mem = total_params * max(4 * num_gpus_per_node, 18 * gpus_factor) * additional_buffer_factor

        else:
            gpu_mem = largest_layer_memory + int(2 * total_params / total_gpus)

            if zero_init:
                cpu_mem = total_params * 16 * gpus_factor * additional_buffer_factor
            else:
                cpu_mem = total_params * max(4 * num_gpus_per_node, 16 * gpus_factor) * additional_buffer_factor
    else:
        gpu_mem = largest_layer_memory + int(18 * total_params / total_gpus)
        if zero_init:
            cpu_mem = largest_layer_params * 4 * num_gpus_per_node * additional_buffer_factor
        else:
            cpu_mem = total_params * 4 * num_gpus_per_node * additional_buffer_factor

    return int(cpu_mem), int(gpu_mem), largest_layer_memory


def model_to_params(model):
    # shared params calculated only once
    total_params = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values())

    largest_layer_params = 0
    for m in model.modules():
        # assuming no shared params within a single layer
        layer_params = sum(p.numel() for p in m.parameters(recurse=False))
        largest_layer_params = max(largest_layer_params, layer_params)

    return total_params, largest_layer_params


def estimate_zero3_model_states_mem_needs_all_live(model,
                                                   num_gpus_per_node=1,
                                                   num_nodes=1,
                                                   additional_buffer_factor=1.5):
    """
    Print out estimates on memory usage requirements for ZeRO 3 params, optim states and gradients
    for a given ``model`` and hardware setup.

    If you have an actual model object, use this function and everything will be derived
    automatically.

    If it's a hypothetical model, use ``estimate_zero3_model_states_mem_needs_all_cold`` where you have to pass
    the ``total_params`` and ``largest_layer_params`` explicitly.

    Args:
        - ``model``: ``nn.Module`` object
        - ``num_gpus_per_node``: how many gpus per node (defaults to 1)
        - ``num_nodes``: how many nodes (defaults to 1),
        - ``additional_buffer_factor``: estimation factor (defaults to 1.5):

    """

    total_params, largest_layer_params = model_to_params(model)

    estimate_zero3_model_states_mem_needs_all_cold(total_params=total_params,
                                                   largest_layer_params=largest_layer_params,
                                                   num_gpus_per_node=num_gpus_per_node,
                                                   num_nodes=num_nodes,
                                                   additional_buffer_factor=additional_buffer_factor)


def estimate_zero3_model_states_mem_needs_all_cold(total_params,
                                                   largest_layer_params,
                                                   num_gpus_per_node=1,
                                                   num_nodes=1,
                                                   additional_buffer_factor=1.5):
    """
    Print out estimates on memory usage requirements for ZeRO 3 params, optim states and gradients
    for a given ``model`` and hardware setup.

    If it's a hypothetical model, use this function where you have to pass
    the ``total_params`` and ``largest_layer_params`` explicitly.

    If you have an actual model object, use ``estimate_zero3_model_states_mem_needs_all_live`` and everything
    will be derived automatically.

    Args:
        - ``total_params``: total  model params
        - ``largest_layer_params``: largest layer's params
        - ``num_gpus_per_node``: how many gpus per node (defaults to 1)
        - ``num_nodes``: how many nodes (defaults to 1),
        - ``additional_buffer_factor``: estimation factor (defaults to 1.5):

    """

    def format_options(cpu_offload, cpu_offload_params, zero_init):
        enabled = []
        padded_cpu_str = f'{OffloadDeviceEnum.cpu:4}'
        param_device = padded_cpu_str if cpu_offload_params else "none"
        enabled.append(f"offload_param={param_device}")
        optimizer_device = padded_cpu_str if cpu_offload else "none"
        enabled.append(f"offload_optimizer={optimizer_device}")
        enabled.append(f"zero_init={1 if zero_init else 0}")
        return ", ".join(enabled)

    nodes_str = "nodes" if num_nodes > 1 else "node"
    gpus_str = "GPUs" if num_gpus_per_node > 1 else "GPU"
    print(
        "Estimated memory needed for params, optim states and gradients for a:\n"
        f"HW: Setup with {num_nodes} {nodes_str}, {num_gpus_per_node} {gpus_str} per node.\n"
        f"SW: Model with {int(total_params/1e6)}M total params, {int(largest_layer_params/1e6)}M largest layer params."
    )
    print("  per CPU  |  per GPU |   Options")
    for cpu_offload in [True, False]:
        for cpu_offload_params in [True, False]:
            if not cpu_offload and cpu_offload_params:
                continue
            for zero_init in [True, False]:
                cpu_mem, gpu_mem, largest_layer_memory = estimate_zero3_model_states_mem_needs(
                    total_params=total_params,
                    largest_layer_params=largest_layer_params,
                    num_gpus_per_node=num_gpus_per_node,
                    num_nodes=num_nodes,
                    cpu_offload=cpu_offload,
                    cpu_offload_params=cpu_offload_params,
                    zero_init=zero_init,
                    additional_buffer_factor=additional_buffer_factor)

                options_str = format_options(cpu_offload=cpu_offload,
                                             cpu_offload_params=cpu_offload_params,
                                             zero_init=zero_init)
                print(f" {cpu_mem/2**30:7.2f}GB | {gpu_mem/2**30:6.2f}GB | {options_str}")