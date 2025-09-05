import torch
import time
import gc
from safetensors import safe_open
import sys

# 把模型代码目录加进来
sys.path.append("/home/mint/Projects/ZipMoE/models")
from deepseek_moe_16b_base.modeling_deepseek import DeepseekMLP


# 定义一个假的 config，包含 DeepseekMLP 需要的字段
class DummyConfig:
    def __init__(self):
        self.hidden_size = 2048
        self.intermediate_size = 1408
        self.num_experts = 1
        self.n_shared_experts = 0
        self.hidden_act = "silu"
        self.use_bias = False
        self.pretraining_tp = 1   # <<< 新增，避免 forward 里报错


def test_single_expert():
    model_path = "/home/mint/Projects/ZipMoE/models/deepseek_moe_16b_base"
    tensor_file = f"{model_path}/model-00001-of-00007.safetensors"

    expert_prefix = "model.layers.1.mlp.experts.0."
    expert_state = {}

    with safe_open(tensor_file, framework="pt", device="cpu") as f:
        for k in f.keys():
            if k.startswith(expert_prefix):
                new_key = k.replace(expert_prefix, "")  # 去掉 experts.0. 前缀
                expert_state[new_key] = f.get_tensor(k)

    if not expert_state:
        raise ValueError(f"在 {tensor_file} 没找到 {expert_prefix} 的参数")

    print(f"成功加载 {len(expert_state)} 个参数：{list(expert_state.keys())}")

    # 构建 DeepseekMLP
    config = DummyConfig()
    expert = DeepseekMLP(config).half().cuda()
    missing, unexpected = expert.load_state_dict(expert_state, strict=False)

    print("missing:", missing)
    print("unexpected:", unexpected)

    # 构造输入
    batch_size, seq_len = 1, 16
    dummy_input = torch.randn(batch_size * seq_len, config.hidden_size, device="cuda", dtype=torch.float16)

    # 预热
    for _ in range(3):
        _ = expert(dummy_input)
    torch.cuda.synchronize()

    # 测速
    num_runs, times = 20, []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = expert(dummy_input)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)

    print(f"平均推理时间: {sum(times)/len(times):.4f} ms")
    print(f"最小: {min(times):.4f} ms, 最大: {max(times):.4f} ms")


if __name__ == "__main__":
    test_single_expert()
