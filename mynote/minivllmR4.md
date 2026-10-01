# R4

```python
src/myvllm/models/qwen3.py |  19 ++--

class Qwen3ForCausalLM(nn.Module):
-    packed_module_mapping = {
-        "q_proj": ('q_proj', 'q'),
-        "k_proj": ('k_proj', 'k'),
-        "v_proj": ('v_proj', 'v'),
-        "gate_up": ('gate_up_proj', '0'),
-        "gate_down": ('gate_down_proj', '1'),
+    # target packed module -> (Hugging Face source module, shard id)
+    packed_modules_mapping = {
+        "qkv_projection": [
+            ("q_proj", "q"),
+            ("k_proj", "k"),
+            ("v_proj", "v"),
+        ],
+        "gate_up": [
+            ("gate_proj", 0),
+            ("up_proj", 1),
+        ],
+    }

# Fuseed Kernel
```

```python
src/myvllm/utils/loader.py 

import os
from collections.abc import Iterable, Iterator
import torch
from safetensors import safe_open
from torch import nn

DEFAULT_PACKED_MODULES_MAPPING = {
    "qkv_projection": [
        ("q_proj", "q"),
        ("k_proj", "k"),
        ("v_proj", "v"),
    ],
    "gate_up": [
        ("gate_proj", 0),
        ("up_proj", 1),
    ],
}

def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
    if param.shape != loaded_weight.shape:
        raise ValueError(
            f"Shape mismatch: model parameter {tuple(param.shape)} vs "
            f"checkpoint tensor {tuple(loaded_weight.shape)}"
        )
    param.data.copy_(loaded_weight)

def _map_weight_name( model: nn.Module, hf_name: str) -> tuple[str, str | int | None]:
	"""把 Hugging Face 仓库里的一个参数名，对应（翻译）成我这个模型里的融合参数名。"""
    
    packed_modules_mapping = getattr(
        model, "packed_modules_mapping", DEFAULT_PACKED_MODULES_MAPPING
    ) # 从model重找packed_modules_mapping,有则返回，若无返回DEFAULT_PACKED_MODULES_MAPPING

    for target_module: str, source_modules: list in packed_modules_mapping.items():
        for source_module: str, shard_id: str in source_modules:
            source = f".{source_module}."
            if source in hf_name:
                target_name = hf_name.replace(
                    source, f".{target_module}.", 1
                )
                return target_name, shard_id
    return hf_name, None
# "model.layers.0.self_attn.q_proj.weight" ->  ("model.layers.0.self_attn.qkv_projection.weight", "q")

def load_weights(model: nn.Module, weights: Iterable[tuple[str, torch.Tensor]],) -> set[str]:
	"""
	将 Hugging Face 格式的权重流（weights），按照模型内部定义的映射规则和切片逻辑，安全地加载到 model 中。

	对于打包的检查点张量（如 q_proj/k_proj/v_proj），它们会被逐个传递给
    同一个目标参数。该参数内部定制的 weight_loader 会根据当前的张量并行
    秩（rank）选择对应的分片，并将其写入正确的打包区域。
    """

	# 获取模型所有参数（允许保留重复参数名，用于共享权重检测）
    params = dict(model.named_parameters(remove_duplicate=False))
    loaded_names: set[str] = set()           # 已加载的目标参数名集合
    loaded_param_ids: set[int] = set()       # 已加载的参数对象内存ID集合（用于去重校验）
    unexpected_names: list[str] = []         # 检查点中有但模型中无对应参数的权重名

    for hf_name: str, loaded_weight: torch.Tensor in weights: 
        target_name, shard_id = _map_weight_name(model, hf_name)
        param = params.get(target_name)
        if param is None:
            unexpected_names.append(hf_name)
            continue

        weight_loader = getattr(param, "weight_loader", default_weight_loader)
        try:
            if shard_id is None:
                weight_loader(param, loaded_weight)
            else:
                weight_loader(param, loaded_weight, shard_id)
        except Exception as error: # 若有任何异常....
            raise RuntimeError(
                f"Failed to load '{hf_name}' into '{target_name}'"
            ) from error

        loaded_names.add(target_name)
        loaded_param_ids.add(id(param))
    """
        hf_name = "layer.q_proj.weight"
        loaded_weight = Tensor_A 
        target_name, shard_id = _map_weight_name(model, hf_name) = "layer.qkv_projection.weight", q
        param = params.get(target_name) = param = <Parameter 对象 (地址 0x7f001)>  
        weight_loader = getattr(param, "weight_loader", default_weight_loader) # 获取方法
        权重加载 ...
    """
    missing_names = [
        name for name, param in params.items()
        if id(param) not in loaded_param_ids
    ]
    if unexpected_names or missing_names:
        messages = []
        if unexpected_names:
            messages.append(
                "checkpoint tensors without a model parameter: "
                + ", ".join(unexpected_names[:10])
            )
        if missing_names:
            messages.append(
                "model parameters without a checkpoint tensor: "
                + ", ".join(missing_names[:10])

            )
        raise RuntimeError("Weight loading is incomplete; " + "; ".join(messages))

    return loaded_names

def _resolve_checkpoint_path(model_name_or_path: str) -> str:
    path = os.path.expanduser(model_name_or_path) # "~/models/Qwen" -> "/home/username/models/Qwen"
    if os.path.isdir(path): # # 如果是本地目录，直接返回，无需联网下载
        return path

    from huggingface_hub import snapshot_download # 本地没有该路径，尝试从 Hugging Face Hub 下载, 延迟导入 snapshot_download，只有在需要下载时才加载该依赖

    try:
        # 调用 HF 官方下载接口
        return snapshot_download(
            repo_id=model_name_or_path,                # 仓库标识符，如 "Qwen/Qwen3-8B"
            allow_patterns=["*.safetensors", "*.json"], # 只下载 safetensors 权重和 json 配置文件
            ignore_patterns=["*.msgpack", "*.h5", "*.bin"], # 忽略旧格式文件，节省时间和空间
        )
    except Exception as error:
        # 如果下载失败（网络错误、仓库不存在、权限问题等），抛出清晰的错误提示
        raise ValueError(
            f"Could not find or download model '{model_name_or_path}'"
        ) from error

def _iter_safetensor_weights(checkpoint_path: str) -> Iterator[tuple[str, torch.Tensor]]:
	"""
	生成器函数：遍历指定目录下的所有 .safetensors 文件，逐个产出权重名称和对应的张量
    参数:
        checkpoint_path (str): 存放 .safetensors 文件的本地目录路径
        
    产出:
        Iterator[tuple[str, torch.Tensor]]: 每个元素为 (权重名称, 张量数据)
	
	"""
    # 1. 列出目录下所有以 .safetensors 结尾的文件，并按文件名排序（保证顺序）
    safetensor_files = sorted(
        file_name for file_name in os.listdir(checkpoint_path)
        if file_name.endswith(".safetensors")
    )
    
    # 2. 如果没有找到任何 safetensors 文件，直接抛出错误
    if not safetensor_files:
        raise ValueError(f"No .safetensors files found in {checkpoint_path}")

    # 3. 初始化一个集合，用于记录已产出过的权重名，用于检测重复
    seen_names: set[str] = set()
    
    # 4. 按顺序遍历每一个分片文件
    for file_name in safetensor_files:
        # 构造文件的完整路径
        file_path = os.path.join(checkpoint_path, file_name)
        
        # 5. 使用 safetensors 库安全打开文件，张量直接加载到 CPU
        with safe_open(file_path, framework="pt", device="cpu") as checkpoint:
            # 6. 遍历当前文件中的所有键（即权重名称）
            for weight_name in checkpoint.keys():
                # 7. 如果该权重名之前已经出现过，说明分片间重复，抛出异常
                if weight_name in seen_names:
                    raise ValueError(
                        f"Duplicate checkpoint tensor '{weight_name}'"
                    )
                # 8. 记录该权重名
                seen_names.add(weight_name)
                # 9. 获取实际张量数据（此时才从磁盘读取），并产出给调用者
                yield weight_name, checkpoint.get_tensor(weight_name)


def load_weights_from_checkpoint(
    model: nn.Module, model_name_or_path: str
) -> set[str]:
    """Load a local or Hugging Face safetensors checkpoint into ``model``."""
    checkpoint_path = _resolve_checkpoint_path(model_name_or_path)
    loaded_names = load_weights(
        model, _iter_safetensor_weights(checkpoint_path)
    )
    print(
        f"Loaded {len(loaded_names)} model parameters from "
        f"{model_name_or_path}"
    )
    return loaded_names
```

```python
python
import os
import sys

import torch
from transformers import Qwen3Config
from transformers import Qwen3ForCausalLM as HFQwen3ForCausalLM

# 将 src 目录加入 Python 路径，以便导入 myvllm 模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from myvllm.models.qwen3 import Qwen3ForCausalLM
from myvllm.utils.loader import load_weights

def test_qwen3_hf_names_use_packed_weight_loaders(monkeypatch):
    """
    测试用例：验证在张量并行（TP=2）下，HF 的分散权重能否正确加载到自定义模型的打包参数中，
    并且只取当前 rank 对应的分片。
    """
    # 1. 模拟分布式环境：world_size=2（2张卡），当前 rank=1（第二张卡）
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 1)

    # 2. 创建自定义模型实例（极小尺寸，方便数值验证）
    model = Qwen3ForCausalLM(
        vocab_size=8,
        hidden_size=8,
        num_heads=4,
        head_dim=2,
        num_kv_heads=2,
        intermediate_size=8,
        num_layers=1,
        qkv_bias=False,
        ffn_bias=False,
        tie_word_embeddings=True,
    )

    # 3. 构造模拟的 HF 权重张量（每个张量都是可预测的固定数值）
    tensors = {
        "model.embed_tokens.weight": torch.arange(64, dtype=torch.float32).reshape(8, 8),
        "model.layers.0.input_layernorm.weight": torch.full((8,), 1.1),
        "model.layers.0.post_attention_layernorm.weight": torch.full((8,), 1.2),
        "model.layers.0.self_attn.q_proj.weight": torch.arange(64, dtype=torch.float32).reshape(8, 8),
        "model.layers.0.self_attn.k_proj.weight": torch.arange(32, dtype=torch.float32).reshape(4, 8) + 100,
        "model.layers.0.self_attn.v_proj.weight": torch.arange(32, dtype=torch.float32).reshape(4, 8) + 200,
        "model.layers.0.self_attn.q_norm.weight": torch.full((2,), 1.3),
        "model.layers.0.self_attn.k_norm.weight": torch.full((2,), 1.4),
        "model.layers.0.self_attn.o_proj.weight": torch.arange(64, dtype=torch.float32).reshape(8, 8) + 300,
        "model.layers.0.mlp.gate_proj.weight": torch.arange(64, dtype=torch.float32).reshape(8, 8) + 400,
        "model.layers.0.mlp.up_proj.weight": torch.arange(64, dtype=torch.float32).reshape(8, 8) + 500,
        "model.layers.0.mlp.down_proj.weight": torch.arange(64, dtype=torch.float32).reshape(8, 8) + 600,
        "model.norm.weight": torch.full((8,), 1.5),
    }

    # 4. 执行加载（将模拟的 HF 权重灌入模型）
    loaded_names = load_weights(model, tensors.items())

    # 5. 验证合并后的 qkv 只包含当前 rank 对应的分片（rank=1 取后半行，即索引 4:8）
    qkv = model.model.layers[0].self_attn.qkv_projection.weight
    # qkv 的形状是 (8, 8)，其中前4行是 q 的后半部分，第4~6行是 k 的后半部分，第6~8行是 v 的后半部分
    torch.testing.assert_close(qkv[:4], tensors["model.layers.0.self_attn.q_proj.weight"][4:8])
    torch.testing.assert_close(qkv[4:6], tensors["model.layers.0.self_attn.k_proj.weight"][2:4])  # k 原始是 (4,8)，后半行是 2:4
    torch.testing.assert_close(qkv[6:8], tensors["model.layers.0.self_attn.v_proj.weight"][2:4])  # v 同 k

    # 验证 gate_up 同样只取后半行
    gate_up = model.model.layers[0].mlp.gate_up.weight
    torch.testing.assert_close(gate_up[:4], tensors["model.layers.0.mlp.gate_proj.weight"][4:8])
    torch.testing.assert_close(gate_up[4:8], tensors["model.layers.0.mlp.up_proj.weight"][4:8])

    # 验证非合并层按列切分（取列的后半部分，即 4:8）
    torch.testing.assert_close(
        model.model.layers[0].self_attn.o_proj.weight,
        tensors["model.layers.0.self_attn.o_proj.weight"][:, 4:8],
    )
    torch.testing.assert_close(
        model.model.layers[0].mlp.down_proj.weight,
        tensors["model.layers.0.mlp.down_proj.weight"][:, 4:8],
    )
    torch.testing.assert_close(
        model.model.embed_tokens.weight,
        tensors["model.embed_tokens.weight"][4:8],   # embed_tokens 也按行切分（取后半行）
    )

    # 验证加载记录中包含合并后的参数名（证明映射生效）
    assert "model.layers.0.self_attn.qkv_projection.weight" in loaded_names
    assert "model.layers.0.mlp.gate_up.weight" in loaded_names

def test_loads_transformers_qwen3_state_dict(monkeypatch):
    """
    测试用例：验证自定义模型能正确加载 Hugging Face 官方 Qwen3 的 state_dict，
    并将分散的 q_proj/k_proj/v_proj 合并为 qkv_projection，gate_proj/up_proj 合并为 gate_up。
    """
    # 1. 模拟单卡环境（无张量并行）
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)

    # 2. 创建 Hugging Face 官方模型（相同配置）
    config = Qwen3Config(
        vocab_size=8,
        hidden_size=8,
        intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=2,
        tie_word_embeddings=True,
    )
    hf_model = HFQwen3ForCausalLM(config)

    # 3. 创建自定义模型（相同配置）
    model = Qwen3ForCausalLM(
        vocab_size=8,
        hidden_size=8,
        num_heads=4,
        head_dim=2,
        num_kv_heads=2,
        intermediate_size=8,
        num_layers=1,
        qkv_bias=False,
        ffn_bias=False,
        tie_word_embeddings=True,
    )

    # 4. 获取官方模型的 state_dict（所有权重）
    hf_weights = hf_model.state_dict()
    # 将 HF 的权重加载到自定义模型中
    load_weights(model, hf_weights.items())

    # 5. 验证合并后的 qkv 是否等于 HF 三个权重沿行拼接的结果
    qkv = model.model.layers[0].self_attn.qkv_projection.weight
    expected_qkv = torch.cat([
        hf_weights["model.layers.0.self_attn.q_proj.weight"],
        hf_weights["model.layers.0.self_attn.k_proj.weight"],
        hf_weights["model.layers.0.self_attn.v_proj.weight"],
    ])   # 沿第0维拼接，形状为 (8+4+4, 8) = (16, 8) ? 但这里 head_dim=2, num_heads=4 => q 输出 4*2=8, k/v 输出 2*2=4，所以总输出 = 8+4+4=16，与模型定义一致
    torch.testing.assert_close(qkv, expected_qkv)

    # 6. 验证 gate_up 是否等于 gate_proj 和 up_proj 沿行拼接
    gate_up = model.model.layers[0].mlp.gate_up.weight
    expected_gate_up = torch.cat([
        hf_weights["model.layers.0.mlp.gate_proj.weight"],
        hf_weights["model.layers.0.mlp.up_proj.weight"],
    ])   # 形状为 (8+8, 8) = (16, 8)
    torch.testing.assert_close(gate_up, expected_gate_up)
```



