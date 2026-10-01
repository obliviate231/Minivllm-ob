# R2：双卡 3090 上跑「1.7B 目标 + 0.6B 草稿」的 gamma 扫描（命令清单）

> 目的：把 R1 里**已验证正确但未验证加速比**的投机解码，在两卡机器上跑出真实收益曲线。
>
> 前置：R1 的代码改动已在仓库里（未提交 git），`tests/test_speculative.py --gpu` 通过。
>
> 预计耗时：权重下载 5–15 分钟；扫描本身约 20–40 分钟。

---

## 0. 先修 R1 遗留的接受率缺陷（**必做，否则整个扫描没意义**）

R1 的自投机接受率只有 0.0013，而自投机下它本应≈1。**没修好就别扫 gamma**——接受率≈0 时曲线会是一条贴在 0.15× 附近的平线，看不出任何东西。

最快的定位方式：用自投机跑 3 个 round，打印**草稿提案**与**目标逐行 argmax**，看第一个 token 是否一致。

```bash
cd ~/infra/MinivLLM-Fixed

cat > /tmp/check_acceptance.py <<'EOF'
"""自投机下：草稿的提案 vs 目标模型自己的 argmax。两者必须一致。"""
import os, sys
os.environ["HF_HUB_OFFLINE"] = "0"
sys.path.insert(0, os.path.expanduser("~/infra/MinivLLM-Fixed/src"))
from myvllm.engine.llm_engine import LLMEngine
from myvllm.sampling_parameters import SamplingParams
from transformers import AutoTokenizer

MODEL = "Qwen/Qwen3-0.6B"
shape = dict(hidden_size=1024, num_heads=16, head_dim=128,
             num_kv_heads=8, intermediate_size=3072)
base = dict(max_num_sequences=8, max_num_batched_tokens=512,
            max_cached_blocks=256, block_size=128, world_size=1,
            model_name_or_path=MODEL, enforce_eager=True, dtype="bfloat16",
            vocab_size=151936, num_layers=28, tie_word_embeddings=True,
            base=1000000, rms_norm_epsilon=1e-6, qkv_bias=False, scale=1,
            max_position=32768, ffn_bias=False, max_model_length=256,
            gpu_memory_utilization=0.7, eos=151645, **shape)
draft = dict(model_name_or_path=MODEL, vocab_size=151936, num_layers=28,
             tie_word_embeddings=True, base=1000000, rms_norm_epsilon=1e-6,
             qkv_bias=False, scale=1.0, max_position=32768, ffn_bias=False,
             **shape)

tok = AutoTokenizer.from_pretrained(MODEL)
prompt = tok.apply_chat_template(
    [{"role": "user", "content": "hi"}], tokenize=False, add_generation_prompt=True)

engine = LLMEngine(config={**base, "speculative": True, "speculative_gamma": 4,
                           "speculative_draft_model_config": draft,
                           "draft_kv_blocks": 32})

_orig = engine._spec_step
n = [0]
def traced(seqs, is_prefill):
    n[0] += 1
    if n[0] > 3:
        raise SystemExit(0)
    drafts = engine.draft_runner.propose(
        [s.last_token for s in seqs], engine.spec_config["gamma"])
    heads = [s.last_token for s in seqs]
    lens = [1 + len(d) for d in drafts]
    engine.model_runner.call("run_verify", seqs, drafts, heads, lens)
    lg = engine.model_runner.verify_logits
    if lg:
        print(f"head        = {heads[0]}")
        print(f"draft 提案   = {drafts[0]}")
        print(f"目标 argmax  = {lg[0].float().argmax(-1).tolist()}")
        print(f"第一个 token 一致? {drafts[0][0] == int(lg[0].float().argmax(-1)[0])}")
    return _orig(seqs, is_prefill)
engine._spec_step = traced

sp = SamplingParams(temperature=1e-5, max_tokens=8, max_model_length=256,
                    ignore_eos=True)
engine.generate([prompt], sp)
EOF

uv run python /tmp/check_acceptance.py
```

**判读**：`第一个 token 一致? True` 才继续。若是 `False`，问题在草稿的 proposal 路径（`DraftModelRunner.propose` 的 context / 位置 / slot 布置），不在验证路径——因为等价性测试已经证明验证路径是对的。

修好后，用自投机确认接受率确实接近 1：

```bash
uv run python tests/test_speculative.py --gpu --model Qwen/Qwen3-0.6B --gamma 4
# 关注 stats 里的 acceptance_rate，应接近 1.0（R1 时是 0.0）
```

---

## 1. 环境与权重

```bash
cd ~/infra/MinivLLM-Fixed

# 依赖
uv sync

# 两张卡都可见吗
nvidia-smi --query-gpu=index,name,memory.total --format=csv

# 权重（1.7B ≈ 4.06 GB，两片 safetensors）
uv run python - <<'EOF'
from huggingface_hub import snapshot_download
for repo in ("Qwen/Qwen3-1.7B", "Qwen/Qwen3-0.6B"):
    p = snapshot_download(repo)
    print(repo, "->", p)
EOF
```

国内直连不稳定的话（R1 实测 `huggingface.co` 会 SSL 中断），用镜像或预下载到本地目录再传路径，例如：

```bash
export HF_ENDPOINT=https://hf-mirror.com
# 或者手动下两片 shard，配合 curl -C - 断点续传：
#   https://huggingface.co/Qwen/Qwen3-1.7B/resolve/main/model-00001-of-00002.safetensors  (3,441,185,608 B)
#   https://huggingface.co/Qwen/Qwen3-1.7B/resolve/main/model-00002-of-00002.safetensors  (622,329,984 B)
```

> 若传本地路径，**目录名必须是 `Qwen3-1.7B` / `Qwen3-0.6B`**——`ModelRunner` 用 basename 匹配模型类型；形状表 `TARGET_SHAPES` 也按 basename 查。

---

## 2. 先跑一次基线，确认显存与吞吐正常

```bash
uv run python tests/benchmarks/benchmark_speculative.py \
  --target Qwen/Qwen3-1.7B --draft Qwen/Qwen3-0.6B \
  --gamma 4 --max-tokens 128 --warmup 1 --repeats 3 \
  --gpu-mem 0.55 \
  --out tests/benchmarks/spec_1p7b_g4.json
```

**看两件事**：
- `baseline X tok/s` —— 这是后面所有加速比的**分母**，两卡 3090 上 1.7B 应该在几百 tok/s 量级；把数字记下来。
- 投机 arm 的 `acceptance rate` —— 期望 0.5–0.8 之间（同词表的 0.6B 草稿 + 1.7B 目标）。若仍是 0.0x，回到第 0 步。

`--gpu-mem 0.55` 给两个模型留了余量。若 OOM，往下调（0.5 → 0.45）；若显存富裕，可上调以提高 batch 上限。

---

## 3. gamma 扫描

```bash
for G in 2 4 6 8; do
  echo "=================== gamma=$G ==================="
  uv run python tests/benchmarks/benchmark_speculative.py \
    --target Qwen/Qwen3-1.7B --draft Qwen/Qwen3-0.6B \
    --gamma "$G" --max-tokens 128 --warmup 1 --repeats 3 \
    --gpu-mem 0.55 \
    --out "tests/benchmarks/spec_1p7b_g${G}.json"
done
```

每次跑完会打印：

```
baseline    <X> tok/s
speculative <Y> tok/s  (<speedup>x)
acceptance rate <a>, mean accepted/round <m>, target forwards <f>
```

**注意 `gamma` 的上限**：`propose()` 每行要 `gamma + 1` 个 slot，且必须落在同一个 block 里（当前 `block_size=128`）。所以 `gamma ≤ 127` 安全；`gamma=8` 远低于此。若将来调小 `block_size`，先确认这个约束。

---

## 4. 画加速曲线

```bash
uv run python - <<'EOF'
import json, glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

files = {}
for g in (2, 4, 6, 8):
    try:
        files[g] = json.load(open(f"tests/benchmarks/spec_1p7b_g{g}.json"))
    except FileNotFoundError:
        pass

base = next(iter(files.values()))["baseline"]["tokens_per_s"]
gs   = sorted(files)
spd  = [files[g]["speedup_tokens_per_s"] for g in gs]
acc  = [files[g]["speculative"]["speculative"]["acceptance_rate"] for g in gs]
macc = [files[g]["speculative"]["speculative"]["mean_accepted_per_round"] for g in gs]

fig, ax = plt.subplots(figsize=(6.5, 4.2))
ax.plot(gs, spd, "o-", color="#1f4e79", lw=2, label="speedup (tokens/s)")
ax.axhline(1.0, ls="--", color="#999999", lw=1, label="no speedup (1.0x)")
best = gs[int(max(range(len(spd)), key=lambda i: spd[i]))]
ax.axhline(spd[gs.index(best)], ls=":", color="#c00000", lw=1,
           label=f"best {spd[gs.index(best)]:.2f}x at gamma={best}")
ax.set_xlabel("gamma (draft tokens per round)")
ax.set_ylabel("speedup over baseline")
ax.set_xticks(gs)
ax.set_title("Qwen3-1.7B target + Qwen3-0.6B draft, 2x RTX 3090")
ax.grid(alpha=0.3)

ax2 = ax.twinx()
ax2.plot(gs, acc, "s--", color="#7f7f7f", lw=1.5, label="acceptance rate")
ax2.set_ylabel("acceptance rate", color="#7f7f7f")
ax2.tick_params(axis="y", colors="#7f7f7f")

h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="best", fontsize=8)

fig.tight_layout()
fig.savefig("tests/benchmarks/spec_gamma_sweep.png", dpi=160)
print(f"baseline {base:.2f} tok/s; speedups:", dict(zip(gs, [round(s, 3) for s in spd])))
print("mean accepted per round:", dict(zip(gs, [round(m, 2) for m in macc])))
print("wrote tests/benchmarks/spec_gamma_sweep.png")
EOF
```

---

## 5. 判读标准

| 现象 | 含义 | 下一步 |
|---|---|---|
| `speedup > 1` 且随 gamma 先升后降 | 正常形态：找到最优点 | 取峰值处的 gamma 作为默认值 |
| 全程 `speedup < 1` | 草稿开销压过收益 | 检查接受率；接受率正常则说明 0.6B 相对 1.7B 还是太大，可换更小草稿或改用树状验证 |
| 接受率仍 ≈0 | 第 0 步没修好 | 回到第 0 步，别继续扫 |
| OOM | 两模型 + 两个 KV 池超显存 | 降 `--gpu-mem`，或降 `draft_kv_blocks`（默认 32，草稿每行只用 1 个 block） |

**理论参考**：贪心投机解码的期望接受长度约 `(1 - a^(gamma+1)) / (1 - a)`（`a` = 单 token 接受率）。加速比的上界由它给出，但要扣掉草稿本身的耗时：

```
speedup ≈ (已接受长度) / (1 + gamma * c_draft/c_target)
```

其中 `c_draft/c_target` 是草稿与目标单次前向的耗时之比（0.6B/1.7B 大约 0.3–0.4）。所以 `gamma` 调大时分子趋于饱和、分母线性增长——**存在最优 gamma**，这正是扫描要找的东西。

---

## 6. 顺带修掉的两个遗留项

1. **统计口径**：R1 里 `tokens_emitted=380 / rounds=95`（每轮 4 个）与实际每轮只前进 1 个 token 对不上，`tokens_emitted` 的累加位置需要核一遍。
2. **多序列批**：R1 的单序列等价性通过，但两序列（不同长度）时第二个序列出过错。双卡上一定跑多并发，先补一条多 prompt 的等价性测试：

```bash
# 把 tests/test_speculative.py 的 prompts 改成 2~4 条不同长度的 prompt，
# 断言 baseline == spec，再上并发压测
uv run python tests/test_speculative.py --gpu --model Qwen/Qwen3-1.7B --gamma 4
```

---

## 7. 想更进一步（对应 L4 简历里的写法）

- **温度采样**：把贪心接受规则换成标准拒绝采样 `min(1, p_target/p_draft)` + 残差重采样。这是支持 `temperature > 0` 的前提。
- **树状验证（DDTree）**：一轮验证多条候选路径，把"线性草稿链"换成树，用 Tree Attention mask 一次验证全树。对应简历里「DDTree 验证机制 / Top-B 候选」的描述。
- **PD 分离 / Chunked Prefill**：与投机解码正交，可并列推进。

---

## 附：本清单里所有命令涉及的路径

| 用途 | 路径 |
|---|---|
| 仓库根 | `~/infra/MinivLLM-Fixed` |
| 投机实现 | `src/myvllm/engine/speculative.py` |
| 验证前向 | `src/myvllm/engine/model_runner.py::run_verify` |
| 引擎接入 | `src/myvllm/engine/llm_engine.py::_spec_step` |
| 正确性测试 | `tests/test_speculative.py` |
| 性能 benchmark | `tests/benchmarks/benchmark_speculative.py` |
| 扫描结果 | `tests/benchmarks/spec_1p7b_g{2,4,6,8}.json` |
| 曲线图 | `tests/benchmarks/spec_gamma_sweep.png` |
