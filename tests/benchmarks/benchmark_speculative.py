"""Benchmark: baseline decode vs. draft-model speculative decode.

Both runs use the same target model, the same prompts and the same sampling
settings; only `speculative` changes. The script reports wall-clock throughput
(tok/s), per-token latency, and the speculative counters (acceptance rate,
target forwards per emitted token) so the speedup can be read off directly
rather than taken on faith.

Run:
    uv run python tests/benchmarks/benchmark_speculative.py
    uv run python tests/benchmarks/benchmark_speculative.py --gamma 3 --max-tokens 256
"""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch
from transformers import AutoTokenizer

from myvllm.engine.llm_engine import LLMEngine
from myvllm.sampling_parameters import SamplingParams

# Defaults are the intended 2-GPU setup (1.7B target, 0.6B draft). On a small
# card, pass --target / --draft the same small model to exercise the full
# speculative path (the accept rule is exact, so self-speculation still has to
# reproduce baseline output token for token).
TARGET_MODEL = "Qwen/Qwen3-1.7B"
DRAFT_MODEL = "Qwen/Qwen3-0.6B"

# Greedy (temperature -> 0 is rejected by SamplingParams, so use the smallest
# legal value) is what the accept rule implements exactly.
TEMPERATURE = 1e-5
_GPU_MEM = 0.55

PROMPTS = [
    "Write a Python function that merges two sorted lists into one sorted list, then explain its time complexity.",
    "Explain how a paged KV cache works in an LLM inference engine, step by step.",
    "List the prime numbers below 100 and say how many there are.",
    "Summarise the causes of the Industrial Revolution in five paragraphs.",
]


def build_config(draft_cfg: dict, gamma: int, model: str = TARGET_MODEL, target_shape=None) -> dict:
    """Target config; the draft model's own config sits nested inside it.

    Keeping the draft's shape in its own sub-dict is what lets one engine
    config describe two models of different sizes without key collisions.
    """
    config = {
        "max_num_sequences": 8,
        "max_num_batched_tokens": 1024,
        "max_cached_blocks": 512,
        "block_size": 128,
        "world_size": 1,
        "model_name_or_path": model,
        "enforce_eager": True,
        "dtype": "bfloat16",
        "vocab_size": 151936,
        **(target_shape or {"hidden_size": 2048, "num_heads": 16, "head_dim": 128,
                            "num_kv_heads": 8, "intermediate_size": 6144}),
        "num_layers": 28,
        "tie_word_embeddings": True,
        "base": 1000000,
        "rms_norm_epsilon": 1e-6,
        "qkv_bias": False,
        "scale": 1,
        "max_position": 32768,
        "ffn_bias": False,
        "max_model_length": 2048,
        "gpu_memory_utilization": _GPU_MEM,
        "eos": 151645,
        # Speculative-decoding knobs. `speculative=False` reproduces the
        # baseline exactly, so the two arms differ in one key only.
        "speculative": False,
        "speculative_gamma": gamma,
        "draft_kv_blocks": 64,
        "draft_gpu_memory_fraction": 0.15,
    }
    if draft_cfg is not None:
        config["speculative"] = True
        config["speculative_draft_model_config"] = draft_cfg
    return config


# Keyed by the checkpoint's directory basename, so a local path such as
# `/models/Qwen3-0.6B` resolves the same way a Hub repo id does.
TARGET_SHAPES = {
    "Qwen3-0.6B": {
        "hidden_size": 1024, "num_heads": 16, "head_dim": 128,
        "num_kv_heads": 8, "intermediate_size": 3072,
    },
    "Qwen3-1.7B": {
        "hidden_size": 2048, "num_heads": 16, "head_dim": 128,
        "num_kv_heads": 8, "intermediate_size": 6144,
    },
}


def shape_for(model: str) -> dict:
    """Look up a model's shape by basename, so local paths work too."""
    from pathlib import Path

    shapes = TARGET_SHAPES.get(Path(model).name) or TARGET_SHAPES.get(model)
    if shapes is None:
        raise ValueError(
            f"No shape entry for {model}; add it to TARGET_SHAPES "
            f"(known: {sorted(TARGET_SHAPES)})"
        )
    return shapes


def draft_config_for(model: str) -> dict:
    """Shape map for the draft model, keyed by the model it names."""
    shapes = shape_for(model)
    return {"model_name_or_path": model, "vocab_size": 151936, **shapes,
            "num_layers": 28,
            "tie_word_embeddings": True, "base": 1000000,
            "rms_norm_epsilon": 1e-6, "qkv_bias": False, "scale": 1.0,
            "max_position": 32768, "ffn_bias": False}


DRAFT_CFG_QWEN3_06B = {
    "model_name_or_path": DRAFT_MODEL,
    "vocab_size": 151936,
    "hidden_size": 1024,
    "num_heads": 16,
    "head_dim": 128,
    "num_kv_heads": 8,
    "intermediate_size": 3072,
    "num_layers": 28,
    "tie_word_embeddings": True,
    "base": 1000000,
    "rms_norm_epsilon": 1e-6,
    "qkv_bias": False,
    "scale": 1.0,
    "max_position": 32768,
    "ffn_bias": False,
}


def run_arm(name: str, config: dict, tokenizer, prompts, max_tokens, warmup, repeats):
    """Run one arm: `warmup` unmeasured generations, then `repeats` measured."""
    llm = LLMEngine(config=config)
    sampling = SamplingParams(
        temperature=TEMPERATURE,
        max_tokens=max_tokens,
        max_model_length=config["max_model_length"],
        ignore_eos=True,
    )
    try:
        for _ in range(warmup):
            llm.generate(prompts, sampling)
            torch.cuda.synchronize()

        times, token_counts, spec_stats = [], [], []
        for _ in range(repeats):
            torch.cuda.synchronize()
            start = time.perf_counter()
            out = llm.generate(prompts, sampling)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
            token_counts.append(sum(len(t) for t in out["token_ids"]))
            stats = llm.get_speculative_stats()
            if stats is not None:
                spec_stats.append(stats)

        best = min(times)
        tokens = max(token_counts)
        result = {
            "name": name,
            "latency_s": best,
            "tokens": tokens,
            "tokens_per_s": tokens / best,
            "latency_per_token_ms": best / tokens * 1000.0,
        }
        if spec_stats:
            merged = spec_stats[-1]
            for key in ("rounds", "draft_tokens", "accepted_tokens", "target_forwards"):
                merged[key] = sum(s[key] for s in spec_stats)
            merged["acceptance_rate"] = (
                merged["accepted_tokens"] / merged["draft_tokens"]
                if merged["draft_tokens"]
                else 0.0
            )
            merged["mean_accepted_per_round"] = (
                merged["accepted_tokens"] / merged["rounds"] if merged["rounds"] else 0.0
            )
            result["speculative"] = merged
        return result
    finally:
        # Two engines cannot coexist on a small card: the baseline's KV cache
        # must be gone before the speculative arm allocates its own.
        llm.exit()
        del llm
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def main():
    global TARGET_MODEL, DRAFT_MODEL, _GPU_MEM

    parser = argparse.ArgumentParser()
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--gamma", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", default="tests/benchmarks/benchmark_speculative.json")
    parser.add_argument("--target", default=TARGET_MODEL)
    parser.add_argument("--draft", default=DRAFT_MODEL)
    parser.add_argument("--gpu-mem", type=float, default=0.55)
    args = parser.parse_args()

    TARGET_MODEL, DRAFT_MODEL = args.target, args.draft
    _GPU_MEM = args.gpu_mem

    tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL)
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in PROMPTS
    ]

    print("=== baseline (no speculation) ===")
    baseline = run_arm(
        "baseline",
        build_config(None, args.gamma, args.target, shape_for(args.target)),
        tokenizer,
        prompts,
        args.max_tokens,
        args.warmup,
        args.repeats,
    )
    print(json.dumps(baseline, indent=2))

    print(f"=== speculative (gamma={args.gamma}, draft={DRAFT_MODEL}) ===")
    spec = run_arm(
        "speculative",
        build_config(
            draft_config_for(args.draft), args.gamma, args.target,
            shape_for(args.target),
        ),
        tokenizer,
        prompts,
        args.max_tokens,
        args.warmup,
        args.repeats,
    )
    print(json.dumps(spec, indent=2))

    speedup = spec["tokens_per_s"] / baseline["tokens_per_s"]
    report = {
        "target_model": TARGET_MODEL,
        "draft_model": DRAFT_MODEL,
        "gamma": args.gamma,
        "max_tokens": args.max_tokens,
        "num_prompts": len(prompts),
        "temperature": TEMPERATURE,
        "baseline": baseline,
        "speculative": spec,
        "speedup_tokens_per_s": speedup,
        "baseline_latency_per_token_ms": baseline["latency_per_token_ms"],
        "speculative_latency_per_token_ms": spec["latency_per_token_ms"],
    }
    print("\n=== summary ===")
    print(f"baseline    {baseline['tokens_per_s']:.2f} tok/s")
    print(f"speculative {spec['tokens_per_s']:.2f} tok/s  ({speedup:.2f}x)")
    if "speculative" in spec:
        s = spec["speculative"]
        print(
            f"acceptance rate {s['acceptance_rate']:.3f}, "
            f"mean accepted/round {s['mean_accepted_per_round']:.2f}, "
            f"target forwards {s['target_forwards']}"
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"wrote {out_path}")

    return report


if __name__ == "__main__":
    main()
