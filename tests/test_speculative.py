"""Correctness tests for draft-model speculative decoding.

`verify_run` is pure logic and runs on CPU, so it is tested exhaustively there.
`test_greedy_equivalence_*` need a GPU and one small model: because the accept
rule is exact for greedy decoding, a chain of speculative rounds must reproduce
the baseline's tokens **exactly**, whatever the draft model's quality.

Run:
    uv run python tests/test_speculative.py            # logic only
    uv run python tests/test_speculative.py --gpu      # + engine equivalence
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from myvllm.engine.speculative import SpeculativeStats, verify_run  # noqa: E402

VOCAB = 128


def _logits(rows, vocab=VOCAB):
    """Logits whose argmax at row `i` is `rows[i]`."""
    out = torch.full((len(rows), vocab), -5.0)
    for i, token in enumerate(rows):
        out[i, token] = 5.0
    return out


def test_accept_rule_basics():
    # All drafts accepted; bonus comes from the row after the last draft.
    run = verify_run(_logits([1, 2, 3, 4]), [1, 2, 3])
    assert run.verified == [1, 2, 3] and run.bonus == 4, run

    # First draft rejected: nothing accepted, but a bonus token still lands.
    run = verify_run(_logits([9, 2, 3, 4]), [1, 2, 3])
    assert run.verified == [] and run.bonus == 9, run

    # Partial acceptance stops at the first disagreement.
    run = verify_run(_logits([1, 2, 7, 4]), [1, 2, 3])
    assert run.verified == [1, 2] and run.bonus == 7, run

    # gamma = 1 works in both directions.
    assert verify_run(_logits([5, 6]), [5]).verified == [5]
    assert verify_run(_logits([5, 6]), [9]).verified == []


def test_accept_rule_matches_longest_common_prefix():
    """The accept rule IS "longest common prefix of drafts and target argmax"."""
    torch.manual_seed(0)
    for _ in range(3000):
        rows = torch.randint(0, VOCAB, (6,)).tolist()
        logits = _logits(rows)
        drafts = [
            rows[j] if torch.rand(1).item() < 0.7 else int(torch.randint(VOCAB, (1,)))
            for j in range(5)
        ]
        run = verify_run(logits, drafts)
        expected = 0
        for j in range(5):
            if drafts[j] != rows[j]:
                break
            expected += 1
        assert run.accepted == expected, (drafts, rows, run)
        assert run.bonus == rows[expected], (drafts, rows, run)


def test_stats_ratios():
    stats = SpeculativeStats(
        rounds=4, draft_tokens=16, accepted_tokens=8,
        target_forwards=4, tokens_emitted=12,
    )
    assert abs(stats.acceptance_rate - 0.5) < 1e-9
    assert abs(stats.mean_accepted_per_round - 2.0) < 1e-9
    assert abs(stats.target_forwards_per_token - 4 / 12) < 1e-9
    empty = SpeculativeStats()
    assert empty.acceptance_rate == 0.0 and empty.target_forwards_per_token == 0.0


def test_bookkeeping_append_truncate_bonus():
    """Append a run, accept a prefix, and check tokens and blocks line up."""
    from myvllm.engine.block_manager import BlockManager
    from myvllm.engine.sequence import Sequence

    block_size = 8
    bm = BlockManager(num_blocks=16, block_size=block_size)
    seq = Sequence(token_ids=list(range(10)), block_size=block_size)
    bm.allocate(seq)

    drafts = [20, 21, 22]
    for token in drafts:
        seq.append_token(token)
        bm.append(seq)

    # Target: agrees with 20 and 21, disagrees on the third proposal.
    run = verify_run(_logits([20, 21, 99, 98], vocab=128), drafts)
    assert run.accepted == 2 and run.bonus == 99, run

    drop = len(drafts) - run.accepted
    if drop:
        del seq.token_ids[-drop:]
        seq.num_tokens -= drop
        seq.last_token = seq.token_ids[-1]
        while len(seq.block_table) > seq.num_blocks:
            block_id = seq.block_table.pop()
            block = bm.blocks[block_id]
            if block.ref_count > 0:
                block.ref_count -= 1
            if block.ref_count == 0:
                bm.used_block_ids.discard(block_id)
                if block_id not in bm.free_block_ids:
                    bm.free_block_ids.append(block_id)
    seq.append_token(run.bonus)
    bm.append(seq)

    assert seq.token_ids == list(range(10)) + [20, 21, 99], seq.token_ids
    assert seq.last_token == 99
    assert len(seq.block_table) >= seq.num_blocks


def test_greedy_equivalence_engine(model: str, gamma: int, max_tokens: int = 32):
    """Speculative decoding must reproduce baseline tokens exactly."""
    from transformers import AutoTokenizer

    from myvllm.engine.llm_engine import LLMEngine
    from myvllm.sampling_parameters import SamplingParams

    shape = {
        "hidden_size": 1024, "num_heads": 16, "head_dim": 128,
        "num_kv_heads": 8, "intermediate_size": 3072,
    }
    base = {
        "max_num_sequences": 8, "max_num_batched_tokens": 512,
        "max_cached_blocks": 256, "block_size": 128, "world_size": 1,
        "model_name_or_path": model, "enforce_eager": True, "dtype": "bfloat16",
        "vocab_size": 151936, **shape, "num_layers": 28,
        "tie_word_embeddings": True, "base": 1000000, "rms_norm_epsilon": 1e-6,
        "qkv_bias": False, "scale": 1, "max_position": 32768, "ffn_bias": False,
        "max_model_length": 256, "gpu_memory_utilization": 0.7, "eos": 151645,
    }
    draft = {"model_name_or_path": model, "vocab_size": 151936, **shape,
             "num_layers": 28, "tie_word_embeddings": True, "base": 1000000,
             "rms_norm_epsilon": 1e-6, "qkv_bias": False, "scale": 1.0,
             "max_position": 32768, "ffn_bias": False}

    tokenizer = AutoTokenizer.from_pretrained(model)
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": "list all prime numbers within 100"}],
            tokenize=False, add_generation_prompt=True,
        )
    ]
    sampling = SamplingParams(
        temperature=1e-5, max_tokens=max_tokens, max_model_length=256,
        ignore_eos=True,
    )

    def run(config):
        import gc
        engine = LLMEngine(config=config)
        ids = [list(t) for t in engine.generate(prompts, sampling)["token_ids"]]
        stats = engine.get_speculative_stats()
        engine.exit()
        del engine
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        return ids, stats

    baseline, _ = run({**base, "speculative": False})
    spec, stats = run(
        {**base, "speculative": True, "speculative_gamma": gamma,
         "speculative_draft_model_config": draft, "draft_kv_blocks": 32}
    )
    assert baseline == spec, (baseline, spec)
    print(f"  equivalence OK at gamma={gamma}; stats={stats}")
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true", help="also run engine tests")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--gamma", type=int, default=4)
    args = parser.parse_args()

    test_accept_rule_basics()
    print("accept-rule basics OK")
    test_accept_rule_matches_longest_common_prefix()
    print("accept-rule equivalence vs longest-common-prefix OK")
    test_stats_ratios()
    print("stats ratios OK")
    test_bookkeeping_append_truncate_bonus()
    print("append/truncate/bonus bookkeeping OK")

    if args.gpu:
        test_greedy_equivalence_engine(args.model, args.gamma)
        print("greedy equivalence (engine) OK")
    else:
        print("(skipped GPU equivalence; pass --gpu to run it)")


if __name__ == "__main__":
    main()
