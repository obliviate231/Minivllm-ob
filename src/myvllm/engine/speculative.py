"""Draft-model speculative decoding for MiniVLLM (the simple version).

The idea in one paragraph: a small draft model proposes `gamma` tokens; the
big target model then verifies all of them in a *single* forward pass instead
of `gamma + 1` passes. Every verified token is therefore free -- it was
already paid for by the one pass -- so the decode loop advances by as many
tokens as were accepted. We use plain greedy sampling, which makes the accept
rule exact and trivial: a draft token is accepted iff the target model's
argmax points at it. No rejection-sampling correction is needed, because for
greedy decoding the accepted prefix's distribution *is* the target's.

Why this fits MiniVLLM unchanged:

  * the sequence's paged KV cache covers positions `0 .. L-1`, where `L-1` is
    the current last token. Verification of a run that *starts* at that last
    token therefore needs no KV write and no block allocation at all -- the
    last token's K/V is reused as the run's first K/V. That is why the accept
    path below is a few lines rather than a rewrite of `BlockManager`.
  * the draft proposal reuses the same verify path on the *draft* model, with
    its own private KV slots. The target's cache is untouched by drafting.

Trade-off: after acceptance we re-decode the run's last token in the
following step (one extra pass per round) so that "last token <-> cached K/V"
stays an invariant of the engine. With `gamma=5` and a realistic acceptance
rate that redundancy is a fraction of one pass, and it buys a two-method
change instead of a scheduler rewrite.
"""

from dataclasses import dataclass

import torch

from myvllm.engine.sequence import Sequence
from myvllm.sampling_parameters import SamplingParams


@dataclass
class SpeculativeStats:
    """Bookkeeping for one speculative-decoding workload.

    `draft_tokens` / `accepted_tokens` describe the draft model's quality;
    `target_forwards` vs `tokens_emitted` describes what it bought us. Plain
    decoding needs one target forward per emitted token, so that ratio is the
    speedup we should see, before paying for the draft model.
    """

    rounds: int = 0
    draft_tokens: int = 0
    accepted_tokens: int = 0
    target_forwards: int = 0
    tokens_emitted: int = 0
    draft_forwards: int = 0

    @property
    def acceptance_rate(self) -> float:
        return (
            self.accepted_tokens / self.draft_tokens if self.draft_tokens else 0.0
        )

    @property
    def mean_accepted_per_round(self) -> float:
        return self.accepted_tokens / self.rounds if self.rounds else 0.0

    @property
    def target_forwards_per_token(self) -> float:
        return (
            self.target_forwards / self.tokens_emitted
            if self.tokens_emitted
            else 0.0
        )

    def merge(self, other: "SpeculativeStats") -> None:
        self.rounds += other.rounds
        self.draft_tokens += other.draft_tokens
        self.accepted_tokens += other.accepted_tokens
        self.target_forwards += other.target_forwards
        self.tokens_emitted += other.tokens_emitted
        self.draft_forwards += other.draft_forwards

    def as_dict(self) -> dict:
        return {
            "rounds": self.rounds,
            "draft_tokens": self.draft_tokens,
            "accepted_tokens": self.accepted_tokens,
            "acceptance_rate": self.acceptance_rate,
            "mean_accepted_per_round": self.mean_accepted_per_round,
            "target_forwards": self.target_forwards,
            "target_forwards_per_token": self.target_forwards_per_token,
            "tokens_emitted": self.tokens_emitted,
            "draft_forwards": self.draft_forwards,
        }


@dataclass
class VerifyRun:
    """One speculative round's outcome for a single sequence.

    `verified` are the draft tokens the target confirmed, in order. `bonus` is
    the target's own next token at the first disagreement, which is exact by
    construction -- so a round always yields at least one new token, even when
    the draft model is wrong on its very first proposal.
    """

    verified: list[int]
    bonus: int

    @property
    def accepted(self) -> int:
        return len(self.verified)


def _restore_context(saved) -> None:
    """Put a previously captured Context back as the engine's global context.

    The draft pass overwrites the process-wide context; the target's verify
    pass and the rest of the engine step depend on the context the engine had
    already built, so it must be restored rather than cleared.
    """
    import myvllm.utils.context as context_module

    context_module._context = saved


def verify_run(target_logits: torch.Tensor, draft_tokens: list[int]) -> VerifyRun:
    """Greedy accept rule: keep the leading draft tokens the target agrees with.

    `target_logits` has one row per position of `[last_token, *draft_tokens]`.
    Row `i` predicts the token at index `i + 1`, so the token the target would
    have produced for index `i + 1` is `argmax(target_logits[i])`. Draft token
    `j` sits at index `j + 1` and is accepted while every preceding row agrees;
    the first disagreement ends the run and its argmax becomes the bonus token.
    """
    argmax = target_logits.float().argmax(dim=-1).tolist()
    accepted = 0
    for j, draft_token in enumerate(draft_tokens):
        if argmax[j] != draft_token:
            break
        accepted += 1
    # `argmax[accepted]` predicts the token after the last accepted one and is
    # always in range because `accepted <= len(draft_tokens)`.
    return VerifyRun(verified=draft_tokens[:accepted], bonus=int(argmax[accepted]))


# ---------------------------------------------------------------------------
# The draft model.
#
# Reusing `ModelRunner` for the draft would drag in the target's process group,
# CUDA graphs and KV sizing, so the draft gets the minimal runner below: load
# weights, warm up, size a small private KV cache from the memory the target
# left free, and expose exactly what speculative decoding needs.


class DraftModelRunner:
    """Minimal eager runner for the draft model (no graphs, no sampling)."""

    def __init__(self, config: dict, rank: int = 0):
        from pathlib import Path

        from myvllm.models.qwen3 import Qwen3ForCausalLM
        from myvllm.utils.loader import load_weights_from_checkpoint, resolve_checkpoint_path

        self.config = config
        self.rank = rank
        self.block_size = config["block_size"]
        dtype_name = str(config.get("draft_dtype", config.get("dtype", "bfloat16")))
        dtype_name = dtype_name.removeprefix("torch.")
        self.dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[dtype_name]

        model_config = config["speculative_draft_model_config"]
        model_name = Path(model_config["model_name_or_path"]).name
        if not model_name.startswith("Qwen3-"):
            raise ValueError(
                f"Unsupported draft model for speculation: {model_name}"
            )
        self.model = Qwen3ForCausalLM(
            vocab_size=model_config["vocab_size"],
            hidden_size=model_config["hidden_size"],
            num_heads=model_config["num_heads"],
            head_dim=model_config["head_dim"],
            scale=model_config.get("scale", 1.0),
            num_kv_heads=model_config["num_kv_heads"],
            rms_norm_epsilon=model_config["rms_norm_epsilon"],
            qkv_bias=model_config["qkv_bias"],
            base=model_config["base"],
            max_position=model_config["max_position"],
            intermediate_size=model_config["intermediate_size"],
            ffn_bias=model_config["ffn_bias"],
            num_layers=model_config["num_layers"],
            tie_word_embeddings=model_config["tie_word_embeddings"],
            block_size=self.block_size,
        )
        self.model = self.model.to(device=f"cuda:{rank}", dtype=self.dtype)
        checkpoint = model_config.get("checkpoint_path") or resolve_checkpoint_path(
            model_config["model_name_or_path"]
        )
        load_weights_from_checkpoint(self.model, checkpoint)
        self._allocate_kv_cache()
        self.model.eval()

    def _allocate_kv_cache(self) -> None:
        """Small private KV pool for the draft model's own autoregressive run.

        Sized from the memory the target left free and capped by
        `draft_kv_blocks`, so a draft model can never starve the target.
        """
        free_mem, _ = torch.cuda.mem_get_info()
        budget = int(free_mem * self.config.get("draft_gpu_memory_fraction", 0.25))
        mc = self.config["speculative_draft_model_config"]
        block_bytes = (
            self.block_size * 2 * mc["num_layers"] * mc["num_kv_heads"]
            * mc["head_dim"] * self.dtype.itemsize
        )
        # The draft pool is stateless across rounds (each row rewrites its own
        # pair), so it only needs one block per concurrently drafted sequence.
        # We ask for `draft_kv_blocks` but never exceed what the memory budget
        # allows -- the budget is a ceiling, not the target.
        wanted = max(1, int(self.config.get("draft_kv_blocks", 256)))
        affordable = max(1, budget // block_bytes)
        num_blocks = min(wanted, affordable) if budget > 0 else wanted
        self.num_blocks = num_blocks
        self.num_slots = num_blocks * self.block_size
        self.kv_cache = torch.zeros(
            2,
            mc["num_layers"],
            num_blocks,
            self.block_size,
            mc["num_kv_heads"],
            mc["head_dim"],
            device=f"cuda:{self.rank}",
            dtype=self.dtype,
        )
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    # ------------------------------------------------------------------
    # Contexts. A draft forward is a decode-shaped pass over one sequence per
    # batch row, reading and writing only its own private slots.

    def _row_block(self, row: int) -> int:
        """Block that row `row` drafts into for one whole proposal round.

        The draft extends one token at a time, so it is an ordinary
        autoregressive decode *within its own block*: step `j` writes the j-th
        slot of the row's block and attends over slots `0 .. j` of it. That
        keeps positions increasing (so the draft conditions on what it already
        wrote) and needs no KV writes outside the row's own block.
        """
        return row % self.num_blocks

    @torch.inference_mode()
    def propose(self, last_tokens: list[int], gamma: int) -> list[list[int]]:
        """Propose `gamma` tokens per row (plain batched greedy drafting)."""
        from myvllm.utils import get_context, set_context

        num_seqs = len(last_tokens)
        if num_seqs > self.config.get("max_num_sequences", 16):
            raise ValueError("draft batch larger than max_num_sequences")
        if num_seqs > self.num_blocks:
            raise ValueError(
                f"draft needs one KV block per batched sequence, but only "
                f"{self.num_blocks} are available (batch = {num_seqs}); "
                "raise draft_kv_blocks"
            )
        if gamma > self.num_blocks // max(1, self.config.get("max_num_sequences", 16)):
            raise ValueError(
                f"draft needs one KV block per step per sequence: gamma="
                f"{gamma} with batch up to "
                f"{self.config.get('max_num_sequences', 16)} needs more than "
                f"{self.num_blocks} blocks; raise draft_kv_blocks"
            )
        proposals: list[list[int]] = [[] for _ in range(num_seqs)]
        row_blocks = [self._row_block(row) for row in range(num_seqs)]
        # The draft borrows the global context object, so whatever was there
        # before must be restored afterwards -- the caller's normal decode
        # context is still needed for the target verify pass that follows.
        saved = get_context()
        try:
            # One spare slot per row holds the incoming token (`last_token`),
            # which is the draft's only context: the draft model does not see
            # the prompt, it only continues from the token the target last
            # produced. Without writing it, the first proposal would attend an
            # empty cache and be meaningless.
            if gamma + 1 > self.block_size:
                raise ValueError(
                    f"gamma={gamma} needs {gamma + 1} slots per row but the "
                    f"draft block_size is {self.block_size}; lower gamma or "
                    "raise block_size"
                )
            # The draft is a plain autoregressive decoder over its own two-slot
            # window. It runs as a *prefill-shaped* multi-token pass so that
            # positions advance with the chain (`context_lens - 1` would pin
            # every step to position 0, which degenerates the draft).
            #
            # Step `j`: the run's tokens so far are `[head, p0 .. p_{j-1}]`, and
            # we ask for the distribution at the last of them. Slot `j` of the
            # row's block holds that token, so `slot_mapping` names slot `j`
            # while `cu_seqlens` spans `j + 1` tokens -- the pass then attends
            # slots `0 .. j` and yields the next proposal.
            if gamma + 1 > self.block_size:
                raise ValueError(
                    f"gamma={gamma} needs {gamma + 1} slots per row but the "
                    f"draft block_size is {self.block_size}; lower gamma or "
                    "raise block_size"
                )
            drafted: list[list[int]] = [[] for _ in range(num_seqs)]
            # Re-feed the whole window each step: the draft's context is short
            # (grow it by one token per step), and re-running it keeps the
            # positions advancing without any cross-step cache bookkeeping.
            window: list[list[int]] = [[tok] for tok in last_tokens]
            for step in range(gamma):
                flat: list[int] = []
                slot_mapping: list[int] = []
                for row, tokens in enumerate(window):
                    block = row_blocks[row] * self.block_size
                    for offset, _ in enumerate(tokens):
                        flat.append(tokens[offset])
                        slot_mapping.append(block + offset)
                tokens_t = torch.tensor(flat, dtype=torch.long, device="cuda")
                widths = [len(t) for t in window]
                cu_q = torch.zeros(num_seqs + 1, dtype=torch.int32, device="cuda")
                cu_q[1:] = torch.tensor(
                    widths, dtype=torch.int32, device="cuda"
                ).cumsum(0)
                set_context(
                    is_prefill=True,
                    cu_seqlens_q=cu_q,
                    cu_seqlens_k=cu_q.clone(),
                    max_seqlen_q=max(widths),
                    max_seqlen_k=max(widths),
                    slot_mapping=torch.tensor(
                        slot_mapping, dtype=torch.long, device="cuda"
                    ),
                    context_lens=None,
                    block_tables=torch.tensor(
                        [[b] for b in row_blocks],
                        dtype=torch.int32,
                        device="cuda",
                    ),
                )
                hidden = self.model(tokens_t)
                logits = self.model.compute_logits(hidden)
                nxt = logits.float().argmax(dim=-1).tolist()
                for row in range(num_seqs):
                    token = int(nxt[row])
                    drafted[row].append(token)
                    window[row].append(token)
            proposals = drafted
        finally:
            _restore_context(saved)
        return proposals

    @torch.inference_mode()
    def run_verify(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.model(input_ids)
        return self.model.compute_logits(hidden)

    def call(self, method_name: str, *args):
        """Mirror `ModelRunner.call` so a TP target can broadcast to workers."""
        method = getattr(self, method_name, None)
        if method is None:
            raise ValueError(f"Unknown draft method: {method_name}")
        return method(*args)


def spec_decode_config(config: dict) -> dict | None:
    """Read the speculative-decoding knobs from the engine config."""
    if not config.get("speculative", False):
        return None
    if not config.get("speculative_draft_model_config"):
        raise ValueError(
            "speculative decoding requires 'speculative_draft_model_config'"
        )
    gamma = int(config.get("speculative_gamma", 5))
    if gamma < 1:
        raise ValueError("speculative_gamma must be at least 1")
    return {"gamma": gamma}


__all__ = [
    "DraftModelRunner",
    "SamplingParams",
    "Sequence",
    "SpeculativeStats",
    "VerifyRun",
    "spec_decode_config",
    "verify_run",
]
