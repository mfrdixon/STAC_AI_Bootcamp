"""Shared block-local attention machinery for the pretrained-model experiments.

`dice_order/run.py` and `waiting_times/run_block.py` were near-identical twins:
both assemble a prefix/records/suffix token sequence, build a causal + block-local
attention mask, reset per-record positions, audit forbidden attention weights via
forward hooks, and read out candidate-token logprobs. That shared skeleton lives
here so the two runners stop drifting apart.

Block tags used throughout: ``-1`` marks prefix tokens, ``-2`` marks the readout
suffix, and a non-negative value marks the record whose block a token belongs to.
"""

import torch


def build_block_sequence(prefix_ids, record_ids, suffix_ids, order):
    """Assemble one ordering into (ids, blocks, positions, suffix_start).

    ``record_ids`` is indexed by ``order``; each record keeps positions that reset
    to ``prefix_length + within_record_index`` so block-local runs are order
    invariant. The suffix retains fixed global positions.
    """
    ids = list(prefix_ids)
    blocks = [-1] * len(prefix_ids)
    positions = list(range(len(prefix_ids)))
    for item in order:
        rec = record_ids[item]
        ids += rec
        blocks += [item] * len(rec)
        positions += list(range(len(prefix_ids), len(prefix_ids) + len(rec)))
    suffix_start = len(ids)
    ids += suffix_ids
    blocks += [-2] * len(suffix_ids)
    positions += list(range(suffix_start, len(ids)))
    return ids, blocks, positions, suffix_start


def block_local_allowed(blocks):
    """Boolean (n, n) matrix of permitted attention under block-local masking.

    Causal, and additionally: the readout suffix (-2) sees everything causal; every
    token may read the prefix (-1); record tokens read only their own block. There
    is no record-to-record dependence.
    """
    n = len(blocks)
    idx = torch.arange(n)
    b = torch.tensor(blocks)
    causal = idx[None, :] <= idx[:, None]
    return causal & (
        (b[:, None] == -2)
        | (b[None, :] == -1)
        | ((b[:, None] >= 0) & (b[:, None] == b[None, :]))
    )


def attention_mask(blocks, mode, dtype=torch.float32):
    """Additive attention mask plus its boolean allowed matrix for one mode.

    ``mode == "ordinary"`` is plain causal; anything else is block-local. Returns
    ``(mask, allowed)`` where ``mask`` is 0 where permitted and ``-inf`` elsewhere.
    """
    n = len(blocks)
    idx = torch.arange(n)
    if mode == "ordinary":
        allowed = idx[None, :] <= idx[:, None]
    else:
        allowed = block_local_allowed(blocks)
    mask = torch.zeros(n, n, dtype=dtype).masked_fill(~allowed, float("-inf"))
    return mask, allowed


def position_ids(positions, mode):
    """Per-record reset positions for block-local mode, else global positions."""
    n = len(positions)
    chosen = positions if mode == "block_local" else list(range(n))
    return torch.tensor(chosen)[None]


class ForbiddenAttentionAuditor:
    """Track the largest attention weight landing on a forbidden cell.

    Registers a forward hook on every self-attention layer. While ``armed`` with a
    boolean forbidden mask, each hook records the max absolute weight the model
    placed on a masked cell; a well-behaved masked run keeps this at zero. Use as a
    context manager so the hooks are always removed.
    """

    def __init__(self, layers):
        self._layers = layers
        self._handles = []
        self._forbidden = None
        self.max_weight = 0.0

    def __enter__(self):
        for layer in self._layers:
            self._handles.append(layer.self_attn.register_forward_hook(self._hook))
        return self

    def __exit__(self, *exc):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        return False

    def _hook(self, module, inputs, output):
        if self._forbidden is None:
            return
        weights = output[1]
        if weights is not None:
            worst = float(
                weights[0].masked_select(self._forbidden[None]).abs().max()
            )
            self.max_weight = max(self.max_weight, worst)

    def arm(self, forbidden):
        """Begin auditing against ``forbidden`` (a boolean ~allowed matrix)."""
        self._forbidden = forbidden
        self.max_weight = 0.0

    def disarm(self):
        """Stop auditing until the next :meth:`arm`."""
        self._forbidden = None


def candidate_logprobs(logits, candidate_ids):
    """Full-vocab log-softmax reduced to one logprob per candidate token.

    ``candidate_ids`` maps a candidate label to its single-token encoding (a list
    whose first element is the token id). Returns ``(logprobs, argmax_id)`` where
    ``logprobs`` is a label -> float dict and ``argmax_id`` is the full-vocabulary
    argmax token id.
    """
    lp = logits.log_softmax(-1)
    logprobs = {label: float(lp[ids[0]]) for label, ids in candidate_ids.items()}
    return logprobs, int(logits.argmax())
