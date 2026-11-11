"""Set semantics test; no training, semantic-accuracy or randomness inference claim.

Two sizes share one evaluation core:
  main_test           10 samples, 20 permutations, 5 repeats, prefix iid_set
  main_expanded_test  20 samples, 100 permutations, 10 repeats, prefix iid_set_expanded
"""

import time

import torch
from torch import nn

from stac_ai_bootcamp.utility import (
    configure_torch,
    experiment_output,
    save_json,
)


class TinySet(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(1, 32)
        self.pos = nn.Embedding(50, 32)
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "attn": nn.MultiheadAttention(
                            32, 4, batch_first=True, dropout=0
                        ),
                        "n1": nn.LayerNorm(32),
                        "n2": nn.LayerNorm(32),
                        "ff": nn.Sequential(
                            nn.Linear(32, 64), nn.GELU(), nn.Linear(64, 32)
                        ),
                    }
                )
                for _ in range(2)
            ]
        )
        self.head = nn.Linear(32, 1)

    def forward(self, x, mode):
        # Each real is one continuous input embedding, not a digit-token string.
        pos = (
            torch.arange(50)
            if mode != "block_set"
            else torch.zeros(50, dtype=torch.long)
        )
        h = (self.embed(x[:, None]) + self.pos(pos))[None]
        mask = (
            torch.triu(torch.ones(50, 50, dtype=torch.bool), 1)
            if mode == "ordinary"
            else ~torch.eye(50, dtype=torch.bool)
        )
        forbidden = []
        for l in self.layers:
            z, w = l["attn"](
                h, h, h, attn_mask=mask, need_weights=True, average_attn_weights=False
            )
            forbidden.append(float(w.masked_select(mask[None, None]).abs().max()))
            h = l["n1"](h + z)
            h = l["n2"](h + l["ff"](h))
        # Symmetric aggregation; no last-item or list-position readout.
        return float(self.head(h[0].double().mean(0).float())), h[0], max(forbidden)


MODES = ["ordinary", "block_global_positions", "block_set"]


def evaluate_modes(model, sample_list, orders, repeats):
    """Run every (sample, mode, permutation, repeat) trial.

    Returns ``(rows, max_equivariance)`` where the latter is the largest observed
    departure from permutation-equivariance of the block_set hidden states.
    """
    rows = []
    max_equivariance = 0
    with torch.inference_mode():
        for si, x in enumerate(sample_list):
            base_h = None
            for mode in MODES:
                for pi, perm in enumerate(orders):
                    assert torch.equal(
                        torch.sort(x[perm]).values, torch.sort(x).values
                    )
                    for repeat in range(repeats):
                        y, h, f = model(x[perm], mode)
                        rows.append(
                            {
                                "sample": si,
                                "mode": mode,
                                "permutation": pi,
                                "repeat": repeat,
                                "output": y,
                                "forbidden_attention_max": f,
                            }
                        )
                        if mode == "block_set":
                            if pi == 0 and repeat == 0:
                                base_h = h.clone()
                            max_equivariance = max(
                                max_equivariance,
                                float((h - base_h[perm]).abs().max()),
                            )
    return rows, max_equivariance


def summarize_modes(rows, samples, permutations):
    """Per-mode output-range statistics over the first repeats and per permutation."""
    summary = {}
    for mode in MODES:
        ranges = []
        repeat_max = 0
        for si in range(samples):
            r = [z for z in rows if z["mode"] == mode and z["sample"] == si]
            ys = [z["output"] for z in r if z["repeat"] == 0]
            ranges.append(max(ys) - min(ys))
            for pi in range(permutations):
                ys = [z["output"] for z in r if z["permutation"] == pi]
                repeat_max = max(repeat_max, max(ys) - min(ys))
        summary[mode] = {
            "max_output_range": max(ranges),
            "mean_output_range": sum(ranges) / samples,
            "first_sample_range": ranges[0],
            "max_repeat_range": repeat_max,
        }
    return summary


def assert_invariance(model, summary, sample_list, max_equivariance):
    """Value-sensitivity probe plus block_set / ordinary assertions.

    Returns ``(a, b)``, the first sample's block_set output and its +1-shifted
    counterpart, recorded in the summary as the value-change control.
    """
    with torch.inference_mode():
        x = sample_list[0]
        a, _, _ = model(x, "block_set")
        b, _, _ = model(x + 1, "block_set")
    assert summary["block_set"]["max_output_range"] < 1e-6
    assert max_equivariance < 1e-6
    assert summary["ordinary"]["max_output_range"] > 1e-5
    assert abs(a - b) > 1e-5
    return a, b


def run_iid_set(samples: int, permutations: int, repeats: int, prefix: str) -> None:
    output_dir = experiment_output("attention_order/block_attention")
    configure_torch(seed=20261021, num_threads=2)
    model = TinySet().eval()
    g = torch.Generator().manual_seed(81234)
    # Independent samples of 50 draws each; identity order plus random permutations.
    sample_list = [torch.randn(50, generator=g) for _ in range(samples)]
    orders = [torch.arange(50)] + [
        torch.randperm(50, generator=g) for _ in range(permutations - 1)
    ]
    start = time.time()
    rows, max_equivariance = evaluate_modes(model, sample_list, orders, repeats)
    summary = summarize_modes(rows, samples, permutations)
    a, b = assert_invariance(model, summary, sample_list, max_equivariance)
    summary.update(
        parameters=sum(p.numel() for p in model.parameters()),
        trained=False,
        seed=20261021,
        data_seed=81234,
        distribution="standard normal, independent pseudorandom draws",
        samples=samples,
        items_per_sample=50,
        permutations=permutations,
        repeats=repeats,
        evaluations=len(rows),
        block_max_hidden_equivariance_error=max_equivariance,
        block_forbidden_attention_max=max(
            z["forbidden_attention_max"] for z in rows if z["mode"] == "block_set"
        ),
        first_sample_original_output=a,
        first_sample_shifted_by_one_output=b,
        value_change_absolute_output=abs(a - b),
        seconds=time.time() - start,
        tolerance=1e-6,
    )
    save_json(output_dir / f"{prefix}_outputs.json", rows)
    save_json(output_dir / f"{prefix}_summary.json", summary, print_data=True)
    save_json(
        output_dir / f"{prefix}_inputs.json",
        {
            "samples": [x.tolist() for x in sample_list],
            "permutations": [o.tolist() for o in orders],
        },
    )


def main_test() -> None:
    run_iid_set(samples=10, permutations=20, repeats=5, prefix="iid_set")


def main_expanded_test() -> None:
    run_iid_set(samples=20, permutations=100, repeats=10, prefix="iid_set_expanded")
