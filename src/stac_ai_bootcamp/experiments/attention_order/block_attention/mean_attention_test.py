"""Analytic attention example: arithmetic mean, no training or model skill claim.
Q=K=0, V=x. Softmax weights are uniform over allowed positions.
"""

import json
import math

from stac_ai_bootcamp.experiments.attention_order.block_attention.input_semantics import (
    compile_input,
)
from stac_ai_bootcamp.utility import experiment_output, save_json


def attention_mean(x, mode):
    if mode == "causal":
        return math.fsum(math.fsum(x[: i + 1]) / (i + 1) for i in range(len(x))) / len(
            x
        )
    if mode == "block_set":
        # One value per block: every attention row has a single allowed entry.
        return math.fsum(x) / len(x)
    if mode == "unrestricted":
        # Every row attends uniformly to all fifty values; pooled rows equal mean.
        h = [math.fsum(x) / len(x)] * len(x)
        return math.fsum(h) / len(h)
    raise ValueError(mode)


MODES = ["causal", "block_set", "unrestricted"]


def evaluate(samples, orders):
    """Compute the analytic attention mean for every sample/permutation/mode/repeat."""
    rows = []
    for si, x in enumerate(samples):
        target = math.fsum(x) / len(x)
        for pi, p in enumerate(orders):
            z = [x[i] for i in p]
            assert sorted(z) == sorted(x)
            assert compile_input({"type": "set", "items": z})["mode"] == "block_set"
            for mode in MODES:
                for repeat in range(10):
                    y = attention_mean(z, mode)
                    rows.append(
                        {
                            "sample": si,
                            "permutation": pi,
                            "repeat": repeat,
                            "mode": mode,
                            "output": y,
                            "target": target,
                            "error": y - target,
                        }
                    )
    return rows


def summarize_and_assert(rows, samples, orders):
    """Per-mode range/error stats, then assert the analytic properties hold."""
    summary = {}
    for mode in MODES:
        ranges = []
        repeat_range = 0
        for si in range(len(samples)):
            r = [z for z in rows if z["mode"] == mode and z["sample"] == si]
            ranges.append(max(z["output"] for z in r) - min(z["output"] for z in r))
            for pi in range(len(orders)):
                v = [z["output"] for z in r if z["permutation"] == pi]
                repeat_range = max(repeat_range, max(v) - min(v))
        first = [z for z in rows if z["mode"] == mode and z["sample"] == 0]
        summary[mode] = {
            "first_sample_min": min(z["output"] for z in first),
            "first_sample_max": max(z["output"] for z in first),
            "first_sample_order0": next(
                z["output"] for z in first if z["permutation"] == 0
            ),
            "first_sample_order1": next(
                z["output"] for z in first if z["permutation"] == 1
            ),
            "max_order_range": max(ranges),
            "max_absolute_error": max(
                abs(z["error"]) for z in rows if z["mode"] == mode
            ),
            "max_repeat_range": repeat_range,
        }
    assert summary["block_set"]["max_absolute_error"] == 0
    assert summary["unrestricted"]["max_absolute_error"] < 1e-14
    assert summary["causal"]["max_order_range"] > 1e-5
    return summary


def main() -> None:
    output_dir = experiment_output("attention_order/block_attention")
    source = json.loads((output_dir / "iid_set_expanded_inputs.json").read_text())
    samples = source["samples"]
    orders = source["permutations"]
    rows = evaluate(samples, orders)
    summary = summarize_and_assert(rows, samples, orders)
    summary.update(
        first_sample_true_mean=math.fsum(samples[0]) / 50,
        samples=len(samples),
        items=50,
        orders=len(orders),
        repeats=10,
        evaluations=len(rows),
        weights="Q=K=0; V=x; uniform softmax within allowed positions; arithmetic mean pooling",
        training=False,
    )
    save_json(output_dir / "mean_attention_summary.json", summary, print_data=True)
    save_json(output_dir / "mean_attention_outputs.json", rows)
