"""Shared core for the dictionary-invariance experiment.

run_qwen.py and run_flan.py differ only in how they load the model and in their
encode()/prompted() callbacks; everything else — the action/case manifest, the
order-invariant cosine cross-attention readout, the evaluation loop, and the
summary aggregation — lives here.
"""

import hashlib
import itertools
import json
import time

import torch
import torch.nn.functional as F
import transformers

from stac_ai_bootcamp.utility import experiment_output, total_variation, winner


actions = {
    "BUY": "Take a long position when eligible actual EPS exceeds consensus EPS.",
    "CASH": "Do not take a position when eligible actual EPS is at or below consensus EPS.",
    "WITHHOLD": "Defer the decision when required evidence is missing or conflicting.",
}
cases = [
    ("positive", "Eligible actual EPS is 1.34. Eligible consensus EPS is 1.07.", "BUY"),
    (
        "negative",
        "Eligible actual EPS is 0.85. Eligible consensus EPS is 1.07.",
        "CASH",
    ),
    ("equal", "Eligible actual EPS is 1.07. Eligible consensus EPS is 1.07.", "CASH"),
    (
        "missing",
        "Actual EPS is unavailable. Eligible consensus EPS is 1.07.",
        "WITHHOLD",
    ),
    (
        "conflict",
        "Two eligible actual EPS sources conflict: 1.34 and 0.85. Eligible consensus EPS is 1.07.",
        "WITHHOLD",
    ),
]


@torch.inference_mode()
def separate(encode, evidence, entries):
    # Each action is encoded independently; entries never share encoder context.
    H = F.normalize(encode(evidence), dim=-1)
    scores = {}
    for name, desc in entries.items():
        q = F.normalize(encode(name + ": " + desc).mean(0), dim=0)
        attn = torch.softmax(H @ q, dim=0)
        z = attn @ H
        scores[name] = float(q @ z)
    # Canonical summation order removes floating-point reduction-order differences.
    names = sorted(scores)
    probs = torch.tensor([scores[n] for n in names], dtype=torch.float64).softmax(0)
    return scores, dict(zip(names, probs.tolist()))


def run_experiment(
    output_name,
    manifest_extra,
    model,
    tok,
    encode,
    prompted,
    summary_fields,
    model_revision,
):
    """Drive the full dictionary-invariance evaluation and write the artifacts.

    output_name     relative experiment path for experiment_output()
    manifest_extra  model-specific keys merged into the manifest (e.g. "model")
    model, tok      the loaded model/tokenizer (for stats and tokenization)
    encode          text -> last_hidden_state[0] callable
    prompted        (evidence, entries) -> dict of row fields (incl. json_prompt_probs, prompt)
    summary_fields  probability fields to aggregate (qwen has three, flan has two)
    model_revision  commit hash recorded in the summary
    """
    output_dir = experiment_output(output_name)
    start = time.time()
    orders = list(itertools.permutations(actions))
    # manifest_extra first (e.g. "model") to preserve the original key order,
    # which the manifest_sha256 in the summary depends on.
    manifest: dict = {
        **manifest_extra,
        "actions": actions,
        "cases": cases,
        "orders": orders,
        "repeats": 5,
        "seed": 20261019,
        "head": "untrained shared cosine cross-attention; no learned projections",
        "scope": "Architectural invariance only. Policy-derived labels are not investment outcomes.",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    rows = []
    for case_id, evidence, expected in cases:
        for oi, order in enumerate(orders):
            for repeat in range(5):
                entries = {name: actions[name] for name in order}
                raw, p = separate(encode, evidence, entries)
                row = {
                    "case": case_id,
                    "expected": expected,
                    "order": list(order),
                    "repeat": repeat,
                    "separate_scores": raw,
                    "separate_probs": p,
                }
                row.update(prompted(evidence, entries))
                rows.append(row)
            print(
                f"{case_id} order {oi + 1}/{len(orders)}: {time.time() - start:.1f}s",
                flush=True,
            )
        (output_dir / "outputs.json").write_text(json.dumps(rows, indent=2))

    summary = {}
    for field in summary_fields:
        order_tv = []
        repeat_tv = []
        flips = 0
        affected = 0
        for case_id, _, _ in cases:
            r = [x for x in rows if x["case"] == case_id]
            first = [x for x in r if x["repeat"] == 0]
            affected += len({winner(x[field]) for x in first}) > 1
            for a, b in itertools.combinations(first, 2):
                order_tv.append(total_variation(a[field], b[field]))
                flips += winner(a[field]) != winner(b[field])
            for oi in range(len(orders)):
                for a, b in itertools.combinations(r[oi * 5 : (oi + 1) * 5], 2):
                    repeat_tv.append(total_variation(a[field], b[field]))
        summary[field] = {
            "order_pairs": len(order_tv),
            "mean_order_tv": sum(order_tv) / len(order_tv),
            "max_order_tv": max(order_tv),
            "order_action_changes": flips,
            "affected_cases": affected,
            "repeat_pairs": len(repeat_tv),
            "max_repeat_tv": max(repeat_tv),
            "policy_matches": sum(winner(x[field]) == x["expected"] for x in rows),
            "runs": len(rows),
        }
    assert summary["separate_probs"]["max_order_tv"] == 0
    assert summary["separate_probs"]["max_repeat_tv"] == 0
    # Evidence sensitivity control: invariant does not mean a constant representation.
    summary["separate_evidence_change_max_tv"] = max(
        total_variation(a["separate_probs"], b["separate_probs"])
        for a, b in itertools.combinations(rows[::30], 2)
    )
    summary.update(
        elapsed_seconds=time.time() - start,
        parameters=sum(p.numel() for p in model.parameters()),
        trained_parameters=0,
        device="CPU",
        torch=torch.__version__,
        transformers=transformers.__version__,
        model_revision=model_revision,
        tokenization={k: tok.encode(k) for k in actions},
        manifest_sha256=hashlib.sha256(
            (output_dir / "manifest.json").read_bytes()
        ).hexdigest(),
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
