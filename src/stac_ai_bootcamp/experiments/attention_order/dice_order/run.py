from stac_ai_bootcamp.utility import set_huggingface_offline  # noqa: I001

set_huggingface_offline()  # must run before transformers/torch import

import hashlib
import itertools
import json
import random
import time

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from stac_ai_bootcamp.experiments.attention_order import block_common
from stac_ai_bootcamp.utility import (
    configure_torch,
    experiment_output,
    get_qwen_model_id,
    get_qwen_model_path,
    get_qwen_model_rev,
    save_json,
    total_variation,
)


QUESTION = "The following 50 numbers were observed. Predict the next number. Reply with exactly one digit from 1 to 6 and nothing else. Observations:\n"


def generate_data(rng):
    """Draw the 50 die values and ten orderings (identity plus nine shuffles).

    Consumes ``rng`` in place; the same generator later shuffles the job list, so
    its draw sequence must stay exactly as it was.
    """
    values = [rng.randint(1, 6) for _ in range(50)]
    orders = [list(range(50))]
    for _ in range(9):
        p = list(range(50))
        rng.shuffle(p)
        orders.append(p)
    return values, orders


def build_manifest(values, orders):
    return {
        "seed": 20261024,
        "values": values,
        "orders": orders,
        "repeats": 3,
        "conditions": ["ordinary", "block_local"],
        "question": QUESTION,
        "source_hidden_from_model": "Independent fair die, pseudorandom draws; true next-value distribution is uniform",
        "record_template": "Observation: {value}.\\n",
        "purpose": "Measure order sensitivity separately from agreement with hidden data-generating distribution",
        "model": get_qwen_model_id(),
        "revision": get_qwen_model_rev(),
    }


def load_model_and_tokenizer():
    path = get_qwen_model_path()
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        path,
        local_files_only=True,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    ).eval()
    return tok, model


def tokenize_fixtures(tok, values):
    """Encode the fixed prompt segments, record templates and candidate digits."""
    enc = lambda s: tok.encode(s, add_special_tokens=False)
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": QUESTION + "RECORDS_INSERTED_HERE"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    pre, suf = rendered.split("RECORDS_INSERTED_HERE")
    prefix = enc(pre)
    suffix = enc(suf)
    records = [enc(f"Observation: {v}.\n") for v in values]
    candidates = {str(i): enc(str(i)) for i in range(1, 7)}
    assert all(len(v) == 1 for v in candidates.values())
    return pre, suf, prefix, suffix, records, candidates


def evaluate_job(
    model, tok, mode, o, r, values, orders, prefix, suffix, records, candidates, auditor
):
    """Run one (mode, order, repeat) trial and return (row, allowed, suffix_start).

    When ``o == 0 and r == 0`` the auditor is armed so the forbidden-attention
    weight is captured; the caller reads it back off the auditor.
    """
    order = orders[o]
    assert sorted(values[i] for i in order) == sorted(values)
    ids, blocks, positions, suffix_start = block_common.build_block_sequence(
        prefix, records, suffix, order
    )
    mask, allowed = block_common.attention_mask(blocks, mode, torch.float32)
    pos = block_common.position_ids(positions, mode)
    check = o == 0 and r == 0
    if check:
        auditor.arm(~allowed)
    result = model(
        input_ids=torch.tensor([ids]),
        position_ids=pos,
        attention_mask={"full_attention": mask[None, None]},
        use_cache=False,
        output_attentions=check,
        logits_to_keep=1,
    )
    logits = result.logits[0, -1].float()
    logprobs, argmax = block_common.candidate_logprobs(logits, candidates)
    vals = torch.tensor(
        [logprobs[str(i)] for i in range(1, 7)], dtype=torch.float64
    )
    ps = vals.softmax(0).tolist()
    rawmass = float(vals.exp().sum())
    row = {
        "mode": mode,
        "order": o,
        "repeat": r,
        "probabilities": dict(zip(map(str, range(1, 7)), ps)),
        "candidate_mass": rawmass,
        "top_candidate": str(ps.index(max(ps)) + 1),
        "full_vocab_argmax_id": argmax,
        "full_vocab_argmax": tok.decode([argmax]),
        "input_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
    }
    del result
    return row, allowed, suffix_start


def summarize(rows, conditions):
    """Per-mode order/repeat total-variation statistics over the first repeats."""
    uniform = {str(i): 1 / 6 for i in range(1, 7)}
    summary = {}
    for mode in conditions:
        first = [
            next(
                x
                for x in rows
                if x["mode"] == mode and x["order"] == o and x["repeat"] == 0
            )
            for o in range(10)
        ]
        repeat_tv = []
        for o in range(10):
            group = [x for x in rows if x["mode"] == mode and x["order"] == o]
            repeat_tv += [
                total_variation(a["probabilities"], b["probabilities"])
                for a, b in itertools.combinations(group, 2)
            ]
        pairs = list(itertools.combinations(first, 2))
        summary[mode] = {
            "original_order_probs": first[0]["probabilities"],
            "max_order_tv": max(
                total_variation(a["probabilities"], b["probabilities"])
                for a, b in pairs
            ),
            "mean_order_tv": sum(
                total_variation(a["probabilities"], b["probabilities"])
                for a, b in pairs
            )
            / len(pairs),
            "top_candidate_changes": sum(
                a["top_candidate"] != b["top_candidate"] for a, b in pairs
            ),
            "order_pairs": len(pairs),
            "max_repeat_tv": max(repeat_tv),
            "min_tv_from_uniform": min(
                total_variation(x["probabilities"], uniform) for x in first
            ),
            "max_tv_from_uniform": max(
                total_variation(x["probabilities"], uniform) for x in first
            ),
            "candidate_mass_min": min(x["candidate_mass"] for x in first),
            "candidate_mass_max": max(x["candidate_mass"] for x in first),
            "top_candidates": sorted({x["top_candidate"] for x in first}),
            "full_vocab_argmaxes": sorted({x["full_vocab_argmax"] for x in first}),
        }
    return summary


def main() -> None:
    output_dir = experiment_output("attention_order/dice_order")
    start = time.time()
    rng = random.Random(20261024)
    values, orders = generate_data(rng)
    manifest = build_manifest(values, orders)
    save_json(output_dir / "manifest.json", manifest)
    configure_torch(seed=20261024)
    tok, model = load_model_and_tokenizer()
    pre, suf, prefix, suffix, records, candidates = tokenize_fixtures(tok, values)
    manifest.update(prefix=pre, suffix=suf, candidate_ids=candidates)
    save_json(output_dir / "manifest.json", manifest)
    jobs = [
        (mode, o, r)
        for mode in manifest["conditions"]
        for o in range(10)
        for r in range(3)
    ]
    rng.shuffle(jobs)
    rows = []
    audit = {}
    auditor = block_common.ForbiddenAttentionAuditor(model.model.layers)
    with torch.inference_mode(), auditor, open(
        output_dir / "outputs.jsonl", "w"
    ) as out:
        for k, (mode, o, r) in enumerate(jobs):
            row, allowed, suffix_start = evaluate_job(
                model,
                tok,
                mode,
                o,
                r,
                values,
                orders,
                prefix,
                suffix,
                records,
                candidates,
                auditor,
            )
            rows.append(row)
            out.write(json.dumps(row) + "\n")
            out.flush()
            if o == 0 and r == 0:
                audit[mode] = {
                    "max_forbidden": auditor.max_weight,
                    "readout_sees_all_records": bool(
                        allowed[-1, len(prefix) : suffix_start].all()
                    ),
                }
                auditor.disarm()
            if (k + 1) % 10 == 0:
                print(k + 1, "/60", round(time.time() - start, 1), "sec", flush=True)
    summary = summarize(rows, manifest["conditions"])
    for o in range(10):
        assert len({x["input_sha256"] for x in rows if x["order"] == o}) == 1
    summary.update(
        audit=audit,
        seconds=time.time() - start,
        total_evaluations=len(rows),
        torch=torch.__version__,
        transformers=transformers.__version__,
        block_invariance_tolerance=1e-5,
        block_within_tolerance=summary["block_local"]["max_order_tv"] < 1e-5,
    )
    save_json(output_dir / "summary.json", summary, print_data=True)
