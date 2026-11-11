from stac_ai_bootcamp.utility import set_huggingface_offline  # noqa: I001

set_huggingface_offline()  # must run before transformers/torch import

import hashlib
import json
import random
import re
import time

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from stac_ai_bootcamp.utility import (
    configure_torch,
    experiment_output,
    get_qwen_model_id,
    get_qwen_model_path,
    get_qwen_model_rev,
    save_json,
)


QUESTION = (
    "These are synthetic waiting times in minutes for 50 different patients. "
    "Count how many values are strictly greater than 30 minutes and calculate their percentage of all 50 patients. "
    "Did MORE THAN HALF of the patients wait more than 30 minutes? "
    "Reply on one line in exactly this format: YES; count=N; percentage=P% or NO; count=N; percentage=P%. "
    "Do not explain. Waiting times: "
)


def build_manifest(rng):
    """Draw the frozen sample, orderings and shuffled job list from ``rng``.

    The same generator draws the data and shuffles the jobs, so its consumption
    order must stay exactly as it was. Returns ``(manifest, values, orders, jobs,
    count, expected)``.
    """
    values = [round(rng.uniform(1, 60), 1) for _ in range(50)]
    orders = [list(range(50))]
    for _ in range(9):
        p = list(range(50))
        rng.shuffle(p)
        orders.append(p)
    count = sum(v > 30 for v in values)
    expected = "YES" if count > 25 else "NO"
    jobs = [(oi, r) for oi in range(10) for r in range(3)]
    rng.shuffle(jobs)
    manifest = {
        "seed": 20261022,
        "values": values,
        "orders": orders,
        "repeats": 3,
        "execution_order": jobs,
        "question": QUESTION,
        "expected_count": count,
        "expected_percentage": 2 * count,
        "expected_majority": expected,
        "model": get_qwen_model_id(),
        "revision": get_qwen_model_rev(),
        "protocol": "Ordinary unmodified model; single fixed sample; raw YES/NO prefix scoring without EOS; unconstrained greedy generation; no tools",
    }
    return manifest, values, orders, jobs, count, expected


def load_model_and_tokenizer():
    modelpath = get_qwen_model_path()
    tok = AutoTokenizer.from_pretrained(modelpath, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        modelpath,
        local_files_only=True,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    ).eval()
    gcfg = GenerationConfig(
        do_sample=False,
        num_beams=1,
        repetition_penalty=1.0,
        max_new_tokens=48,
        bos_token_id=tok.bos_token_id,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.eos_token_id,
    )
    return tok, model, gcfg


def score_and_generate(model, tok, gcfg, oi, repeat, values, orders):
    """Score YES/NO by prefix logprob, greedily generate, and parse one trial."""
    nums = [values[i] for i in orders[oi]]
    assert sorted(nums) == sorted(values)
    prompt = QUESTION + json.dumps(nums)
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    prefix = tok.encode(rendered, add_special_tokens=False)
    x = torch.tensor([prefix])
    scores = {}
    pieces = {}
    for answer in ["YES", "NO"]:
        full = tok.encode(rendered + answer, add_special_tokens=False)
        assert full[: len(prefix)] == prefix
        c = full[len(prefix) :]
        lp = (
            model(
                input_ids=torch.tensor([full]),
                use_cache=False,
                logits_to_keep=len(c) + 1,
            )
            .logits[0, :-1]
            .float()
            .log_softmax(-1)
        )
        scores[answer] = float(lp[torch.arange(len(c)), torch.tensor(c)].sum())
        pieces[answer] = c
    pyes = float(
        torch.tensor([scores["YES"], scores["NO"]], dtype=torch.float64).softmax(0)[0]
    )
    generated = model.generate(
        input_ids=x, attention_mask=torch.ones_like(x), generation_config=gcfg
    )
    ids = generated[0, len(prefix) :].tolist()
    answer = tok.decode(ids, skip_special_tokens=True)
    m = re.fullmatch(
        r"\s*(YES|NO);\s*count=(\d+);\s*percentage=(\d+(?:\.\d+)?)%\s*", answer
    )
    parsed = (
        {"majority": m[1], "count": int(m[2]), "percentage": float(m[3])}
        if m
        else None
    )
    return {
        "order": oi,
        "repeat": repeat,
        "prompt": prompt,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "answer": answer,
        "parsed": parsed,
        "p_yes": pyes,
        "logprobs": scores,
        "candidate_token_ids": pieces,
        "generated_ids": ids,
        "truncated": len(ids) >= 48 and ids[-1] != tok.eos_token_id,
    }


def summarize(rows, count, expected):
    """Order/repeat spread plus answer-format and exact-correct counts."""
    first = [
        next(z for z in rows if z["order"] == o and z["repeat"] == 0)
        for o in range(10)
    ]
    repeat_prob = max(
        max(z["p_yes"] for z in rows if z["order"] == o)
        - min(z["p_yes"] for z in rows if z["order"] == o)
        for o in range(10)
    )
    return {
        "expected_count": count,
        "expected_percentage": count * 2,
        "expected_majority": expected,
        "orders": 10,
        "repeats": 3,
        "total": 30,
        "valid_format": sum(z["parsed"] is not None for z in rows),
        "truncated": sum(z["truncated"] for z in rows),
        "distinct_generated_outputs": len({z["answer"] for z in first}),
        "distinct_majority_answers": sorted(
            {z["parsed"]["majority"] for z in first if z["parsed"]}
        ),
        "distinct_counts": sorted({z["parsed"]["count"] for z in first if z["parsed"]}),
        "distinct_percentages": sorted(
            {z["parsed"]["percentage"] for z in first if z["parsed"]}
        ),
        "exact_correct": sum(
            z["parsed"]
            == {"majority": expected, "count": count, "percentage": float(count * 2)}
            for z in rows
        ),
        "p_yes_min": min(z["p_yes"] for z in first),
        "p_yes_max": max(z["p_yes"] for z in first),
        "max_identical_prompt_probability_range": repeat_prob,
        "repeat_text_changes": sum(
            len({z["answer"] for z in rows if z["order"] == o}) > 1 for o in range(10)
        ),
    }


def main() -> None:
    output_dir = experiment_output("attention_order/waiting_times")
    start = time.time()
    rng = random.Random(20261022)
    manifest, values, orders, jobs, count, expected = build_manifest(rng)
    save_json(output_dir / "manifest.json", manifest)
    print("Frozen manifest:", count, expected, flush=True)
    configure_torch(seed=20261022)
    tok, model, gcfg = load_model_and_tokenizer()
    rows = []
    with torch.inference_mode(), open(output_dir / "outputs.jsonl", "w") as out:
        for k, (oi, repeat) in enumerate(jobs):
            row = score_and_generate(
                model, tok, gcfg, oi, repeat, values, orders
            )
            out.write(json.dumps(row) + "\n")
            out.flush()
            rows.append(row)
            print(
                f"{k + 1}/30: order {oi} repeat {repeat}: {row['answer']!r} | P(YES)={row['p_yes']:.6f} | {time.time() - start:.1f}s",
                flush=True,
            )
    summary = summarize(rows, count, expected)
    summary.update(
        elapsed_seconds=time.time() - start,
        torch=torch.__version__,
        transformers=transformers.__version__,
    )
    save_json(output_dir / "summary.json", summary, print_data=True)
