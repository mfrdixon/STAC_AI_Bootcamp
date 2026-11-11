"""Offline architectural proof, not a trained investment decision system.

Two models share one evaluation core (see common.run_experiment):
  main_qwen  Qwen2.5-0.5B-Instruct, causal LM, loaded from QWEN_MODEL_PATH
  main_flan  google/flan-t5-base, seq2seq, loaded from the HuggingFace cache
"""

from stac_ai_bootcamp.utility import set_huggingface_offline  # noqa: I001

set_huggingface_offline()  # must run before transformers/torch import

import functools
import json

import torch
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

from stac_ai_bootcamp.experiments.attention_order.dictionary_invariance import common
from stac_ai_bootcamp.utility import (
    configure_torch,
    get_flan_model_id,
    get_flan_model_rev,
    get_qwen_model_path,
    get_qwen_model_rev,
)


def build_prompt(evidence, entries):
    """The shared instruction prompt both models receive for one case/order."""
    return (
        "Choose the action using the evidence and these definitions. "
        "Reply with exactly BUY, CASH or WITHHOLD.\nEvidence: "
        + evidence
        + "\nAction definitions: "
        + json.dumps(entries)
    )


@torch.inference_mode()
def qwen_encode(model, tok, text):
    x = tok(text, return_tensors="pt")
    return model.model(**x, use_cache=False).last_hidden_state[0]


@torch.inference_mode()
def qwen_prompted(model, tok, evidence, entries):
    prompt = build_prompt(evidence, entries)
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    prefix = tok.encode(rendered, add_special_tokens=False)
    names = sorted(common.actions)
    scores = []
    prefix_scores = []
    token_scores = {}
    for name in names:
        full = tok.encode(rendered + name, add_special_tokens=False)
        assert full[: len(prefix)] == prefix
        candidate = full[len(prefix) :] + [tok.eos_token_id]
        x = torch.tensor([prefix + candidate])
        logits = (
            model(input_ids=x, use_cache=False, logits_to_keep=len(candidate) + 1)
            .logits[0]
            .float()
        )
        lp = logits[:-1].log_softmax(-1)
        vals = lp[torch.arange(len(candidate)), torch.tensor(candidate)]
        scores.append(float(vals.sum()))
        prefix_scores.append(float(vals[:-1].sum()))
        token_scores[name] = {"ids": candidate, "logprobs": vals.tolist()}
    complete = dict(
        zip(names, torch.tensor(scores, dtype=torch.float64).softmax(0).tolist())
    )
    partial = dict(
        zip(
            names,
            torch.tensor(prefix_scores, dtype=torch.float64).softmax(0).tolist(),
        )
    )
    return {
        "json_prompt_probs": complete,
        "prompt": prompt,
        "json_prefix_probs": partial,
        "candidate_tokens": token_scores,
    }


@torch.inference_mode()
def flan_encode(model, tok, text):
    x = tok(text, return_tensors="pt")
    return model.encoder(**x).last_hidden_state[0]


@torch.inference_mode()
def flan_prompted(model, tok, evidence, entries):
    prompt = build_prompt(evidence, entries)
    names = sorted(common.actions)
    inputs = tok([prompt] * 3, return_tensors="pt", padding=True)
    targets = tok(names, return_tensors="pt", padding=True)
    labels = targets.input_ids.clone()
    labels[targets.attention_mask == 0] = -100
    logits = model(**inputs, labels=labels).logits.float().log_softmax(-1)
    ids = labels.clamp_min(0)
    lp = logits.gather(-1, ids.unsqueeze(-1)).squeeze(-1)
    scores = (lp * (labels != -100)).sum(-1).double()
    # Includes EOS: complete answer probabilities, normalized over three answers.
    return {
        "json_prompt_probs": dict(zip(names, scores.softmax(0).tolist())),
        "prompt": prompt,
    }


def main_qwen() -> None:
    configure_torch(seed=20261019)
    model_path = get_qwen_model_path()
    tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        attn_implementation="eager",
        torch_dtype=torch.float32,
    ).eval()
    common.run_experiment(
        output_name="attention_order/dictionary_invariance/qwen",
        manifest_extra={"model": str(model_path)},
        model=model,
        tok=tok,
        encode=functools.partial(qwen_encode, model, tok),
        prompted=functools.partial(qwen_prompted, model, tok),
        summary_fields=["separate_probs", "json_prompt_probs", "json_prefix_probs"],
        model_revision=get_qwen_model_rev(),
    )


def main_flan() -> None:
    configure_torch(seed=20261019)
    flan_id: str = get_flan_model_id()
    tok = AutoTokenizer.from_pretrained(flan_id, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        flan_id, local_files_only=True
    ).eval()
    common.run_experiment(
        output_name="attention_order/dictionary_invariance/flan",
        manifest_extra={"model": flan_id},
        model=model,
        tok=tok,
        encode=functools.partial(flan_encode, model, tok),
        prompted=functools.partial(flan_prompted, model, tok),
        summary_fields=["separate_probs", "json_prompt_probs"],
        model_revision=get_flan_model_rev(),
    )
