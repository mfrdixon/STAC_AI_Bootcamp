# Action-order isolation and Block Attention

Two separate demonstrations: independent encoding with pretrained Qwen/FLAN, and a tiny random transformer using an explicit block mask. Neither demonstrates better investment decisions. Block masks and set-equivariant architectures are established; no foundational novelty is claimed.

## Run the small tests (CPU, no model download)

```bash
python -m pip install torch
python experiments/attention_order/block_attention/run.py
python experiments/attention_order/block_attention/input_semantics.py
python experiments/attention_order/block_attention/iid_set_expanded_test.py
python experiments/attention_order/block_attention/mean_attention_test.py
```

The expanded test generates 20 fixed samples of 50 standard-normal draws. Each sample is reused exactly over 100 permutations and ten repeats. The script asserts equal input multisets. Ten repeats test execution variation, not additional independent cases. `iid_set_expanded_inputs.json` records all actual inputs and permutations. The scalar readout has no statistical meaning.

The arithmetic-mean example is intentionally analytic: causal uniform prefix attention followed by averaging is NOT a correct mean algorithm. Blocking and unrestricted uniform attention both compute the true mean. It explains aggregation bias, not a learned model failing a task. See the paper for formulas and alternative controls.

## Explicit input semantics

```json
{"type":"set", "items":[0.12,-0.83,1.47], "duplicates":"preserve"}
```

`input_semantics.py` compiles a set declaration into singleton block IDs, local positions and mean pooling. `sequence` retains global positions and one causal block. Duplicates are preserved: this is a multiset/sample. No model infers unorderedness from the data. The tiny-model ablations explicitly select modes so mask-only and ordinary controls remain testable.

## Pretrained comparisons

Install `requirements.txt`. FLAN runner loads an already cached `google/flan-t5-base` with offline mode. Download it first with Hugging Face if necessary. Qwen runner expects `dictionary_invariance/qwen_model/` populated with revision `7ae557604adf67be50417f59c2c2f167def9a775` of `Qwen/Qwen2.5-0.5B-Instruct`. Model weights are intentionally excluded. Then run the appropriate `run.py` (writes results beside itself).

Both runners freeze weights and use independent hidden-vector encoding plus an untrained shared attention scorer. JSON baseline likelihoods include EOS; Qwen also records prefix scores without EOS. Scores are not calibrated probabilities. This test does not independently evaluate generated answers.

## Results

- Qwen JSON: 32/75 pairwise order comparisons change highest-ranked action; independent encoding: 0/75.
- Tiny random transformer: ordinary 52/75, mask-only 51/75, mask plus local positions 0/75.
- Expanded 50-number diagnostic: 60,000 evaluations; maximum ordinary output range 0.187697, mask-only 0.067648, full construction 0. Identical-repeat variation zero in all conditions.

All results are controlled synthetic tests with deterministic inference. See [working paper](../../papers/block_attention/block_attention_working_paper.pdf), with detailed methodology, limitations and preliminary results. Local scripts record PyTorch 2.13.0 and Transformers 4.57.3; tolerate numerical differences across platforms.
