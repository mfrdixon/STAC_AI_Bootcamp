# Exploratory tiny numerical adaptation (outside the action-focused paper)

Train an 18,818-parameter numerical transformer on 8,000 synthetic waiting-time sets; evaluate 1,000 independent held-out sets under ten permutations. Values are continuous embeddings, not language tokens. No pretrained language model or LoRA is involved. All weights train. The blocked model adapts for 300 additional steps after 500 ordinary steps; there is no matched additional-training ordinary control, so do not attribute an improvement solely to architecture or compare training efficiency.

Run `python run.py` with PyTorch installed. Training, test and permutation seeds are fixed in manifest.json. The output head learns a fraction estimate and binary majority decision; no hand-coded threshold is applied to its fraction prediction to obtain the majority decision.

Measured majority accuracy: ordinary 97.78%, blocked without adaptation 99.70%, blocked after adaptation 99.70%. Mean fraction errors: 1.721, 3.193 and 2.024 percentage points respectively. Blocking improves the majority metric here but worsens fraction regression; adaptation partly recovers fraction accuracy without surpassing the ordinary model. Ordinary answer flips occur on 54/1,000 sets; both blocked configurations have zero flips. Blocked residual floating-point score variation is about 1e-7. Identical repeats are exact. One seed and one data distribution are not a robust performance comparison. Checkpoints and recorded predictions are included; loading PyTorch checkpoint files should only be done from a trusted source.

This exploratory result is intentionally excluded from the action-focused working paper and slides.
