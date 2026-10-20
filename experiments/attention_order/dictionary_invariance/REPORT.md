# Dictionary order invariance: small local proof

## Scope and execution

Completed offline on CPU in 51.64 seconds after loading a cached model. FLAN-T5-base has 247,577,856 parameters; no parameters were trained. Revision and environment are recorded in summary.json. Run with the Python environment containing torch and transformers: `python run.py`.

Five constructed EPS cases (above consensus, below, equal, missing and conflicting) were tested under all six dictionary orders, with five identical repeats: 150 evaluations per method. These are five distinct cases, not 150 independent observations. Labels reflect the stipulated EPS policy, not market outcomes.

## Methods

**JSON in the prompt:** the full dictionary is serialized in insertion order. The existing model scores BUY, CASH and WITHHOLD by teacher forcing all tokens, including EOS. Scores are normalized over these three complete answers. We measure ranked answers, not independently generated answers.

**Dictionary processed by application code:** evidence is encoded separately. Each action name and description is independently encoded through the same frozen encoder; its mean hidden vector is normalized. Normalized evidence token vectors H provide keys and values. For each action q, use attention weights softmax(Hq), context z = H^T softmax(Hq), and score q^T z. This is an untrained shared cosine-attention scorer, not a learned decision head. Each input is recomputed on each run; caching is not responsible for invariance. Final normalization uses canonical action-name order, and ties use an action-name tie-break, eliminating ordering effects from summation and ties.

## Results

| Measure | JSON in prompt | Separate encoding/scoring |
|---|---:|---:|
| Mean total-variation distance across orders | 0.054584 | 0 |
| Maximum total-variation distance across orders | 0.148761 | 0 |
| Changed top action / 75 within-case order pairs | 0 | 0 |
| Maximum distance / 300 identical-repeat pairs | 0 | 0 |
| Stipulated-policy matches / 150 runs | 30 | 30 |

Total variation is half the sum of absolute changes in named-action probabilities. The separate scorer's softmax values are normalized compatibility scores, not language-generation probabilities or calibrated beliefs. The JSON values are conditional sequence probabilities over three answers. Their absolute levels should not be compared as if they shared a probabilistic meaning.

For missing EPS, simply changing dictionary order changed the JSON-prompt BUY probability from 76.45% to 61.57%; the top action stayed BUY. Identical repeats were exact. Separately encoded scores were bit-for-bit unchanged across orders. Evidence changes did alter that scorer's distribution (maximum distance 0.008432), so its vector representation was not entirely constant, but this does not establish useful evidence responsiveness.

## What this establishes

Application-side independent encoding and shared scoring can eliminate dictionary-order dependence without training an LLM. The symmetry follows from each action depending only on its own content and fixed evidence, with no cross-action positional input. Numerical and tie handling must also be order independent.

## What this does not establish

Neither method made useful decisions here: each matched the stipulated policy on only one of five distinct cases. The separate scorer is intentionally untrained. This proof does not demonstrate a trained attention architecture, economic value, calibrated confidence, improved accuracy, or invariance to changed wording or evidence order. It does not identify the neural cause of the JSON model's sensitivity. A trained head and held-out decision tests remain necessary before claiming a practical decision system.

Artifacts: manifest.json, outputs.json, summary.json, run.py, run.log.
