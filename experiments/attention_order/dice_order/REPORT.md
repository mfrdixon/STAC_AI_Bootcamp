# Hidden-source dice test: ordinary versus Block Attention

One sample of 50 independent pseudorandom fair-die draws, seed 20261024. The model is NOT told these are dice rolls or independent/fair draws. It sees the same 50 numbers in ten permutations, repeated three times, and is asked to predict the next digit from 1 to 6. Both conditions use identical record-formatted token sequences, fixed Qwen2.5-0.5B weights and its original output head. No training or sampling. Sixty forward calls took 24.27 seconds on CPU.

## Results

| Measure | Ordinary | Block + local positions |
|---|---:|---:|
| Highest-ranked digit changes / 45 order pairs | 16 | 0 |
| Maximum order TV | 0.155478 | 0.000003171 |
| Identical-repeat TV | 0 | 0 |
| Top digits across orders | 4, 5 | 1 |
| Raw probability mass on digits 1–6 | 93.93–95.53% | 1.93646–1.93654% |
| Unrestricted vocabulary argmax | 4 or 5 | `Observ` |

First ordering, normalized over six digits:

| Digit | Ordinary | Block + local positions | True hidden process |
|---|---:|---:|---:|
| 1 | 11.7892% | 64.3762% | 16.6667% |
| 2 | 5.6526% | 5.5056% | 16.6667% |
| 3 | 25.9069% | 6.3672% | 16.6667% |
| 4 | 26.2371% | 5.4086% | 16.6667% |
| 5 | 18.9917% | 7.0223% | 16.6667% |
| 6 | 11.4225% | 11.3202% | 16.6667% |

The original output head is evaluated at the assistant-prefix token; digit candidates are single tokens. Scores are normalized over those six candidates, without EOS. No full answer generation was performed. Audited forbidden attention weights are exactly zero; the readout can access all number records. Identical token hashes are asserted across conditions within each order.

## Interpretation

Ordinary Qwen exhibits order sensitivity here. Block Attention produces near-invariant restricted scores, within 1e-5 TV tolerance, but DOES NOT produce calibrated fair-die probabilities. It substantially disrupts the output distribution: the model would begin with `Observ`, not a digit, under unrestricted greedy decoding. Candidate normalization hides this unless raw mass is also reported. Do not present conditional digit 1 as the model's unrestricted answer.

The experimenter knows the true distribution is uniform. The model was deliberately not told the data-generating process. Therefore nonuniform probabilities alone do not establish a logical error or irrationality; the model may infer an ordered process. The intervention imposes set semantics externally. One sample and one model are not a general robustness result.

Block architecture: instruction prefix causal; each record sees prefix and its own causal tokens; records cannot see each other; final readout sees all records. Local rotary positions reset per record, and suffix positions remain fixed. No weights are changed.

## Reproduce
Set QWEN_MODEL_PATH to the local pinned Qwen2.5-0.5B directory and run `python run.py`. `manifest.json` contains values, orders, token IDs, templates and model revision; `outputs.jsonl` contains every trial; `summary.json` contains results.
