# Matched Qwen Block Attention: waiting-time evidence

## Question and design
Same 50 waiting times as the ordinary counting control; exact answer is 19, 38%, NO. Ten permutations, three identical repeats, three attention/position conditions: 90 forward calls in 58.21 seconds on CPU. Weights, vocabulary output head, all input token IDs, questions, values, answer conventions and execution settings are held fixed within each order. Conditions are randomly interleaved. The script asserts identical input-token hashes across conditions. No training, tools or generation processors.

**Serialization changed from the earlier JSON-list experiment:** each number is represented as `Waiting time: {value:.1f} minutes.\n`, with each segment tokenized separately. This makes record boundaries and within-record tokens identical across permutations. Accordingly the proper ordinary baseline is the new 21.73–23.57% range, not the earlier 23.75–26.80% range.

## Attention graph
- The instruction prefix retains causal self-attention.
- Each number record can attend to the shared prefix and its own preceding tokens, but never another number record.
- The fixed assistant/readout suffix can attend to the prefix, all records and its own preceding suffix tokens.
- `ordinary`: ordinary causal mask and global rotary positions.
- `mask_only`: restricted graph above, but global rotary positions retained.
- `block_local`: same graph; every number record resets rotary positions to prefix length + its local index. Prefix and suffix positions are fixed across permutations.

All 596 tokens and original Qwen2.5-0.5B weights are reused. The readout is Qwen's original vocabulary head at the final assistant-prefix token. YES and NO each tokenize to one token. Their raw log probabilities are normalized over these two uppercase candidates. No EOS or self-reported confidence is used. This is not full-answer generation.

## Results

| Condition | Min P(YES) | Max P(YES) | Range, percentage points | Repeat variation |
|---|---:|---:|---:|---:|
| Ordinary | 21.734693% | 23.566553% | 1.831860 | 0 |
| Mask only | 9.217633% | 9.741331% | 0.523698 | 0 |
| Block + local positions | 19.726548% | 19.726937% | 0.000389 | 0 |

The full construction is invariant within the predeclared 1e-5 absolute-probability tolerance, not bitwise identical. The residual 3.89e-6 is consistent with numerical reduction effects; no higher-precision diagnosis was run. In audited order-zero runs, every masked attention entry was exactly zero across all layers. Checks confirmed that the readout can see every record.

## Material side effect: conditional probabilities are not ordinary output behavior

The combined raw probability mass of uppercase YES and NO is 95.71–96.77% for ordinary attention, 11.26–12.99% for mask only, and 8.6926–8.6929% for Block Attention. Uppercase NO wins in all three restricted-candidate comparisons. However, the full-vocabulary argmax is NO in every ordinary run, NO in 24 and COUNT in 6 mask-only runs, and title-case **Yes** in all 30 Block Attention runs. Therefore the change isolates order effects but substantially changes the output distribution, and it does not preserve the original answer behavior. Do not label conditional NO dominance as an unconstrained model answer. Generated counting accuracy was not evaluated with these masks.

One fixed sample and one model do not establish general robustness or calibrated beliefs. The result supports a masked architecture's numerical order-invariance property under the specified readout, not a ready-to-use counting system.

## Reproduce
Set QWEN_MODEL_PATH to the pinned Qwen2.5-0.5B directory, then run `python run.py`. Parent `manifest.json` is the original waiting-times manifest. Files: manifest.json, outputs.jsonl, summary.json. Model revision is recorded in the nested source manifest. No downloaded weights are included in GitHub.
