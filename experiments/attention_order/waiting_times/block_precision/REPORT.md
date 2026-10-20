# Why the residual is nonzero: precision diagnostic

Same waiting-time record inputs, masks, local positions, original Qwen weights and original output head as block_matched. Ten orders and three repeats; no training. Two additional numerical configurations were run:

| Numerical configuration | Max P(YES) range, probability units |
|---|---:|
| Original float32 | 3.8886923e-6 |
| Float64 model, default internal float32 casts | 3.6016731e-7 |
| Float64 model + dtype-preserving attention softmax and RMSNorm | 4.4964032e-15 |

The source implementation explicitly casts attention softmax and RMSNorm inputs to float32, even with float64 model weights. The final diagnostic removes those two casts while retaining the formulas, weights, masks and inputs. Rotary embedding calculations retain the library implementation; identical local positions make those features repeat identically across orderings. This is a precision diagnostic, not training or a semantic architectural change.

Identical repeats had exactly zero variation. Forbidden attention weights remained zero. The final range is within the declared 1e-12 tolerance. This strongly supports a floating-point explanation for the initial residual, rather than a meaningful remaining dependence on list order. Different arithmetic reduction orders can yield slightly different rounding.

The mean probability itself also changes slightly under precision changes. This does not fix the separate output-format problem: restricted uppercase YES/NO candidates still receive only about 8.69% of raw probability mass. No improvement in counting correctness is claimed.

Runtime: 63.25 seconds for the final 30-call diagnostic. Use QWEN_MODEL_PATH and run.py as in block_matched. Parent manifest is the original waiting-time manifest. block_float64 records the intermediate diagnostic without removal of the internal casts.
