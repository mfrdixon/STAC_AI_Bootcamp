# Why attention in LLMs can sometimes work against you

If the evidence stays the same, should moving BUY to the end of a list change an AI's preferred action?

In our small Qwen2.5-0.5B experiment, it did.

We tested five financial-evidence cases, all six orders of BUY / CASH / WITHHOLD, and five identical repeats. We scored all three answers, rather than relying on self-reported confidence.

| Qwen: action-order test | JSON in prompt | Independent encoding |
|---|---:|---:|
| Changed highest-scoring action | 32/75 order comparisons | 0/75 |
| Largest distribution shift (total variation) | 0.5975 | 0 |
| Identical-repeat variation | 0 | 0 |

The workaround: encode each action description separately, read the model's internal vectors, and apply the same action-to-evidence scoring calculation. We bypass answer generation—not attention within the transformer. An action no longer sees the alternatives listed before it.

We then implemented **Block Attention** in a tiny, untrained transformer: one structured input, attention restricted to each evidence/action block, positions reset within each block, and a shared scoring head.

| Tiny model: 22,144 parameters | Changed rankings / 75 |
|---|---:|
| Ordinary attention | 52 |
| Block mask, but global positions retained | 51 |
| Block mask + local positions | 0 |

**The mask alone was not enough. Position handling mattered too.** Forbidden attention weights were exactly zero; changing BUY's text did not change CASH's representation.

This is a proof of order isolation—not a claim of better investment decisions or a diagnosis of which attention head caused Qwen's behavior. The action-block experiment used random weights in a tiny model; it is not a trained replacement for Qwen’s answer generator. Separate scoring produces normalized compatibility scores, whereas the prompt comparison uses answer likelihoods. Neither method improved agreement with the stipulated action rule in these five cases.

These ideas have precedents:

- **Set Transformer** lets unordered elements interact without arbitrary list-order dependence. Attention need not be removed altogether. [Lee et al., ICML 2019](https://proceedings.mlr.press/v97/lee19d.html)
- **Fusion-in-Decoder** encodes passages separately and combines them during generation; **Atlas** explicitly uses it. [FiD](https://aclanthology.org/2021.eacl-main.74/) · [Atlas](https://jmlr.org/papers/volume24/23-0037/23-0037.pdf)
- **UniLM** trains different attention visibility patterns within a shared transformer. [Dong et al., 2019](https://arxiv.org/abs/1905.03197)
- **Llama 3** used masks to isolate documents during pretraining. That does not make action lists in its chat prompts order-invariant. [Meta](https://ai.meta.com/blog/meta-llama-3/)

“Block Attention” is our descriptive label for this construction, not a claim to have invented masking or set attention. Public sources reviewed do not establish that GPT, Claude or Gemini use these specific architectures.

The design question: **which parts of your workflow should interact—and which should be structurally independent?**

[Working paper: methodology, equations and preliminary results](https://github.com/mfrdixon/STAC_AI_Bootcamp/blob/main/papers/block_attention/block_attention_working_paper.pdf)

Figures: `partial_attention_figure.png` (pretrained-model workaround) and `block_attention_figure.png` (masked tiny-model proof).

The paper and code are available in the linked GitHub repository.
