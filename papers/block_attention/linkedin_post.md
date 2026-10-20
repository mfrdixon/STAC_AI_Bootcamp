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

This is a proof of order isolation—not a claim of better investment decisions or a diagnosis of which attention head caused Qwen's behavior. The Block Attention test used random weights; it has not yet been applied inside pretrained Qwen. The construction builds on established set-based attention ideas.

The design question: **which parts of your workflow should interact—and which should be structurally independent?**

[Working paper: methodology, equations and preliminary results](https://github.com/mfrdixon/STAC_AI_Bootcamp/blob/main/papers/block_attention/block_attention_working_paper.pdf)

Figures: `partial_attention_figure.png` (pretrained-model workaround) and `block_attention_figure.png` (masked tiny-model proof).

The paper and code are available in the linked GitHub repository.
