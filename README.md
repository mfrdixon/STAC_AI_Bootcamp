# STAC AI Bootcamp

**Matthew Dixon, PhD, FRM · BeliefLens**  
STAC AI Workshop · NYC · 21 October 2026

Seven example Python notebooks, each with four learning objectives, runnable experiments, learning checks and a recap.

| Section | Notebook | Experiment |
|---|---|---|
| 1 | [Financial language-model architectures](notebooks/01_architectures.ipynb) | Model context, retrieval and calculator comparisons |
| 2 | [Grounded financial RAG](notebooks/02_grounded_rag.ipynb) | Provenance, timestamp eligibility and missing evidence |
| 3 | [Post-training](notebooks/03_post_training.ipynb) | SFT loss, low-rank updates and numerical DPO optimization |
| 4 | [Adaptation lab](notebooks/04_adaptation_lab.ipynb) | Instruction examples, held-out issuers and actual logistic model training |
| 5 | [Alpha and risk](notebooks/05_alpha_risk.ipynb) | Signal replay, costs, look-ahead leakage and vintage joins |
| 6 | [Evaluation](notebooks/06_evaluation.ipynb) | Evidence graphs, fault tests and held-out calibration |
| 7 | [Governance](notebooks/07_governance.ipynb) | Approval revocation, Bayesian review and expected loss |

## Quick start

Requires Python 3.10+ and Node.js 20+. No GPU, model weights or API key is needed for the default exercises.

```bash
git clone https://github.com/mfrdixon/STAC_AI_Bootcamp.git
cd STAC_AI_Bootcamp
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m jupyterlab notebooks
```

On Windows activate with `.venv\Scripts\activate` instead. Open a notebook and run the cells in order. Keep the complete repository: notebooks import `lab_client.py`, which invokes the shared JavaScript case engine through Node. If Node is not on your PATH, set `NODE_BINARY` to its executable path.

## Execution modes

**Offline (default):** fictional financial data and deterministic teaching outputs. Numerical experiments demonstrate training objectives, statistical evaluation and decision policies. They are not measurements of a language model.

**Optional live:** commented notebook cells can call a separately running BeliefLens Decision Lab server. Set `LAB_URL` to its base URL (default `http://127.0.0.1:8787/decision-lab`) and enter a temporary workshop key using `getpass`. The server holds `TOGETHER_API_KEY`; participants should not put that provider credential into notebooks. This repository includes the offline engine, not the web server.

A real fine-tuned financial LLM adapter is not yet included. Section 4 trains a numerical classifier; it does not claim to fine-tune an LLM. The graph uses a bounded case template rather than general automatic extraction of model reasoning. Approval is simulated and no financial action is executed.

## Verification

```bash
python scripts/check_notebooks.py
```

All seven notebooks were executed successfully before packaging. Outcomes on synthetic examples do not establish production reliability. Actual live inference requires a configured server and provider account.

## Background

- [Together API](https://docs.together.ai/reference/chat-completions)
- [LoRA](https://arxiv.org/abs/2106.09685)
- [QLoRA](https://arxiv.org/abs/2305.14314)
- [DPO](https://arxiv.org/abs/2305.18290)
