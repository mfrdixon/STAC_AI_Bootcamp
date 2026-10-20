# STAC AI Bootcamp

**[Open the public web exercises](https://belieflens.org/decision-lab)** · No account or API key needed for the seven default exercises.

# STAC AI Workshop · NYC · 21 October 2026

Matthew Dixon, PhD, FRM · Participant workbook

## Start here

1. Extract the complete `BeliefLens_Workshop_Notebooks.zip`.
2. Install Python 3.10+ and JupyterLab (`python -m pip install jupyterlab`).
3. Open a terminal in the extracted `belieflens-decision-lab` folder and run `python -m jupyterlab`.
4. Open the numbered notebook in `notebooks/`. Run cells in order.

Keep `workshop_offline.py`, `data/`, and the package helper files in their original locations. Downloading an individual notebook without its companion files is insufficient. Default exercises require no GPU, API key, Node.js or model weights. The optional legacy classifier reference uses Node.js; it is not the main Section 4 exercise.

| Section | Main notebook | What you do |
|---|---|---|
| 1 | `01_architectures.ipynb` | Compare the simple definition, annex, programmed branch and missing annex; inspect what each path supports. |
| 2 | `02_grounded_rag.ipynb` | Check time eligibility and sufficiency, trace branching impact, and verify 4 valid / 17 invalid repairs. |
| 3 | `03_post_training.ipynb` | Calculate the same SFT, LoRA and DPO illustrations as the web interface; inspect the unchanged evidence gap. |
| 4 | `04_adaptation_lab.ipynb` | Diagnose a controlled workflow through supplied interventions. See the additional recorded-adaptation notebooks below. |
| 5 | `05_alpha_risk.ipynb` | Compare time-valid and future-derived signals, fees, a gate and the no-trade baseline. |
| 6 | `06_evaluation.ipynb` | Inspect 1,200 saved framing assessments, recompute evidence checks and compare exact responses. |
| 7 | `07_governance.ipynb` | Trace Yankees/Patriots sponsorship checks and approval; then enforce current authority and idempotency at commit. |

## Section 4 alternatives retained

- `04_actual_adaptation_readable.ipynb`: readable recorded base-versus-adapter comparison.
- `04_actual_adaptation.ipynb`: detailed recorded comparison with explicit optional local inference.
- `04_instructor_lora_demo.ipynb`: instructor demonstration; use the supplied guide.
- `04_workflow_diagnosis.ipynb`: dedicated controlled diagnosis notebook.
- `04_adaptation_lab_classifier_reference.ipynb`: optional legacy numerical classifier exercise; requires Node.js. It is not LLM fine-tuning.

Instructor instructions are retained in `instructor-sandbox/INSTRUCTOR_LORA_DEMO.md` and `REMOTE_JUPYTER.md`. No model weights or credentials are distributed. Any opt-in fresh inference requires separately provisioned approved artifacts; it does not run by default.

## Interpretation

Sections 1–3, 5 and 7 are declared synthetic teaching computations. Section 6 replays measured model responses to synthetic cases. The Section 4 recorded adapter comparison is separately labelled. Do not interpret synthetic positive returns as market alpha, a lower training loss as calibrated confidence, or model-reported citations as causal attribution.

The portable Python checks in Sections 1–3 and 5–7 are compared against saved outputs exported from the browser engines. Regeneration for maintainers uses `scripts/export_notebook_fixtures.mjs` and `scripts/align_offline_notebooks.py`; participants do not run those scripts.

## Clone and run

```bash
git clone https://github.com/mfrdixon/STAC_AI_Bootcamp.git
cd STAC_AI_Bootcamp
python3 -m pip install -r requirements.txt
python3 -m jupyterlab notebooks
```

Live model calls and fresh GPU training are optional instructor extensions. The default notebooks use bundled fixtures, numerical illustrations and explicitly labelled recorded responses. They do not contact a model provider.
