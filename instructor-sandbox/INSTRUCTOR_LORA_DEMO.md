# STAC AI Workshop · NYC · 21 October 2026

## Instructor LoRA demonstration · Matthew Dixon, PhD, FRM

## Reliable class route: recorded run

Open the workshop folder locally. Launch Jupyter with the Python environment in which JupyterLab is installed:

```bash
cd /Users/matthewdixon_1/Downloads/codex-workspace/admin/apps/belieflens-decision-lab
python -m jupyterlab notebooks/04_instructor_lora_demo.ipynb
```

If that environment has no JupyterLab, use your existing Jupyter installation or install it in a dedicated virtual environment. No GPU, cloud access or model download is needed for the notebook's default cells.

1. Inspect a real supervised example: system instruction, financial evidence, one-token label.
2. Show q_proj/v_proj, rank 8, alpha 16, response-only loss and 540,672 adapter parameters.
3. Show saved losses; epoch 2 was selected on validation. Training took 28.6 minutes on the previous T4 run; this is not a timing promise for another run.
4. Compare a saved held-out pair under identical evidence. Reveal the outcome and independent evidence checks.
5. Recompute all 16-case metrics. The adapter changed behaviour; it did not establish better forecasting or alpha.

The ordinary participant notebook is `04_actual_adaptation.ipynb`. Both read the same canonical `public/decision-lab/data/recorded-adaptation.json`. No regenerated rationale is substituted for the actual class-token outputs.

## Optional live mechanics

The remote Jupyter Docker workflow is documented in `REMOTE_JUPYTER.md`. It forwards a Unix socket over SSH, preserving the container’s disabled network. In the instructor notebook, `RUN_GPU_DEMO=True` runs the isolated ten-step training helper directly inside the container. Run only one GPU job at a time.

### Alternative: local notebook + SSH to offline GPU Docker

These instructions do not start EC2. Power is managed separately for an authorized demonstration. Start/use the GPU only when separately authorized and ready to demonstrate. Account `043588880476`, region `us-east-1`, instance `i-0c688e37141272dbb`. Obtain its current public IP after it is running; the prior IP may have changed.

From a hotel, update the associated SSH rule in security group `sg-0085835a8ba055040` to **My IP /32**, TCP 22. Do not open all addresses. The existing EC2 Instance Connect browser route is a terminal fallback if local SSH is unavailable; it is not a local notebook tunnel.

For this alternative route, keep Jupyter local and execute training through SSH. The private key remains on this computer. Run the following from the workshop folder, substituting the current IP:

```bash
STAC_EC2_IP='CURRENT_PUBLIC_IP'
ssh -o BatchMode=yes -o StrictHostKeyChecking=yes \
  -o HostKeyAlias=35.175.242.8 \
  -o UserKnownHostsFile=instructor-sandbox/runtime/ec2_host_candidate \
  -i instructor-sandbox/runtime/ec2_instructor_ed25519 \
  ubuntu@"$STAC_EC2_IP" 'nvidia-smi; docker --version; docker compose version'
```

The alias refers to the previously verified host key. If verification fails, verify the host fingerprint through the authenticated console; do not disable host-key checking.

The server was provisioned under `/home/ubuntu/belieflens-instructor`. Copy the new helper (the existing base, training data, image and original configuration must already be present):

```bash
scp -o StrictHostKeyChecking=yes -o HostKeyAlias=35.175.242.8 \
  -o UserKnownHostsFile=instructor-sandbox/runtime/ec2_host_candidate \
  -i instructor-sandbox/runtime/ec2_instructor_ed25519 \
  instructor-sandbox/scripts/run_lora_smoke.py \
  ubuntu@"$STAC_EC2_IP":/home/ubuntu/belieflens-instructor/scripts/
```

First prepare an isolated configuration, without training:

```bash
ssh -o StrictHostKeyChecking=yes -o HostKeyAlias=35.175.242.8 \
  -o UserKnownHostsFile=instructor-sandbox/runtime/ec2_host_candidate \
  -i instructor-sandbox/runtime/ec2_instructor_ed25519 \
  ubuntu@"$STAC_EC2_IP" \
  'cd /home/ubuntu/belieflens-instructor && python3 scripts/run_lora_smoke.py --steps 10'
```

When ready, repeat the last command with `--steps 10 --run`. The helper stops competing lab containers, runs the existing hash/token-length/CUDA preflight, and invokes `llamafactory-cli train` inside Docker. Compose uses `network_mode: none`, read-only base/data mounts and a writable artifact directory. The new smoke YAML is under that writable mount, so the existing image can read it without rebuilding.

Before running, inspect `docker image inspect belieflens-instructor:0.1.0`, free GPU memory from `nvidia-smi`, and available disk space. The preflight verifies the known base-weight SHA-256, single-token A/B/C coding, and training/validation prompt lengths. The helper records source-config, generated-config and training-data hashes. The saved run's software versions are visible in the notebook manifest; compare the active image's PyTorch/Transformers/PEFT versions before interpreting a new comparison.

## What the short run demonstrates

- Real SFT backpropagation into q/v LoRA matrices; base weights remain frozen.
- Ten optimizer steps, at most 64 training rows, rank 8, alpha 16, original 5,120-token cutoff.
- A fresh timestamped output folder; the evaluated `alpha_v2/adapter` is never overwritten.
- No evaluation-based adapter selection or performance claim from the smoke run.

The full previous run is described by `config/alpha_v2_lora.yaml`. Do not launch it directly over the existing output folder. A new experiment needs a fresh output path and untouched evaluation data. The `RUN_GPU_DEMO` notebook switch is for a notebook running on an already provisioned GPU host; when Jupyter is local, use the SSH terminal commands above instead.

## Failure fallback

If preflight fails, memory is exhausted or the class is out of time, stop the smoke training container and return to recorded results. Keep any partial output labelled incomplete. Do not replace the saved measured adapter with it. No synthetic loss curve or output is used as a substitute. Stop the cloud instance after the authorized demonstration to avoid idle cost.

Optional fresh inference in the participant notebook is also disabled by default. It requires local base/adapter files, verifies their hashes and the rendered prompt, and uses installed packages without network downloads. CPU inference may be slow. The compact workshop download intentionally contains no model weights or private keys.
