# Remote Jupyter · STAC AI Workshop NYC · 21 October 2026

The instructor opens JupyterLab in a laptop browser, but Python and GPU training run on EC2 inside Docker. The container has `network_mode: none`, no published ports and no Docker socket. SSH forwards laptop **127.0.0.1:8888** to a private Unix socket on EC2. Do not open inbound port 8888 in the security group.

## One-time preparation on EC2

Use account `043588880476`, region `us-east-1`, instance `i-0c688e37141272dbb`. Start it only when required, and obtain its current public IP; that address changes after stop/start. The Ubuntu username is `ubuntu`. Security-group SSH access should be restricted to the instructor's current IP or the approved Instance Connect source.

Copy the updated instructor-sandbox folder into `/home/ubuntu/belieflens-instructor`, preserving the existing models, data and artifacts. The notebook sources belong under `notebooks/`, including `adaptation_lab.py` and its bundled recorded JSON.

```bash
cd /home/ubuntu/belieflens-instructor
bash scripts/jupyter_mode.sh prepare
bash scripts/jupyter_mode.sh build
bash scripts/jupyter_mode.sh start
```

The build step accesses package repositories; model/data provisioning is also an explicit online preparation step. Runtime inference/training has no external network. For an already built base image, build only the derivative with `docker compose --profile jupyter build jupyter`.

The preparation command generates `runtime/jupyter-token` with owner-only permissions. It does not print the secret. Jupyter reads this file, requires token authentication and suppresses startup token URLs. Do not place the token in slides, source control, shell arguments or shared logs.

## Open from the instructor laptop

From the local `instructor-sandbox` folder, with the existing pinned host key and SSH private key in `runtime/`:

```bash
bash scripts/connect_jupyter.sh CURRENT_PUBLIC_IP
```

Keep that terminal open. Open **http://127.0.0.1:8888/lab**. In a separate private SSH session, read `runtime/jupyter-token` and paste it into Jupyter's login form. Do not share that terminal or copy the token into a notebook. The script pins the previously verified host key under `HostKeyAlias=35.175.242.8`; a changed IP does not disable host verification.

The Unix-socket forward is equivalent to reaching the remote service through SSH, without creating a remote TCP listener. The browser only connects to laptop loopback. The boundary includes EC2, SSH and the instructor's browser, which necessarily receives displayed outputs. This is an instructor environment, not a multi-user JupyterHub deployment.

## Run the workshop notebooks

1. Open `04_actual_adaptation.ipynb`. Run the saved-result cells; no training or inference starts by default.
2. Open `04_instructor_lora_demo.ipynb` for the GPU demonstration. Check the prerequisite/model paths. The initial switch is false.
3. Set `RUN_GPU_DEMO = True` only for the authorized short demonstration. The direct helper runs ten training steps inside the existing GPU container, with a new output directory. It does not overwrite the selected workshop adapter.
4. Return to the evaluation notebook. A smoke-training completion is not evidence of better accuracy or alpha.

Notebook sources are mounted read-only at `/opt/workshop-notebooks` and copied into `/workspace` on first use. Edited copies persist in host `notebook-workspace/`. Existing copies are not silently overwritten by later source updates; copy updated versions deliberately when needed. Data and base models are read-only; artifacts are writable. Training output belongs under `/app/artifacts/instructor_smoke/`.

No provider API keys or host cloud credentials are mounted. Hugging Face offline flags are set. The browser Content Security Policy blocks external resources; notebook output rendering can require adjustments for optional rich widgets. No fresh data download is part of runtime.

## Stop and avoid continuing compute charges

```bash
cd /home/ubuntu/belieflens-instructor
bash scripts/jupyter_mode.sh stop
```

Close the SSH tunnel. **Stopping Jupyter does not stop EC2 billing.** After ensuring no training job remains, stop the instance through EC2. Do not terminate it or delete its volumes.

## Configuration references

- [Jupyter authentication](https://jupyter-server.readthedocs.io/en/latest/operators/security.html)
- [Server configuration and Unix sockets](https://jupyter-server.readthedocs.io/en/stable/other/full-config.html)

The remote validation report records which build, kernel execution and GPU checks were actually completed; configuration alone is not a test result.
