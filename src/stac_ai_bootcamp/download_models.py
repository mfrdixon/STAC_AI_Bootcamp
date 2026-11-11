"""Download the pinned Qwen / FLAN models so the offline experiments can run.

Configuration comes entirely from environment variables (see .env.example); there
are no CLI arguments. This is the one script that must run with HuggingFace ONLINE,
so it deliberately does NOT call set_huggingface_offline().

Qwen is fetched into QWEN_MODEL_PATH (a local directory the runners load with
local_files_only=True); FLAN is fetched into the default HuggingFace cache, which
is where run_flan.py resolves it from by repo id.
"""

from huggingface_hub import snapshot_download

from stac_ai_bootcamp.utility import (
    get_flan_model_id,
    get_qwen_model_id,
    get_qwen_model_path,
    get_qwen_model_rev,
)


def main() -> None:
    qwen_id = get_qwen_model_id()
    qwen_rev = get_qwen_model_rev()
    qwen_path = get_qwen_model_path()
    flan_id = get_flan_model_id()

    print(f"Downloading {qwen_id}@{qwen_rev[:12]} -> {qwen_path}", flush=True)
    snapshot_download(repo_id=qwen_id, revision=qwen_rev, local_dir=qwen_path)

    print(f"Downloading {flan_id} -> HuggingFace cache", flush=True)
    flan_path = snapshot_download(repo_id=flan_id)

    print("Models ready:", flush=True)
    print(f"  qwen: {qwen_path}", flush=True)
    print(f"  flan: {flan_path}", flush=True)
