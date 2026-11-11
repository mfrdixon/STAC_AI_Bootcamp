import json
import os
from pathlib import Path


PROJECT_ROOT: Path = Path(__file__).parents[2].resolve()


def experiment_output(relative: str) -> Path:
    """Return the output directory for an experiment, creating it.

    The path is anchored to PROJECT_ROOT/experiments/<exp-name>/output.
    """
    output_dir = PROJECT_ROOT / "experiments" / relative
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def get_qwen_model_path() -> Path:
    """Local directory holding the Qwen weights the offline experiments load.

    Honors the QWEN_MODEL_PATH environment variable, falling back to the shared
    experiments/attention_order/qwen_model location that download_models writes to.
    """
    return Path(
        os.environ.get(
            "QWEN_MODEL_PATH",
            str(PROJECT_ROOT / "experiments/attention_order/qwen_model"),
        )
    )


def get_flan_model_id() -> str:
    """Repo id for the FLAN seq2seq model (env FLAN_MODEL_ID)."""
    return os.environ.get("FLAN_MODEL_ID", "google/flan-t5-base")


def get_flan_model_rev() -> str:
    """Commit SHA of the locally cached FLAN snapshot.

    Transformers 5.18+ no longer exposes config._commit_hash; resolve the
    revision from the offline HuggingFace cache instead. Honors FLAN_MODEL_REV to
    pin explicitly, otherwise reads the cached snapshot directory name (its SHA).
    """
    flan_rev = os.environ.get("FLAN_MODEL_REV")
    if flan_rev:
        return flan_rev
    from huggingface_hub import snapshot_download

    flan_id: str = get_flan_model_id()
    flan_rev: str = Path(snapshot_download(flan_id, local_files_only=True)).name
    return flan_rev


def get_qwen_model_id() -> str:
    """Repo id for the Qwen instruct model (env QWEN_MODEL_ID)."""
    return os.environ.get("QWEN_MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")


def get_qwen_model_rev() -> str:
    """Pinned Qwen revision (env QWEN_MODEL_REV)."""
    return os.environ.get(
        "QWEN_MODEL_REV", "7ae557604adf67be50417f59c2c2f167def9a775"
    )


def save_json(path: Path, data: dict, print_data: bool = False) -> None:
    """Serialize `data` to `path` as pretty-printed JSON (indent=2)."""
    data_json: str = json.dumps(data, indent=2)
    path.write_text(data_json)
    if print_data:
        print(data_json, flush=True)


def total_variation(a: dict, b: dict) -> float:
    """Total-variation distance between two categorical distributions keyed alike."""
    return sum(abs(a[k] - b[k]) for k in a) / 2


def winner(probabilities: dict):
    """Return the highest-probability key, breaking ties by key order."""
    return min(probabilities, key=lambda k: (-probabilities[k], k))


def set_huggingface_offline() -> None:
    """Force HuggingFace fully offline — no downloads, no tokenizer forking

    Must be called BEFORE importing torch/transformers/peft — HuggingFace reads
    these environment variables at import time.
    """
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def configure_torch(
    seed: int = 20261019, num_threads: int = 4, deterministic: bool = True
) -> None:
    """Apply the shared deterministic torch setup.

    Each experiment passes its own seed/thread count; the defaults match the
    most common values. torch is imported inside the function so this module
    stays importable before set_huggingface_offline() has run.
    """
    import torch

    torch.manual_seed(seed)
    torch.set_num_threads(num_threads)
    torch.use_deterministic_algorithms(deterministic)
