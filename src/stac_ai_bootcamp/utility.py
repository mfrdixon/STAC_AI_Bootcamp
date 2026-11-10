import os
from pathlib import Path


PROJECT_ROOT: Path = Path(__file__).parents[2].resolve()


def experiment_io(relative: str) -> tuple[Path, Path]:
    """Return (input_dir, output_dir) for an experiment, creating both.

    Paths are anchored to the legacy tree at PROJECT_ROOT/experiments/<relative>
    rather than to each script's on-disk location, so experiment code can live
    anywhere under src/ and still read/write a stable, shared data tree. The
    <relative> key mirrors the old folder layout, which keeps the outputs of
    now-flattened scripts (e.g. the flan/qwen runners) in distinct directories.
    """
    base = PROJECT_ROOT / "experiments" / relative
    input_dir = base / "input"
    output_dir = base / "output"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    return input_dir, output_dir


def set_huggingface_offline() -> None:
    """Force HuggingFace fully offline — no downloads, no tokenizer forking

    Must be called BEFORE importing torch/transformers/peft — HuggingFace reads
    these environment variables at import time.
    """
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
