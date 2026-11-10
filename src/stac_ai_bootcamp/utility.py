import os
from pathlib import Path


PROJECT_ROOT: Path = Path(__file__).parents[2].resolve()


def experiment_output(relative: str) -> Path:
    """Return the output directory for an experiment, creating it.

    The path is anchored to PROJECT_ROOT/experiments/<exp-name>/output.
    """
    output_dir = PROJECT_ROOT / "experiments" / relative / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


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
