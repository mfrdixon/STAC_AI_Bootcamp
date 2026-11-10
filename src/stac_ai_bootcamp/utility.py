import os
from pathlib import Path


PROJECT_ROOT: Path = Path(__file__).parents[2].resolve()


def set_huggingface_offline() -> None:
    """Force HuggingFace fully offline — no downloads, no tokenizer forking

    Must be called BEFORE importing torch/transformers/peft — HuggingFace reads
    these environment variables at import time.
    """
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_io_directories(file_path: str) -> tuple[Path, Path]:
    """Return input and output directory paths for given file path

    Args:
        file_path (str): pass __file__ of target file

    Returns:
        tuple[Path, Path]: input_dir, output_dir
    """
    parent_dir: Path = Path(file_path).parent.resolve()
    input_dir: Path = parent_dir / "input"
    output_dir: Path = parent_dir / "output"

    return input_dir, output_dir
