from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted((ROOT / "notebooks").glob("*.ipynb"))


def test_notebooks_discovered():
    # Fail loudly if the glob matches nothing, so a broken path cannot pass silently.
    assert NOTEBOOKS, f"No notebooks found under {ROOT / 'notebooks'}"


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=[path.name for path in NOTEBOOKS])
def test_notebook_executes(notebook):
    # A notebook is correct when it runs top-to-bottom without raising.
    # Run with `uv run pytest -n auto` after adding pytest-xdist to parallelize
    # these independent cases when notebooks run in live (network-bound) mode.
    nb = nbformat.read(notebook, as_version=4)
    NotebookClient(
        nb,
        timeout=120,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    ).execute()
