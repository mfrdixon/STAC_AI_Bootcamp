import importlib.metadata


# Retrieve the version from package metadata (pyproject.toml)
try:
    __version__ = importlib.metadata.version("stac-ai-bootcamp")
except importlib.metadata.PackageNotFoundError:
    __version__ = None


def main() -> None:
    print("Hello from stac-ai-bootcamp!")
