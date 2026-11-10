.PHONY: all
all: help


.venv:
	@echo "Installing project dependencies.."
	@uv sync --group dev

jupyter:
	@echo "Running JupyterLab.."
	@if [ -f .env ]; then \
		uv run --env-file=.env jupyter lab; \
	else \
		uv run jupyter lab; \
	fi

server:
	@echo "Serving Decision Lab at.."
	@uv run python -m http.server 8000 --directory public

test:
	@echo "Running tests.."
	@uv run pytest -v

test-cov:
	@echo "Running tests with coverage.."
	@uv run pytest --cov --cov-report=term-missing

lint:
	@echo "Linting code.."
	@uv run ruff check src/ tests/

format:
	@echo "Formatting code.."
	@uv run ruff check --fix src/ tests/
	@uv run ruff format src/ tests/


clean:
	@echo "Removing build artifacts.."
	@rm -rf .venv dist/ build/ *.egg-info .coverage htmlcov/
	@rm -rf `find -type d -name __pycache__`
	@rm -rf `find -type d -name .pytest_cache`

help:
	@echo "Available make targets:"
	@echo "  make help          - Print help"
	@echo "  make .venv         - Install project dependencies"
	@echo "  make jupyter       - Run JupyterLab"
	@echo "  make server        - Serve the Decision Lab web app"
	@echo "  make test          - Run tests"
	@echo "  make test-cov      - Run tests with coverage"
	@echo "  make lint          - Lint code"
	@echo "  make format        - Format code"
	@echo "  make clean         - Remove build artifacts"
	@echo ""
