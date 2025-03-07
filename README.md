# diffusion-language-model

## Description
Python project diffusion-language-model

## Installation
```bash
# Clone the repository
git clone <repository-url>
cd diffusion-language-model

# Create and activate virtual environment
uv venv .venv
source .venv/bin/activate

# Install dependencies
uv pip install -e .
```

## Development
This project uses:
- uv for dependency management
- ruff for formatting and linting
- mypy for type checking
- pytest for testing
- pre-commit for code quality checks

Setup development environment:
```bash
# Install dev dependencies
uv pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

## Running Tests
```bash
pytest
```
