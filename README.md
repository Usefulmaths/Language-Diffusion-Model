# Diffusion Language Model

## Description

This project implements a diffusion-based approach to language modeling based on the Large Language Diffusion Model paper, focusing specifically on the pretraining phase. It's designed for experimentation and research purposes.

The implementation includes:

- A transformer-based architecture for text generation
- Diffusion-based token masking and denoising strategies
- Configurable training parameters via YAML configuration

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

## Configuration

Before running the model, you need to set up Accelerate for distributed training:

```bash
uv run accelerate config
```

This will guide you through configuring distributed training settings based on your hardware.

## Running the Model

To train the model:

```bash
uv run accelerate launch main.py
```

The training parameters can be customized by modifying the `configs/config.yaml` file.

## Project Structure

- `src/`: Core implementation
  - `model.py`: Diffusion language model implementation
  - `transformer.py`: Transformer architecture
  - `masking_strategy.py`: Token masking strategies
  - `trainer.py`: Training loop and evaluation
  - `scheduler.py`: Learning rate schedulers
  - `tokenizer.py`: Text tokenization
  - `dataset.py`: Data loading and processing
  - `config.py`: Configuration management

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
