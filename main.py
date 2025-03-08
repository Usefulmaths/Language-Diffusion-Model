#!/usr/bin/env python3
"""Diffusion Language Model training script.

This script trains a diffusion-based language model using a configurable
architecture with transformer-based backbone. Configuration is loaded from
a YAML file to allow for flexible experimentation.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Literal, cast

import torch
import yaml
from torch.optim import AdamW

from src.config import Config
from src.dataset import create_question_dataloaders
from src.masking_strategy import create_masking_strategy
from src.model import DiffusionLanguageModel
from src.scheduler import create_scheduler
from src.tokenizer import DiffusionTokenizer, TokenizerConfig
from src.trainer import DiffusionLanguageModelTrainer, TrainingMetrics
from src.transformer import DiffusionTransformer, DiffusionTransformerConfig


def setup_logging() -> None:
    """Configure logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def main(config_path: str | Path) -> list[TrainingMetrics]:
    """Main training function.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        List of training metrics for each epoch

    Raises:
        FileNotFoundError: If the config file doesn't exist
        yaml.YAMLError: If the YAML configuration is invalid
        ValueError: If configuration is invalid
    """
    # Convert string path to Path object
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info(f"Loading configuration from {config_file}")

    try:
        # Load configuration
        config = Config.from_yaml(str(config_file))
    except yaml.YAMLError as e:
        logger.error(f"Invalid configuration: {e}")
        raise

    # Set random seed for reproducibility
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    # Initialize tokenizer
    tokenizer_config = TokenizerConfig(**vars(config.tokenizer))
    tokenizer = DiffusionTokenizer(config=tokenizer_config)
    logger.info(f"Initialized tokenizer with vocab size: {tokenizer.vocab_size}")

    # Update vocab_size in transformer config based on tokenizer
    config.transformer.vocab_size = tokenizer.vocab_size

    # Initialize transformer
    transformer_config = DiffusionTransformerConfig(**vars(config.transformer))
    transformer = DiffusionTransformer(transformer_config)
    logger.info(
        f"Initialized transformer with {transformer_config.num_layers} layers, "
        f"{transformer_config.d_model} dimensions"
    )

    # Initialize masking strategy
    masking_strategy = create_masking_strategy(
        cast(Literal["random", "low_confidence"], config.masking.strategy),
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    logger.info(f"Using masking strategy: {config.masking.strategy}")

    # Initialize model
    model = DiffusionLanguageModel(transformer, tokenizer, masking_strategy)
    logger.info(
        f"Model initialized with {sum(p.numel() for p in model.parameters())} "
        f"parameters"
    )

    # Create dataloaders
    train_file_path = Path(config.data.train_file)
    if not train_file_path.exists():
        raise FileNotFoundError(f"Training file not found: {train_file_path}")

    val_file: Path | None = None
    if config.data.val_file:
        val_file = Path(config.data.val_file)
        if not val_file.exists():
            raise FileNotFoundError(f"Validation file not found: {val_file}")

    # Calculate train/val split ratio only if no validation file provided
    train_ratio: float | None = None
    if val_file is None:
        train_ratio = config.data.train_ratio

    logger.info(f"Loading data from {train_file_path}")
    train_loader, val_loader = create_question_dataloaders(
        file_path=str(train_file_path),
        tokenizer=tokenizer,
        batch_size=config.training.batch_size,
        max_length=config.training.max_length,
        train_ratio=train_ratio,
        shuffle=config.data.shuffle,
        num_workers=config.training.num_workers,
    )
    logger.info(f"Created train dataloader with {len(train_loader)} batches")
    if val_loader:
        logger.info(f"Created validation dataloader with {len(val_loader)} batches")

    # Initialize optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    # Initialize scheduler with simplified API
    total_steps = len(train_loader) * config.training.num_epochs
    scheduler_config = config.training.scheduler

    # Set peak_lr to learning_rate if not specified
    peak_lr = (
        scheduler_config.peak_lr
        if scheduler_config.peak_lr is not None
        else config.training.learning_rate
    )

    scheduler = create_scheduler(
        optimizer=optimizer,
        total_steps=total_steps,
        warmup_steps=scheduler_config.warmup_steps,
        initial_lr=scheduler_config.initial_lr,
        peak_lr=peak_lr,
        stable_lr=scheduler_config.stable_lr,
        final_lr=scheduler_config.final_lr,
    )

    logger.info(
        f"Using LLaDA scheduler with: "
        f"warmup_steps={scheduler_config.warmup_steps}, "
        f"peak_lr={peak_lr}, "
        f"stable_lr={scheduler_config.stable_lr or peak_lr * 0.25}, "
        f"final_lr={scheduler_config.final_lr or peak_lr * 0.025}"
    )

    # Create checkpoint directory
    checkpoint_dir = Path(config.training.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Determine device
    device = config.training.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available, falling back to CPU")
        device = "cpu"

    # Initialize trainer
    trainer = DiffusionLanguageModelTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        checkpoint_dir=str(checkpoint_dir),
        gradient_clip_val=config.training.gradient_clip_val,
        use_amp=config.training.use_amp and device == "cuda",
        log_interval=config.training.log_interval,
    )

    # Log training configuration
    logger.info(
        f"Starting training with:\n"
        f"  - Epochs: {config.training.num_epochs}\n"
        f"  - Batch size: {config.training.batch_size}\n"
        f"  - Learning rate: {config.training.learning_rate}\n"
        f"  - Device: {device}\n"
        f"  - Mixed precision: {config.training.use_amp and device == 'cuda'}\n"
    )

    # Train the model
    try:
        history = trainer.train(
            train_dataloader=train_loader,
            num_epochs=config.training.num_epochs,
            mask_prob=config.masking.mask_ratio,
            val_dataloader=val_loader,
            early_stopping_patience=config.training.early_stopping_patience,
        )
        logger.info("Training completed successfully")

        # Save final model
        final_model_path = checkpoint_dir / "final_model.pt"
        trainer.save_checkpoint(str(final_model_path))
        logger.info(f"Final model saved to {final_model_path}")

        # Save final configuration
        config_save_path = checkpoint_dir / "final_config.yaml"
        config.save(str(config_save_path))
        logger.info(f"Final configuration saved to {config_save_path}")

        return history
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        # Save interrupted model
        interrupted_path = checkpoint_dir / "interrupted_model.pt"
        trainer.save_checkpoint(str(interrupted_path))
        logger.info(f"Interrupted model saved to {interrupted_path}")
        return trainer.get_training_history()
    except Exception as e:
        logger.exception(f"Error during training: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train diffusion language model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config file",
    )
    args = parser.parse_args()

    try:
        main(args.config)
    except (FileNotFoundError, yaml.YAMLError) as e:
        logging.error(f"Configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Training failed: {e}")
        sys.exit(2)
