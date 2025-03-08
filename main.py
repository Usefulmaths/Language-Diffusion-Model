#!/usr/bin/env python3
"""Diffusion Language Model training script with diagnostics.

This script trains a diffusion-based language model using a configurable
architecture with transformer-based backbone. Configuration is loaded from
a YAML file to allow for flexible experimentation.
"""

import argparse
import logging
import sys
import traceback
from pathlib import Path
from typing import Literal, cast

import torch
import yaml
from accelerate import Accelerator
from torch.optim import AdamW

from src.config import Config
from src.dataset import create_question_dataloaders
from src.masking_strategy import create_masking_strategy
from src.model import DiffusionLanguageModel
from src.scheduler import create_scheduler
from src.tokenizer import DiffusionTokenizer, TokenizerConfig
from src.trainer import DiffusionLanguageModelTrainer, TrainingMetrics
from src.transformer import DiffusionTransformer, DiffusionTransformerConfig


def setup_logging(log_file: str | None = None, level: int = logging.INFO) -> None:
    """Configure logging for the application.

    Args:
        log_file: Optional path to save logs to a file
        level: Logging level
    """
    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file:
        handlers.append(logging.FileHandler(log_file, mode="w"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )


def get_accelerator(use_amp: bool) -> Accelerator:
    """Create an accelerator instance with the appropriate settings.

    Args:
        use_amp: Whether to use automatic mixed precision.

    Returns:
        An Accelerator instance.
    """
    # Determine mixed precision setting
    mixed_precision = "no"
    if use_amp and torch.cuda.is_available():
        mixed_precision = "fp16"

    # Create a single accelerator instance
    try:
        # First, try to create with specified settings
        return Accelerator(mixed_precision=mixed_precision)
    except RuntimeError:
        # If accelerator is already initialized, create with default settings
        # and log a warning
        logging.warning(
            "Accelerator already initialized. Using existing instance. "
            "Restart your runtime for full control over acceleration settings."
        )
        return Accelerator()


def main(config_path: str | Path, debug: bool = False) -> list[TrainingMetrics]:
    """Main training function.

    Args:
        config_path: Path to the YAML configuration file
        debug: Enable detailed diagnostics

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

    # Create diagnostics directory
    diagnostics_dir = Path("diagnostics")
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    # Set up logging with file output for diagnostics
    log_file = diagnostics_dir / "training_diagnostics.log" if debug else None
    setup_logging(log_file=str(log_file) if log_file else None)

    logger = logging.getLogger(__name__)
    logger.info(f"Loading configuration from {config_file}")
    if debug:
        logger.info("DEBUG MODE ENABLED - detailed diagnostics will be logged")

    try:
        # Load configuration
        config = Config.from_yaml(str(config_file))
    except yaml.YAMLError as e:
        logger.error(f"Invalid configuration: {e}")
        raise

    # Get the appropriate accelerator
    accelerator = get_accelerator(use_amp=config.training.use_amp)

    # Log the device and mixed precision information
    device_type = "GPU" if torch.cuda.is_available() else "CPU"
    precision_mode = accelerator.mixed_precision
    if accelerator.is_local_main_process:
        logger.info(f"Using {device_type} with precision mode: {precision_mode}")

    # Set random seed for reproducibility
    seed = config.seed
    logger.info(f"Setting random seed to {seed}")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Initialize tokenizer
    tokenizer_config = TokenizerConfig(**vars(config.tokenizer))
    tokenizer = DiffusionTokenizer(config=tokenizer_config)

    if accelerator.is_local_main_process:
        logger.info(f"Initialized tokenizer with vocab size: {tokenizer.vocab_size}")

        # Log token ID information for diagnostics
        if debug:
            logger.info(f"  - PAD token ID: {tokenizer.pad_token_id}")
            logger.info(f"  - MASK token ID: {tokenizer.mask_token_id}")
            logger.info(f"  - BOS token ID: {tokenizer.bos_token_id}")
            logger.info(f"  - EOS token ID: {tokenizer.eos_token_id}")

    # Update vocab_size in transformer config based on tokenizer
    config.transformer.vocab_size = tokenizer.vocab_size

    # Initialize transformer
    transformer_config = DiffusionTransformerConfig(**vars(config.transformer))
    transformer = DiffusionTransformer(transformer_config)

    if accelerator.is_local_main_process:
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

    if accelerator.is_local_main_process:
        logger.info(f"Using masking strategy: {config.masking.strategy}")
        if debug:
            logger.info(f"  - Mask token ID: {tokenizer.mask_token_id}")
            logger.info(
                f"  - Special tokens excluded from masking: BOS={tokenizer.bos_token_id}, EOS={tokenizer.eos_token_id}, PAD={tokenizer.pad_token_id}"
            )

    # Initialize model
    model = DiffusionLanguageModel(transformer, tokenizer, masking_strategy)

    if accelerator.is_local_main_process:
        param_count = sum(p.numel() for p in model.parameters())
        trainable_param_count = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        logger.info(f"Model initialized with {param_count:,} parameters")
        logger.info(f"Trainable parameters: {trainable_param_count:,}")

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

    if accelerator.is_local_main_process:
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

    if accelerator.is_local_main_process:
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

    if accelerator.is_local_main_process:
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

    # Initialize trainer passing our accelerator instance
    trainer = DiffusionLanguageModelTrainer(
        model=model,
        optimizer=optimizer,
        accelerator=accelerator,  # Pass the existing accelerator
        scheduler=scheduler,
        checkpoint_dir=str(checkpoint_dir),
        gradient_clip_val=config.training.gradient_clip_val,
        log_interval=config.training.log_interval,
        debug_mode=debug,  # Enable diagnostics based on the debug flag
    )

    # Log device and mixed precision information
    if accelerator.is_local_main_process:
        logger.info(
            f"Starting training with:\n"
            f"  - Epochs: {config.training.num_epochs}\n"
            f"  - Batch size: {config.training.batch_size}\n"
            f"  - Learning rate: {config.training.learning_rate}\n"
            f"  - Device: {device_type}\n"
            f"  - Mixed precision: {precision_mode}\n"
            f"  - Gradient clipping: {config.training.gradient_clip_val}\n"
            f"  - Mask ratio: {config.masking.mask_ratio if hasattr(config.masking, 'mask_ratio') else 'random'}\n"
        )

    # Special token IDs to exclude from loss calculation
    special_token_ids = [
        tokenizer.pad_token_id,
        tokenizer.bos_token_id,
        tokenizer.eos_token_id,
    ]

    if debug and accelerator.is_local_main_process:
        logger.info(f"Special token IDs excluded from loss: {special_token_ids}")

        # Check first batch from dataloader
        first_batch = next(iter(train_loader))
        logger.info("First batch statistics:")
        logger.info(f"  - input_ids shape: {first_batch['input_ids'].shape}")
        if "attention_mask" in first_batch:
            logger.info(
                f"  - attention_mask shape: {first_batch['attention_mask'].shape}"
            )
            mask_sum = first_batch["attention_mask"].sum().item()
            total = first_batch["attention_mask"].numel()
            logger.info(
                f"  - attention_mask active: {mask_sum}/{total} ({mask_sum / total:.2%})"
            )

    # Train the model
    try:
        history = trainer.train(
            train_dataloader=train_loader,
            num_epochs=config.training.num_epochs,
            mask_prob=config.masking.mask_ratio
            if hasattr(config.masking, "mask_ratio")
            else None,
            special_token_ids=special_token_ids,
            val_dataloader=val_loader,
            early_stopping_patience=config.training.early_stopping_patience,
        )

        if accelerator.is_local_main_process:
            logger.info("Training completed successfully")

            # Save final model (only from main process)
            final_model_path = checkpoint_dir / "final_model.pt"
            trainer.save_checkpoint(str(final_model_path))
            logger.info(f"Final model saved to {final_model_path}")

            # Save final configuration
            config_save_path = checkpoint_dir / "final_config.yaml"
            config.save(str(config_save_path))
            logger.info(f"Final configuration saved to {config_save_path}")

        return history
    except KeyboardInterrupt:
        if accelerator.is_local_main_process:
            logger.info("Training interrupted by user")
            # Save interrupted model
            interrupted_path = checkpoint_dir / "interrupted_model.pt"
            trainer.save_checkpoint(str(interrupted_path))
            logger.info(f"Interrupted model saved to {interrupted_path}")
        return trainer.get_training_history()
    except Exception as e:
        if accelerator.is_local_main_process:
            logger.error(f"Training failed: {e}")
            logger.error(f"Detailed traceback: {traceback.format_exc()}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train diffusion language model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable detailed diagnostics for debugging",
    )
    args = parser.parse_args()

    try:
        main(args.config, debug=args.debug)
    except (FileNotFoundError, yaml.YAMLError) as e:
        logging.error(f"Configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Training failed: {e}")
        sys.exit(2)
