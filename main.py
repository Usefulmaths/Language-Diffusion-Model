#!/usr/bin/env python3
"""Main script to train a diffusion language model with a train/val split using Accelerate."""

import os
import random

import click
import torch
from accelerate import Accelerator
from torch.optim import AdamW

from src.config import Config
from src.dataset import create_question_dataloaders
from src.masking_strategy import create_masking_strategy, create_remasking_strategy
from src.model import DiffusionLanguageModel
from src.scheduler import create_scheduler
from src.tokenizer import DiffusionTokenizer, TokenizerConfig
from src.trainer import DiffusionLanguageModelTrainer
from src.transformer import DiffusionTransformer, DiffusionTransformerConfig


@click.command()
@click.option(
    "--config",
    "-c",
    default="configs/config.yaml",
    help="Path to the configuration file",
)
def main(config):
    """Train a diffusion language model with the specified configuration."""
    # Load configuration
    if os.path.exists(config):
        config_obj = Config.from_yaml(config)
        print(f"Loaded configuration from {config}")
    else:
        raise FileNotFoundError(f"Config file not found at {config}")

    # Set random seed for reproducibility.
    torch.manual_seed(config_obj.seed)
    random.seed(config_obj.seed)

    # Initialize tokenizer with minimal configuration.
    tokenizer_config = TokenizerConfig(
        tokenizer_name=config_obj.tokenizer.tokenizer_name,
        pad_token=config_obj.tokenizer.pad_token,
        bos_token=config_obj.tokenizer.bos_token,
        eos_token=config_obj.tokenizer.eos_token,
        mask_token=config_obj.tokenizer.mask_token,
    )
    tokenizer = DiffusionTokenizer(config=tokenizer_config)

    # Create a minimal transformer configuration.
    transformer_config = DiffusionTransformerConfig(
        vocab_size=tokenizer.vocab_size,
        num_layers=config_obj.transformer.num_layers,
        d_model=config_obj.transformer.d_model,
        nhead=config_obj.transformer.nhead,
        dim_feedforward=config_obj.transformer.dim_feedforward,
        dropout=config_obj.transformer.dropout,
        layer_norm_eps=config_obj.transformer.layer_norm_eps,
        max_position_embeddings=config_obj.transformer.max_position_embeddings,
    )
    transformer = DiffusionTransformer(transformer_config)

    # Create masking strategies.
    masking_strategy = create_masking_strategy(
        config_obj.masking.strategy,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        mask_token_id=tokenizer.mask_token_id,
    )
    remasking_strategy = create_remasking_strategy(
        config_obj.remasking.strategy,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        mask_token_id=tokenizer.mask_token_id,
    )

    # Initialize the diffusion language model.
    model = DiffusionLanguageModel(
        transformer, tokenizer, masking_strategy, remasking_strategy
    )

    # Create optimizer.
    optimizer = AdamW(
        model.parameters(),
        lr=config_obj.training.learning_rate,
        weight_decay=config_obj.training.weight_decay,
    )

    # Create dataloaders with a train/val split.
    train_loader, val_loader = create_question_dataloaders(
        file_path=config_obj.data.train_file,
        tokenizer=tokenizer,
        batch_size=config_obj.training.batch_size,
        max_length=config_obj.training.max_length,
        train_ratio=config_obj.data.train_ratio,
        shuffle=config_obj.data.shuffle,
        num_workers=config_obj.training.num_workers,
    )

    # Initialize Accelerator and prepare model, optimizer, and dataloaders.
    accelerator = Accelerator()
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    # Set up a learning rate scheduler.
    total_steps = len(train_loader) * config_obj.training.num_epochs
    scheduler = create_scheduler(
        optimizer=optimizer,
        total_steps=total_steps,
        warmup_steps=config_obj.training.scheduler.warmup_steps,
        initial_lr=config_obj.training.scheduler.initial_lr,
        peak_lr=config_obj.training.scheduler.peak_lr,
        stable_lr=config_obj.training.scheduler.stable_lr,
        final_lr=config_obj.training.scheduler.final_lr,
    )

    # Pass the accelerator instance into the trainer.
    trainer = DiffusionLanguageModelTrainer(
        model, optimizer, scheduler, accelerator=accelerator
    )

    # Train with periodic evaluation.
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        num_epochs=config_obj.training.num_epochs,
        eval_steps=config_obj.training.evaluation.eval_steps,
        eval_mask_ratio=config_obj.training.evaluation.eval_mask_ratio,
        num_examples=config_obj.training.evaluation.num_examples,
    )

    print("Training completed.")


if __name__ == "__main__":
    main()
