#!/usr/bin/env python3
"""Main script to train a diffusion language model with a train/val split using Accelerate."""

import os
import random

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


def main():
    # Load configuration
    config_path = "configs/config.yaml"
    if os.path.exists(config_path):
        config = Config.from_yaml(config_path)
        print(f"Loaded configuration from {config_path}")
    else:
        raise FileNotFoundError(f"Config file not found at {config_path}")

    # Set random seed for reproducibility.
    torch.manual_seed(config.seed)
    random.seed(config.seed)

    # Initialize tokenizer with minimal configuration.
    tokenizer_config = TokenizerConfig(
        tokenizer_name=config.tokenizer.tokenizer_name,
        pad_token=config.tokenizer.pad_token,
        bos_token=config.tokenizer.bos_token,
        eos_token=config.tokenizer.eos_token,
        mask_token=config.tokenizer.mask_token,
    )
    tokenizer = DiffusionTokenizer(config=tokenizer_config)

    # Create a minimal transformer configuration.
    transformer_config = DiffusionTransformerConfig(
        vocab_size=tokenizer.vocab_size,
        num_layers=config.transformer.num_layers,
        d_model=config.transformer.d_model,
        nhead=config.transformer.nhead,
        dim_feedforward=config.transformer.dim_feedforward,
        dropout=config.transformer.dropout,
        layer_norm_eps=config.transformer.layer_norm_eps,
        max_position_embeddings=config.transformer.max_position_embeddings,
    )
    transformer = DiffusionTransformer(transformer_config)

    # Create masking strategies.
    masking_strategy = create_masking_strategy(
        config.masking.strategy,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        mask_token_id=tokenizer.mask_token_id,
    )
    remasking_strategy = create_remasking_strategy(
        config.remasking.strategy,
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
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    # Create dataloaders with a train/val split.
    train_loader, val_loader = create_question_dataloaders(
        file_path=config.data.train_file,
        tokenizer=tokenizer,
        batch_size=config.training.batch_size,
        max_length=config.training.max_length,
        train_ratio=config.data.train_ratio,
        shuffle=config.data.shuffle,
        num_workers=config.training.num_workers,
    )

    # Initialize Accelerator and prepare model, optimizer, and dataloaders.
    accelerator = Accelerator()
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    # Set up a learning rate scheduler.
    total_steps = len(train_loader) * config.training.num_epochs
    scheduler = create_scheduler(
        optimizer=optimizer,
        total_steps=total_steps,
        warmup_steps=config.training.scheduler.warmup_steps,
        initial_lr=config.training.scheduler.initial_lr,
        peak_lr=config.training.scheduler.peak_lr,
        stable_lr=config.training.scheduler.stable_lr,
        final_lr=config.training.scheduler.final_lr,
    )

    # Pass the accelerator instance into the trainer.
    trainer = DiffusionLanguageModelTrainer(
        model, optimizer, scheduler, accelerator=accelerator
    )

    # Train with periodic evaluation.
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        num_epochs=config.training.num_epochs,
        eval_steps=config.training.evaluation.eval_steps,
        eval_mask_ratio=config.training.evaluation.eval_mask_ratio,
        num_examples=config.training.evaluation.num_examples,
    )

    print("Training completed.")


if __name__ == "__main__":
    main()
