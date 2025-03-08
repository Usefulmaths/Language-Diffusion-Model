"""Diffusion Language Model training script."""

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.dataset import create_question_dataloaders
from src.datatypes import DiffusionTransformerConfig
from src.masking_strategy import RandomMaskingStrategy
from src.model import DiffusionLanguageModel
from src.tokenizer import DiffusionTokenizer
from src.trainer import DiffusionLanguageModelTrainer
from src.transformer import DiffusionTransformer

if __name__ == "__main__":
    # Optionally, set the device explicitly
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize components
    tokenizer = DiffusionTokenizer()
    config = DiffusionTransformerConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=32,
        nhead=2,
        num_layers=2,
        dim_feedforward=64,
        dropout=0.1,
        layer_norm_eps=1e-12,
        max_position_embeddings=64,
    )
    transformer = DiffusionTransformer(config)
    masking_strategy = RandomMaskingStrategy(
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = DiffusionLanguageModel(transformer, tokenizer, masking_strategy)

    # Create dataloaders
    train_loader, val_loader = create_question_dataloaders(
        file_path="data/small_questions.txt",
        tokenizer=tokenizer,
        batch_size=8,
        max_length=64,
        num_workers=0,
    )

    # Initialize optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=len(train_loader))

    # Initialize trainer with mixed precision enabled (use_amp=True)
    trainer = DiffusionLanguageModelTrainer(
        model=model,
        optimizer=optimizer,
        device=device,  # Explicitly pass the device if desired
        scheduler=scheduler,
        checkpoint_dir="checkpoints",
        gradient_clip_val=1.0,
        use_amp=True,  # Enable mixed precision training
    )

    # Train the model
    trainer.train(
        train_dataloader=train_loader,
        num_epochs=10,
        val_dataloader=val_loader,
        early_stopping_patience=3,
    )
