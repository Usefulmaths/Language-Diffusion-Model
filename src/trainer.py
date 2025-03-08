"""Trainer for diffusion language models with optional mixed precision support."""

import logging
import os
import time
from typing import Any

import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.datatypes import TrainingMetrics
from src.loss import variational_lower_bound_loss
from src.model import DiffusionLanguageModel


class DiffusionLanguageModelTrainer:
    """Trainer for diffusion language models with optional mixed precision support."""

    def __init__(
        self,
        model: DiffusionLanguageModel,
        optimizer: Optimizer,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        scheduler: LRScheduler | None = None,
        checkpoint_dir: str | None = None,
        log_interval: int = 100,
        gradient_clip_val: float | None = None,
        use_amp: bool = False,
    ):
        """Initialize the trainer.

        Args:
            model: Diffusion language model to train.
            optimizer: Optimizer for parameter updates.
            device: Device to train on.
            scheduler: Learning rate scheduler.
            checkpoint_dir: Directory to save checkpoints.
            log_interval: Number of steps between logging.
            gradient_clip_val: Maximum gradient norm for gradient clipping.
            use_amp: Whether to use mixed precision training.
        """
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.checkpoint_dir = checkpoint_dir
        self.log_interval = log_interval
        self.gradient_clip_val = gradient_clip_val
        self.use_amp = use_amp

        self.model.to(self.device)
        self.logger = logging.getLogger(__name__)

        # Mixed precision scaler, if using AMP and on CUDA
        if self.use_amp and self.device.startswith("cuda"):
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

        # Initialize metrics tracking
        self.history: list[TrainingMetrics] = []
        self.best_loss: float = float("inf")
        self.global_step: int = 0

        # Create checkpoint directory if needed
        if self.checkpoint_dir and not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir, exist_ok=True)

    def train_step(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        mask_prob: float | None = None,
        special_token_ids: list[int] | None = None,
    ) -> dict[str, float]:
        """Perform a single training step.

        Args:
            input_ids: Input token ids of shape [batch_size, seq_len].
            attention_mask: Attention mask of shape [batch_size, seq_len].
            mask_prob: Optional probability of masking each token.
            special_token_ids: Optional list of special token IDs to exclude from loss.

        Returns:
            Dictionary with training metrics (contains at least 'loss').
        """
        # Move tensors to device
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        # Zero gradients
        self.optimizer.zero_grad()

        if self.use_amp and self.scaler is not None:
            # Mixed precision forward pass
            with torch.cuda.amp.autocast():
                logits, masked_input_ids, mask_indices = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    mask_prob=mask_prob,
                )
                loss = variational_lower_bound_loss(
                    logits=logits,
                    targets=input_ids,
                    masked_input_ids=masked_input_ids,
                    mask_token_id=self.model.tokenizer.mask_token_id,
                    attention_mask=attention_mask,
                    pad_token_id=self.model.tokenizer.pad_token_id,
                    special_token_ids=special_token_ids,
                    mask_ratio=mask_prob if mask_prob is not None else 0.5,
                )

            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()

            if self.gradient_clip_val is not None:
                # Unscale gradients before clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clip_val
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Standard precision forward pass
            logits, masked_input_ids, mask_indices = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                mask_prob=mask_prob,
            )
            loss = variational_lower_bound_loss(
                logits=logits,
                targets=input_ids,
                masked_input_ids=masked_input_ids,
                mask_token_id=self.model.tokenizer.mask_token_id,
                attention_mask=attention_mask,
                pad_token_id=self.model.tokenizer.pad_token_id,
                special_token_ids=special_token_ids,
                mask_ratio=mask_prob if mask_prob is not None else 0.5,
            )

            loss.backward()

            if self.gradient_clip_val is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clip_val
                )

            self.optimizer.step()

        metrics = {"loss": loss.item()}
        return metrics

    def train_epoch(
        self,
        dataloader: DataLoader[Any],
        epoch: int,
        mask_prob: float | None = None,
        special_token_ids: list[int] | None = None,
    ) -> TrainingMetrics:
        """Train for one epoch.

        Args:
            dataloader: DataLoader for training data.
            epoch: Current epoch number.
            mask_prob: Fixed masking probability for this epoch (if None, uses random).
            special_token_ids: Optional list of special token IDs to exclude from loss.

        Returns:
            TrainingMetrics object containing metrics for this epoch.
        """
        self.model.train()
        epoch_loss = 0.0
        epoch_start_time = time.time()

        # Get current learning rate
        current_lr = self.optimizer.param_groups[0]["lr"]

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")
        for step, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"]
            attention_mask = batch.get("attention_mask", None)

            step_metrics = self.train_step(
                input_ids=input_ids,
                attention_mask=attention_mask,
                mask_prob=mask_prob,
                special_token_ids=special_token_ids,
            )

            self.global_step += 1
            epoch_loss += step_metrics["loss"]
            avg_loss = epoch_loss / (step + 1)
            progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})

            if self.global_step % self.log_interval == 0:
                self.logger.info(
                    f"Step {self.global_step}, Loss: {step_metrics['loss']:.4f},\
                        Avg Loss: {avg_loss:.4f}"
                )

        epoch_time = time.time() - epoch_start_time
        avg_loss = epoch_loss / len(dataloader)

        if self.scheduler is not None:
            self.scheduler.step()

        train_metrics = TrainingMetrics(
            epoch=epoch + 1,
            loss=avg_loss,
            learning_rate=current_lr,
            epoch_time=epoch_time,
        )
        self.history.append(train_metrics)
        self.logger.info(f"Epoch {epoch + 1} completed: {train_metrics}")

        if avg_loss < self.best_loss:
            self.best_loss = avg_loss
            if self.checkpoint_dir:
                self.save_checkpoint(os.path.join(self.checkpoint_dir, "best_model.pt"))

        if self.checkpoint_dir:
            self.save_checkpoint(
                os.path.join(self.checkpoint_dir, f"epoch_{epoch + 1}.pt")
            )

        return train_metrics

    def train(
        self,
        train_dataloader: DataLoader[Any],
        num_epochs: int,
        mask_prob: float | None = None,
        special_token_ids: list[int] | None = None,
        val_dataloader: DataLoader[Any] | None = None,
        early_stopping_patience: int | None = None,
    ) -> list[TrainingMetrics]:
        """Train the model for multiple epochs.

        Args:
            train_dataloader: DataLoader for training data.
            num_epochs: Number of epochs to train.
            mask_prob: Fixed masking probability (if None, uses random per batch).
            special_token_ids: Optional list of special token IDs to exclude from loss.
            val_dataloader: Optional validation dataloader.
            early_stopping_patience: Number of epochs with no improvement before
            stopping.

        Returns:
            List of TrainingMetrics objects containing metrics for each epoch.
        """
        self.logger.info(f"Starting training for {num_epochs} epochs")
        no_improvement_count = 0
        best_val_loss = float("inf")

        for epoch in range(num_epochs):
            train_metrics = self.train_epoch(
                dataloader=train_dataloader,
                epoch=epoch,
                mask_prob=mask_prob,
                special_token_ids=special_token_ids,
            )
            self.logger.info(f"Train metrics for epoch {epoch + 1}: {train_metrics}")

            if val_dataloader:
                val_loss = self.evaluate(val_dataloader)
                self.logger.info(f"Validation loss: {val_loss:.4f}")

                if early_stopping_patience is not None:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        no_improvement_count = 0
                        if self.checkpoint_dir:
                            self.save_checkpoint(
                                os.path.join(self.checkpoint_dir, "best_val_model.pt")
                            )
                    else:
                        no_improvement_count += 1
                        if no_improvement_count >= early_stopping_patience:
                            self.logger.info(f"Early stopping after {epoch + 1} epochs")
                            break

        self.logger.info("Training completed")
        return self.history

    def evaluate(
        self,
        dataloader: DataLoader[Any],
        mask_prob: float = 0.15,
        special_token_ids: list[int] | None = None,
    ) -> float:
        """Evaluate the model on a dataset.

        Args:
            dataloader: DataLoader for evaluation data.
            mask_prob: Masking probability for evaluation.
            special_token_ids: Optional list of special token IDs to exclude from loss.

        Returns:
            Average loss on the dataset.
        """
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                if isinstance(batch, dict):
                    input_ids = batch["input_ids"]
                    attention_mask = batch.get("attention_mask")
                else:
                    input_ids = batch[0]
                    attention_mask = batch[1] if len(batch) > 1 else None

                input_ids = input_ids.to(self.device)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)

                if self.use_amp and self.scaler is not None:
                    with torch.cuda.amp.autocast():
                        logits, masked_input_ids, mask_indices = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            mask_prob=mask_prob,
                        )
                        loss = variational_lower_bound_loss(
                            logits=logits,
                            targets=input_ids,
                            masked_input_ids=masked_input_ids,
                            mask_token_id=self.model.tokenizer.mask_token_id,
                            attention_mask=attention_mask,
                            pad_token_id=self.model.tokenizer.pad_token_id,
                            special_token_ids=special_token_ids,
                            mask_ratio=mask_prob,
                        )
                else:
                    logits, masked_input_ids, mask_indices = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        mask_prob=mask_prob,
                    )
                    loss = variational_lower_bound_loss(
                        logits=logits,
                        targets=input_ids,
                        masked_input_ids=masked_input_ids,
                        mask_token_id=self.model.tokenizer.mask_token_id,
                        attention_mask=attention_mask,
                        pad_token_id=self.model.tokenizer.pad_token_id,
                        special_token_ids=special_token_ids,
                        mask_ratio=mask_prob,
                    )

                total_loss += loss.item()

        return total_loss / len(dataloader)

    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint.

        Args:
            path: Path to save checkpoint.
        """
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict()
            if self.scheduler
            else None,
            "global_step": self.global_step,
            "best_loss": self.best_loss,
            "history": self.history,
        }
        torch.save(checkpoint, path)
        self.logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint.

        Args:
            path: Path to load checkpoint from.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler and checkpoint["scheduler_state_dict"]:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.best_loss = checkpoint["best_loss"]
        self.history = checkpoint["history"]
        self.logger.info(f"Checkpoint loaded from {path}")

    def get_training_history(self) -> list[TrainingMetrics]:
        """Get the training history.

        Returns:
            List of TrainingMetrics objects for all completed epochs.
        """
        return self.history
