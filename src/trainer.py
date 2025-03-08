"""Trainer for diffusion language models with diagnostics for loss investigation."""

import logging
import os
import time
from dataclasses import dataclass

import torch
from accelerate import Accelerator
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.loss import variational_lower_bound_loss
from src.model import DiffusionLanguageModel


@dataclass
class TrainingMetrics:
    """Container for training metrics."""

    epoch: int
    loss: float
    learning_rate: float
    epoch_time: float

    def __str__(self) -> str:
        """Convert metrics to string for logging."""
        return (
            f"Epoch: {self.epoch}, "
            f"Loss: {self.loss:.4f}, "
            f"Learning Rate: {self.learning_rate:.8f}, "
            f"Time: {self.epoch_time:.2f}s"
        )


class DiffusionLanguageModelTrainer:
    """Trainer for diffusion language models with accelerate."""

    def __init__(
        self,
        model: DiffusionLanguageModel,
        optimizer: Optimizer,
        accelerator: Accelerator | None = None,
        scheduler: LRScheduler | None = None,
        checkpoint_dir: str | None = None,
        log_interval: int = 100,
        gradient_clip_val: float | None = None,
        debug_mode: bool = True,
    ):
        """Initialize the trainer.

        Args:
            model: Diffusion language model to train.
            optimizer: Optimizer for parameter updates.
            accelerator: Pre-initialized accelerator instance.
            scheduler: Learning rate scheduler.
            checkpoint_dir: Directory to save checkpoints.
            log_interval: Number of steps between logging.
            gradient_clip_val: Maximum gradient norm for gradient clipping.
            debug_mode: Enable detailed diagnostics for debugging.
        """
        # Use provided accelerator or create a new one if not provided
        # This avoids re-initializing the accelerator if it already exists
        self.accelerator = accelerator or Accelerator()

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.checkpoint_dir = checkpoint_dir
        self.log_interval = log_interval
        self.gradient_clip_val = gradient_clip_val
        self.debug_mode = debug_mode

        # Create checkpoint directory if needed
        if self.checkpoint_dir:
            os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.logger = logging.getLogger(__name__)

        # Initialize metrics tracking
        self.history: list[TrainingMetrics] = []
        self.best_loss: float = float("inf")
        self.global_step: int = 0

        # Prepare model and optimizer with accelerator (separately to avoid tuple unpacking issues)
        self.model = self.accelerator.prepare(model)
        self.optimizer = self.accelerator.prepare(optimizer)

        # Prepare scheduler if provided
        if self.scheduler is not None:
            self.scheduler = self.accelerator.prepare(scheduler)

    def _log_diagnostics(
        self,
        step: int,
        input_ids: Tensor,
        masked_input_ids: Tensor,
        logits: Tensor,
        mask_prob: float | None,
        loss: Tensor,
    ) -> None:
        """Log diagnostic information for debugging.

        Args:
            step: Current step number.
            input_ids: Original input token ids.
            masked_input_ids: Input token ids with masks applied.
            logits: Model predictions.
            mask_prob: Mask probability used.
            loss: Loss value from criterion.
        """
        if not self.debug_mode or step % self.log_interval != 0:
            return

        # Count masks and calculate actual mask ratio
        mask_token_id = self.model.tokenizer.mask_token_id
        mask_indices = masked_input_ids == mask_token_id
        num_masks = mask_indices.sum().item()
        total_tokens = input_ids.numel()
        actual_mask_ratio = num_masks / total_tokens if total_tokens > 0 else 0

        self.logger.info(f"DIAGNOSTICS for step {step}:")
        self.logger.info(
            f"  - Specified mask_prob: {mask_prob if mask_prob is not None else 'random'}"
        )
        self.logger.info(
            f"  - Actual mask ratio: {actual_mask_ratio:.4f} ({num_masks}/{total_tokens})"
        )
        self.logger.info(f"  - Loss value: {loss.item():.6f}")

        # Check predictions for a few masked tokens
        with torch.no_grad():
            predicted_tokens = logits.argmax(dim=-1)

            # Find masked positions
            batch_idx = 0  # First batch
            masked_positions = (masked_input_ids[batch_idx] == mask_token_id).nonzero()

            if len(masked_positions) > 0:
                # Log up to 3 predictions
                num_to_log = min(3, len(masked_positions))

                self.logger.info("  - Sample predictions:")
                for i in range(num_to_log):
                    pos = masked_positions[i].item()
                    true_token = input_ids[batch_idx, pos].item()
                    pred_token = predicted_tokens[batch_idx, pos].item()

                    # Get string representation
                    try:
                        true_str = self.model.tokenizer.decode([true_token])
                        pred_str = self.model.tokenizer.decode([pred_token])
                    except:
                        true_str = "<unknown>"
                        pred_str = "<unknown>"

                    self.logger.info(
                        f"    - Pos {pos}: True: {true_token} ('{true_str}') → Pred: {pred_token} ('{pred_str}')"
                    )

    def _log_gradient_stats(self) -> None:
        """Log statistics about the gradients for debugging."""
        if not self.debug_mode:
            return

        # Calculate gradient norm
        total_norm = 0.0
        max_norm = 0.0
        min_norm = float("inf")

        for name, p in self.model.named_parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2).item()
                total_norm += param_norm**2
                max_norm = max(max_norm, param_norm)
                min_norm = min(min_norm, param_norm)

        total_norm = total_norm**0.5

        self.logger.info("  - Gradient stats:")
        self.logger.info(f"    - Total norm: {total_norm:.6f}")
        self.logger.info(
            f"    - Min/Max per-parameter norm: {min_norm:.6f}/{max_norm:.6f}"
        )

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
        # Zero gradients
        self.optimizer.zero_grad()

        # Forward pass - accelerate handles device placement automatically
        logits, masked_input_ids, mask_indices = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            mask_prob=mask_prob,
        )

        # Save pre-loss state for diagnostics
        if self.debug_mode and self.global_step % self.log_interval == 0:
            batch_size, seq_len = input_ids.shape
            vocab_size = logits.shape[-1]
            self.logger.info(
                f"  - Input shape: [{batch_size}, {seq_len}], Logits shape: [{batch_size}, {seq_len}, {vocab_size}]"
            )

        # Calculate loss
        loss = variational_lower_bound_loss(
            logits=logits,
            targets=input_ids,
            masked_input_ids=masked_input_ids,
            mask_token_id=self.model.tokenizer.mask_token_id,
            attention_mask=attention_mask,
            pad_token_id=self.model.tokenizer.pad_token_id,
            special_token_ids=special_token_ids,
            mask_ratio=mask_prob if mask_prob is not None else 0.5,
            debug_mode=self.debug_mode,
            step=self.global_step,
            log_interval=self.log_interval,
            logger=self.logger,
        )

        # Log diagnostics before backward pass
        self._log_diagnostics(
            step=self.global_step,
            input_ids=input_ids,
            masked_input_ids=masked_input_ids,
            logits=logits,
            mask_prob=mask_prob,
            loss=loss,
        )

        # Backward pass - accelerate handles mixed precision
        self.accelerator.backward(loss)

        # Log gradient statistics
        if self.debug_mode and self.global_step % self.log_interval == 0:
            self._log_gradient_stats()

        if self.gradient_clip_val is not None:
            self.accelerator.clip_grad_norm_(
                self.model.parameters(), self.gradient_clip_val
            )
            if self.debug_mode and self.global_step % self.log_interval == 0:
                self.logger.info(
                    f"  - Applied gradient clipping with value: {self.gradient_clip_val}"
                )

        self.optimizer.step()

        return {"loss": loss.item()}

    def train_epoch(
        self,
        dataloader: DataLoader,
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

        # Show progress bar only on main process
        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch {epoch + 1}",
            disable=not self.accelerator.is_local_main_process,
        )

        for step, batch in enumerate(progress_bar):
            step_metrics = self.train_step(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask", None),
                mask_prob=mask_prob,
                special_token_ids=special_token_ids,
            )

            self.global_step += 1
            epoch_loss += step_metrics["loss"]
            avg_loss = epoch_loss / (step + 1)
            progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})

            if (
                self.global_step % self.log_interval == 0
                and self.accelerator.is_local_main_process
            ):
                self.logger.info(
                    f"Step {self.global_step}, Loss: {step_metrics['loss']:.4f}, "
                    f"Avg Loss: {avg_loss:.4f}"
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

        # Only log and save checkpoints from main process
        if self.accelerator.is_local_main_process:
            self.logger.info(f"Epoch {epoch + 1} completed: {train_metrics}")

            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                if self.checkpoint_dir:
                    self.save_checkpoint(
                        os.path.join(self.checkpoint_dir, "best_model.pt")
                    )

            if self.checkpoint_dir:
                self.save_checkpoint(
                    os.path.join(self.checkpoint_dir, f"epoch_{epoch + 1}.pt")
                )

        return train_metrics

    def train(
        self,
        train_dataloader: DataLoader,
        num_epochs: int,
        mask_prob: float | None = None,
        special_token_ids: list[int] | None = None,
        val_dataloader: DataLoader | None = None,
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
        # Prepare dataloaders with accelerate (separately to avoid issues)
        train_dataloader = self.accelerator.prepare(train_dataloader)
        if val_dataloader is not None:
            val_dataloader = self.accelerator.prepare(val_dataloader)

        if self.accelerator.is_local_main_process:
            self.logger.info(f"Starting training for {num_epochs} epochs")
            if mask_prob is not None:
                self.logger.info(f"Using fixed mask probability: {mask_prob}")
            else:
                self.logger.info("Using random mask probability for each batch")

        no_improvement_count = 0
        best_val_loss = float("inf")

        for epoch in range(num_epochs):
            train_metrics = self.train_epoch(
                dataloader=train_dataloader,
                epoch=epoch,
                mask_prob=mask_prob,
                special_token_ids=special_token_ids,
            )

            if self.accelerator.is_local_main_process:
                self.logger.info(
                    f"Train metrics for epoch {epoch + 1}: {train_metrics}"
                )

            if val_dataloader:
                val_loss = self.evaluate(val_dataloader, mask_prob, special_token_ids)

                if self.accelerator.is_local_main_process:
                    self.logger.info(f"Validation loss: {val_loss:.4f}")

                if early_stopping_patience is not None:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        no_improvement_count = 0
                        if (
                            self.checkpoint_dir
                            and self.accelerator.is_local_main_process
                        ):
                            self.save_checkpoint(
                                os.path.join(self.checkpoint_dir, "best_val_model.pt")
                            )
                    else:
                        no_improvement_count += 1
                        if no_improvement_count >= early_stopping_patience:
                            if self.accelerator.is_local_main_process:
                                self.logger.info(
                                    f"Early stopping after {epoch + 1} epochs"
                                )
                            break

        if self.accelerator.is_local_main_process:
            self.logger.info("Training completed")

        return self.history

    def evaluate(
        self,
        dataloader: DataLoader,
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
            progress_bar = tqdm(
                dataloader,
                desc="Evaluating",
                disable=not self.accelerator.is_local_main_process,
            )

            for batch in progress_bar:
                if isinstance(batch, dict):
                    input_ids = batch["input_ids"]
                    attention_mask = batch.get("attention_mask")
                else:
                    input_ids = batch[0]
                    attention_mask = batch[1] if len(batch) > 1 else None

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
                    debug_mode=False,  # Disable debugging for evaluation
                )

                total_loss += loss.item()

        return total_loss / len(dataloader)

    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint.

        Args:
            path: Path to save checkpoint.
        """
        # Get unwrapped model for saving
        unwrapped_model = self.accelerator.unwrap_model(self.model)

        checkpoint = {
            "model_state_dict": unwrapped_model.state_dict(),
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
        checkpoint = torch.load(path, map_location="cpu")  # Load to CPU first

        # Always apply to unwrapped model
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.load_state_dict(checkpoint["model_state_dict"])

        # Load optimizer and scheduler states
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
