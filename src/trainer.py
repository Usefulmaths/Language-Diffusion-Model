"""Trainer for diffusion language models using Accelerate."""

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from src.loss import variational_lower_bound_loss
from src.model import DiffusionLanguageModel


class DiffusionLanguageModelTrainer:
    """Minimal trainer for diffusion language models with a learning rate scheduler
    that uses an Accelerator instance for device management and gradient scaling.

    Note: This trainer now requires an Accelerator instance (from Hugging Face Accelerate)
          to be passed in. It assumes that the model, optimizer, and dataloaders have been
          prepared (via accelerator.prepare(...)) in your main training script.
    """

    def __init__(
        self,
        model: DiffusionLanguageModel,
        optimizer: Optimizer,
        scheduler: LRScheduler | None,
        accelerator,  # Required Accelerator instance.
    ):
        """Args:
        model: The diffusion language model.
        optimizer: Optimizer for updating model parameters.
        scheduler: Learning rate scheduler (optional).
        accelerator: An Accelerator instance for handling device placement
                     and backward passes.
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.accelerator = accelerator
        self.history = []  # Tracks average loss per epoch
        self.global_step = 0

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train the model for one epoch.

        Assumes that the dataloader outputs are already on the correct device.

        Returns:
            Average loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0

        for batch in dataloader:
            # With Accelerate, we assume the inputs are already on the correct device.
            input_ids = batch["input_ids"]
            attention_mask = batch.get("attention_mask", None)

            self.optimizer.zero_grad()

            # Forward pass.
            logits, masked_input_ids, mask_indices = self.model(
                input_ids=input_ids, attention_mask=attention_mask
            )

            loss = variational_lower_bound_loss(
                logits=logits,
                targets=input_ids,
                masked_input_ids=masked_input_ids,
                attention_mask=attention_mask,
                mask_token_id=self.model.tokenizer.mask_token_id,
            )

            # Backward pass using Accelerate.
            self.accelerator.backward(loss)
            self.optimizer.step()

            total_loss += loss.item()
            self.global_step += 1
            print(f"Step {self.global_step} Loss: {loss.item():.4f}")

        epoch_loss = total_loss / len(dataloader)
        self.history.append(epoch_loss)

        if self.scheduler is not None:
            self.scheduler.step()

        current_lr = self.optimizer.param_groups[0]["lr"]
        print(f"Epoch {len(self.history)} Loss: {epoch_loss:.4f}, LR: {current_lr:.8f}")
        return epoch_loss

    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader = None,
        num_epochs: int = 1,
        eval_steps: int = 100,
        eval_mask_ratio: float = 0.15,
        num_examples: int = 3,
        sample_steps: int = 10,  # Number of diffusion steps for sampling
    ) -> list[float]:
        """Train the model for multiple epochs with periodic evaluation.

        Args:
            train_dataloader: DataLoader for training data.
            val_dataloader: DataLoader for validation data.
            num_epochs: Number of epochs to train.
            eval_steps: Frequency of evaluation in steps.
            eval_mask_ratio: Fixed mask ratio for evaluation.
            num_examples: Number of examples to generate during evaluation.
            sample_steps: Number of diffusion steps for sampling.

        Returns:
            A list of average losses for each epoch.
        """
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0.0

            for batch_idx, batch in enumerate(train_dataloader):
                input_ids = batch["input_ids"]
                attention_mask = batch.get("attention_mask", None)

                self.optimizer.zero_grad()
                logits, masked_input_ids, mask_indices = self.model(
                    input_ids=input_ids, attention_mask=attention_mask
                )

                loss = variational_lower_bound_loss(
                    logits=logits,
                    targets=input_ids,
                    masked_input_ids=masked_input_ids,
                    attention_mask=attention_mask,
                    mask_token_id=self.model.tokenizer.mask_token_id,
                )

                self.accelerator.backward(loss)
                self.optimizer.step()

                total_loss += loss.item()
                self.global_step += 1

                if batch_idx % 10 == 0:
                    print(
                        f"Epoch {epoch + 1}/{num_epochs} | Batch {batch_idx}/{len(train_dataloader)} | "
                        f"Loss: {loss.item():.4f}"
                    )

                # Periodic evaluation.
                if val_dataloader is not None and self.global_step % eval_steps == 0:
                    eval_results = self.evaluate(
                        val_dataloader,
                        num_examples=num_examples,
                        mask_ratio=eval_mask_ratio,
                        sample_steps=sample_steps,
                    )
                    print(f"\nEvaluation at step {self.global_step}:")
                    print(f"Validation Loss: {eval_results['loss']:.4f}")
                    print(f"Mask Prediction Accuracy: {eval_results['accuracy']:.4f}")

                    print("\nGeneration Examples:")
                    for i, example in enumerate(eval_results["examples"]):
                        print(f"\nExample {i + 1}:")
                        print(f"Original: {example['original']}")
                        print(f"Masked  : {example['masked']}")
                        print(f"Generated: {example['generated']}")
                        print(f"Predicted: {example['predicted']}")
                    print("\n")
                    self.model.train()

            epoch_loss = total_loss / len(train_dataloader)
            self.history.append(epoch_loss)

            if self.scheduler is not None:
                self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch + 1}/{num_epochs} | Loss: {epoch_loss:.4f} | LR: {current_lr:.8f}"
            )

            if val_dataloader is not None:
                eval_results = self.evaluate(
                    val_dataloader,
                    num_examples=num_examples,
                    mask_ratio=eval_mask_ratio,
                    sample_steps=sample_steps,
                )
                print(f"Epoch {epoch + 1} Validation Loss: {eval_results['loss']:.4f}")
                print(
                    f"Epoch {epoch + 1} Mask Prediction Accuracy: {eval_results['accuracy']:.4f}"
                )

        return self.history

    def evaluate(
        self,
        dataloader: DataLoader,
        num_examples: int = 3,
        mask_ratio: float = 0.15,
        sample_steps: int = 10,
        temperature: float = 0.8,
    ) -> dict:
        """Evaluate the model on validation data.

        Assumes that the dataloader outputs are already on the correct device.

        Returns:
            A dictionary with evaluation metrics and example generations.
        """
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_masked = 0
        example_inputs = []
        example_masked = []
        example_preds = []
        example_generated = []
        example_targets = []

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                input_ids = batch["input_ids"]
                attention_mask = batch.get("attention_mask", None)

                # Create fixed mask ratio tensor on the same device as input_ids.
                t = torch.ones(1, device=input_ids.device) * mask_ratio

                logits, masked_input_ids, mask_indices = self.model.forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    t=t,  # Use fixed t for evaluation.
                )

                loss = variational_lower_bound_loss(
                    logits=logits,
                    targets=input_ids,
                    masked_input_ids=masked_input_ids,
                    attention_mask=attention_mask,
                    mask_token_id=self.model.tokenizer.mask_token_id,
                )

                total_loss += loss.item()

                predictions = logits.argmax(dim=-1)
                mask_positions = masked_input_ids == self.model.tokenizer.mask_token_id
                if attention_mask is not None:
                    mask_positions = mask_positions & (attention_mask == 1)

                correct = (
                    (predictions[mask_positions] == input_ids[mask_positions])
                    .sum()
                    .item()
                )
                total_correct += correct
                total_masked += mask_positions.sum().item()

                if i < num_examples and i < len(dataloader):
                    example_idx = 0  # use the first sequence in batch.
                    example_inputs.append(input_ids[example_idx].cpu().tolist())
                    example_masked.append(masked_input_ids[example_idx].cpu().tolist())
                    example_preds.append(predictions[example_idx].cpu().tolist())
                    example_targets.append(input_ids[example_idx].cpu().tolist())

                    prompt_length = min(5, input_ids.size(1))
                    prompt = input_ids[example_idx : example_idx + 1, :prompt_length]
                    generated = self.model.sample(
                        prompt=prompt,
                        num_steps=sample_steps,
                        seq_length=input_ids.size(1),
                        temperature=temperature,
                    )
                    example_generated.append(generated[0].cpu().tolist())

        avg_loss = total_loss / len(dataloader)
        accuracy = total_correct / total_masked if total_masked > 0 else 0

        examples = []
        for i in range(min(num_examples, len(example_inputs))):
            input_seq = example_inputs[i]
            masked_seq = example_masked[i]
            pred_seq = example_preds[i]
            generated_seq = example_generated[i] if i < len(example_generated) else []

            original_text = self.model.tokenizer.decode(
                input_seq, skip_special_tokens=True
            )
            masked_text = self.model.tokenizer.decode(
                masked_seq, skip_special_tokens=False
            )
            generated_text = (
                self.model.tokenizer.decode(generated_seq, skip_special_tokens=True)
                if generated_seq
                else "No generation available"
            )

            combined_seq = masked_seq.copy()
            mask_token_id = self.model.tokenizer.mask_token_id
            for j, token_id in enumerate(masked_seq):
                if token_id == mask_token_id:
                    combined_seq[j] = pred_seq[j]

            predicted_text = self.model.tokenizer.decode(
                combined_seq, skip_special_tokens=True
            )

            examples.append(
                {
                    "original": original_text,
                    "masked": masked_text,
                    "predicted": predicted_text,
                    "generated": generated_text,
                }
            )

        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "examples": examples,
        }
