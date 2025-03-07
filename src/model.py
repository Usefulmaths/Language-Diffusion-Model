"""Core diffusion language model implementation based on the LLaDA paper."""

import random

import torch
import torch.nn as nn
from torch import Tensor

from src.masking_strategy import MaskingStrategy, RemaskingStrategy
from src.tokenizer import DiffusionTokenizer
from src.transformer import DiffusionTransformer


class DiffusionLanguageModel(nn.Module):
    """Diffusion language model using transformer architecture.

    As described in the LLaDA paper, this model implements:
    1. A forward process that gradually masks tokens
    2. A reverse process that predicts masked tokens
    """

    def __init__(
        self,
        transformer: DiffusionTransformer,
        tokenizer: DiffusionTokenizer,
        masking_strategy: MaskingStrategy,
        remasking_strategy: RemaskingStrategy,
    ) -> None:
        """Initialize the diffusion language model.

        Args:
            transformer: Transformer model for token prediction.
            tokenizer: Tokenizer for encoding and decoding.
            masking_strategy: Strategy for masking tokens.
            remasking_strategy: Strategy for remasking tokens.
        """
        super().__init__()
        self.transformer = transformer
        self.tokenizer = tokenizer
        self.masking_strategy = masking_strategy
        self.remasking_strategy = remasking_strategy

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        t: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Forward pass through the diffusion model.

        Args:
            input_ids: Tensor of token ids [batch_size, seq_len].
            attention_mask: Optional attention mask [batch_size, seq_len].
            t: Optional fixed mask probability (for evaluation).

        Returns:
            A tuple (logits, masked_input_ids, mask_indices).
        """
        if t is None:
            mask_prob = random.random()
        else:
            mask_prob = (
                t.item() if (isinstance(t, torch.Tensor) and t.numel() == 1) else t
            )

        masked_input_ids, mask_indices = self.masking_strategy.apply_random_mask(
            input_ids=input_ids, mask_prob=mask_prob
        )
        logits = self.transformer(
            input_ids=masked_input_ids, attention_mask=attention_mask
        )
        return logits, masked_input_ids, mask_indices

    def generate(self, input_ids: Tensor, temperature: float = 1.0) -> Tensor:
        """Fill in masked tokens in the input sequence.

        Args:
            input_ids: Tensor of shape [batch_size, seq_len] containing token IDs.
                Assumes input_ids is already on the correct device.
            temperature: Sampling temperature. If 0, uses greedy decoding.

        Returns:
            A tensor of shape [batch_size, seq_len] with mask tokens replaced by predictions.
        """
        # Assume input_ids is already on the desired device.
        attention_mask = torch.ones_like(input_ids)  # device inferred from input_ids

        with torch.no_grad():
            logits = self.transformer(
                input_ids=input_ids, attention_mask=attention_mask
            )

        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            predictions = torch.multinomial(
                probs.reshape(-1, probs.shape[-1]), num_samples=1
            ).reshape(input_ids.shape)
        else:
            predictions = torch.argmax(logits, dim=-1)

        mask_token_id = self.tokenizer.mask_token_id
        mask_positions = input_ids == mask_token_id

        output = input_ids.clone()
        output[mask_positions] = predictions[mask_positions]
        return output

    def sample(
        self,
        prompt: Tensor = None,
        num_steps: int = 20,
        seq_length: int = 32,
        temperature: float = 1.0,
    ) -> Tensor:
        """Generate text using the reverse diffusion process.

        Args:
            prompt: Optional prompt tokens to condition generation on [batch_size, prompt_length].
                Assumes prompt is already on the correct device.
            num_steps: Number of diffusion steps to use.
            seq_length: Length of the sequence to generate.
            temperature: Sampling temperature (0 for greedy decoding).

        Returns:
            Generated sequence of token IDs [batch_size, seq_length].
        """
        if prompt is not None:
            seq = prompt  # Already on the correct device.
            batch_size = seq.size(0)
            if seq.size(1) < seq_length:
                padding = torch.full(
                    (batch_size, seq_length - seq.size(1)),
                    self.tokenizer.mask_token_id,
                    dtype=torch.long,
                    device=seq.device,
                )
                seq = torch.cat([seq, padding], dim=1)
            else:
                seq = seq[:, :seq_length]
        else:
            # In a fully Accelerate-prepared workflow, the model and any created tensors
            # are expected to be on the correct device.
            # Here, we simply create the tensor without manually setting the device.
            batch_size = 1
            seq = torch.full(
                (batch_size, seq_length),
                self.tokenizer.mask_token_id,
                dtype=torch.long,
            )
            seq[:, 0] = self.tokenizer.bos_token_id

        attention_mask = torch.ones_like(seq)
        prompt_positions = torch.zeros_like(seq, dtype=torch.bool)
        if prompt is not None:
            prompt_length = min(prompt.size(1), seq_length)
            prompt_positions[:, :prompt_length] = True

        for step in range(num_steps):
            t = 1.0 - (step / (num_steps - 1))
            next_t = (
                1.0 - ((step + 1) / (num_steps - 1)) if step < num_steps - 1 else 0.0
            )

            with torch.no_grad():
                logits = self.transformer(input_ids=seq, attention_mask=attention_mask)

            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                predictions = torch.multinomial(
                    probs.reshape(-1, probs.shape[-1]), num_samples=1
                ).reshape(seq.shape)
            else:
                predictions = torch.argmax(logits, dim=-1)

            mask_positions = seq == self.tokenizer.mask_token_id
            seq = seq.clone()
            seq[mask_positions] = predictions[mask_positions]

            if step < num_steps - 1:
                remask_ratio = next_t / t
                seq = self.remasking_strategy.apply_remask(
                    input_ids=seq,
                    logits=logits,
                    mask_indices=mask_positions,
                    remask_ratio=remask_ratio,
                    prompt_positions=prompt_positions,
                )

        return seq
