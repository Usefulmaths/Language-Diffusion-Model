"""Core diffusion language model implementation based on the LLaDA paper."""

import random

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from src.masking_strategy import MaskingStrategy, RandomMaskingStrategy
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
        masking_strategy: MaskingStrategy | None = None,
    ) -> None:
        """Initialize the diffusion language model.

        Args:
            transformer: Transformer model for token prediction
            tokenizer: Tokenizer for encoding and decoding
            masking_strategy: Strategy for masking tokens
            (defaults to RandomMaskingStrategy)
        """
        super().__init__()

        self.transformer = transformer
        self.tokenizer = tokenizer
        self.masking_strategy = masking_strategy or RandomMaskingStrategy()

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        mask_prob: float | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Forward pass through the diffusion model.

        Args:
            input_ids: Tensor of token ids of shape [batch_size, seq_len]
            attention_mask: Attention mask of shape [batch_size, seq_len]
            mask_prob: Probability of masking each token
            (if None, sample uniformly from [0, 1])

        Returns:
            Tuple containing:
            - logits: Output logits of shape [batch_size, seq_len, vocab_size]
            - masked_input_ids: Input ids with masks applied
            - mask_indices: Boolean tensor indicating which tokens were masked
        """
        # Sample mask probability uniformly if not provided
        mask_prob_value = mask_prob if mask_prob is not None else random.random()

        # Apply masking using the provided strategy
        masked_input_ids, mask_indices = self.masking_strategy.apply_random_mask(
            input_ids=input_ids,
            mask_token_id=self.tokenizer.mask_token_id,
            mask_prob=mask_prob_value,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        # Get predictions from transformer
        logits = self.transformer(
            input_ids=masked_input_ids,
            attention_mask=attention_mask,
        )

        return logits, masked_input_ids, mask_indices

    def generate(
        self,
        prompt_ids: Tensor | None = None,
        prompt_attention_mask: Tensor | None = None,
        response_length: int = 20,
        num_steps: int = 20,
        temperature: float = 1.0,
    ) -> Tensor:
        """Generate a sequence using the diffusion model.

        Implements a simplified reverse process from the LLaDA paper.

        Args:
            prompt_ids: Optional prompt token ids of shape [batch_size, prompt_len]
            prompt_attention_mask: Optional prompt attention mask
            response_length: Length of the response to generate
            num_steps: Number of diffusion steps
            temperature: Temperature for sampling (0 = greedy, >0 = sampling)

        Returns:
            Generated sequence as tensor of shape [batch_size, seq_len]
        """
        # Get device from the model's parameters
        device = next(self.parameters()).device

        # Determine batch size from prompt or default to 1
        batch_size = prompt_ids.shape[0] if prompt_ids is not None else 1

        # Handle the case with or without a prompt
        if prompt_ids is not None:
            prompt_len = prompt_ids.shape[1]
            # Create masked response tokens
            response_ids = torch.full(
                (batch_size, response_length),
                self.tokenizer.mask_token_id,
                dtype=torch.long,
                device=device,
            )

            # Combine prompt and response
            input_ids = torch.cat([prompt_ids, response_ids], dim=1)

            # Create attention mask if prompt attention mask is provided
            if prompt_attention_mask is not None:
                response_attention_mask = torch.ones_like(response_ids)
                attention_mask = torch.cat(
                    [prompt_attention_mask, response_attention_mask], dim=1
                )
            else:
                attention_mask = None
        else:
            prompt_len = 0
            # Without prompt, just generate masked tokens
            input_ids = torch.full(
                (batch_size, response_length),
                self.tokenizer.mask_token_id,
                dtype=torch.long,
                device=device,
            )
            attention_mask = None

        # Create timesteps for the reverse process (from t=1 to t=0)
        timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)[:-1]

        # Start with the initial sequence
        sequence = input_ids.clone()

        # Reverse diffusion process
        for i in range(num_steps):
            # Current and next timestep
            t = timesteps[i].item()
            s = max(timesteps[i + 1].item() if i + 1 < num_steps else 0.0, 0.0)

            # Find currently masked tokens
            mask_indices = sequence == self.tokenizer.mask_token_id

            # If nothing is masked, we're done
            if not mask_indices.any():
                break

            # Get predictions for masked tokens
            with torch.no_grad():
                logits = self.transformer(
                    input_ids=sequence,
                    attention_mask=attention_mask,
                )

            # Sample from logits based on temperature
            if temperature > 0:
                # Apply temperature to logits and convert to probabilities
                probs = F.softmax(logits / temperature, dim=-1)
                # Sample from the distribution
                predictions = torch.multinomial(
                    probs.reshape(-1, probs.size(-1)), 1
                ).reshape(batch_size, -1)
            else:
                # Greedy decoding (argmax)
                predictions = torch.argmax(logits, dim=-1)

            # Create new sequence with predictions
            new_sequence = sequence.clone()
            new_sequence[mask_indices] = predictions[mask_indices]

            # Calculate how many tokens to remask based on the timestep ratio
            remask_ratio = s / t

            # Only consider response tokens for remasking (not prompt tokens)
            response_range = slice(prompt_len, sequence.size(1))
            response_mask = torch.zeros_like(sequence, dtype=torch.bool)
            response_mask[:, response_range] = True

            # Calculate number of tokens to remask
            num_tokens_to_remask = int(response_mask.sum().item() * remask_ratio)

            if num_tokens_to_remask > 0:
                # Create a flat tensor of response token positions
                flat_indices = torch.nonzero(response_mask.reshape(-1), as_tuple=True)[
                    0
                ]

                # Randomly select tokens to remask
                if len(flat_indices) > 0:
                    # Create random permutation for selecting indices
                    perm = torch.randperm(len(flat_indices), device=device)
                    remask_indices = flat_indices[perm[:num_tokens_to_remask]]

                    # Convert flat indices back to 2D indices
                    batch_indices = remask_indices // sequence.size(1)
                    token_indices = remask_indices % sequence.size(1)

                    # Remask selected tokens
                    new_sequence[
                        batch_indices, token_indices
                    ] = self.tokenizer.mask_token_id

            # Update sequence for next iteration
            sequence = new_sequence

        return sequence
