"""Masking strategies for diffusion language models."""

from typing import Protocol

import torch
from torch import Tensor


class MaskingStrategy(Protocol):
    """Protocol defining the interface for masking strategies in the LLaDA model.

    A masking strategy determines how tokens are masked during the forward process
    and how they are remasked during the reverse process of diffusion.
    """

    def apply_random_mask(
        self,
        input_ids: Tensor,
        mask_token_id: int,
        mask_prob: float,
        pad_token_id: int | None = None,
        special_token_ids: list[int] | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Apply random masking to input tokens.

        Args:
            input_ids: Input token IDs of shape [batch_size, seq_len]
            mask_token_id: ID of the mask token
            mask_prob: Probability of masking each token
            pad_token_id: ID of the padding token (optional)
            special_token_ids: Additional special token IDs to exclude from masking

        Returns:
            Tuple containing:
            - masked_input: Input with masks applied
            - mask_indices: Boolean tensor indicating which tokens were masked
        """
        ...

    def remask_tokens(
        self,
        sequence: Tensor,
        predictions: Tensor,
        logits: Tensor,
        mask_indices: Tensor,
        mask_token_id: int,
        remask_ratio: float,
        prompt_len: int = 0,
        use_confidence: bool = True,
    ) -> Tensor:
        """Remask tokens during the reverse diffusion process.

        Args:
            sequence: Current sequence with masks
            predictions: Predicted tokens from the model
            logits: Raw logits from the model
            mask_indices: Boolean tensor indicating which tokens are masked
            mask_token_id: ID of the mask token
            remask_ratio: Ratio of tokens to be remasked
            prompt_len: Length of the prompt-tokens before prompt_len won't be remasked
            use_confidence: Whether to use token confidence for remasking

        Returns:
            Next sequence with remasked tokens
        """
        ...


class RandomMaskingStrategy:
    """Implements random masking and remasking for the LLaDA model."""

    def __init__(
        self,
        bos_token_id: int = 101,  # Default BERT [CLS]
        eos_token_id: int = 102,  # Default BERT [SEP]
        cls_token_id: int | None = None,
        sep_token_id: int | None = None,
        pad_token_id: int | None = 0,
        default_special_token_ids: list[int] | None = None,
    ):
        """Initialize the masking strategy.

        Args:
            bos_token_id: ID of the beginning of sequence token
            eos_token_id: ID of the end of sequence token
            cls_token_id: ID of the classification token (if different from BOS)
            sep_token_id: ID of the separator token (if different from EOS)
            pad_token_id: ID of the padding token
            default_special_token_ids: Additional special token IDs to always exclude
            from masking
        """
        # Initialize list of special tokens to exclude from masking
        self.special_token_ids = [bos_token_id, eos_token_id]

        # Add cls_token_id if provided and not already in the list
        if cls_token_id is not None and cls_token_id not in self.special_token_ids:
            self.special_token_ids.append(cls_token_id)

        # Add sep_token_id if provided and not already in the list
        if sep_token_id is not None and sep_token_id not in self.special_token_ids:
            self.special_token_ids.append(sep_token_id)

        # Store pad_token_id separately as it's often handled differently
        self.pad_token_id = pad_token_id

        # Add any additional special tokens
        if default_special_token_ids:
            for token_id in default_special_token_ids:
                if token_id not in self.special_token_ids:
                    self.special_token_ids.append(token_id)

    def apply_random_mask(
        self,
        input_ids: Tensor,
        mask_token_id: int,
        mask_prob: float,
        pad_token_id: int | None = None,
        special_token_ids: list[int] | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Apply random masking to input tokens.

        Randomly masks tokens with probability mask_prob, excluding padding and
        special tokens. Completely vectorized implementation.

        Args:
            input_ids: Input token IDs of shape [batch_size, seq_len]
            mask_token_id: ID of the mask token
            mask_prob: Probability of masking each token
            pad_token_id: ID of the padding token (override instance default)
            special_token_ids: Additional special token IDs to exclude from masking

        Returns:
            Tuple containing:
            - masked_input: Input with masks applied
            - mask_indices: Boolean tensor indicating which tokens were masked
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        masked_input = input_ids.clone()

        # Start with an empty special tokens mask
        special_tokens_mask = torch.zeros_like(input_ids, dtype=torch.bool)

        # Add class-level special tokens to mask
        for token_id in self.special_token_ids:
            special_tokens_mask = special_tokens_mask | (input_ids == token_id)

        # Add function parameter special tokens to mask
        if special_token_ids:
            for token_id in special_token_ids:
                special_tokens_mask = special_tokens_mask | (input_ids == token_id)

        # Add PAD tokens to excluded tokens - use parameter first, then instance default
        pad_token_id_to_use = (
            pad_token_id if pad_token_id is not None else self.pad_token_id
        )
        if pad_token_id_to_use is not None:
            special_tokens_mask = special_tokens_mask | (
                input_ids == pad_token_id_to_use
            )

        # Create mask for eligible tokens (non-special tokens)
        eligible_mask = ~special_tokens_mask

        # Generate random values and set non-eligible tokens to infinity
        rand = torch.rand_like(input_ids, dtype=torch.float)
        rand = rand.masked_fill(~eligible_mask, float("inf"))

        # Create sequence-relative ranking
        # This is the key to a truly vectorized approach
        values, _ = torch.sort(rand, dim=1)

        # Get threshold values for each sequence (the nth smallest value
        # where n is the number to mask)
        num_to_mask = (eligible_mask.sum(dim=1) * mask_prob).long()
        max_masks = int(num_to_mask.max().item())

        # Handle case where some sequences need fewer masks than others
        # by padding the values tensor with infinities
        if max_masks > 0:
            # Create indices for gathering thresholds
            indices = num_to_mask - 1  # -1 because 0-indexed
            indices = torch.clamp(indices, min=0)  # Handle case where num_to_mask=0

            # Ensure we have enough values for maximum masks
            if values.size(1) < max_masks:
                padding = torch.full(
                    (batch_size, max_masks - values.size(1)),
                    float("inf"),
                    device=device,
                )
                values = torch.cat([values, padding], dim=1)

            # Get threshold for each sequence
            thresholds = values.gather(1, indices.unsqueeze(1)).expand(-1, seq_len)

            # Create mask based on thresholds
            # Only mask eligible tokens with random values <= their sequence threshold
            # AND only if the sequence actually needs masks (num_to_mask > 0)
            needs_masks = (num_to_mask > 0).unsqueeze(1).expand(-1, seq_len)
            masked_indices = (rand <= thresholds) & eligible_mask & needs_masks

            # Apply masking
            masked_input[masked_indices] = mask_token_id
        else:
            # If no masks needed, return empty mask
            masked_indices = torch.zeros_like(input_ids, dtype=torch.bool)

        return masked_input, masked_indices

    def remask_tokens(
        self,
        sequence: Tensor,
        predictions: Tensor,
        logits: Tensor,
        mask_indices: Tensor,
        mask_token_id: int,
        remask_ratio: float,
        prompt_len: int = 0,
        use_confidence: bool = False,  # Default for random strategy
    ) -> Tensor:
        """Remask tokens during the reverse diffusion process.

        For the random strategy, we randomly select tokens to remask across all batches
        in a fully vectorized approach.

        Args:
            sequence: Current sequence with masks
            predictions: Predicted tokens from the model
            logits: Raw logits from the model
            mask_indices: Boolean tensor indicating which tokens are masked
            mask_token_id: ID of the mask token
            remask_ratio: Ratio of tokens to be remasked
            prompt_len: Length of the prompt-tokens before prompt_len won't be remasked
            use_confidence: Whether to use token confidence for remasking
            (unused in random strategy)

        Returns:
            Next sequence with remasked tokens
        """
        next_sequence = sequence.clone()
        device = sequence.device

        # Fill in predictions for masked tokens
        next_sequence[mask_indices] = predictions[mask_indices]

        # Create mask for valid tokens (after prompt and not special tokens)
        # 1. Identify tokens after prompt
        valid_tokens = torch.ones_like(sequence, dtype=torch.bool)
        if prompt_len > 0:
            valid_tokens[:, :prompt_len] = False

        # 2. Exclude special tokens - using the class-level special tokens
        special_tokens_tensor = torch.tensor(
            [0] + self.special_token_ids, device=device
        )
        special_tokens = torch.isin(sequence, special_tokens_tensor)
        valid_tokens = valid_tokens & (~special_tokens)

        # Count total valid tokens
        num_valid = valid_tokens.sum().item()

        # If we have valid tokens to potentially mask
        if num_valid > 0:
            # Calculate number of tokens to mask across all batches
            num_to_mask = int(num_valid * remask_ratio)

            if num_to_mask > 0:
                # Create random values for all valid tokens
                rand = torch.rand_like(sequence.float())
                # Set random values to infinity for invalid tokens so they won't
                # be selected
                rand[~valid_tokens] = float("inf")

                # Find the k smallest random values (positions to mask)
                # This returns flattened indices
                flat_indices = torch.topk(rand.view(-1), num_to_mask, largest=False)[1]

                # Convert flat indices back to 2D indices
                batch_size, seq_len = sequence.shape
                batch_indices = flat_indices // seq_len
                token_indices = flat_indices % seq_len

                # Apply masking in one operation
                next_sequence[batch_indices, token_indices] = mask_token_id

        return next_sequence
