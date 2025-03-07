"""Masking strategies for diffusion language models."""

from typing import Literal, Protocol

import torch
from torch import Tensor


class MaskingStrategy(Protocol):
    """Protocol defining the interface for masking strategies during forward process.

    A masking strategy determines how tokens are masked during training.
    """

    def apply_random_mask(
        self,
        input_ids: Tensor,
        mask_prob: float,
    ) -> tuple[Tensor, Tensor]:
        """Apply random masking to input tokens.

        Args:
            input_ids: Input token IDs of shape [batch_size, seq_len]
            mask_prob: Probability of masking each token

        Returns:
            Tuple containing:
            - masked_input: Input with masks applied
            - mask_indices: Boolean tensor indicating which tokens were masked
        """
        ...


class RemaskingStrategy(Protocol):
    """Protocol defining the interface for remasking strategies during reverse process.

    A remasking strategy determines how tokens are remasked during generation.
    """

    def apply_remask(
        self,
        input_ids: Tensor,
        logits: Tensor,
        mask_indices: Tensor,
        remask_ratio: float,
        prompt_positions: Tensor = None,
    ) -> Tensor:
        """Apply remasking during the reverse diffusion process.

        Args:
            input_ids: Current token IDs of shape [batch_size, seq_len]
            logits: Model predictions of shape [batch_size, seq_len, vocab_size]
            mask_indices: Boolean tensor indicating which tokens were
                previously masked
            remask_ratio: Ratio of previously masked tokens to remask again
            prompt_positions: Optional boolean tensor indicating positions that
                are part of the prompt

        Returns:
            Updated input_ids with remasking applied
        """
        ...


class RandomMaskingStrategy:
    """Random masking strategy for the forward process in diffusion language models."""

    def __init__(
        self,
        bos_token_id: int = 101,
        eos_token_id: int = 102,
        pad_token_id: int = 0,
        mask_token_id: int = 103,
    ):
        """Initialize with token IDs for special tokens.

        Args:
            bos_token_id: Beginning of sequence token ID
            eos_token_id: End of sequence token ID
            pad_token_id: Padding token ID
            mask_token_id: Mask token ID
        """
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id

    def apply_random_mask(
        self, input_ids: Tensor, mask_prob: float
    ) -> tuple[Tensor, Tensor]:
        """Randomly mask eligible tokens with probability mask_prob.

        Eligible tokens are those that are not BOS, EOS, or PAD.

        Args:
            input_ids: Tensor of shape [batch_size, seq_len] with token IDs.
            mask_prob: Probability for each eligible token to be masked.

        Returns:
            A tuple (masked_input, mask_indices) where:
              - masked_input: A copy of input_ids with masked tokens replaced.
              - mask_indices: A boolean tensor indicating masked positions.
        """
        masked_input = input_ids.clone()
        eligible_mask = torch.ones_like(input_ids, dtype=torch.bool)
        eligible_mask &= input_ids != self.bos_token_id
        eligible_mask &= input_ids != self.eos_token_id
        eligible_mask &= input_ids != self.pad_token_id

        rand = torch.rand_like(input_ids, dtype=torch.float)
        mask_indices = (rand < mask_prob) & eligible_mask
        masked_input[mask_indices] = self.mask_token_id

        return masked_input, mask_indices


class RandomRemaskingStrategy:
    """Random remasking strategy for the reverse process."""

    def __init__(
        self,
        bos_token_id: int = 101,
        eos_token_id: int = 102,
        pad_token_id: int = 0,
        mask_token_id: int = 103,
    ):
        """Initialize with token IDs for special tokens.

        Args:
            bos_token_id: Beginning of sequence token ID
            eos_token_id: End of sequence token ID
            pad_token_id: Padding token ID
            mask_token_id: Mask token ID
        """
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id

    def apply_remask(
        self,
        input_ids: Tensor,
        logits: Tensor,
        mask_indices: Tensor,
        remask_ratio: float,
        prompt_positions: Tensor = None,
    ) -> Tensor:
        """Apply random remasking during the reverse diffusion process.

        Args:
            input_ids: Current token IDs [batch_size, seq_len]
            logits: Model predictions [batch_size, seq_len, vocab_size]
            mask_indices: Which tokens were previously masked
            remask_ratio: Ratio of previously masked tokens to remask
            prompt_positions: Optional tensor marking prompt positions

        Returns:
            Updated input_ids with remasking applied
        """
        seq = input_ids.clone()
        batch_size, seq_length = seq.shape
        device = seq.device

        # Don't remask special tokens or prompt tokens
        valid_for_remasking = torch.ones_like(seq, dtype=torch.bool)
        valid_for_remasking &= seq != self.bos_token_id
        valid_for_remasking &= seq != self.eos_token_id
        valid_for_remasking &= seq != self.pad_token_id

        # Don't remask prompt positions if provided
        if prompt_positions is not None:
            valid_for_remasking &= ~prompt_positions

        # Only consider positions that were masked in the previous step
        valid_for_remasking &= mask_indices

        # Count how many tokens are eligible for remasking
        num_eligible = valid_for_remasking.sum().item()

        if num_eligible > 0 and remask_ratio > 0:
            # Calculate how many tokens to remask
            num_to_remask = int(num_eligible * remask_ratio)

            if num_to_remask > 0:
                # Get positions eligible for remasking
                flat_indices = valid_for_remasking.nonzero(as_tuple=True)

                # Randomly select positions to remask
                perm = torch.randperm(num_eligible, device=device)
                remask_indices = perm[:num_to_remask]

                # Get the actual positions to remask
                remask_positions = tuple(idx[remask_indices] for idx in flat_indices)

                # Apply remasking
                seq[remask_positions] = self.mask_token_id

        return seq


def create_masking_strategy(
    strategy_name: Literal["random"],
    bos_token_id: int,
    eos_token_id: int,
    pad_token_id: int,
    mask_token_id: int,
) -> MaskingStrategy:
    """Create a masking strategy based on the configuration.

    Args:
        strategy_name: Type of masking strategy to create
        bos_token_id: Beginning of sentence token ID
        eos_token_id: End of sentence token ID
        pad_token_id: Padding token ID
        mask_token_id: Mask token ID

    Returns:
        A concrete masking strategy implementation

    Raises:
        ValueError: If an unsupported masking strategy is specified
    """
    if strategy_name == "random":
        return RandomMaskingStrategy(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            mask_token_id=mask_token_id,
        )
    else:
        raise ValueError(f"Unsupported masking strategy: {strategy_name}")


def create_remasking_strategy(
    strategy_name: Literal["random"],
    bos_token_id: int,
    eos_token_id: int,
    pad_token_id: int,
    mask_token_id: int,
) -> RemaskingStrategy:
    """Create a remasking strategy based on the configuration.

    Args:
        strategy_name: Type of remasking strategy to create
        bos_token_id: Beginning of sentence token ID
        eos_token_id: End of sentence token ID
        pad_token_id: Padding token ID
        mask_token_id: Mask token ID

    Returns:
        A concrete remasking strategy implementation

    Raises:
        ValueError: If an unsupported remasking strategy is specified
    """
    if strategy_name == "random":
        return RandomRemaskingStrategy(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            mask_token_id=mask_token_id,
        )
    else:
        raise ValueError(f"Unsupported remasking strategy: {strategy_name}")
