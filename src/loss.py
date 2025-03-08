"""Loss functions for diffusion language models."""

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor


def masked_cross_entropy_loss(
    logits: Tensor,
    targets: Tensor,
    mask_indices: Tensor,
    attention_mask: Tensor | None = None,
    pad_token_id: int | None = None,
    special_token_ids: list[int] | None = None,
    mask_ratio: float = 1.0,
) -> Tensor:
    """Calculate cross entropy loss only on valid masked tokens.

    Args:
        logits: Predicted logits of shape [batch_size, seq_len, vocab_size]
        targets: Target token ids of shape [batch_size, seq_len]
        mask_indices: Boolean tensor indicating which tokens were masked
        attention_mask: Attention mask indicating valid (non-padding) positions
        pad_token_id: ID of the padding token to exclude
        special_token_ids: List of special token IDs to exclude from loss calculation
        mask_ratio: Ratio of tokens that were masked (t value in the paper)

    Returns:
        Scaled masked cross entropy loss
    """
    batch_size, seq_len, vocab_size = logits.shape

    # Create a valid token mask (excluding padding and possibly special tokens)
    valid_tokens_mask = torch.ones_like(targets, dtype=torch.bool)

    # Exclude padding tokens if attention mask is provided
    if attention_mask is not None:
        valid_tokens_mask = valid_tokens_mask & (attention_mask == 1)

    # Alternatively, exclude padding tokens using pad_token_id
    if pad_token_id is not None:
        valid_tokens_mask = valid_tokens_mask & (targets != pad_token_id)

    # Exclude special tokens if specified
    if special_token_ids is not None:
        for token_id in special_token_ids:
            valid_tokens_mask = valid_tokens_mask & (targets != token_id)

    # Only calculate loss on tokens that are both valid and were masked
    eligible_mask = valid_tokens_mask & mask_indices

    # Count eligible tokens to normalize the loss correctly
    num_eligible_tokens = eligible_mask.sum().item()
    if num_eligible_tokens == 0:
        # Return zero loss if no eligible tokens (avoid division by zero)
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    # Select eligible logits and targets
    flat_logits = logits.reshape(-1, vocab_size)
    flat_targets = targets.reshape(-1)
    flat_eligible_mask = eligible_mask.reshape(-1)

    eligible_logits = flat_logits[flat_eligible_mask]
    eligible_targets = flat_targets[flat_eligible_mask]

    # Calculate cross entropy loss on eligible tokens
    loss = F.cross_entropy(eligible_logits, eligible_targets, reduction="mean")

    # Scale by 1/t as per the LLaDA paper formulation
    if mask_ratio > 0:
        # Adjust by the proportion of tokens that were eligible vs. all masked tokens
        # This ensures we properly normalize the loss regardless of filtering
        total_masked = mask_indices.sum().item()
        if total_masked > 0:
            eligibility_ratio = num_eligible_tokens / total_masked
            loss = loss / (mask_ratio * eligibility_ratio)
    elif mask_ratio == 0:
        # Handle the zero mask_ratio case by setting a large but finite value
        loss = loss * 1e6  # Large but finite value

    # Ensure the loss maintains gradient tracking
    loss = loss.clone().requires_grad_()

    return loss


def variational_lower_bound_loss(
    logits: Tensor,
    targets: Tensor,
    masked_input_ids: Tensor,
    mask_token_id: int,
    attention_mask: Tensor | None = None,
    pad_token_id: int | None = None,
    special_token_ids: list[int] | None = None,
    mask_ratio: float = 1.0,
) -> Tensor:
    """Calculate the variational lower bound loss as defined in the LLaDA paper.

    This implements the loss function from the LLaDA paper:
    L(θ) = -E_t,x_0,x_t[ (1/t) * sum_i=1^L 1[x_t^i = M] log p_θ(x_0^i|x_t) ]

    Args:
        logits: Predicted logits from model [batch_size, seq_len, vocab_size]
        targets: Original clean token ids [batch_size, seq_len]
        masked_input_ids: Input token ids with masks applied [batch_size, seq_len]
        mask_token_id: ID of the mask token
        attention_mask: Attention mask for valid positions [batch_size, seq_len]
        pad_token_id: ID of the padding token to exclude
        special_token_ids: List of token IDs to exclude from loss calculation
        mask_ratio: Ratio of tokens that were masked (t value in paper)

    Returns:
        Variational lower bound loss (upper bound on negative log-likelihood)
    """
    # Create mask indices tensor - only tokens that were masked
    mask_indices = masked_input_ids == mask_token_id

    # Calculate the masked cross entropy loss with all token filtering
    loss = masked_cross_entropy_loss(
        logits=logits,
        targets=targets,
        mask_indices=mask_indices,
        attention_mask=attention_mask,
        pad_token_id=pad_token_id,
        special_token_ids=special_token_ids,
        mask_ratio=mask_ratio,
    )

    return loss
