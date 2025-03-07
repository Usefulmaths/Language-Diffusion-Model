"""This module contains functions to compute the loss for the masked language model."""

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor


def masked_cross_entropy_loss(
    logits: Tensor, targets: Tensor, mask_indices: Tensor
) -> Tensor:
    """Compute the average cross-entropy loss over masked tokens."""
    batch_size, seq_len, vocab_size = logits.shape
    # Flatten tensors to compute loss on each token.
    logits_flat = logits.view(-1, vocab_size)
    targets_flat = targets.view(-1)
    mask_flat = mask_indices.view(-1)
    # If no tokens were masked, return zero loss.
    if mask_flat.sum() == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    masked_logits = logits_flat[mask_flat]
    masked_targets = targets_flat[mask_flat]
    loss = F.cross_entropy(masked_logits, masked_targets)

    # Ensure loss has requires_grad=True
    if not loss.requires_grad:
        loss = loss.clone().detach().requires_grad_(True)

    return loss


def variational_lower_bound_loss(
    logits: Tensor,
    targets: Tensor,
    masked_input_ids: Tensor,
    mask_token_id: int,
    attention_mask: Tensor | None = None,
) -> Tensor:
    """Compute the variational lower bound loss as described in the paper.

    This loss is calculated as the average cross-entropy loss over the positions where
    the input was masked (i.e. replaced by mask_token_id). If an attention mask is
    provided, only valid (non-padding) positions are considered.

    Args:
        logits (Tensor): Predicted logits with shape [batch_size, seq_len, vocab_size].
        targets (Tensor): Original token IDs with shape [batch_size, seq_len].
        masked_input_ids (Tensor): Input token IDs after masking, shape
        [batch_size, seq_len].
        mask_token_id (int): The ID of the mask token.
        attention_mask (Tensor, optional): Attention mask indicating valid tokens
        (1 for valid, 0 for padding),
                                           with shape [batch_size, seq_len].
                                           Defaults to None.

    Returns:
        Tensor: A scalar tensor representing the average loss over the masked tokens.
    """
    # Determine which positions were masked.
    mask_indices = masked_input_ids == mask_token_id

    # If an attention mask is provided, only consider positions where attention_mask==1.
    if attention_mask is not None:
        mask_indices = mask_indices & (attention_mask == 1)

    return masked_cross_entropy_loss(logits, targets, mask_indices)
