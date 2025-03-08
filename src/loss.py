"""Loss functions for diffusion language models with diagnostics."""

import logging

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
    debug_mode: bool = False,
    step: int = 0,
    log_interval: int = 100,
    logger: logging.Logger | None = None,
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
        debug_mode: Whether to enable diagnostic logging
        step: Current step number (for logging)
        log_interval: Interval between logging
        logger: Logger instance for debug output

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

    # Diagnostics for understanding eligible tokens
    if debug_mode and step % log_interval == 0 and logger is not None:
        total_tokens = targets.numel()
        total_masked = mask_indices.sum().item()
        total_valid = valid_tokens_mask.sum().item()

        logger.info("  - Loss calculation diagnostics:")
        logger.info(f"    - Total tokens: {total_tokens}")
        logger.info(
            f"    - Masked tokens: {total_masked} ({total_masked / total_tokens:.2%})"
        )
        logger.info(
            f"    - Valid tokens: {total_valid} ({total_valid / total_tokens:.2%})"
        )
        logger.info(
            f"    - Eligible tokens: {num_eligible_tokens} ({num_eligible_tokens / total_tokens:.2%} of all, {num_eligible_tokens / max(1, total_masked):.2%} of masked)"
        )

    if num_eligible_tokens == 0:
        # Return zero loss if no eligible tokens (avoid division by zero)
        if debug_mode and logger is not None:
            logger.warning(
                "    - No eligible tokens found for loss calculation! Returning zero loss."
            )
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    # Select eligible logits and targets
    flat_logits = logits.reshape(-1, vocab_size)
    flat_targets = targets.reshape(-1)
    flat_eligible_mask = eligible_mask.reshape(-1)

    eligible_logits = flat_logits[flat_eligible_mask]
    eligible_targets = flat_targets[flat_eligible_mask]

    # Calculate raw cross entropy loss on eligible tokens
    raw_loss = F.cross_entropy(eligible_logits, eligible_targets, reduction="mean")

    # Store the raw loss before scaling for diagnostics
    loss = raw_loss

    # Calculate scaling factor
    scaling_factor = 1.0
    if mask_ratio > 0:
        # Adjust by the proportion of tokens that were eligible vs. all masked tokens
        # This ensures we properly normalize the loss regardless of filtering
        total_masked = mask_indices.sum().item()
        if total_masked > 0:
            eligibility_ratio = num_eligible_tokens / total_masked
            scaling_factor = mask_ratio * eligibility_ratio
            loss = loss / scaling_factor
    elif mask_ratio == 0:
        # Handle the zero mask_ratio case by setting a large but finite value
        scaling_factor = 1e-6
        loss = loss / scaling_factor

    # Diagnostics for loss scaling
    if debug_mode and step % log_interval == 0 and logger is not None:
        logger.info(f"    - Raw loss (before scaling): {raw_loss.item():.6f}")
        logger.info(f"    - Mask ratio: {mask_ratio:.6f}")
        logger.info(f"    - Scaling factor: {scaling_factor:.6f}")
        logger.info(f"    - Final scaled loss: {loss.item():.6f}")

        # Add breakdown of predictions vs targets for eligible tokens
        if len(eligible_targets) > 0:
            with torch.no_grad():
                predictions = eligible_logits.argmax(dim=-1)
                correct = (predictions == eligible_targets).sum().item()
                accuracy = correct / len(eligible_targets)
                logger.info(
                    f"    - Token prediction accuracy: {accuracy:.2%} ({correct}/{len(eligible_targets)})"
                )

                # Examine loss values for some samples
                sample_size = min(5, len(eligible_targets))
                logger.info(
                    f"    - Sample-level cross entropy (first {sample_size} tokens):"
                )
                for i in range(sample_size):
                    token_logits = eligible_logits[i].unsqueeze(0)
                    token_target = eligible_targets[i].unsqueeze(0)
                    token_loss = F.cross_entropy(
                        token_logits, token_target, reduction="mean"
                    )
                    logger.info(f"      - Token {i}: loss={token_loss.item():.4f}")

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
    debug_mode: bool = False,
    step: int = 0,
    log_interval: int = 100,
    logger: logging.Logger | None = None,
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
        debug_mode: Whether to enable diagnostic logging
        step: Current step number (for logging)
        log_interval: Interval between logging
        logger: Logger instance for debug output

    Returns:
        Variational lower bound loss (upper bound on negative log-likelihood)
    """
    # Create mask indices tensor - only tokens that were masked
    mask_indices = masked_input_ids == mask_token_id

    # Diagnostic logging for masked tokens
    if debug_mode and step % log_interval == 0 and logger is not None:
        # Check if we have any masked tokens
        num_masked = mask_indices.sum().item()
        if num_masked == 0:
            logger.warning(
                f"Step {step}: No masked tokens found! Check mask generation."
            )

        # Log some example masked token predictions
        batch_size, seq_len = targets.shape
        if batch_size > 0 and num_masked > 0:
            logger.info(f"  - Variational Lower Bound Loss - step {step}")
            logger.info(f"    - Input shape: [{batch_size}, {seq_len}]")
            logger.info(f"    - Number of masked tokens: {num_masked}")

            # Find first few masked positions
            batch_idx = 0
            masked_pos = mask_indices[batch_idx].nonzero().squeeze()
            if masked_pos.dim() > 0 and len(masked_pos) > 0:
                pos = (
                    masked_pos[0].item() if masked_pos.dim() > 0 else masked_pos.item()
                )
                logger.info(f"    - First masked position in batch 0: {pos}")

    # Calculate the masked cross entropy loss with all token filtering
    loss = masked_cross_entropy_loss(
        logits=logits,
        targets=targets,
        mask_indices=mask_indices,
        attention_mask=attention_mask,
        pad_token_id=pad_token_id,
        special_token_ids=special_token_ids,
        mask_ratio=mask_ratio,
        debug_mode=debug_mode,
        step=step,
        log_interval=log_interval,
        logger=logger,
    )

    return loss
