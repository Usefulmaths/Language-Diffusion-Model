"""Test suite for diffusion language model loss functions."""

from typing import TypedDict

import pytest
import torch
from torch import Tensor

from src.loss import masked_cross_entropy_loss, variational_lower_bound_loss


class BasicInputs(TypedDict):
    """Basic inputs for loss function testing."""

    logits: Tensor
    targets: Tensor
    mask_indices: Tensor
    masked_input_ids: Tensor
    attention_mask: Tensor
    mask_token_id: int
    batch_size: int
    seq_len: int
    vocab_size: int


class TestLossFunctions:
    """Test suite for diffusion language model loss functions."""

    @pytest.fixture
    def basic_inputs(self) -> BasicInputs:
        """Create basic inputs for loss function testing."""
        batch_size: int = 2
        seq_len: int = 5
        # Increase vocabulary size to accommodate token IDs
        vocab_size: int = 150

        # Set a specific seed for reproducibility
        torch.manual_seed(42)

        # Sample logits, targets, and mask indices
        logits: Tensor = torch.randn(batch_size, seq_len, vocab_size)
        targets: Tensor = torch.randint(0, vocab_size - 100, (batch_size, seq_len))

        # Create mask indices with some specific pattern
        mask_indices: Tensor = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        mask_indices[:, 1] = True  # Second token in each sequence
        mask_indices[:, 3] = True  # Fourth token in each sequence

        # Create masked input ids
        mask_token_id: int = 103
        masked_input_ids: Tensor = targets.clone()
        masked_input_ids[mask_indices] = mask_token_id

        # Create an attention mask (all tokens are valid in this case)
        attention_mask: Tensor = torch.ones(batch_size, seq_len)

        return {
            "logits": logits,
            "targets": targets,
            "mask_indices": mask_indices,
            "masked_input_ids": masked_input_ids,
            "attention_mask": attention_mask,
            "mask_token_id": mask_token_id,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "vocab_size": vocab_size,
        }

    def test_masked_cross_entropy_loss_basic(self, basic_inputs: BasicInputs) -> None:
        """Test basic functionality of masked cross entropy loss."""
        loss: Tensor = masked_cross_entropy_loss(
            logits=basic_inputs["logits"],
            targets=basic_inputs["targets"],
            mask_indices=basic_inputs["mask_indices"],
        )

        # Loss should be a scalar tensor
        assert isinstance(loss, torch.Tensor)
        assert loss.shape == torch.Size([])

        # Loss should be positive for non-perfect predictions
        assert loss.item() > 0

    def test_masked_cross_entropy_loss_no_eligible_tokens(
        self, basic_inputs: BasicInputs
    ) -> None:
        """Test handling of case where no tokens are eligible for loss calculation."""
        # Empty mask indices
        empty_mask_indices: Tensor = torch.zeros_like(basic_inputs["mask_indices"])

        loss: Tensor = masked_cross_entropy_loss(
            logits=basic_inputs["logits"],
            targets=basic_inputs["targets"],
            mask_indices=empty_mask_indices,
        )

        # Should return zero loss
        assert loss.item() == 0.0

    def test_variational_lower_bound_loss_basic(
        self, basic_inputs: BasicInputs
    ) -> None:
        """Test basic functionality of variational lower bound loss."""
        loss: Tensor = variational_lower_bound_loss(
            logits=basic_inputs["logits"],
            targets=basic_inputs["targets"],
            masked_input_ids=basic_inputs["masked_input_ids"],
            mask_token_id=basic_inputs["mask_token_id"],
        )

        direct_loss: Tensor = masked_cross_entropy_loss(
            logits=basic_inputs["logits"],
            targets=basic_inputs["targets"],
            mask_indices=(
                basic_inputs["masked_input_ids"] == basic_inputs["mask_token_id"]
            ),
        )

        # The variational lower bound loss should be equivalent to
        # masked cross entropy loss with derived mask_indices
        assert torch.isclose(loss, direct_loss)

    def test_variational_lower_bound_loss_with_attention_mask(
        self, basic_inputs: BasicInputs
    ) -> None:
        """Test handling of attention mask in variational lower bound loss."""
        # Create an attention mask with the last token as padding
        attention_mask: Tensor = torch.ones_like(basic_inputs["attention_mask"])
        attention_mask[:, -1] = 0  # Mark last token as padding

        # Create masked IDs with a token masked in the padded position
        masked_input_ids = basic_inputs["targets"].clone()
        masked_input_ids[:, -1] = basic_inputs["mask_token_id"]  # Mask the padded token

        # Add the original masked positions as well
        masked_input_ids[basic_inputs["mask_indices"]] = basic_inputs["mask_token_id"]

        # Test with attention mask (should ignore the masked token in padding)
        loss_with_attn_mask: Tensor = variational_lower_bound_loss(
            logits=basic_inputs["logits"],
            targets=basic_inputs["targets"],
            masked_input_ids=masked_input_ids,
            mask_token_id=basic_inputs["mask_token_id"],
            attention_mask=attention_mask,
        )

        # Test without attention mask (should include all masked tokens)
        loss_without_attn_mask: Tensor = variational_lower_bound_loss(
            logits=basic_inputs["logits"],
            targets=basic_inputs["targets"],
            masked_input_ids=masked_input_ids,
            mask_token_id=basic_inputs["mask_token_id"],
        )

        # The losses should be different since tokens are being excluded by
        # the attention mask
        assert loss_with_attn_mask.item() != loss_without_attn_mask.item()
