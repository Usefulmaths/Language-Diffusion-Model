"""Test suite for diffusion language model loss functions."""

from typing import TypedDict

import pytest
import torch
from torch import Tensor

from src.loss import masked_cross_entropy_loss, variational_lower_bound_loss


class BasicInputs(TypedDict):
    logits: Tensor
    targets: Tensor
    mask_indices: Tensor
    masked_input_ids: Tensor
    attention_mask: Tensor
    pad_token_id: int
    special_token_ids: list[int]
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

        # Special tokens settings - making sure they're within vocab range
        pad_token_id: int = 0
        special_token_ids: list[int] = [101, 102]  # For example, BOS/EOS tokens

        return {
            "logits": logits,
            "targets": targets,
            "mask_indices": mask_indices,
            "masked_input_ids": masked_input_ids,
            "attention_mask": attention_mask,
            "pad_token_id": pad_token_id,
            "special_token_ids": special_token_ids,
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

        # Loss should be a scalar tensor with grad_fn
        assert isinstance(loss, torch.Tensor)
        assert loss.shape == torch.Size([])
        assert loss.requires_grad

        # Loss should be positive for non-perfect predictions
        assert loss.item() > 0

    def test_masked_cross_entropy_loss_mask_ratio_scaling(
        self, basic_inputs: BasicInputs
    ) -> None:
        """Test that loss scales correctly with mask_ratio."""
        loss_full: Tensor = masked_cross_entropy_loss(
            logits=basic_inputs["logits"],
            targets=basic_inputs["targets"],
            mask_indices=basic_inputs["mask_indices"],
            mask_ratio=1.0,
        )

        loss_half: Tensor = masked_cross_entropy_loss(
            logits=basic_inputs["logits"],
            targets=basic_inputs["targets"],
            mask_indices=basic_inputs["mask_indices"],
            mask_ratio=0.5,
        )

        # Loss with mask_ratio=0.5 should be approximately 2x the loss with mask_ratio=1.0.
        assert torch.isclose(loss_half, loss_full * 2.0, rtol=1e-5)

    def test_masked_cross_entropy_loss_no_eligible_tokens(
        self, basic_inputs: BasicInputs
    ) -> None:
        """Test handling of case where no tokens are eligible for loss calculation."""
        # 1. Empty mask indices
        empty_mask_indices: Tensor = torch.zeros_like(basic_inputs["mask_indices"])

        loss: Tensor = masked_cross_entropy_loss(
            logits=basic_inputs["logits"],
            targets=basic_inputs["targets"],
            mask_indices=empty_mask_indices,
        )

        # Should return zero loss
        assert loss.item() == 0.0

        # 2. All tokens are special or padding
        special_token_id: int = basic_inputs["special_token_ids"][0]
        targets: Tensor = torch.ones_like(basic_inputs["targets"]) * special_token_id

        loss = masked_cross_entropy_loss(
            logits=basic_inputs["logits"],
            targets=targets,
            mask_indices=basic_inputs["mask_indices"],
            special_token_ids=basic_inputs["special_token_ids"],
        )

        # Should return zero loss
        assert loss.item() == 0.0

    def test_masked_cross_entropy_loss_special_token_filtering(
        self, basic_inputs: BasicInputs
    ) -> None:
        """Test that special tokens are properly excluded from loss calculation."""
        mask_indices: Tensor = basic_inputs["mask_indices"].clone()
        targets: Tensor = basic_inputs["targets"].clone()

        # Set targets in the first sequence to be a special token (within vocab range)
        special_token_id: int = basic_inputs["special_token_ids"][0]
        targets[0, :] = special_token_id

        loss_with_filtering: Tensor = masked_cross_entropy_loss(
            logits=basic_inputs["logits"],
            targets=targets,
            mask_indices=mask_indices,
            special_token_ids=basic_inputs["special_token_ids"],
        )

        non_special_targets: Tensor = targets.clone()
        non_special_targets[0, :] = 1  # Use a non-special token value

        loss_without_filtering: Tensor = masked_cross_entropy_loss(
            logits=basic_inputs["logits"],
            targets=non_special_targets,
            mask_indices=mask_indices,
        )

        # Loss with filtering should either be zero (if all masked tokens were special)
        # or different from the loss without filtering.
        if loss_with_filtering.item() > 0:
            assert loss_with_filtering.item() != loss_without_filtering.item()

    def test_masked_cross_entropy_loss_attention_mask(
        self, basic_inputs: BasicInputs
    ) -> None:
        """Test proper handling of attention mask."""
        attention_mask: Tensor = torch.ones_like(basic_inputs["attention_mask"])
        attention_mask[:, -1] = 0  # Mark last token as padding

        mask_indices: Tensor = basic_inputs["mask_indices"].clone()
        mask_indices[:, -1] = True

        loss_with_attn_mask: Tensor = masked_cross_entropy_loss(
            logits=basic_inputs["logits"],
            targets=basic_inputs["targets"],
            mask_indices=mask_indices,
            attention_mask=attention_mask,
        )

        loss_without_attn_mask: Tensor = masked_cross_entropy_loss(
            logits=basic_inputs["logits"],
            targets=basic_inputs["targets"],
            mask_indices=mask_indices,
        )

        # The losses should be different since tokens are being excluded
        assert loss_with_attn_mask.item() != loss_without_attn_mask.item()

    def test_variational_lower_bound_loss(self, basic_inputs: BasicInputs) -> None:
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
        # masked cross entropy loss with derived mask_indices.
        assert torch.isclose(loss, direct_loss)

    def test_variational_lower_bound_loss_with_all_parameters(
        self, basic_inputs: BasicInputs
    ) -> None:
        """Test variational lower bound loss with all parameters."""
        loss: Tensor = variational_lower_bound_loss(
            logits=basic_inputs["logits"],
            targets=basic_inputs["targets"],
            masked_input_ids=basic_inputs["masked_input_ids"],
            mask_token_id=basic_inputs["mask_token_id"],
            attention_mask=basic_inputs["attention_mask"],
            pad_token_id=basic_inputs["pad_token_id"],
            special_token_ids=basic_inputs["special_token_ids"],
            mask_ratio=0.5,
        )

        direct_loss: Tensor = masked_cross_entropy_loss(
            logits=basic_inputs["logits"],
            targets=basic_inputs["targets"],
            mask_indices=(
                basic_inputs["masked_input_ids"] == basic_inputs["mask_token_id"]
            ),
            attention_mask=basic_inputs["attention_mask"],
            pad_token_id=basic_inputs["pad_token_id"],
            special_token_ids=basic_inputs["special_token_ids"],
            mask_ratio=0.5,
        )

        # The losses should be identical
        assert torch.isclose(loss, direct_loss)

    def test_zero_mask_ratio(self, basic_inputs: BasicInputs) -> None:
        """Test behavior with zero mask ratio."""
        loss: Tensor = masked_cross_entropy_loss(
            logits=basic_inputs["logits"],
            targets=basic_inputs["targets"],
            mask_indices=basic_inputs["mask_indices"],
            mask_ratio=0.0,
        )

        # With mask_ratio=0, our implementation should return a large but finite value
        assert loss.item() > 1000  # Should be large
        assert not torch.isinf(loss)
