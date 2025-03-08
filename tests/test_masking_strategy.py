"""Test suite for the RandomMaskingStrategy class."""

import pytest
import torch
from torch import Tensor

from src.masking_strategy import MaskingStrategy, RandomMaskingStrategy


class TestRandomMaskingStrategy:
    """Test suite for the RandomMaskingStrategy class."""

    @pytest.fixture
    def strategy(self) -> MaskingStrategy:
        """Create a RandomMaskingStrategy instance for testing."""
        return RandomMaskingStrategy(
            bos_token_id=101,  # BERT [CLS]
            eos_token_id=102,  # BERT [SEP]
            pad_token_id=0,
            default_special_token_ids=[103],  # BERT [MASK]
        )

    @pytest.fixture
    def sample_batch(self) -> tuple[Tensor, int, int, int]:
        """Create a sample batch of token IDs for testing."""
        batch_size: int = 2
        seq_len: int = 10

        torch.manual_seed(42)
        input_ids: Tensor = torch.randint(1, 100, (batch_size, seq_len))
        input_ids[:, 0] = 101  # BOS token
        input_ids[:, -1] = 102  # EOS token

        pad_token_id: int = 0
        mask_token_id: int = 103

        return input_ids, mask_token_id, pad_token_id, seq_len

    def test_apply_random_mask_basic(
        self,
        strategy: RandomMaskingStrategy,
        sample_batch: tuple[Tensor, int, int, int],
    ) -> None:
        """Test the apply_random_mask method with basic inputs."""
        input_ids, mask_token_id, pad_token_id, seq_len = sample_batch
        mask_prob: float = 0.15

        masked_input, mask_indices = strategy.apply_random_mask(
            input_ids, mask_token_id, mask_prob, pad_token_id
        )

        assert torch.all(masked_input[mask_indices] == mask_token_id)
        assert torch.all(masked_input[~mask_indices] == input_ids[~mask_indices])

    def test_apply_random_mask_special_tokens(
        self,
        strategy: RandomMaskingStrategy,
        sample_batch: tuple[Tensor, int, int, int],
    ) -> None:
        """Test that special tokens are not masked."""
        input_ids, mask_token_id, pad_token_id, seq_len = sample_batch
        mask_prob: float = 0.5  # High probability to ensure regular tokens get masked

        # Add special tokens in the middle
        special_token_id = 104
        input_ids[:, 4] = special_token_id

        masked_input, mask_indices = strategy.apply_random_mask(
            input_ids,
            mask_token_id,
            mask_prob,
            pad_token_id,
            special_token_ids=[special_token_id],
        )

        # Check that special tokens weren't masked
        assert not torch.any(mask_indices[:, 0])  # BOS not masked
        assert not torch.any(mask_indices[:, -1])  # EOS not masked
        assert not torch.any(mask_indices[:, 4])  # Custom special token not masked

    def test_apply_random_mask_with_padding(
        self,
        strategy: RandomMaskingStrategy,
        sample_batch: tuple[Tensor, int, int, int],
    ) -> None:
        """Test that padding tokens are not masked."""
        input_ids, mask_token_id, pad_token_id, seq_len = sample_batch
        mask_prob: float = 0.5  # High probability to ensure regular tokens get masked

        # Add padding at the end
        input_ids[:, -2:] = pad_token_id

        masked_input, mask_indices = strategy.apply_random_mask(
            input_ids, mask_token_id, mask_prob, pad_token_id
        )

        # Check that pad tokens weren't masked
        assert not torch.any(mask_indices[:, -2:])

    def test_remask_tokens_basic(
        self,
        strategy: RandomMaskingStrategy,
        sample_batch: tuple[Tensor, int, int, int],
    ) -> None:
        """Test the remask_tokens method with basic inputs."""
        input_ids, mask_token_id, pad_token_id, seq_len = sample_batch
        sequence: Tensor = input_ids.clone()
        mask_indices: Tensor = torch.zeros_like(sequence, dtype=torch.bool)
        mask_indices[:, 2:8:2] = True
        sequence[mask_indices] = mask_token_id

        predictions: Tensor = torch.randint(1, 100, sequence.shape)
        logits: Tensor = torch.randn(sequence.shape + (100,))

        remask_ratio: float = 0.5
        prompt_len: int = 1

        next_sequence: Tensor = strategy.remask_tokens(
            sequence,
            predictions,
            logits,
            mask_indices,
            mask_token_id,
            remask_ratio,
            prompt_len=prompt_len,
        )

        # Check prompt tokens aren't masked
        assert torch.all(next_sequence[:, :prompt_len] != mask_token_id)

    def test_remask_tokens_with_special_tokens(
        self,
        strategy: RandomMaskingStrategy,
        sample_batch: tuple[Tensor, int, int, int],
    ) -> None:
        """Test that special tokens are not remasked."""
        input_ids, mask_token_id, pad_token_id, seq_len = sample_batch
        sequence: Tensor = input_ids.clone()

        # Add a special token in the middle
        special_token_id = 104
        sequence[:, 5] = special_token_id

        # Create mask indices
        mask_indices: Tensor = torch.zeros_like(sequence, dtype=torch.bool)
        mask_indices[:, 2:8:2] = True
        sequence[mask_indices] = mask_token_id

        predictions: Tensor = torch.randint(1, 100, sequence.shape)
        logits: Tensor = torch.randn(sequence.shape + (100,))

        remask_ratio: float = 0.5
        prompt_len: int = (
            0  # No prompt to ensure special token protection comes from strategy
        )

        # Update strategy special tokens
        strategy_with_special = RandomMaskingStrategy(
            bos_token_id=101,
            eos_token_id=102,
            pad_token_id=0,
            default_special_token_ids=[103, special_token_id],
        )

        next_sequence: Tensor = strategy_with_special.remask_tokens(
            sequence,
            predictions,
            logits,
            mask_indices,
            mask_token_id,
            remask_ratio,
            prompt_len=prompt_len,
        )

        # Verify special token wasn't remasked
        assert torch.all(next_sequence[:, 5] == special_token_id)

    def test_remask_tokens_deterministic(
        self,
        strategy: RandomMaskingStrategy,
        sample_batch: tuple[Tensor, int, int, int],
    ) -> None:
        """Test that remasking is deterministic."""
        input_ids, mask_token_id, pad_token_id, seq_len = sample_batch
        sequence: Tensor = input_ids.clone()
        mask_indices: Tensor = torch.zeros_like(sequence, dtype=torch.bool)
        predictions: Tensor = torch.randint(1, 100, sequence.shape)
        logits: Tensor = torch.randn(sequence.shape + (100,))

        remask_ratio: float = 0.4
        prompt_len: int = 1

        torch.manual_seed(42)
        result1: Tensor = strategy.remask_tokens(
            sequence,
            predictions,
            logits,
            mask_indices,
            mask_token_id,
            remask_ratio,
            prompt_len,
        )

        torch.manual_seed(42)
        result2: Tensor = strategy.remask_tokens(
            sequence,
            predictions,
            logits,
            mask_indices,
            mask_token_id,
            remask_ratio,
            prompt_len,
        )

        assert torch.all(result1 == result2)

        torch.manual_seed(43)
        result3: Tensor = strategy.remask_tokens(
            sequence,
            predictions,
            logits,
            mask_indices,
            mask_token_id,
            remask_ratio,
            prompt_len,
        )

        assert not torch.all(result1 == result3)
