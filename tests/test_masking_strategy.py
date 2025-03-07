import torch

from src.masking_strategy import RandomMaskingStrategy, RandomRemaskingStrategy


class TestMaskingStrategy:
    def test_random_masking(self):
        # Test the forward masking process
        strategy = RandomMaskingStrategy(
            bos_token_id=101, eos_token_id=102, pad_token_id=0, mask_token_id=103
        )

        # Create a sample input
        input_ids = torch.tensor(
            [
                [101, 450, 231, 102, 0],  # Normal sequence with padding
                [101, 450, 231, 450, 102],  # Sequence without padding
            ]
        )

        # Test with 100% mask probability to make the test deterministic
        masked_input, mask_indices = strategy.apply_random_mask(
            input_ids, mask_prob=1.0
        )

        # Check that special tokens are not masked
        assert torch.all(masked_input[:, 0] == 101)  # BOS not masked
        assert torch.all(masked_input[0, 3] == 102)  # EOS not masked
        assert torch.all(masked_input[1, 4] == 102)  # EOS not masked
        assert torch.all(masked_input[0, 4] == 0)  # PAD not masked

        # Check that all eligible tokens are masked
        assert torch.all(masked_input[0, 1:3] == 103)  # All normal tokens masked
        assert torch.all(masked_input[1, 1:4] == 103)  # All normal tokens masked

        # Check mask_indices
        expected_mask = torch.tensor(
            [[False, True, True, False, False], [False, True, True, True, False]]
        )
        assert torch.all(mask_indices == expected_mask)

        # Test with 0% mask probability
        masked_input, mask_indices = strategy.apply_random_mask(
            input_ids, mask_prob=0.0
        )

        # Check that no tokens are masked
        assert torch.all(masked_input == input_ids)
        assert not torch.any(mask_indices)


class TestRemaskingStrategy:
    def test_apply_remask_basic(self):
        """Test basic remasking functionality with a simple example."""
        strategy = RandomRemaskingStrategy(
            bos_token_id=101, eos_token_id=102, pad_token_id=0, mask_token_id=103
        )

        # Create a sample input where some tokens have already been predicted
        input_ids = torch.tensor(
            [
                [101, 450, 231, 102, 0],  # Normal sequence with padding
                [101, 450, 231, 450, 102],  # Sequence without padding
            ]
        )

        # Create mask indices (tokens that were masked in the previous step)
        mask_indices = torch.tensor(
            [[False, True, True, False, False], [False, True, True, True, False]]
        )

        # Create dummy logits (not used in this test)
        vocab_size = 1000
        logits = torch.zeros((2, 5, vocab_size))

        # Test with 100% remask ratio (all previously masked tokens should be remasked)
        remasked = strategy.apply_remask(
            input_ids=input_ids,
            logits=logits,
            mask_indices=mask_indices,
            remask_ratio=1.0,
            prompt_positions=None,
        )

        # Check that all previously masked tokens are remasked
        expected = torch.tensor(
            [
                [101, 103, 103, 102, 0],  # Masked tokens replaced with mask token
                [101, 103, 103, 103, 102],  # Masked tokens replaced with mask token
            ]
        )
        assert torch.all(remasked == expected)

        # Test with 0% remask ratio (no tokens should be remasked)
        remasked = strategy.apply_remask(
            input_ids=input_ids,
            logits=logits,
            mask_indices=mask_indices,
            remask_ratio=0.0,
            prompt_positions=None,
        )

        # Check that no tokens are remasked
        assert torch.all(remasked == input_ids)

    def test_apply_remask_with_prompt(self):
        """Test that prompt positions are not remasked."""
        strategy = RandomRemaskingStrategy(
            bos_token_id=101, eos_token_id=102, pad_token_id=0, mask_token_id=103
        )

        # Create a sample input
        input_ids = torch.tensor(
            [
                [101, 450, 231, 450, 102],  # All tokens are valid
            ]
        )

        # All tokens were masked in the previous step
        mask_indices = torch.tensor(
            [
                [False, True, True, True, False],  # BOS and EOS weren't masked
            ]
        )

        # Mark the first two tokens as prompt (should not be remasked)
        prompt_positions = torch.tensor(
            [
                [True, True, False, False, False],
            ]
        )

        # Create dummy logits
        vocab_size = 1000
        logits = torch.zeros((1, 5, vocab_size))

        # Test with 100% remask ratio
        remasked = strategy.apply_remask(
            input_ids=input_ids,
            logits=logits,
            mask_indices=mask_indices,
            remask_ratio=1.0,
            prompt_positions=prompt_positions,
        )

        # Check that prompt positions are not remasked
        expected = torch.tensor(
            [
                [101, 450, 103, 103, 102],  # Only non-prompt masked tokens are remasked
            ]
        )
        assert torch.all(remasked == expected)

    def test_apply_remask_partial(self):
        """Test partial remasking with controlled randomness."""
        strategy = RandomRemaskingStrategy(
            bos_token_id=101, eos_token_id=102, pad_token_id=0, mask_token_id=103
        )

        # Create a sample with many tokens to test partial remasking
        input_ids = torch.tensor([[101, 201, 202, 203, 204, 205, 206, 207, 208, 102]])

        # All tokens except BOS and EOS were masked in the previous step
        mask_indices = torch.tensor(
            [[False, True, True, True, True, True, True, True, True, False]]
        )

        # Create dummy logits
        vocab_size = 1000
        logits = torch.zeros((1, 10, vocab_size))

        # Use a fixed seed for deterministic testing
        torch.manual_seed(42)

        # Test with 50% remask ratio
        remasked = strategy.apply_remask(
            input_ids=input_ids,
            logits=logits,
            mask_indices=mask_indices,
            remask_ratio=0.5,
            prompt_positions=None,
        )

        # Count how many tokens were remasked
        num_remasked = torch.sum(remasked == 103).item()
        num_eligible = torch.sum(mask_indices).item()

        # Check that approximately 50% of eligible tokens were remasked
        # Due to randomness, we allow some flexibility
        assert (
            3 <= num_remasked <= 5
        ), f"Expected ~4 remasked tokens, got {num_remasked}"

        # Verify that only previously masked tokens were remasked
        for i in range(10):
            if not mask_indices[0, i]:
                assert (
                    remasked[0, i] == input_ids[0, i]
                ), f"Non-masked token at position {i} was changed"

    def test_remask_ratio_edge_cases(self):
        """Test edge cases for remask ratio."""
        strategy = RandomRemaskingStrategy(
            bos_token_id=101, eos_token_id=102, pad_token_id=0, mask_token_id=103
        )

        # Create a sample input
        input_ids = torch.tensor([[101, 450, 231, 450, 102]])

        # All tokens except BOS and EOS were masked
        mask_indices = torch.tensor([[False, True, True, True, False]])

        # Create dummy logits
        vocab_size = 1000
        logits = torch.zeros((1, 5, vocab_size))

        # Test with negative remask ratio (should be treated as 0)
        remasked = strategy.apply_remask(
            input_ids=input_ids,
            logits=logits,
            mask_indices=mask_indices,
            remask_ratio=-0.1,
            prompt_positions=None,
        )
        assert torch.all(remasked == input_ids)

        # Test with remask ratio > 1 (should be capped at 1)
        remasked = strategy.apply_remask(
            input_ids=input_ids,
            logits=logits,
            mask_indices=mask_indices,
            remask_ratio=1.5,
            prompt_positions=None,
        )
        expected = torch.tensor([[101, 103, 103, 103, 102]])
        assert torch.all(remasked == expected)

        # Test with no eligible tokens
        no_eligible = torch.tensor([[False, False, False, False, False]])
        remasked = strategy.apply_remask(
            input_ids=input_ids,
            logits=logits,
            mask_indices=no_eligible,
            remask_ratio=1.0,
            prompt_positions=None,
        )
        assert torch.all(remasked == input_ids)
