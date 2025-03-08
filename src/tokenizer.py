"""Tokenizer for the diffusion language model."""

from dataclasses import dataclass
from typing import Any, Literal, TypedDict

import torch
from torch import Tensor
from transformers import BertTokenizer


class TokenizerOutput(TypedDict):
    """Output structure for the tokenizer."""

    input_ids: Tensor  # Token IDs, shape: [batch_size, seq_len]
    attention_mask: Tensor  # Attention mask, shape: [batch_size, seq_len]


@dataclass
class TokenizerConfig:
    """Configuration for the diffusion tokenizer."""

    tokenizer_name: str = "bert-base-uncased"
    mask_token: str = "[MASK]"
    pad_token: str = "[PAD]"
    bos_token: str = "[CLS]"
    eos_token: str = "[SEP]"


class DiffusionTokenizer:
    """Wrapper for tokenizer used in diffusion language model."""

    def __init__(
        self,
        config: TokenizerConfig | None = None,
    ):
        """Initialize the tokenizer.

        Args:
            config: Configuration for the tokenizer. If None, default config is used.
        """
        if config is None:
            config = TokenizerConfig()

        self.config = config
        self.tokenizer = BertTokenizer.from_pretrained(config.tokenizer_name)

        # Get token IDs
        self.mask_token_id = self.tokenizer.convert_tokens_to_ids(config.mask_token)
        self.pad_token_id = self.tokenizer.convert_tokens_to_ids(config.pad_token)
        self.bos_token_id = self.tokenizer.convert_tokens_to_ids(config.bos_token)
        self.eos_token_id = self.tokenizer.convert_tokens_to_ids(config.eos_token)

        # Set vocabulary size
        self.vocab_size = len(self.tokenizer)

    def encode(
        self,
        text: str,
        padding: bool | Literal["max_length"] = True,
        truncation: bool = True,
        max_length: int = 512,
        return_batch_dimension: bool = True,
    ) -> TokenizerOutput:
        """Encode text into input_ids and attention_mask.

        Args:
            text: The text to encode
            padding: Whether to pad sequences or padding strategy
            truncation: Whether to truncate sequences
            max_length: Maximum sequence length
            return_batch_dimension: If True, returns tensors with shape [1, seq_len]
                                   If False, returns tensors with shape [seq_len]

        Returns:
            Dictionary containing input_ids and attention_mask.
        """
        padding_strategy = (
            "max_length" if padding == "max_length" else "longest" if padding else False
        )

        encoding = self.tokenizer(
            text,
            padding=padding_strategy,
            truncation=truncation,
            max_length=max_length,
            return_tensors="pt" if return_batch_dimension else None,
        )

        # Convert to tensors if return_batch_dimension is False
        if not return_batch_dimension:
            encoding = {k: torch.tensor(v) for k, v in encoding.items()}

        # Ensure we have the expected keys
        result: TokenizerOutput = {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
        }

        return result

    def batch_encode(
        self,
        texts: list[str],
        padding: bool | Literal["max_length"] = True,
        truncation: bool = True,
        max_length: int = 512,
    ) -> TokenizerOutput:
        """Encode a batch of texts.

        Args:
            texts: List of texts to encode
            padding: Whether to pad sequences or padding strategy
            truncation: Whether to truncate sequences
            max_length: Maximum sequence length

        Returns:
            Dictionary containing batched input_ids and attention_mask
            with shape [batch_size, seq_len]
        """
        padding_strategy = (
            "max_length" if padding == "max_length" else "longest" if padding else False
        )

        encoding = self.tokenizer(
            texts,
            padding=padding_strategy,
            truncation=truncation,
            max_length=max_length,
            return_tensors="pt",
        )

        # Ensure we have the expected keys
        result: TokenizerOutput = {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
        }

        return result

    def decode(self, token_ids: Tensor, skip_special_tokens: bool = True) -> Any:
        """Decode token ids back to text.

        Args:
            token_ids: Tensor to decode
            skip_special_tokens: Whether to skip special tokens

        Returns:
            Decoded text
        """
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
