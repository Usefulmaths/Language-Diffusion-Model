"""Tokenizer for the diffusion language model."""

from typing import cast

from torch import Tensor
from transformers import BertTokenizer


class DiffusionTokenizer:
    """Wrapper for tokenizer used in diffusion language model."""

    def __init__(
        self,
        tokenizer_name: str = "bert-base-uncased",
        mask_token: str = "[MASK]",
        pad_token: str = "[PAD]",
        bos_token: str = "[CLS]",
        eos_token: str = "[SEP]",
    ):
        """Initialize the tokenizer.

        Args:
            tokenizer_name: The name of the pretrained tokenizer to use
            mask_token: The token used for masking
            pad_token: The token used for padding
            bos_token: The token used for beginning of sequence
            eos_token: The token used for end of sequence
        """
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.mask_token = mask_token
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token

        # Get token IDs
        self.mask_token_id = self.tokenizer.convert_tokens_to_ids(mask_token)
        self.pad_token_id = self.tokenizer.convert_tokens_to_ids(pad_token)
        self.bos_token_id = self.tokenizer.convert_tokens_to_ids(bos_token)
        self.eos_token_id = self.tokenizer.convert_tokens_to_ids(eos_token)

        # Set vocabulary size
        self.vocab_size = len(self.tokenizer)

    def encode(
        self,
        text: str,
        padding: bool = True,
        truncation: bool = True,
        max_length: int = 512,
    ) -> dict[str, Tensor]:
        """Encode text into input_ids and attention_mask."""
        # IMPORTANT: BertTokenizer.encode() returns a list of IDs, not a dictionary
        # Use the __call__ method instead which returns a dictionary with input_ids
        # and attention_mask
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=truncation,
            max_length=max_length,
            return_tensors="pt",
        )
        return cast(dict[str, Tensor], encoding)

    def batch_encode(
        self,
        texts: list[str],
        padding: bool = True,
        truncation: bool = True,
        max_length: int = 512,
    ) -> dict[str, Tensor]:
        """Encode a batch of texts.

        Args:
            texts: List of texts to encode
            padding: Whether to pad sequences to the maximum length
            truncation: Whether to truncate sequences to the maximum length
            max_length: The maximum length of the sequence

        Returns:
            Dictionary containing input_ids and attention_mask
        """
        encoding = self.tokenizer(
            texts,
            padding="max_length",  # Change from True to 'max_length'
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        # Cast to the expected return type for mypy
        return cast(dict[str, Tensor], encoding)

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        """Decode token ids back to text.

        Args:
            token_ids: List of token ids to decode
            skip_special_tokens: Whether to skip special tokens like [PAD], [CLS], etc.

        Returns:
            Decoded text
        """
        result = self.tokenizer.decode(
            token_ids, skip_special_tokens=skip_special_tokens
        )
        # Cast to ensure type safety
        return cast(str, result)
