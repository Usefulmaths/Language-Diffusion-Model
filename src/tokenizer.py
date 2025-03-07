"""A simple tokenizer wrapper for diffusion language models."""

from dataclasses import dataclass

import torch
from transformers import BertTokenizer


@dataclass
class TokenizerConfig:
    """Configuration for the DiffusionTokenizer.

    Attributes:
        tokenizer_name (str): Name of the pretrained tokenizer model.
        mask_token (str): Token used to represent masked positions.
        pad_token (str): Token used for padding.
        bos_token (str): Token marking the beginning of a sequence.
        eos_token (str): Token marking the end of a sequence.
    """

    tokenizer_name: str = "bert-base-uncased"
    mask_token: str = "[MASK]"
    pad_token: str = "[PAD]"
    bos_token: str = "[CLS]"
    eos_token: str = "[SEP]"


class DiffusionTokenizer:
    """A simple tokenizer wrapper for diffusion language models.

    This class wraps a pretrained BertTokenizer to provide minimal functionality:
    - Encoding text into token IDs and an attention mask
    - Manually inserting BOS and EOS tokens
    - Padding (or truncating) sequences to a fixed length
    - Decoding token IDs back to text
    """

    def __init__(self, config: TokenizerConfig = None):
        """Initialize the DiffusionTokenizer.

        Args:
            config (TokenizerConfig, optional): Configuration for the tokenizer.
                If None, a default configuration is used.
        """
        if config is None:
            config = TokenizerConfig()
        self.config = config
        self.tokenizer = BertTokenizer.from_pretrained(config.tokenizer_name)
        self.mask_token_id = self.tokenizer.convert_tokens_to_ids(config.mask_token)
        self.pad_token_id = self.tokenizer.convert_tokens_to_ids(config.pad_token)
        self.bos_token_id = self.tokenizer.convert_tokens_to_ids(config.bos_token)
        self.eos_token_id = self.tokenizer.convert_tokens_to_ids(config.eos_token)
        self.vocab_size = len(self.tokenizer)

    def encode(
        self, text: str, max_length: int = 512, return_tensors: bool = False
    ) -> dict:
        """Encode a text string into 'input_ids' and 'attention_mask'.

        This method tokenizes the input without adding special tokens. It then
        manually adds BOS and EOS tokens, and pads (or truncates) the sequence
        to exactly `max_length`. An attention mask is generated accordingly.

        Args:
            text (str): The input text to encode.
            max_length (int, optional): The sequence length (including BOS/EOS).
                Defaults to 512.
            return_tensors (bool, optional): If True, returns PyTorch tensors
                with shape [1, max_length]. Otherwise, returns lists. Defaults
                to False.

        Returns:
            dict: A dict with "input_ids" and "attention_mask". If
                return_tensors is True, values are torch.Tensors; otherwise,
                they are lists of ints.
        """
        effective_length = max_length - 2
        encoding = self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=effective_length,
            return_tensors="pt" if return_tensors else None,
        )

        if return_tensors:
            input_ids = encoding["input_ids"].squeeze(0).tolist()
            attention_mask = encoding["attention_mask"].squeeze(0).tolist()
        else:
            input_ids = encoding["input_ids"]
            attention_mask = encoding["attention_mask"]

        input_ids = [self.bos_token_id] + input_ids + [self.eos_token_id]
        attention_mask = [1] + attention_mask + [1]

        if len(input_ids) < max_length:
            pad_len = max_length - len(input_ids)
            input_ids += [self.pad_token_id] * pad_len
            attention_mask += [0] * pad_len
        else:
            input_ids = input_ids[:max_length]
            attention_mask = attention_mask[:max_length]

        if return_tensors:
            input_ids = torch.tensor([input_ids], dtype=torch.long)
            attention_mask = torch.tensor([attention_mask], dtype=torch.long)

        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        """Decode a list of token IDs back into a text string.

        Args:
            token_ids (list[int]): The token IDs to decode.
            skip_special_tokens (bool, optional): Whether to skip special
                tokens. Defaults to True.

        Returns:
            str: The decoded text.
        """
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
