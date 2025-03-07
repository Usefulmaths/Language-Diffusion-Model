"""Custom dataset for loading questions."""

from collections.abc import Callable
from typing import TypedDict

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from src.tokenizer import DiffusionTokenizer


class DatasetItem(TypedDict):
    """Dataset item structure.

    Attributes:
        input_ids (torch.Tensor): 1D tensor of token IDs, shape [seq_len].
        attention_mask (torch.Tensor): 1D tensor of attention mask values,
        shape [seq_len].
    """

    input_ids: torch.Tensor
    attention_mask: torch.Tensor


def collate_batch(batch: list[DatasetItem]) -> DatasetItem:
    """Collate function for variable length sequences.

    Pads a list of dataset items (each containing input_ids and attention_mask)
    to the maximum sequence length in the batch.

    Args:
        batch: List of dataset items.

    Returns:
        A batched dataset item with keys "input_ids" and "attention_mask",
        where each value is a padded tensor.
    """
    input_ids = [item["input_ids"] for item in batch]
    attention_masks = [item["attention_mask"] for item in batch]

    # Pad sequences to the maximum length in the batch.
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_masks_padded = pad_sequence(
        attention_masks, batch_first=True, padding_value=0
    )

    return {"input_ids": input_ids_padded, "attention_mask": attention_masks_padded}


class QuestionDataset(Dataset[DatasetItem]):
    """Dataset for loading questions from a formatted text file.

    Each line in the file is expected to have parts separated by the @@ delimiter.
    The second part (index 1) is used as the question.
    """

    def __init__(
        self,
        file_path: str,
        tokenizer: DiffusionTokenizer,
        max_length: int = 512,
        transform: Callable[[DatasetItem], DatasetItem] | None = None,
    ):
        """Initialize the dataset.

        Args:
            file_path (str): Path to the text file containing questions.
            tokenizer (DiffusionTokenizer): Tokenizer to encode the questions.
            max_length (int, optional): Maximum sequence length for encoding. Defaults
            to 512.
            transform (Callable, optional): Optional transform to apply to each dataset
            item.
        """
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.transform = transform

        # Load and parse questions.
        self.questions = self._load_questions()

    def _load_questions(self) -> list[str]:
        """Load questions from the file.

        Returns:
            list[str]: A list of questions.
        """
        questions = []
        with open(self.file_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Split line by @@ delimiter.
                parts = line.split("@@")
                # Extract question (second element) if available.
                if len(parts) >= 2:
                    question = parts[1]
                    questions.append(question)
        return questions

    def __len__(self) -> int:
        """Return the number of questions in the dataset."""
        return len(self.questions)

    def __getitem__(self, idx: int) -> DatasetItem:
        """Get a question by index.

        Encodes the question using the tokenizer without padding (to allow the collate
        function to handle variable lengths). The resulting lists are converted to 1D
        tensors.

        Args:
            idx (int): Index of the question.

        Returns:
            DatasetItem: A dictionary with keys "input_ids" and "attention_mask", each
            a 1D tensor.
        """
        question = self.questions[idx]
        encoding = self.tokenizer.encode(
            question, max_length=self.max_length, return_tensors=False
        )
        # Convert lists to tensors.
        input_ids = torch.tensor(encoding["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(encoding["attention_mask"], dtype=torch.long)
        result: DatasetItem = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if self.transform is not None:
            result = self.transform(result)
        return result


def create_question_dataloaders(
    file_path: str,
    tokenizer: DiffusionTokenizer,
    batch_size: int = 16,
    max_length: int = 512,
    train_ratio: float = 0.9,
    shuffle: bool = True,
    num_workers: int = 4,
) -> tuple[DataLoader[DatasetItem], DataLoader[DatasetItem] | None]:
    """Create dataloaders for training and validation.

    Args:
        file_path (str): Path to the file containing questions.
        tokenizer (DiffusionTokenizer): Tokenizer to use for encoding questions.
        batch_size (int, optional): Batch size. Defaults to 16.
        max_length (int, optional): Maximum sequence length. Defaults to 512.
        train_ratio (float, optional): Ratio of data to use for training.
        Defaults to 0.9.
        shuffle (bool, optional): Whether to shuffle the training data.
        Defaults to True.
        num_workers (int, optional): Number of worker threads for data loading.
        Defaults to 4.

    Returns:
        tuple: A tuple containing the train DataLoader and validation DataLoader
        (or None if no validation split).
    """
    dataset = QuestionDataset(
        file_path=file_path,
        tokenizer=tokenizer,
        max_length=max_length,
    )

    dataset_size = len(dataset)
    train_size = int(train_ratio * dataset_size)
    val_size = dataset_size - train_size

    if val_size > 0:
        train_subset, val_subset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_batch,
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_batch,
        )
    else:
        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_batch,
        )
        val_loader = None

    return (train_loader, val_loader)
