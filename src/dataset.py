"""Custom dataset for loading questions."""

from collections.abc import Callable
from typing import TypeVar

import torch
from torch.utils.data import DataLoader, Dataset

from src.tokenizer import DiffusionTokenizer

T = TypeVar("T")


class QuestionDataset(Dataset[dict[str, torch.Tensor]]):
    """Dataset for loading questions from a formatted text file.

    Each line in the file is formatted as:
    /url-path@@Question text@@ID@@[tags]@@[related_questions]
    """

    def __init__(
        self,
        file_path: str,
        tokenizer: DiffusionTokenizer,
        max_length: int = 512,
        transform: Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]]
        | None = None,
    ):
        """Initialize the dataset.

        Args:
            file_path: Path to the question file
            tokenizer: Tokenizer for encoding questions
            max_length: Maximum sequence length
            transform: Optional transform to apply to the tokenized input
        """
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.transform = transform

        # Load and parse questions
        self.questions = self._load_questions()

    def _load_questions(self) -> list[str]:
        """Load questions from the file.

        Returns:
            List of questions
        """
        questions = []

        with open(self.file_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # Split line by @@ delimiter
                parts = line.split("@@")

                # Extract question (second element)
                if len(parts) >= 2:
                    question = parts[1]
                    questions.append(question)

        return questions

    def __len__(self) -> int:
        """Return the number of questions in the dataset.

        Returns:
            Number of questions
        """
        return len(self.questions)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a question by index.

        Args:
            idx: Index of the question

        Returns:
            Dictionary containing tokenized inputs
        """
        question = self.questions[idx]

        # Tokenize the question
        encoding = self.tokenizer.encode(
            question,
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        # Ensure input_ids and attention_mask have correct shape [batch_size, seq_len]
        if (
            "input_ids" in encoding
            and len(encoding["input_ids"].shape) == 2
            and encoding["input_ids"].shape[0] == 1
        ):
            encoding["input_ids"] = encoding["input_ids"].squeeze(0)

        if (
            "attention_mask" in encoding
            and len(encoding["attention_mask"].shape) == 2
            and encoding["attention_mask"].shape[0] == 1
        ):
            encoding["attention_mask"] = encoding["attention_mask"].squeeze(0)

        # Apply transform if provided
        if self.transform is not None:
            encoding = self.transform(encoding)

        return encoding


def create_question_dataloaders(
    file_path: str,
    tokenizer: DiffusionTokenizer,
    batch_size: int = 16,
    max_length: int = 512,
    train_ratio: float = 0.9,
    shuffle: bool = True,
    num_workers: int = 4,
) -> tuple[
    DataLoader[dict[str, torch.Tensor]], DataLoader[dict[str, torch.Tensor]] | None
]:
    """Create dataloaders for training and validation.

    Args:
        file_path: Path to the question file
        tokenizer: Tokenizer for encoding questions
        batch_size: Batch size for dataloaders
        max_length: Maximum sequence length
        train_ratio: Ratio of data to use for training (rest for validation)
        shuffle: Whether to shuffle the data
        num_workers: Number of workers for data loading

    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    # Create dataset
    dataset = QuestionDataset(
        file_path=file_path,
        tokenizer=tokenizer,
        max_length=max_length,
    )

    # Split into train and validation sets
    dataset_size = len(dataset)
    train_size = int(train_ratio * dataset_size)
    val_size = dataset_size - train_size

    # Split into train and validation sets
    dataset_size = len(dataset)
    train_size = int(train_ratio * dataset_size)
    val_size = dataset_size - train_size

    # First create the train dataloader (always created)
    if val_size > 0:
        # Split the dataset
        train_subset, val_subset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )

        # Create train dataloader from the subset
        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )

        # Create validation dataloader
        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
    else:
        # Use full dataset for training
        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )
        val_loader = None

    # Type annotation for return value
    result: tuple[
        DataLoader[dict[str, torch.Tensor]],
        DataLoader[dict[str, torch.Tensor]] | None,
    ] = (train_loader, val_loader)

    return result
