"""Custom dataset for loading questions."""

from collections.abc import Callable
from typing import TypedDict

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from src.tokenizer import DiffusionTokenizer


class DatasetItem(TypedDict):
    """Dataset item structure with 1D tensors."""

    input_ids: torch.Tensor  # Shape: [seq_len]
    attention_mask: torch.Tensor  # Shape: [seq_len]


# Custom collate function to handle variable length sequences
def collate_batch(batch: list[DatasetItem]) -> DatasetItem:
    """Collate function for variable length sequences.

    Args:
        batch: List of dataset items

    Returns:
        Batched dataset item with padded sequences
    """
    input_ids = [item["input_ids"] for item in batch]
    attention_masks = [item["attention_mask"] for item in batch]

    # Pad sequences to the maximum length in the batch
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_masks_padded = pad_sequence(
        attention_masks, batch_first=True, padding_value=0
    )

    return {"input_ids": input_ids_padded, "attention_mask": attention_masks_padded}


class QuestionDataset(Dataset[DatasetItem]):
    """Dataset for loading questions from a formatted text file."""

    def __init__(
        self,
        file_path: str,
        tokenizer: DiffusionTokenizer,
        max_length: int = 512,
        transform: Callable[[DatasetItem], DatasetItem] | None = None,
    ):
        """Initialize the dataset."""
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.transform = transform

        # Load and parse questions
        self.questions = self._load_questions()

    def _load_questions(self) -> list[str]:
        """Load questions from the file."""
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
        """Return the number of questions in the dataset."""
        return len(self.questions)

    def __getitem__(self, idx: int) -> DatasetItem:
        """Get a question by index."""
        question = self.questions[idx]

        # Tokenize the question - Don't pad here, let the collate function handle it
        encoding = self.tokenizer.encode(
            question,
            padding=False,  # Don't pad individual items
            truncation=True,
            max_length=self.max_length,
            return_batch_dimension=False,  # Get 1D tensors
        )

        # Create a properly typed result
        result: DatasetItem = {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
        }

        # Apply transform if provided
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
    """Create dataloaders for training and validation."""
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

    # Create dataloaders with the custom collate function
    if val_size > 0:
        # Split the dataset
        train_subset, val_subset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )

        # Create train dataloader with custom collate function
        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_batch,  # Use our custom collate function
        )

        # Create validation dataloader with custom collate function
        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_batch,
        )
    else:
        # Use full dataset for training
        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_batch,
        )
        val_loader = None

    return (train_loader, val_loader)
