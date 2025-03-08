"""Dataclasses for configuration and metrics."""

from dataclasses import dataclass


@dataclass
class DiffusionTransformerConfig:
    """Configuration for DiffusionTransformer."""

    vocab_size: int = 30522  # Default BERT vocab size
    d_model: int = 768
    nhead: int = 12
    num_layers: int = 12
    dim_feedforward: int = 3072
    dropout: float = 0.1
    layer_norm_eps: float = 1e-12
    max_position_embeddings: int = 512


@dataclass
class TrainingMetrics:
    """Container for training metrics."""

    epoch: int
    loss: float
    learning_rate: float
    epoch_time: float

    def __str__(self) -> str:
        """Convert metrics to string for logging."""
        return (
            f"Epoch: {self.epoch}, "
            f"Loss: {self.loss:.4f}, "
            f"Learning Rate: {self.learning_rate:.8f}, "
            f"Time: {self.epoch_time:.2f}s"
        )
