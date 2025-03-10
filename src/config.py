"""Configuration management for diffusion language model."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml


@dataclass
class TokenizerConfig:
    """Configuration for the tokenizer."""

    tokenizer_name: str = "bert-base-uncased"
    mask_token: str = "[MASK]"
    pad_token: str = "[PAD]"
    bos_token: str = "[CLS]"
    eos_token: str = "[SEP]"


@dataclass
class TransformerConfig:
    """Configuration for the transformer model."""

    vocab_size: int = 30522  # Will be set from tokenizer
    d_model: int = 768
    nhead: int = 12
    num_layers: int = 12
    dim_feedforward: int = 3072
    dropout: float = 0.1
    layer_norm_eps: float = 1e-12
    max_position_embeddings: int = 512


@dataclass
class MaskingConfig:
    """Configuration for masking strategy."""

    strategy: Literal["random"] = "random"
    mask_ratio: float | None = None  # If None, random mask ratio per batch


@dataclass
class RemaskingConfig:
    """Configuration for remasking strategy."""

    strategy: Literal["random"] = "random"
    remask_ratio: float = 0.15  # Ratio of tokens to remask


@dataclass
class SchedulerConfig:
    """Configuration for the Warmup-Stable-Decay learning rate scheduler."""

    warmup_steps: int = 2000  # From the paper
    initial_lr: float = 0.0
    peak_lr: float | None = None  # If None, uses training.learning_rate
    stable_lr: float | None = None  # If None, uses 0.25 * peak_lr
    final_lr: float | None = None  # If None, uses 0.025 * peak_lr


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation."""

    eval_steps: int = 100  # Evaluate every N steps
    eval_mask_ratio: float = 0.15  # Fixed mask ratio for evaluation
    num_examples: int = 3  # Number of generation examples to show during evaluation


@dataclass
class TrainingConfig:
    """Configuration for training."""

    batch_size: int = 16
    learning_rate: float = 4e-4  # Updated to match paper's 4 × 10^-4
    weight_decay: float = 0.01
    num_epochs: int = 10
    max_length: int = 512
    num_workers: int = 4
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    checkpoint_dir: str = "checkpoints"


@dataclass
class DataConfig:
    """Configuration for dataset."""

    train_file: str = "data/questions.txt"
    train_ratio: float = 0.9
    shuffle: bool = True


@dataclass
class Config:
    """Main configuration for the diffusion language model."""

    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    masking: MaskingConfig = field(default_factory=MaskingConfig)
    remasking: RemaskingConfig = field(default_factory=RemaskingConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    seed: int = 42

    @classmethod
    def from_yaml(cls, yaml_file: str) -> "Config":
        """Load configuration from YAML file.

        Args:
            yaml_file: Path to the YAML configuration file

        Returns:
            Config object populated from YAML

        Raises:
            IOError: If file cannot be read
            yaml.YAMLError: If YAML parsing fails
        """
        with open(yaml_file) as f:
            # Use safe_load for security
            config_dict = yaml.safe_load(f)

        if not config_dict:
            # Handle empty or invalid YAML file
            return cls()

        config = cls()

        # Update tokenizer config
        if "tokenizer" in config_dict:
            config.tokenizer = TokenizerConfig(**config_dict["tokenizer"])

        # Update transformer config
        if "transformer" in config_dict:
            config.transformer = TransformerConfig(**config_dict["transformer"])

        # Update masking config
        if "masking" in config_dict:
            config.masking = MaskingConfig(**config_dict["masking"])

        # Update remasking config
        if "remasking" in config_dict:
            config.remasking = RemaskingConfig(**config_dict["remasking"])

        # Update training config
        if "training" in config_dict:
            training_dict = config_dict["training"].copy()

            # Handle nested scheduler config
            if "scheduler" in training_dict:
                scheduler_dict = training_dict.pop("scheduler")
                scheduler_config = SchedulerConfig(**scheduler_dict)
                training_dict["scheduler"] = scheduler_config

            # Handle nested evaluation config
            if "evaluation" in training_dict:
                eval_dict = training_dict.pop("evaluation")
                eval_config = EvaluationConfig(**eval_dict)
                training_dict["evaluation"] = eval_config

            config.training = TrainingConfig(**training_dict)

        # Update data config
        if "data" in config_dict:
            config.data = DataConfig(**config_dict["data"])

        # Update seed
        if "seed" in config_dict:
            config.seed = config_dict["seed"]

        return config

    def save(self, yaml_file: str) -> None:
        """Save configuration to YAML file."""
        # Create directory if it doesn't exist
        Path(yaml_file).parent.mkdir(parents=True, exist_ok=True)

        # Convert dataclass to dict
        config_dict: dict[str, Any] = {
            "seed": self.seed,
            "tokenizer": dict(vars(self.tokenizer)),
            "transformer": dict(vars(self.transformer)),
            "masking": {k: v for k, v in vars(self.masking).items() if v is not None},
            "remasking": {
                k: v for k, v in vars(self.remasking).items() if v is not None
            },
            "training": {
                **{
                    k: v
                    for k, v in vars(self.training).items()
                    if k not in ["scheduler", "evaluation"] and v is not None
                },
                "scheduler": dict(vars(self.training.scheduler)),
                "evaluation": dict(vars(self.training.evaluation)),
            },
            "data": {k: v for k, v in vars(self.data).items() if v is not None},
        }

        # Save to YAML file
        with open(yaml_file, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)
