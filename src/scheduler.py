"""LLaDA scheduler implementation.

This module contains the learning rate scheduler implementation as described in the
LLaDA paper (Warmup-Stable-Decay pattern).
"""

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class WarmupStableDecayLR(_LRScheduler):
    """Learning rate scheduler that implements the Warmup-Stable-Decay pattern.

    The scheduler follows these phases:
    1. Warmup: Linear increase from initial_lr to peak_lr
    2. First stable: Maintain peak_lr
    3. Middle decay: Linear decay from peak_lr to stable_lr
    4. Second stable: Maintain stable_lr
    5. Final decay: Linear decay from stable_lr to final_lr

    Note: While this implementation follows PyTorch's convention of returning a list of
    learning rates (one per parameter group), the LLaDA paper uses a single learning
    rate
    for the entire model.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        stable_steps: int,
        middle_decay_steps: int,
        final_decay_steps: int,
        initial_lr: float = 0.0,
        peak_lr: float | None = None,
        stable_lr: float | None = None,
        final_lr: float = 1e-7,
        last_epoch: int = -1,
    ):
        """Initialize the scheduler.

        Args:
            optimizer: The optimizer to schedule
            warmup_steps: Number of steps for linear warmup
            stable_steps: Number of steps to maintain peak_lr
            middle_decay_steps: Number of steps to decay from peak_lr to stable_lr
            final_decay_steps: Number of steps for final decay from stable_lr to
            final_lr
            initial_lr: Initial learning rate at start of training
            peak_lr: Learning rate after warmup (defaults to base_lr if None)
            stable_lr: Learning rate during stable phase (defaults to 0.25 * peak_lr
            if None)
            final_lr: Final learning rate at end of training
            last_epoch: The index of the last epoch
        """
        self.warmup_steps = warmup_steps
        self.stable_steps = stable_steps
        self.middle_decay_steps = middle_decay_steps
        self.final_decay_steps = final_decay_steps

        # Set default peak_lr from optimizer if not provided
        base_lr = optimizer.param_groups[0]["lr"]
        self.initial_lr = initial_lr
        self.peak_lr = peak_lr if peak_lr is not None else base_lr
        self.stable_lr = stable_lr if stable_lr is not None else self.peak_lr * 0.25
        self.final_lr = final_lr

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        """Calculate the learning rate for the current step.

        Returns:
            List of learning rates, one for each parameter group
        """
        step = self.last_epoch

        # Warmup phase: Linear increase from initial_lr to peak_lr
        if step < self.warmup_steps:
            factor = step / self.warmup_steps if self.warmup_steps > 0 else 1.0
            return [
                self.initial_lr + factor * (self.peak_lr - self.initial_lr)
                for _ in self.base_lrs
            ]

        # First stable phase: maintain peak_lr
        step = step - self.warmup_steps
        if step < self.stable_steps:
            return [self.peak_lr for _ in self.base_lrs]

        # Middle decay phase: Linear decay from peak_lr to stable_lr
        step = step - self.stable_steps
        if step < self.middle_decay_steps:
            factor = (
                step / self.middle_decay_steps if self.middle_decay_steps > 0 else 1.0
            )
            return [
                self.peak_lr - factor * (self.peak_lr - self.stable_lr)
                for _ in self.base_lrs
            ]

        # Second stable phase: maintain stable_lr
        step = step - self.middle_decay_steps
        if step < 0:
            return [self.stable_lr for _ in self.base_lrs]

        # Final decay phase: Linear decay from stable_lr to final_lr
        if step < self.final_decay_steps:
            factor = (
                step / self.final_decay_steps if self.final_decay_steps > 0 else 1.0
            )
            return [
                self.stable_lr - factor * (self.stable_lr - self.final_lr)
                for _ in self.base_lrs
            ]

        # Beyond final decay: maintain final_lr
        return [self.final_lr for _ in self.base_lrs]


def create_scheduler(
    optimizer: Optimizer,
    total_steps: int,
    warmup_steps: int = 2000,
    initial_lr: float = 0.0,
    peak_lr: float | None = None,
    stable_lr: float | None = None,
    final_lr: float | None = None,
) -> WarmupStableDecayLR:
    """Create a scheduler following the paper's Warmup-Stable-Decay pattern.

    Implements the exact learning rate schedule described in the LLaDA paper:
    - Linear warmup from initial_lr to peak_lr over warmup_steps
    - Maintain peak_lr for 52% of remaining steps
    - Linear decay to stable_lr and hold for 35% of remaining steps
    - Linear decay to final_lr for final 13% of remaining steps

    Args:
        optimizer: The optimizer to schedule
        total_steps: Total number of training steps
        warmup_steps: Number of steps for linear warmup
        initial_lr: Initial learning rate (0.0 in paper)
        peak_lr: Learning rate after warmup (4e-4 in paper, defaults to optimizer lr)
        stable_lr: Learning rate during stable phase (1e-4 in paper, defaults
        to 0.25*peak_lr)
        final_lr: Final learning rate (1e-5 in paper, defaults to 0.025*peak_lr)

    Returns:
        WarmupStableDecayLR scheduler configured per the paper
    """
    # Ensure warmup_steps doesn't exceed total steps
    warmup_steps = min(warmup_steps, total_steps)

    # Calculate remaining steps after warmup
    remaining_steps = total_steps - warmup_steps

    # Calculate phase durations based on paper's ratios
    # Paper used 52% stable, 35% middle decay, 13% final decay
    stable_steps = int(remaining_steps * 0.52)
    final_decay_steps = int(remaining_steps * 0.13)
    middle_decay_steps = remaining_steps - stable_steps - final_decay_steps

    # Set learning rates to paper defaults if not specified
    if peak_lr is None:
        peak_lr = optimizer.param_groups[0]["lr"]  # Use optimizer's learning rate

    if stable_lr is None:
        stable_lr = peak_lr * 0.25  # Default to 25% of peak_lr

    if final_lr is None:
        final_lr = peak_lr * 0.025  # Default to 2.5% of peak_lr

    return WarmupStableDecayLR(
        optimizer=optimizer,
        warmup_steps=warmup_steps,
        stable_steps=stable_steps,
        middle_decay_steps=middle_decay_steps,
        final_decay_steps=final_decay_steps,
        initial_lr=initial_lr,
        peak_lr=peak_lr,
        stable_lr=stable_lr,
        final_lr=final_lr,
    )
