import numpy as np


class MomentumScheduler:
    """
    A scheduler for momentum values that follows a cosine decay pattern.
    """

    def __init__(self, base_momentum, warmup_steps, total_steps):
        """
        Parameters:
        ----------
        base_momentum : float
            The initial momentum value (1 - mm).
        warmup_steps : int
            The number of steps for the warmup phase.
        total_steps : int
            The total number of steps for the scheduler.
        """
        self.base_momentum = base_momentum
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def get(self, step):
        """
        Get the momentum value for a given step.

        Parameters:
        ----------
        step : int
            The current step.

        Returns:
        -------
        float
            The momentum value for the given step.
        """
        if step < self.warmup_steps:
            return self.base_momentum * step / self.warmup_steps
        elif self.warmup_steps <= step <= self.total_steps:
            return (
                self.base_momentum
                * (
                    1
                    + np.cos(
                        (step - self.warmup_steps) * np.pi / (self.total_steps - self.warmup_steps)
                    )
                )
                / 2
            )
        else:
            raise ValueError(f"Step ({step}) > total number of steps ({self.total_steps}).")
