import numpy as np
from torch.optim.lr_scheduler import _LRScheduler


class WarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, base_lr, warmup_steps, after_scheduler):
        """
        A learning rate scheduler with a warmup phase.

        Parameters:
        ----------
        optimizer : torch.optim.Optimizer
            The optimizer for which to adjust the learning rate.
        base_lr : float
            The target learning rate after the warmup phase.
        warmup_steps : int
            The number of steps for the warmup phase.
        after_scheduler : _LRScheduler
            The scheduler to use after the warmup phase.
        """
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.after_scheduler = after_scheduler
        self.finished_warmup = False
        super().__init__(optimizer)

    def get_lr(self):
        """
        Compute the learning rate for the current step.

        Returns:
        -------
        List[float]
            A list of learning rates for each parameter group.
        """
        if not self.finished_warmup:
            # linear warmup
            return [base_lr * (self.last_epoch / self.warmup_steps) for base_lr in self.base_lrs]
        # after warmup
        return self.after_scheduler.get_last_lr()

    def step(self, epoch=None):
        """
        Update the learning rate for the current step.

        Parameters:
        ----------
        epoch : Optional[int]
            The current epoch number. If None, the scheduler uses the internal counter.
        """
        if not self.finished_warmup:
            if self.last_epoch < self.warmup_steps:
                super().step(epoch)
            else:
                self.finished_warmup = True
                self.after_scheduler.base_lrs = self.get_lr()
        if self.finished_warmup:
            self.after_scheduler.step(epoch)


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
