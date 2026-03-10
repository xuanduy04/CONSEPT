from typing import TYPE_CHECKING, Literal, SupportsFloat


if TYPE_CHECKING:
    pass


from .base_scheduler import CompletionLengthScheduler


if TYPE_CHECKING:
    from multiprocessing.sharedctypes import Synchronized


class IncreaseCompletionLengthOnVictory(CompletionLengthScheduler):
    """Increase completion length when a metric has "consistently surpassed" a threshold (i.e. victorious).

    This scheduler reads a metric's quantity and if it has continuously stayed higher/lower
    than a `threshold` for a `patience` number of steps, the completion length is increased.

    Args:
        completion_length (multiprocessing.sharedctypes.Synchronized): The completion length
            this scheduler will adjust the values of.
        max_completion_length (int): The `completion_length`'s upper bound.
        threshold (float):
            The metric needs to be "better" than threshold to be considered a victory.
        mode (str, One of [`min`, `max`]): In `min` mode, victory is achieved when metric is **less** than `threshold`;
        in `max` mode , it is achieved when metric is **more** than `threshold`. Default: 'max'.
        factor (float): Factor by which the `completion length` will be increased,
            rounded down after every multiplication. `new_completion_length = new_completion_length * factor`.
            Default: 2.0.
        patience (int): The number of allowed steps of continual victory after
            which the completion length will be increased.
            Default: 10.
        eps (float):
            When the difference between current `metric` and `threshold` is less than this value,
            they are considered equal (and thus counted as a victory). Default: 1e-8
        cooldown (int): Number of steps to wait before resuming
            normal operation after `completion_length` has been increased. Default: 0.
    """

    def __init__(
        self,
        completion_length: "Synchronized",
        max_completion_length: int,
        threshold: float,
        mode: Literal["min", "max"] = "max",
        factor: float = 2.0,
        patience: int = 10,
        eps: float = 1e-8,
        cooldown: int = 0,
    ) -> None:
        if mode not in {"min", "max"}:
            raise ValueError("mode " + mode + " is unknown!")
        if factor <= 1.0:
            raise ValueError("Scheduler's `factor` should be > 1.0.")
        super().__init__(completion_length=completion_length, max_completion_length=max_completion_length)
        self.threshold = threshold
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.eps = eps
        self.cooldown = cooldown
        self._reset()

    def _reset(self) -> None:
        """Reset `num_bad_steps` counter and `cooldown_counter`."""
        self.cooldown_counter = 0
        self.num_bad_steps = 0

    def step(self, metrics: SupportsFloat) -> None:  # type: ignore[override]
        """Perform a step."""
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)
        self._step_count += 1

        if self._is_victory(current):
            self.num_bad_steps += 1
        else:
            self.num_bad_steps = 0

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_steps = 0  # ignore any bad steps in cooldown

        if self.num_bad_steps > self.patience:
            with self.completion_length.get_lock():
                # Technically only read-modify-write methods require get_lock,
                # but better be safe than sorry.
                self.completion_length.value = min(
                    int(self.completion_length.value * self.factor), self.max_completion_length
                )
            self.cooldown_counter = self.cooldown
            self.num_bad_steps = 0

    @property
    def in_cooldown(self):
        return self.cooldown_counter > 0

    def _is_victory(self, metric):
        # Verify the math via making the equation: 0 < metric - self.threshold < self.eps
        if self.mode == "min":
            return metric < self.threshold + self.eps
        else:  # mode == 'max':
            return metric + self.eps > self.threshold
