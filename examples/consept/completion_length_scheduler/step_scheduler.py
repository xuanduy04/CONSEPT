from typing import TYPE_CHECKING

from typing_extensions import override

from .base_scheduler import CompletionLengthScheduler


if TYPE_CHECKING:
    from multiprocessing.sharedctypes import Synchronized


class StepCompletionLengthScheduler(CompletionLengthScheduler):
    """Multiplies the completion length of each parameter group by `factor` every `step_size` steps.
    Values are always rounded down to the nearest integer after every multiplication.

    Notice that such increase can happen simultaneously with other changes to the completion length
    from outside this scheduler.

    Args:
        completion_length (multiprocessing.sharedctypes.Synchronized): The completion length
            this scheduler will adjust the values of.
        max_completion_length (int): The `completion_length`'s upper bound.
        step_size (int): Period of completion length decay.
        factor (float): Multiplicative factor of completion length decay.
            Default: 2.0
    """

    def __init__(
        self,
        completion_length: "Synchronized",
        max_completion_length: int,
        step_size: int,
        factor: float = 2.0,
    ) -> None:
        super().__init__(completion_length=completion_length, max_completion_length=max_completion_length)
        if factor <= 1.0:
            raise ValueError("Scheduler's `factor` should be > 1.0.")
        self.step_size = step_size
        self.factor = factor
        self.last_multiplication_step = self._step_count

    @override
    def get_completion_length(self) -> int:
        if self._step_count - self.last_multiplication_step > self.step_size:
            self.last_multiplication_step = self._step_count
            return int(self.completion_length.value * self.factor)
        return self.completion_length.value
