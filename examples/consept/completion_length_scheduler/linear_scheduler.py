from typing import TYPE_CHECKING

from typing_extensions import override

from .base_scheduler import CompletionLengthScheduler


if TYPE_CHECKING:
    from multiprocessing.sharedctypes import Synchronized


class LinearCompletionLengthScheduler(CompletionLengthScheduler):
    """Linearly increases the completion length from `initial_completion_length`
    to `max_completion_length` over `warmup_steps` steps, then holds at
    `max_completion_length` for all subsequent steps.

    Args:
        completion_length (multiprocessing.sharedctypes.Synchronized): The completion length
            this scheduler will adjust the values of.
        max_completion_length (int): The `completion_length`'s upper bound.
            This value is reached at `warmup_steps` and maintained thereafter.
        initial_completion_length (int): The value of `completion_length` at
            step 0, i.e. the starting point of the linear ramp.
        warmup_steps (int): The step at which `completion_length.value` first
            reaches `max_completion_length`. The increment per step is
            `(max_completion_length - initial_completion_length) / warmup_steps`.
    """

    def __init__(
        self,
        completion_length: "Synchronized",
        max_completion_length: int,
        initial_completion_length: int,
        warmup_steps: int,
    ) -> None:
        super().__init__(completion_length=completion_length, max_completion_length=max_completion_length)
        self.initial_completion_length = initial_completion_length
        self.warmup_steps = warmup_steps

    @override
    def get_completion_length(self) -> int:
        if self._step_count >= self.warmup_steps:
            return self.max_completion_length

        progress = self._step_count / self.warmup_steps
        length = self.initial_completion_length + progress * (
            self.max_completion_length - self.initial_completion_length
        )
        return int(length)
