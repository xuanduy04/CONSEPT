from multiprocessing.sharedctypes import Synchronized
from typing import Any


class CompletionLengthScheduler:
    r"""Base class for all completion length schedulers.

    Subclasses implement :meth:`get_completion_length` and optionally override :meth:`step` to
    define scheduling behavior.

    Args:
        completion_length (multiprocessing.sharedctypes.Synchronized): The completion length
            this scheduler will adjust the values of.
        max_completion_length (int): The `completion_length`'s upper bound.
    """

    def __init__(
        self,
        completion_length: Synchronized,
        max_completion_length: int,
    ) -> None:
        # Attach completion_length
        if not isinstance(completion_length, Synchronized):
            raise TypeError(
                f"{type(completion_length).__name__} is not a multiprocessing.sharedctypes.Synchronized object."
            )
        self.completion_length = completion_length
        self.max_completion_length = max_completion_length

        self._step_count = 0

    def state_dict(self) -> dict[str, Any]:
        """Return the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in ``self.__dict__``,
        and saves the `value` field of the `completion_length` attribute.
        """
        state = {k: v for k, v in self.__dict__.items() if k != "completion_length"}
        state["completion_length"] = self.v.value  # save just the integer value
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load the scheduler's state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        completion_length_value = state_dict.pop("completion_length")
        self.__dict__.update(state_dict)
        with self.completion_length.get_lock():
            # Technically only read-modify-write methods require get_lock,
            # but better be safe than sorry.
            self.completion_length.value = completion_length_value

    def get_last_completion_length(self) -> int:
        r"""Get the most recent completion length computed by this scheduler.

        Returns:
            int: The most recent completion length's value

        .. note::
            The returned :class:`~torch.Tensor`\s are copies, and never alias
            the optimizer's ``group["lr"]``\s.
        """
        return int(self.completion_length.value)

    def get_completion_length(self) -> int:
        r"""Compute the next completion length value

        Returns:
            int: The completion length's value of the next "step"

        .. note::
            If you're trying to inspect the most recent completion length, use
            :meth:`get_last_completion_length()` instead.
        """
        raise NotImplementedError

    def step(self, **kwargs) -> None:
        """Step the scheduler."""
        self._step_count += 1
        self._update_completion_length()

    def _update_completion_length(self) -> None:
        with self.completion_length.get_lock():
            # Technically only read-modify-write methods require get_lock,
            # but better be safe than sorry.
            self.completion_length.value = min(self.get_completion_length(), self.max_completion_length)
