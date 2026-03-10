from .base_scheduler import CompletionLengthScheduler


class ConstantCompletionLengthScheduler(CompletionLengthScheduler):
    r"""A completion length scheduler that doesn't change the current value of `completion_length`. Ever.

    Args:
        completion_length (multiprocessing.sharedctypes.Synchronized): The completion length
            this scheduler will adjust the values of.
        max_completion_length (int): The `completion_length`'s upper bound. (exists for compatibility)
    """

    def get_completion_length(self) -> int:
        r"""Compute the next completion length value

        Returns:
            int: The completion length's value of the next "step", which... is exactly the same.
        """
        return self.get_last_completion_length()
