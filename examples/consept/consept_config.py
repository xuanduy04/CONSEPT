from dataclasses import dataclass, field
from typing import Optional

from trl import GRPOConfig


@dataclass
class CONSEPTConfig(GRPOConfig):
    r"""
    Configuration class for the [`CONSEPTTrainer`].
    """

    # Parameters whose default values are overridden from GRPOConfig
    max_prompt_length: Optional[int] = field(
        default=0xFFFFFFFFFFFFFFFF,
        metadata={
            "help": "Maximum length of the prompt. If the prompt is longer than this value, it will be truncated left."
            "We dynamically set the prompt length so this really should be infinite. (Defaults to the maximum 64-bit unsigned integer value)"
        },
    )

    # Parameters that control the dynamic completion length
    initial_completion_length: Optional[int] = field(
        default=1,
        metadata={"help": "At first, what is the number of tokens that the model will try to generate."},
    )

    prompt_length_remove_threshold: Optional[int] = field(
        default=100,
        metadata={"help": "If the prompt is less than this number, we remove the sample instead."},
    )

    # default for template
    # _name_: Optional[str] = field(
    #     default=False,
    #     metadata={"help": ""},
    # )

    def __post_init__(self):
        super().__post_init__()

        if self.initial_completion_length <= 0:
            raise ValueError("'initial_completion_length' must be a non-negative integer.")
        if self.max_completion_length < self.initial_completion_length:
            raise ValueError("'max_completion_length' must be at least 'initial_completion_length'.")

        if self.prompt_length_remove_threshold <= 0:
            raise ValueError("'prompt_length_remove_threshold' must be a non-negative integer.")
