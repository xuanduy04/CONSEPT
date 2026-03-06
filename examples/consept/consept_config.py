from dataclasses import dataclass, field
from typing import Optional, Union

from trl import GRPOConfig


@dataclass
class CONSEPTConfig(GRPOConfig):
    r"""
    Configuration class for the [`CONSEPTTrainer`].
    """
    # Parameters whose default values are overridden from GRPOConfig
    max_prompt_length: Optional[int] = field(
        default=0xffffffffffffffff,
        metadata={
            "help": "Maximum length of the prompt. If the prompt is longer than this value, it will be truncated left."
            "We dynamically set the prompt length so this really should be infinite. (Defaults to the maximum 64-bit unsigned integer value)"
        },
    )

    # Parameters that control the dynamic completion length
    initial_completion_length: Optional[int] = field(
        default=64,
        metadata={
            "help": "At first, what is the number of tokens that the model will try to generate"
        },
    )

    # completion_length_factor: Optional[float] = field(
    #     default=2.0,
    #     metadata={
    #         "help": ""
    #     },
    # )

    # default for template
    _name_: Optional[str] = field(
        default=False,
        metadata={
            "help": ""
        },
    )
