from dataclasses import dataclass, field
from typing import Optional, Union

from trl import GRPOConfig


@dataclass
class CONSEPTConfig(GRPOConfig):
    r"""
    Configuration class for the [`CONSEPTTrainer`].
    """
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
    name: Optional[str] = field(
        default=False,
        metadata={
            "help": ""
        },
    )
