from dataclasses import dataclass, field
from typing import Literal, Optional

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
        default=64,
        metadata={"help": "At first, what is the number of tokens that the model will try to generate."},
    )

    prompt_length_remove_threshold: Optional[int] = field(
        default=100,
        metadata={"help": "If the prompt is less than this number, we remove the sample instead."},
    )

    # completion_length_scheduler: Literal["constant", "increase_on_victory", "linear", "step"] = field(
    #     default="constant",
    #     metadata={
    #         "help": "The completion length scheduler to use. Defaults to 'constant', "
    #         "i.e. no change in completion length throughout training."
    #     },
    # )

    # default for template
    # _name_: Optional[str] = field(
    #     default=False,
    #     metadata={"help": ""},
    # )
