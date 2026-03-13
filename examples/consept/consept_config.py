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
            "help": "Maximum length of the prompt. If the prompt is longer than this value, it will be truncated left. "
            "We dynamically set the `max_prompt_length` following (`max_length` - `current_completion_length`) so setting this is useless "
            "(Defaults to the maximum 64-bit unsigned integer value, to be overriden during training)."
        },
    )

    max_length: int = field(
        # The name is taken from SFTTrainer, out of the possible choices:
        #   - tokenizer's `model_max_len`,
        #   - vLLM's `max_model_len` (seen in GRPOTrainer)
        default=4096,
        metadata={
            "help": "When `current_completion_length` changes, we will use this value to dynamically set `max_prompt_length`. "
            "Ensuring: `max_prompt_length` + `current_completion_length` = `max_length` (Defaults to 4096)."
        },
    )

    # Parameters that control the dynamic completion length
    initial_completion_length: Optional[int] = field(
        default=8,
        metadata={"help": "At step 0, this is the number of tokens that the model will try to generate."},
    )

    min_prompt_length: Optional[int] = field(
        default=100,
        metadata={"help": "If the prompt length (in tokens) is less than this number, we remove the sample instead."},
    )

    # default for template
    # _name_: Optional[str] = field(
    #     default=False,
    #     metadata={"help": ""},
    # )

    def __post_init__(self):
        super().__post_init__()
        if self.max_length <= 0:
            raise ValueError("'max_length' must be a non-negative integer.")

        if self.initial_completion_length <= 0:
            raise ValueError("'initial_completion_length' must be a non-negative integer.")

        if self.min_prompt_length <= 0:
            raise ValueError("'min_prompt_length' must be a non-negative integer.")

        if self.max_completion_length < self.initial_completion_length:
            raise ValueError("'max_completion_length' must be at least 'initial_completion_length'.")
        if self.max_completion_length + self.min_prompt_length > self.max_length:
            raise ValueError(
                "'max_completion_length' + 'min_prompt_length' must be at most 'max_length'. "
                "Else, prompts will (eventually) all be pruned."
            )
