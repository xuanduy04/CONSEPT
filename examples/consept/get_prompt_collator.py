from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from multiprocessing.sharedctypes import Synchronized

    from transformers import PreTrainedTokenizerBase


class GetPromptCollator:
    """Wrap the data collator to remove "masked" portions, based on a variable"""

    def __init__(
        self,
        data_collator,
        tokenizer: "PreTrainedTokenizerBase",
        completion_length: "Synchronized[int]",
    ):
        self.data_collator = data_collator
        self.tokenizer = tokenizer
        self.completion_length = completion_length

    def _make_prompt(self, feature: dict) -> dict:
        if not isinstance(feature, dict):
            return feature

        if "prompt" in feature:
            text_column = "prompt"
        elif "solution" in feature:
            text_column = "solution"
        elif "text" in feature:
            text_column = "text"
        else:
            raise ValueError("Could not find raw text column")

        tokens = self.tokenizer.encode(feature[text_column])
        trimmed_tokens = tokens[: len(tokens) - min(self.completion_length.value, len(tokens))]
        feature["prompt"] = self.tokenizer.decode(trimmed_tokens)
        return feature

    def __call__(self, features: list[dict]):
        return self.data_collator([self._make_prompt(feature) for feature in features])
