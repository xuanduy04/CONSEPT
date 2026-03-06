from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from multiprocessing.sharedctypes import Synchronized

    from transformers import PreTrainedTokenizerBase


class PromptSolutionCollator:
    """Wrap the data collator to create "prompt" and "solution" columns from raw "text"."""

    def __init__(
        self,
        data_collator,
        tokenizer: "PreTrainedTokenizerBase",
        completion_length: "Synchronized[int]",
        prompt_length_remove_threshold: int,
    ):
        self.data_collator = data_collator
        self.tokenizer = tokenizer
        self.completion_length = completion_length
        self.prompt_length_remove_threshold = prompt_length_remove_threshold

    def _make_prompt(self, feature: dict) -> dict:
        if not isinstance(feature, dict):
            return feature

        if "text" not in feature:
            raise ValueError("Could not find raw text column")

        tokens = self.tokenizer.encode(feature["text"])
        feature["prompt"] = self.tokenizer.decode(tokens[: len(tokens) - self.completion_length.value])
        feature["solution"] = self.tokenizer.decode(tokens[len(tokens) - self.completion_length.value :])

        return feature

    def __call__(self, features: list[dict]):
        return self.data_collator([self._make_prompt(feature) for feature in features])


if __name__ == "__main__":
    import multiprocessing
    from pprint import pprint

    from datasets import load_dataset
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    from trl.trainer.utils import identity

    from .sampler import DynamicRepeatSampler

    prompt_length_remove_threshold = 100
    completion_length = multiprocessing.Value("i", 1000)
    processing_class = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")

    def valid_item_fn(item: str) -> bool:
        tokens = processing_class.encode(item)
        return len(tokens) - completion_length.value <= prompt_length_remove_threshold

    data_collator = identity
    data_collator = PromptSolutionCollator(
        data_collator, processing_class, completion_length, prompt_length_remove_threshold
    )

    train_dataset = load_dataset("HuggingFaceTB/cosmopedia", name="openstax", split="train")
    train_dataset = train_dataset.remove_columns([c for c in train_dataset.column_names if c != "text"])

    data_loader = DataLoader(
        train_dataset,
        sampler=DynamicRepeatSampler(train_dataset, valid_item_fn=valid_item_fn),
        collate_fn=data_collator,
    )

    for i, data in enumerate(data_loader):
        tokenized_length_data = dict()
        for k, v in data[0].items():
            tokenized_length_data[k] = len(processing_class.encode(v))

        pprint(tokenized_length_data)

        if i > 10:
            break

    print(f"{processing_class.eos_token=}")
