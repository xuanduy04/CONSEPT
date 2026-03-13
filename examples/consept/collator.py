from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from multiprocessing.sharedctypes import Synchronized

    from transformers import PreTrainedTokenizerBase


class PromptSolutionCollator:
    """Wraps the data collator to create "prompt" and "solution" columns from raw "text"."""

    def __init__(
        self,
        data_collator,
        tokenizer: "PreTrainedTokenizerBase",
        completion_length: "Synchronized[int]",
        max_length: int,
    ):
        self.data_collator = data_collator
        self.tokenizer = tokenizer
        self.completion_length = completion_length
        self.max_length = max_length

    def _make_prompt(self, feature: dict) -> dict:
        if not isinstance(feature, dict):
            return feature
        if "text" not in feature:
            raise ValueError("Could not find raw text column")

        completion_length = self.completion_length.value

        tokens = self.tokenizer.encode(
            feature["text"],
            # This ensures that total tokens will not exceed max_length
            # Note that the min_prompt_length requirement is assured via the sampler-level, not here.
            max_length=self.max_length,
            truncation=True,
        )
        # Maybe.. # Snap to nearest whitespace to avoid partial BPE word fragments?
        # prompt_text = self.tokenizer.decode(tokens[:-completion_length])
        # first_space = prompt_text.find(" ")
        # if first_space != -1 and not prompt_text[0].isspace():
        #     prompt_text = prompt_text[first_space:]
        # # then set "prompt": prompt_text

        return {
            "prompt": self.tokenizer.decode(tokens[:-completion_length]),
            "solution": self.tokenizer.decode(tokens[-completion_length:]),
        }  # I decided to remove the full 'text' column as it is quite a lot of text.
        # GRPO also natively duplicates all columns too, so it might be a significant amount of memory

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

    min_prompt_length = 100
    completion_length = multiprocessing.Value("i", 1000)
    processing_class = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")

    def valid_item_fn(item: str) -> bool:
        tokens = processing_class.encode(item)
        return len(tokens) - completion_length.value <= min_prompt_length

    data_collator = identity
    data_collator = PromptSolutionCollator(data_collator, processing_class, completion_length)

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
