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
        prompt_length_remove_threshold: int,
    ):
        self.data_collator = data_collator
        self.tokenizer = tokenizer
        self.completion_length = completion_length
        self.prompt_length_remove_threshold = prompt_length_remove_threshold

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
        if len(trimmed_tokens) <= self.prompt_length_remove_threshold:
            feature["prompt"] = self.tokenizer.eos_token
        else:
            feature["prompt"] = self.tokenizer.decode(trimmed_tokens)

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

    prompt_length_remove_threshold = 100
    completion_length = multiprocessing.Value("i", 1000)
    processing_class = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")

    data_collator = identity
    data_collator = GetPromptCollator(data_collator, processing_class, completion_length, prompt_length_remove_threshold)

    train_dataset = load_dataset(
        "nvidia/Nemotron-Pretraining-Dataset-sample", name="Nemotron-CC-High-Quality", split="train[:5%]"
    ).rename_column("text", "solution").remove_columns("id")
    data_loader = DataLoader(train_dataset, collate_fn=data_collator)

    for i, data in enumerate(data_loader):
        tokenized_length_data = dict()
        for k,v in data[0].items():
            tokenized_length_data[k] = len(processing_class.encode(v)) if v != processing_class.eos_token else v

        pprint(tokenized_length_data)

        if i > 5:
            break

    print(f"{processing_class.eos_token=}")
