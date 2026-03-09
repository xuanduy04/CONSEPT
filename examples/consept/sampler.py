from typing import TYPE_CHECKING, Any, Callable, Optional

import torch
from torch.utils.data import Sampler


if TYPE_CHECKING:
    from datasets import Dataset


class DynamicRepeatSampler(Sampler):
    """
    Sampler that repeats the indices of a dataset in a structured manner.
    This is *dynamic*, i.e. will only return samples deemed valid by a function.

    Args:
        data_source (`datasets.Dataset`):
            Dataset to sample from.
        valid_item_fn(`Callable[[Any], bool],`):
            Function that inputs an item and returns whether this item is valid for sampling
        mini_repeat_count (`int`):
            Number of times to repeat each index per batch.
        batch_size (`int`, *optional*, defaults to `1`):
            Number of unique indices per batch.
        repeat_count (`int`, *optional*, defaults to `1`):
            Number of times to repeat the full sampling process.
        shuffle (`bool`, *optional*, defaults to `True`):
            Whether to shuffle the dataset.
        seed (`int`, *optional*):
            Random seed for reproducibility (only affects this sampler).
        text_column (`str`, *optional*, defaults to 'text'):
            The `data_source`'s column to sample from (and check validity of).
    """

    def __init__(
        self,
        data_source: "Dataset",
        valid_item_fn: Callable[[Any], bool],
        mini_repeat_count: int = 1,
        batch_size: int = 1,
        repeat_count: int = 1,
        shuffle: bool = True,
        seed: Optional[int] = None,
        text_column: str = "text",
    ):
        self.data_source = data_source
        self.is_valid_item = valid_item_fn
        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)
        self.shuffle = shuffle
        self.seed = seed
        self.text_column = text_column

        if shuffle:
            self.generator = torch.Generator()  # Create a local random generator
            if seed is not None:
                self.generator.manual_seed(seed)

    def __iter__(self):
        if self.shuffle:
            # E.g., [2, 4, 3, 1, 0, 6, 5] (num_samples = 7)
            indexes = torch.randperm(self.num_samples, generator=self.generator).tolist()
        else:
            indexes = list(range(self.num_samples))

        batch = []
        for index in indexes:
            if self.is_valid_item(self.data_source[self.text_column][index]):
                batch.append(index)
            if len(batch) == self.batch_size:
                # Same as original RepeatSampler
                for _ in range(self.repeat_count):
                    for index in batch:
                        for _ in range(self.mini_repeat_count):
                            yield index
                # Reset batch
                batch = []

    def __len__(self):
        raise TypeError(
            "DynamicSampler cannot pre-determine the number of valid samples,"
            "as `valid_item_fn` can change (to be more and more restrictive) during runtime."
        )


if __name__ == "__main__":
    import multiprocessing
    from pprint import pprint

    from datasets import load_dataset
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    prompt_length_remove_threshold = 100
    completion_length = multiprocessing.Value("i", 1000)
    processing_class = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")

    def valid_item_fn(item: str) -> bool:
        tokens = processing_class.encode(item)
        return len(tokens) - completion_length.value <= prompt_length_remove_threshold

    train_dataset = load_dataset("HuggingFaceTB/cosmopedia", name="openstax", split="train")
    train_dataset = train_dataset.remove_columns([c for c in train_dataset.column_names if c != "text"])

    data_loader = DataLoader(train_dataset, sampler=DynamicRepeatSampler(train_dataset, valid_item_fn=valid_item_fn))

    for i, data in enumerate(data_loader):
        tokenized_length_data = dict()
        for k, v in data.items():
            tokenized_length_data[k] = len(processing_class.encode(v))

        pprint(tokenized_length_data)

        completion_length.value += 1
        if i > 10:
            break

    print(f"{processing_class.eos_token=}")
