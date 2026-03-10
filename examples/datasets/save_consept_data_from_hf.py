import math
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset
from transformers import HfArgumentParser


@dataclass
class ScriptArguments:
    r"""
    Arguments for the script.

    Args:
        repo_id (`str`, *optional*, defaults to `"trl-lib/hh-rlhf-helpful-base"`):
            Hugging Face repository ID to push the dataset to.
        dataset_num_proc (`int`, *optional*):
            Number of workers to use for dataset processing.
        streaming (`bool`, *optional*, defaults to `False`):
            Whether to load the dataset in streaming mode.
        save_to_local (`str`, *optional*):
            Local path to save the dataset to. If not provided, the dataset will be pushed to the Hub.
        percent (`float`, *optional*):
            Percentage of the dataset files to load (e.g. 0.1 for 10%). Mutually exclusive with `num_samples`.
        num_samples (`int`, *optional*):
            Number of samples to load from the dataset. Mutually exclusive with `percent`.
        shuffle (`bool`, *optional*, defaults to `False`):
            Whether to shuffle before selecting the subset. Applies to both `percent` and `num_samples`.
        shuffle_seed (`int`, *optional*, defaults to `42`):
            Random seed to use when shuffling.
    """

    repo_id: str = field(
        default="trl-lib/hh-rlhf-helpful-base", metadata={"help": "Hugging Face repository ID to push the dataset to."}
    )
    dataset_num_proc: Optional[int] = field(
        default=None, metadata={"help": "Number of workers to use for dataset processing."}
    )
    streaming: bool = field(default=False, metadata={"help": "Whether to load the dataset in streaming mode."})
    save_to_local: Optional[str] = field(
        default=None,
        metadata={
            "help": "Local path to save the dataset as JSONL. If not provided, the dataset will be pushed to the Hub."
        },
    )
    percent: Optional[float] = field(
        default=None,
        metadata={
            "help": "Percentage of the dataset to load (e.g. 0.1 for 10%%). Mutually exclusive with num_samples."
        },
    )
    num_samples: Optional[int] = field(
        default=None, metadata={"help": "Number of samples to load. Mutually exclusive with percent."}
    )
    shuffle: bool = field(
        default=False,
        metadata={"help": "Whether to shuffle before selecting the subset. Applies to both percent and num_samples."},
    )
    shuffle_seed: int = field(default=42, metadata={"help": "Random seed to use when shuffling."})


def filter_metadata(example):
    """Keep only the 'text' column from an example."""
    return {"text": example["text"]}


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    if script_args.percent is not None and script_args.num_samples is not None:
        raise ValueError("percent and num_samples are mutually exclusive, please specify only one.")

    if script_args.percent is not None and not (0.0 < script_args.percent <= 1.0):
        raise ValueError("percent must be between 0 (exclusive) and 1 (inclusive).")

    # Build a split string to load only a percentage of the files if requested,
    # e.g. "train[:10%]" — this avoids downloading the full 20GB when not needed
    if script_args.percent is not None:
        pct = math.ceil(script_args.percent * 100)
        split = f"train[:{pct}%]"
    else:
        split = "train"

    dataset = load_dataset(
        "havisdino/nem_meo_dataset",
        split=split,
        streaming=script_args.streaming,
        num_proc=script_args.dataset_num_proc if not script_args.streaming else None,
    )

    # Keep only the 'text' column
    dataset = dataset.remove_columns([c for c in dataset.column_names if c != "text"])

    # Shuffle and/or select a fixed number of samples if requested
    if script_args.shuffle:
        dataset = dataset.shuffle(seed=script_args.shuffle_seed)

    if script_args.num_samples is not None:
        dataset = dataset.select(range(script_args.num_samples))

    if script_args.save_to_local:
        # Save the dataset locally as JSONL
        dataset.to_json(script_args.save_to_local, lines=True)
        print(f"Dataset saved locally to: {script_args.save_to_local}")
    else:
        print("Dataset loaded successfully (not saved) as `save_to_local` is `False`.")
