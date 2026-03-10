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
    """

    repo_id: Optional[str] = field(
        default=None, metadata={"help": "Hugging Face repository ID to push the dataset to."}
    )
    dataset_num_proc: Optional[int] = field(
        default=None, metadata={"help": "Number of workers to use for dataset processing."}
    )
    streaming: bool = field(default=False, metadata={"help": "Whether to load the dataset in streaming mode."})
    save_to_local: Optional[str] = field(
        default=None,
        metadata={"help": "Local path to save the dataset to."},
    )


def filter_metadata(example):
    """Keep only the 'text' column from an example."""
    return {"text": example["text"]}


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    # Load the dataset, optionally in streaming mode
    dataset = load_dataset(
        "havisdino/nem_meo_dataset",
        split="train",
        streaming=script_args.streaming,
        num_proc=script_args.dataset_num_proc if not script_args.streaming else None,
    )

    # Keep only the 'text' column
    dataset = dataset.remove_columns([c for c in dataset.column_names if c != "text"])

    if script_args.save_to_local:
        # Save the dataset to a local directory in Arrow format
        dataset.save_to_disk(script_args.save_to_local)
        print(f"Dataset saved locally to: {script_args.save_to_local}")
