import random
from typing import Optional

from transformers.utils import (
    is_rich_available,
)


if is_rich_available():
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text


def print_prompt_text_completions_sample(
    prompts: list,
    completions: list,
    rewards: dict[str, list[float]],
    advantages: list[float],
    step: int,
    eos_token: str,
    num_samples: Optional[int] = None,
) -> None:
    """
    Print out a sample of model completions to the console with multiple reward metrics.

    This function creates a nicely formatted table showing prompt-completion pairs, useful for monitoring model outputs
    during training. It requires the `rich` library to be installed.

    Args:
        prompts (`list[str]`):
            List of prompts.
        completions (`list`):
            List of completions corresponding to the prompts.
        rewards (`dict[str, list[float]]`):
            Dictionary where keys are reward names and values are lists of rewards.
        advantages (`list[float]`):
            List of advantages corresponding to the prompts and completions.
        step (`int`):
            Current training step number, used in the output title.
        eos_token (`str`):
            The tokenizer's end-of-sentence token, to not print empty samples
        num_samples (`int`, *optional*):
            Number of random samples to display. If `None` (default), all items will be displayed.

    Example:
    ```python
    >>> from trl.trainer.utils import print_prompt_completions_sample

    >>> prompts = ["The sky is", "The sun is"]
    >>> completions = [" blue.", " in the sky."]
    >>> rewards = {"Correctness": [0.123, 0.456], "Format": [0.789, 0.101]}
    >>> advantages = [0.987, 0.654]
    >>> print_prompt_completions_sample(prompts, completions, rewards, advantages, 42)
    ╭──────────────────────────── Step 42 ─────────────────────────────╮
    │ ┏━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━┓ │
    │ ┃ Prompt     ┃ Completion   ┃ Correctness ┃ Format ┃ Advantage ┃ │
    │ ┡━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━┩ │
    │ │ The sky is │  blue.       │        0.12 │   0.79 │      0.99 │ │
    │ ├────────────┼──────────────┼─────────────┼────────┼───────────┤ │
    │ │ The sun is │  in the sky. │        0.46 │   0.10 │      0.65 │ │
    │ └────────────┴──────────────┴─────────────┴────────┴───────────┘ │
    ╰──────────────────────────────────────────────────────────────────╯
    ```
    """
    if not is_rich_available():
        raise ImportError(
            "The function `print_prompt_text_completions_sample` requires the `rich` library. Please install it with "
            "`pip install rich`."
        )

    # filter empty prompts
    valid_prompt_indices = [i for i, prompt in enumerate(prompts) if prompt != eos_token]
    prompts = [prompts[i] for i in valid_prompt_indices]
    completions = [completions[i] for i in valid_prompt_indices]
    rewards = {key: [val[i] for i in valid_prompt_indices] for key, val in rewards.items()}
    advantages = [advantages[i] for i in valid_prompt_indices]

    # === Begin main code === #
    console = Console()
    table = Table(show_header=True, header_style="bold white", expand=True)

    if len(prompts) == 0:
        console.print("This sample has only empty prompts")
        return

    # Add columns
    table.add_column("Prompt", style="bright_yellow")
    table.add_column("Completion", style="bright_green")
    for reward_name in rewards.keys():
        table.add_column(reward_name, style="bold cyan", justify="right")
    table.add_column("Advantage", style="bold magenta", justify="right")

    def format_entry(entry) -> Text:
        t = Text()
        if isinstance(entry, list) and all(isinstance(m, dict) for m in entry):
            for j, msg in enumerate(entry):
                role = msg.get("role", "")
                if "content" in msg:
                    # Chat message
                    t.append(f"{role.upper()}\n", style="bold red")
                    t.append(msg["content"])
                elif "name" in msg and "args" in msg:
                    # Tool call
                    t.append(f"{role.upper()}\n", style="bold red")
                    t.append(f"{msg['name']}({msg['args']})")
                else:
                    # Fallback
                    t.append(str(msg))
                if j < len(entry) - 1:
                    t.append("\n\n")
        else:
            t.append(str(entry))
        return t

    # Some basic input validation
    if num_samples is not None:
        if num_samples >= len(prompts):
            num_samples = None
        elif num_samples <= 0:
            return

    # Subsample data if num_samples is specified
    if num_samples is not None:
        indices = random.sample(range(len(prompts)), num_samples)
        prompts = [prompts[i] for i in indices]
        completions = [completions[i] for i in indices]
        rewards = {key: [val[i] for i in indices] for key, val in rewards.items()}
        advantages = [advantages[i] for i in indices]

    for i in range(len(prompts)):
        reward_values = [f"{rewards[key][i]:.2f}" for key in rewards.keys()]  # 2 decimals
        table.add_row(
            format_entry(prompts[i]),
            format_entry(completions[i]),
            *reward_values,
            f"{advantages[i]:.2f}",
        )
        table.add_section()  # Adds a separator between rows

    panel = Panel(table, expand=False, title=f"Step {step}", border_style="bold white")
    console.print(panel)
