from typing import TYPE_CHECKING

import torch


if TYPE_CHECKING:
    from transformers.utils import is_accelerate_available

    if is_accelerate_available():
        from accelerate import Accelerator


def validate_accelerator_config(accelerator: "Accelerator"):
    """
    Validate that the accelerate configuration matches the available GPU count.
    """
    assert accelerator.num_processes == torch.cuda.device_count(), (
        f"GPU count mismatch: expected {accelerator.num_processes} from accelerator, "
        f"but `CUDA_VISIBLE_DEVICES` shows {torch.cuda.device_count()} GPU(s) (CUDA_VISIBLE_DEVICES='{os.environ.get('CUDA_VISIBLE_DEVICES')}')."
    )
