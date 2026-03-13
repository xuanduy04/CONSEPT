import os
from pathlib import Path


def validate_accelerator_config():
    """
    Validate that the accelerate configuration matches the available GPU count.
    """
    # 1. ACCELERATE_CONFIG_FILE must be set
    config_path = os.environ.get("ACCELERATE_CONFIG_FILE")
    assert config_path is not None, (
        "ACCELERATE_CONFIG_FILE is not set. Please call `accelerate launch` "
        "with one of the config files in 'examples/accelerate_configs'."
    )

    config_file = Path(config_path)

    # 2. The config file must reside in 'examples/accelerate_configs'
    #    (allow both absolute and relative paths)
    expected_dir = Path("accelerate_configs")
    assert expected_dir in config_file.parents, (
        f"Config file {config_file} is not inside the expected directory 'examples/accelerate_configs'."
    )

    # 3. Extract the first character of the filename stem (e.g., '2' from '2gpu.yaml')
    stem = config_file.stem
    assert stem, "Config filename stem is empty."
    first_char = stem[0]
    assert first_char.isdigit(), (
        f"First character of config filename '{stem}' is not a digit."
        "Expected a digit indicating the number of GPUs."
    )
    expected_gpu_count = int(first_char)

    # 4. Count the GPUs visible via CUDA_VISIBLE_DEVICES
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    assert cuda_visible_devices is not None, "`CUDA_VISIBLE_DEVICES` is not set."
    # Remove empty strings in case of trailing commas, etc.
    visible_gpu_count = len([d for d in cuda_visible_devices.split(",") if d.strip()])

    # 5. Assert they match
    assert visible_gpu_count == expected_gpu_count, (
        f"GPU count mismatch: expected {expected_gpu_count} from config filename, "
        f"but `CUDA_VISIBLE_DEVICES` shows {visible_gpu_count} GPU(s) (CUDA_VISIBLE_DEVICES='{cuda_visible_devices}')."
    )
