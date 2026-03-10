import json
import os
from typing import Any


def save_dict_to_json(dictionary: dict[str, Any], json_path: str) -> None:
    """Save the content of dictionary in JSON format inside `json_path`."""
    try:
        json_string = json.dumps(dictionary, indent=2, sort_keys=True) + "\n"
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(json_string)
    except (OSError, TypeError) as e:
        raise OSError(f"Failed to save JSON to {json_path}: {e}")


def load_dict_from_json(json_path: str) -> dict[str, Any]:
    """Load JSON content from `json_path` and return as dictionary."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    try:
        with open(json_path, encoding="utf-8") as f:
            return json.load(f)  # json.load() is more efficient than read+loads
    except (OSError, json.JSONDecodeError) as e:
        raise ValueError(f"Failed to load JSON from {json_path}: {e}")
