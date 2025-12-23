from datetime import datetime
from pathlib import Path
from string import Template
from typing import Any, Tuple
import json
import logging
import yaml

logger = logging.getLogger(__name__)


def get_today_str() -> str:
    """Get current date in a human-readable format."""
    return datetime.now().strftime("%a %b %d, %Y").replace(" 0", " ")


def load_prompt(path: str | Path, **vars: Any) -> str:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")

    text = path.read_text(encoding="utf-8")

    # Prefer Python's str.format for `{var}` placeholders, but fall back to
    # string.Template for `$var` placeholders for backward compatibility.
    try:
        return text.format(**vars)
    except KeyError:
        # If `{}`-style formatting fails due to missing keys, try $-style.
        template = Template(text)
        try:
            return template.substitute(**vars)
        except KeyError as e:
            raise ValueError(f"Missing variable {e} for prompt: {path.name}") from None
    except Exception as e:
        # Re-raise with context for debugging other formatting issues.
        raise ValueError(f"Error formatting prompt {path.name}: {e}") from e


def read_yaml(file_path: str) -> dict:
    """
    Reads a YAML file and returns its contents as a dictionary.

    Args:
        file_path (str): Path to the YAML file.

    Returns:
        dict: Parsed YAML content.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data
    except FileNotFoundError:
        logger.info(f"Error: File '{file_path}' not found.")
        return {}
    except yaml.YAMLError as e:
        logger.info(f"Error parsing YAML file: {e}")
        return {}
