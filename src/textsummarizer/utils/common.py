import os
from box.exceptions import BoxValueError
import yaml
from textsummarizer.logging import logger
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    Reads a YAML file and returns 

    Args:
        path_to_yaml (str): path like input
    Raises:
        ValueError: If the YAML file is empty or cannot be parsed.
        e : empty file
    Returns:
        ConfigBox: A ConfigBox type
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"YAML file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError :
        raise ValueError(f"YAML file is empty")
    except Exception as e:
        raise e
    