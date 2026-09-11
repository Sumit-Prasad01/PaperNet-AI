import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import torch
import yaml

from utils.custom_exception import CustomException
from utils.logger import logger


def read_yaml(path_to_yaml: Union[str, Path]) -> Dict[str, Any]:
    """Reads a yaml file and returns its content as a dictionary.

    Args:
        path_to_yaml: Path to the YAML file.

    Returns:
        dict: Parsed content of the YAML file.

    Raises:
        CustomException: If file loading or parsing fails.
    """
    try:
        path = Path(path_to_yaml)
        with open(path, "r", encoding="utf-8") as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"YAML file loaded successfully from: {path}")
            return content or {}
    except Exception as e:
        logger.error(f"Failed to read YAML file at: {path_to_yaml}")
        raise CustomException(e, sys)


def create_directories(
    path_to_directories: List[Union[str, Path]], verbose: bool = True
) -> None:
    """Creates a list of directories if they do not exist.

    Args:
        path_to_directories: List of directory paths.
        verbose: Whether to log directory creation.
    """
    try:
        for path in path_to_directories:
            os.makedirs(path, exist_ok=True)
            if verbose:
                logger.info(f"Created directory at: {path}")
    except Exception as e:
        logger.error("Failed to create directory")
        raise CustomException(e, sys)


def save_json(path: Union[str, Path], data: Dict[str, Any]) -> None:
    """Saves a dictionary into a JSON file.

    Args:
        path: Path to the output JSON file.
        data: Data to be serialized.
    """
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        logger.info(f"JSON file saved at: {p}")
    except Exception as e:
        logger.error(f"Failed to save JSON file at: {path}")
        raise CustomException(e, sys)


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    """Loads data from a JSON file.

    Args:
        path: Path to the JSON file.

    Returns:
        dict: Loaded JSON data.
    """
    try:
        p = Path(path)
        with open(p, "r", encoding="utf-8") as f:
            content = json.load(f)
        logger.info(f"JSON file loaded from: {p}")
        return content
    except Exception as e:
        logger.error(f"Failed to load JSON file at: {path}")
        raise CustomException(e, sys)


def set_seed(seed: int = 42) -> None:
    """Sets random seeds across random, numpy, and torch for reproducibility."""
    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info(f"Random seed set to {seed}")
    except Exception as e:
        logger.error("Failed to set seed")
        raise CustomException(e, sys)


def get_device(device_str: str = "auto") -> torch.device:
    """Resolves and returns the torch.device to use."""
    try:
        if device_str == "auto":
            selected_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            selected_device = torch.device(device_str)
        logger.info(f"Device selected: {selected_device}")
        return selected_device
    except Exception as e:
        logger.error(f"Failed to configure device: {device_str}")
        raise CustomException(e, sys)
