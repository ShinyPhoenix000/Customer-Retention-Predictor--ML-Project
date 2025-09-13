"""
Utility functions for the Customer Retention Predictor project.
This module contains helper functions used across the project.
"""

import logging
import yaml
from pathlib import Path
from typing import Dict, Any
import json
import datetime


def setup_logging(log_path: str = "logs/training.log") -> None:
    """
    Set up logging configuration for the project.

    Args:
        log_path (str): Path to the log file
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        filename=str(log_path),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filemode='a'
    )
    logging.info("Logging is set up.")


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.

    Args:
        config_path (str): Path to the configuration file

    Returns:
        Dict[str, Any]: Configuration parameters

    Raises:
        FileNotFoundError: If the config file doesn't exist
        yaml.YAMLError: If the config file is not valid YAML
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        if config is None:
            raise ValueError(f"Empty configuration file: {config_path}")

        # Provide default values if keys are missing
        defaults = {
            'results_dir': 'results/',
            'data_path': 'data/',
            'model_path': 'models/',
            'logs_dir': 'logs/',
            'random_state': 42
        }

        for key, default_value in defaults.items():
            if key not in config:
                config[key] = default_value

        return config

    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML configuration file: {e}")


def save_metrics(metrics: Dict[str, float], save_path: str) -> None:
    """
    Save model metrics to a JSON file.

    Args:
        metrics (Dict[str, float]): Dictionary of metric names and values
        save_path (str): Path to save the metrics
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    logging.info(f"Metrics saved to {save_path}")


def create_experiment_folder(base_dir: str) -> Path:
    """
    Creates a timestamped experiment folder under the base_dir and returns its Path.

    Args:
        base_dir (str): Base directory where experiment folders are created

    Returns:
        Path: Path object of the created experiment folder
    """
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = base_path / f"experiment_{timestamp}"
    experiment_dir.mkdir(exist_ok=True)

    logging.info(f"Experiment folder created at {experiment_dir}")
    return experiment_dir