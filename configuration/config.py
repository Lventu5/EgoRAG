import yaml
import os
from pathlib import Path
from typing import Any, Dict


class Config:
    """
    Configuration object that loads settings from config.yaml.
    Provides dot notation access to config values.
    """
    def __init__(self, config_dict: Dict[str, Any]):
        self._config = config_dict
        # Convert nested dicts to Config objects for dot notation access
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)
    
    def __getitem__(self, key):
        """Allow dictionary-style access"""
        return self._config[key]
    
    def __contains__(self, key):
        return key in self._config
    
    def get(self, key, default=None):
        """Get with default value"""
        return self._config.get(key, default)
    
    def __repr__(self):
        return f"Config({self._config})"

    def to_dict(self) -> Dict[str, Any]:
        """
        Recursively convert the configuration into a plain dict for serialization.
        """
        def _convert(value):
            if isinstance(value, Config):
                return value.to_dict()
            if isinstance(value, dict):
                return {k: _convert(v) for k, v in value.items()}
            return value

        return {k: _convert(v) for k, v in self._config.items()}


def load_config(config_path: str = None) -> Config:
    """
    Load configuration from YAML file.
    If config_path is None, looks for config.yaml in the configuration directory.
    """
    if config_path is None:
        # Get the directory of this file
        config_dir = Path(__file__).parent
        config_path = config_dir / "config.yaml"
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f) or {}
    
    return Config(config_dict)


CONFIG = load_config()


def save_config_snapshot(config: Config, output_path: str, extra_info: Dict[str, Any] | None = None) -> Path:
    """
    Persist the current configuration (and optional experiment metadata) to a text file.

    Args:
        config: Config instance to serialize.
        output_path: Destination path for the snapshot file.
        extra_info: Optional metadata (e.g., experiment name, modalities) to store alongside config.

    Returns:
        Path to the written file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {"config": config.to_dict()}
    if extra_info:
        payload["experiment"] = extra_info

    with output_path.open("w") as f:
        yaml.safe_dump(payload, f, default_flow_style=False)

    return output_path
