"""
Configuration module for Image Colorization project.

This module handles configuration management using YAML files
and environment variables.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Configuration for the colorization model."""
    name: str = "timbrooks/instruct-pix2pix"
    device: Optional[str] = None
    cache_dir: Optional[str] = None
    num_inference_steps: int = 20
    guidance_scale: float = 7.5
    prompt: str = "colorize this image"


@dataclass
class UIConfig:
    """Configuration for the user interface."""
    port: int = 8501
    host: str = "localhost"
    debug: bool = False
    theme: str = "light"


@dataclass
class LoggingConfig:
    """Configuration for logging."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: Optional[str] = None


@dataclass
class AppConfig:
    """Main application configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # Paths
    data_dir: str = "data"
    models_dir: str = "models"
    output_dir: str = "outputs"
    
    # Other settings
    max_image_size: int = 1024
    supported_formats: list = field(default_factory=lambda: [".jpg", ".jpeg", ".png", ".bmp", ".tiff"])


class ConfigManager:
    """Manages application configuration."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path or "config/config.yaml"
        self.config = self._load_config()
    
    def _load_config(self) -> AppConfig:
        """Load configuration from file and environment variables."""
        config_dict = self._load_from_file()
        config_dict = self._override_with_env(config_dict)
        return self._dict_to_config(config_dict)
    
    def _load_from_file(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        config_path = Path(self.config_path)
        
        if not config_path.exists():
            # Create default config file
            self._create_default_config(config_path)
        
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            print(f"Warning: Could not load config file {config_path}: {e}")
            return {}
    
    def _override_with_env(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Override configuration with environment variables."""
        env_mappings = {
            'MODEL_NAME': 'model.name',
            'MODEL_DEVICE': 'model.device',
            'UI_PORT': 'ui.port',
            'UI_HOST': 'ui.host',
            'LOG_LEVEL': 'logging.level',
            'DATA_DIR': 'data_dir',
            'OUTPUT_DIR': 'output_dir'
        }
        
        for env_var, config_path in env_mappings.items():
            if env_var in os.environ:
                self._set_nested_value(config_dict, config_path, os.environ[env_var])
        
        return config_dict
    
    def _set_nested_value(self, config_dict: Dict[str, Any], path: str, value: Any) -> None:
        """Set a nested value in the configuration dictionary."""
        keys = path.split('.')
        current = config_dict
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Convert string values to appropriate types
        if key == 'port':
            value = int(value)
        elif key == 'debug':
            value = value.lower() in ('true', '1', 'yes', 'on')
        elif key == 'num_inference_steps':
            value = int(value)
        elif key == 'guidance_scale':
            value = float(value)
        
        current[keys[-1]] = value
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> AppConfig:
        """Convert dictionary to AppConfig object."""
        model_config = ModelConfig(**config_dict.get('model', {}))
        ui_config = UIConfig(**config_dict.get('ui', {}))
        logging_config = LoggingConfig(**config_dict.get('logging', {}))
        
        return AppConfig(
            model=model_config,
            ui=ui_config,
            logging=logging_config,
            data_dir=config_dict.get('data_dir', 'data'),
            models_dir=config_dict.get('models_dir', 'models'),
            output_dir=config_dict.get('output_dir', 'outputs'),
            max_image_size=config_dict.get('max_image_size', 1024),
            supported_formats=config_dict.get('supported_formats', [".jpg", ".jpeg", ".png", ".bmp", ".tiff"])
        )
    
    def _create_default_config(self, config_path: Path) -> None:
        """Create a default configuration file."""
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        default_config = {
            'model': {
                'name': 'timbrooks/instruct-pix2pix',
                'device': None,
                'cache_dir': None,
                'num_inference_steps': 20,
                'guidance_scale': 7.5,
                'prompt': 'colorize this image'
            },
            'ui': {
                'port': 8501,
                'host': 'localhost',
                'debug': False,
                'theme': 'light'
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'file': None
            },
            'data_dir': 'data',
            'models_dir': 'models',
            'output_dir': 'outputs',
            'max_image_size': 1024,
            'supported_formats': ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False, indent=2)
        
        print(f"Created default configuration file: {config_path}")
    
    def get_config(self) -> AppConfig:
        """Get the current configuration."""
        return self.config
    
    def reload_config(self) -> None:
        """Reload configuration from file."""
        self.config = self._load_config()


# Global configuration instance
config_manager = ConfigManager()
config = config_manager.get_config()
