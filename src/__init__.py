"""
Image Colorization Package

A modern, production-ready image colorization application using
state-of-the-art deep learning models.
"""

__version__ = "1.0.0"
__author__ = "AI Projects"
__email__ = "ai@projects.com"

from .image_colorizer import ImageColorizer, create_sample_images
from .config import ConfigManager, AppConfig, config

__all__ = [
    "ImageColorizer",
    "create_sample_images", 
    "ConfigManager",
    "AppConfig",
    "config"
]
