"""
Modern Image Colorization Module

This module provides a modern, type-safe implementation of image colorization
using state-of-the-art deep learning models from Hugging Face.
"""

import logging
from pathlib import Path
from typing import Optional, Union, Tuple, List
import warnings

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from transformers import pipeline, AutoImageProcessor, AutoModelForImageToImage
import requests
from io import BytesIO

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageColorizer:
    """
    A modern image colorization class using Hugging Face transformers.
    
    This class provides methods to colorize grayscale images using pre-trained
    deep learning models. It supports multiple model backends and includes
    error handling, logging, and type safety.
    """
    
    def __init__(
        self, 
        model_name: str = "timbrooks/instruct-pix2pix",
        device: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize the ImageColorizer.
        
        Args:
            model_name: Name of the Hugging Face model to use for colorization
            device: Device to run inference on ('cpu', 'cuda', 'mps', or None for auto)
            cache_dir: Directory to cache downloaded models
        """
        self.model_name = model_name
        self.device = self._get_device(device)
        self.cache_dir = cache_dir
        
        # Initialize model and processor
        self._model = None
        self._processor = None
        self._pipeline = None
        
        logger.info(f"Initializing ImageColorizer with model: {model_name}")
        self._load_model()
    
    def _get_device(self, device: Optional[str]) -> str:
        """Determine the best available device."""
        if device is not None:
            return device
        
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def _load_model(self) -> None:
        """Load the colorization model and processor."""
        try:
            logger.info(f"Loading model {self.model_name} on device {self.device}")
            
            # Try to load as a pipeline first (simpler approach)
            try:
                self._pipeline = pipeline(
                    "image-to-image",
                    model=self.model_name,
                    device=self.device,
                    cache_dir=self.cache_dir
                )
                logger.info("Successfully loaded model as pipeline")
            except Exception as e:
                logger.warning(f"Pipeline loading failed: {e}")
                # Fallback to manual loading
                self._processor = AutoImageProcessor.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir
                )
                self._model = AutoModelForImageToImage.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir
                ).to(self.device)
                logger.info("Successfully loaded model manually")
                
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Could not load model {self.model_name}: {e}")
    
    def load_image(self, image_path: Union[str, Path]) -> Image.Image:
        """
        Load an image from file path or URL.
        
        Args:
            image_path: Path to image file or URL
            
        Returns:
            PIL Image object
        """
        image_path = str(image_path)
        
        try:
            if image_path.startswith(('http://', 'https://')):
                logger.info(f"Loading image from URL: {image_path}")
                response = requests.get(image_path, timeout=30)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content))
            else:
                logger.info(f"Loading image from file: {image_path}")
                image = Image.open(image_path)
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            logger.info(f"Successfully loaded image: {image.size}")
            return image
            
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            raise ValueError(f"Could not load image from {image_path}: {e}")
    
    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Preprocess image for colorization.
        
        Args:
            image: Input PIL Image
            
        Returns:
            Preprocessed PIL Image
        """
        # Convert to grayscale if not already
        if image.mode != 'L':
            image = image.convert('L')
        
        # Convert back to RGB (3-channel grayscale)
        image = image.convert('RGB')
        
        logger.info(f"Preprocessed image: {image.size}")
        return image
    
    def colorize(
        self, 
        image: Union[str, Path, Image.Image],
        prompt: str = "colorize this image",
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5
    ) -> Image.Image:
        """
        Colorize a grayscale image.
        
        Args:
            image: Input image (path, URL, or PIL Image)
            prompt: Text prompt for colorization
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale for generation
            
        Returns:
            Colorized PIL Image
        """
        try:
            # Load and preprocess image
            if isinstance(image, (str, Path)):
                image = self.load_image(image)
            
            processed_image = self.preprocess_image(image)
            
            logger.info("Starting colorization process")
            
            if self._pipeline is not None:
                # Use pipeline approach
                result = self._pipeline(
                    processed_image,
                    prompt=prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale
                )
                
                if isinstance(result, list) and len(result) > 0:
                    colorized_image = result[0]['image']
                else:
                    colorized_image = result['image']
            else:
                # Use manual model approach
                inputs = self._processor(
                    processed_image,
                    prompt,
                    return_tensors="pt"
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self._model.generate(
                        **inputs,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale
                    )
                
                colorized_image = self._processor.decode(outputs[0])
            
            logger.info("Colorization completed successfully")
            return colorized_image
            
        except Exception as e:
            logger.error(f"Colorization failed: {e}")
            raise RuntimeError(f"Colorization failed: {e}")
    
    def batch_colorize(
        self, 
        images: List[Union[str, Path, Image.Image]],
        prompt: str = "colorize this image",
        **kwargs
    ) -> List[Image.Image]:
        """
        Colorize multiple images in batch.
        
        Args:
            images: List of input images
            prompt: Text prompt for colorization
            **kwargs: Additional arguments for colorize method
            
        Returns:
            List of colorized PIL Images
        """
        logger.info(f"Starting batch colorization of {len(images)} images")
        results = []
        
        for i, image in enumerate(images):
            try:
                logger.info(f"Processing image {i+1}/{len(images)}")
                result = self.colorize(image, prompt, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to colorize image {i+1}: {e}")
                results.append(None)
        
        logger.info(f"Batch colorization completed: {len([r for r in results if r is not None])} successful")
        return results
    
    def visualize_comparison(
        self, 
        original: Image.Image, 
        colorized: Image.Image,
        save_path: Optional[Union[str, Path]] = None,
        figsize: Tuple[int, int] = (12, 6)
    ) -> None:
        """
        Create a side-by-side comparison visualization.
        
        Args:
            original: Original grayscale image
            colorized: Colorized image
            save_path: Optional path to save the comparison
            figsize: Figure size for the plot
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Original image
        axes[0].imshow(original, cmap='gray' if original.mode == 'L' else None)
        axes[0].set_title("Original Grayscale", fontsize=14)
        axes[0].axis('off')
        
        # Colorized image
        axes[1].imshow(colorized)
        axes[1].set_title("Colorized Output", fontsize=14)
        axes[1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Comparison saved to: {save_path}")
        
        plt.show()
    
    def save_image(self, image: Image.Image, path: Union[str, Path]) -> None:
        """
        Save image to file.
        
        Args:
            image: PIL Image to save
            path: Output file path
        """
        try:
            image.save(path)
            logger.info(f"Image saved to: {path}")
        except Exception as e:
            logger.error(f"Failed to save image to {path}: {e}")
            raise RuntimeError(f"Could not save image to {path}: {e}")


def create_sample_images(output_dir: Union[str, Path]) -> List[Path]:
    """
    Create sample grayscale images for testing.
    
    Args:
        output_dir: Directory to save sample images
        
    Returns:
        List of paths to created sample images
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a simple synthetic grayscale image
    size = (256, 256)
    image = Image.new('L', size, 128)  # Gray background
    
    # Add some simple shapes
    from PIL import ImageDraw
    draw = ImageDraw.Draw(image)
    
    # Draw a circle
    draw.ellipse([50, 50, 150, 150], fill=255)
    
    # Draw a rectangle
    draw.rectangle([100, 100, 200, 200], fill=64)
    
    # Draw some lines
    for i in range(0, 256, 20):
        draw.line([(i, 0), (i, 256)], fill=192)
    
    sample_path = output_dir / "sample_grayscale.png"
    image.save(sample_path)
    
    logger.info(f"Created sample image: {sample_path}")
    return [sample_path]


if __name__ == "__main__":
    # Example usage
    colorizer = ImageColorizer()
    
    # Create sample data
    sample_images = create_sample_images("data/samples")
    
    # Colorize sample image
    for image_path in sample_images:
        try:
            original = colorizer.load_image(image_path)
            colorized = colorizer.colorize(image_path)
            colorizer.visualize_comparison(original, colorized)
        except Exception as e:
            logger.error(f"Example failed: {e}")
