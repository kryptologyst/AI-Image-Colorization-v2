#!/usr/bin/env python3
"""
Example script demonstrating image colorization usage.

This script shows how to use the ImageColorizer class programmatically.
"""

import logging
from pathlib import Path
from src.image_colorizer import ImageColorizer, create_sample_images

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main example function."""
    logger.info("Starting image colorization example")
    
    try:
        # Initialize colorizer
        logger.info("Initializing colorizer...")
        colorizer = ImageColorizer()
        
        # Create sample data if it doesn't exist
        data_dir = Path("data/samples")
        data_dir.mkdir(parents=True, exist_ok=True)
        
        sample_files = list(data_dir.glob("*.png"))
        if not sample_files:
            logger.info("Creating sample images...")
            sample_files = create_sample_images(data_dir)
        
        # Process each sample image
        for sample_file in sample_files:
            logger.info(f"Processing: {sample_file}")
            
            try:
                # Load original image
                original = colorizer.load_image(sample_file)
                
                # Colorize the image
                colorized = colorizer.colorize(
                    sample_file,
                    prompt="colorize this image with realistic colors"
                )
                
                # Save result
                output_dir = Path("outputs")
                output_dir.mkdir(parents=True, exist_ok=True)
                
                output_path = output_dir / f"colorized_{sample_file.name}"
                colorizer.save_image(colorized, output_path)
                
                logger.info(f"Successfully colorized: {output_path}")
                
                # Show comparison (if matplotlib is available)
                try:
                    colorizer.visualize_comparison(original, colorized)
                except Exception as e:
                    logger.warning(f"Could not display comparison: {e}")
                
            except Exception as e:
                logger.error(f"Failed to process {sample_file}: {e}")
        
        logger.info("Example completed successfully!")
        
    except Exception as e:
        logger.error(f"Example failed: {e}")
        raise


if __name__ == "__main__":
    main()
