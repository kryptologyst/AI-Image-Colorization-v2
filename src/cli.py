"""
Command Line Interface for Image Colorization

This module provides a command-line interface for batch processing
and automation of image colorization tasks.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

from image_colorizer import ImageColorizer, create_sample_images
from config import config


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Setup logging configuration."""
    level = getattr(logging, log_level.upper())
    
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format=config.logging.format,
        handlers=handlers
    )


def colorize_single_image(
    input_path: str,
    output_path: str,
    colorizer: ImageColorizer,
    prompt: str = "colorize this image",
    **kwargs
) -> bool:
    """
    Colorize a single image.
    
    Args:
        input_path: Path to input image
        output_path: Path to save colorized image
        colorizer: ImageColorizer instance
        prompt: Colorization prompt
        **kwargs: Additional arguments for colorize method
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger = logging.getLogger(__name__)
        logger.info(f"Processing: {input_path}")
        
        # Colorize image
        colorized_image = colorizer.colorize(input_path, prompt=prompt, **kwargs)
        
        # Save result
        colorizer.save_image(colorized_image, output_path)
        
        logger.info(f"Successfully saved: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to process {input_path}: {e}")
        return False


def colorize_batch(
    input_dir: str,
    output_dir: str,
    colorizer: ImageColorizer,
    prompt: str = "colorize this image",
    **kwargs
) -> None:
    """
    Colorize all images in a directory.
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save colorized images
        colorizer: ImageColorizer instance
        prompt: Colorization prompt
        **kwargs: Additional arguments for colorize method
    """
    logger = logging.getLogger(__name__)
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all supported image files
    image_files = []
    for ext in config.supported_formats:
        image_files.extend(input_path.glob(f"*{ext}"))
        image_files.extend(input_path.glob(f"*{ext.upper()}"))
    
    if not image_files:
        logger.warning(f"No supported image files found in {input_dir}")
        return
    
    logger.info(f"Found {len(image_files)} images to process")
    
    # Process each image
    successful = 0
    failed = 0
    
    for image_file in image_files:
        output_file = output_path / f"colorized_{image_file.name}"
        
        if colorize_single_image(
            str(image_file),
            str(output_file),
            colorizer,
            prompt,
            **kwargs
        ):
            successful += 1
        else:
            failed += 1
    
    logger.info(f"Batch processing complete: {successful} successful, {failed} failed")


def create_samples(output_dir: str) -> None:
    """Create sample images for testing."""
    logger = logging.getLogger(__name__)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Creating sample images in {output_dir}")
    sample_files = create_sample_images(output_path)
    
    logger.info(f"Created {len(sample_files)} sample images")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="AI Image Colorization CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Colorize a single image
  python cli.py colorize input.jpg output.jpg
  
  # Colorize all images in a directory
  python cli.py batch input_dir/ output_dir/
  
  # Create sample images
  python cli.py samples data/samples/
  
  # Use custom settings
  python cli.py colorize input.jpg output.jpg --prompt "vibrant colors" --steps 30
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Colorize single image command
    colorize_parser = subparsers.add_parser('colorize', help='Colorize a single image')
    colorize_parser.add_argument('input', help='Input image path')
    colorize_parser.add_argument('output', help='Output image path')
    colorize_parser.add_argument('--prompt', default=config.model.prompt, help='Colorization prompt')
    colorize_parser.add_argument('--steps', type=int, default=config.model.num_inference_steps, help='Number of inference steps')
    colorize_parser.add_argument('--guidance', type=float, default=config.model.guidance_scale, help='Guidance scale')
    
    # Batch colorize command
    batch_parser = subparsers.add_parser('batch', help='Colorize all images in a directory')
    batch_parser.add_argument('input_dir', help='Input directory path')
    batch_parser.add_argument('output_dir', help='Output directory path')
    batch_parser.add_argument('--prompt', default=config.model.prompt, help='Colorization prompt')
    batch_parser.add_argument('--steps', type=int, default=config.model.num_inference_steps, help='Number of inference steps')
    batch_parser.add_argument('--guidance', type=float, default=config.model.guidance_scale, help='Guidance scale')
    
    # Create samples command
    samples_parser = subparsers.add_parser('samples', help='Create sample images')
    samples_parser.add_argument('output_dir', help='Output directory for sample images')
    
    # Global options
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='Logging level')
    parser.add_argument('--log-file', help='Log file path')
    parser.add_argument('--model', default=config.model.name, help='Model name')
    parser.add_argument('--device', help='Device to use (cpu, cuda, mps)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    logger = logging.getLogger(__name__)
    
    try:
        if args.command == 'samples':
            create_samples(args.output_dir)
            return
        
        # Initialize colorizer for colorization commands
        logger.info(f"Initializing colorizer with model: {args.model}")
        colorizer = ImageColorizer(
            model_name=args.model,
            device=args.device,
            cache_dir=config.model.cache_dir
        )
        
        # Prepare colorization arguments
        colorize_kwargs = {
            'num_inference_steps': args.steps,
            'guidance_scale': args.guidance
        }
        
        if args.command == 'colorize':
            success = colorize_single_image(
                args.input,
                args.output,
                colorizer,
                args.prompt,
                **colorize_kwargs
            )
            sys.exit(0 if success else 1)
        
        elif args.command == 'batch':
            colorize_batch(
                args.input_dir,
                args.output_dir,
                colorizer,
                args.prompt,
                **colorize_kwargs
            )
    
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
