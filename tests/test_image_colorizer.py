"""
Test suite for Image Colorization application.

This module contains unit tests for the core functionality.
"""

import pytest
import tempfile
import os
from pathlib import Path
from PIL import Image
from unittest.mock import Mock, patch

from src.image_colorizer import ImageColorizer, create_sample_images
from src.config import ConfigManager, AppConfig


class TestImageColorizer:
    """Test cases for ImageColorizer class."""
    
    def test_init(self):
        """Test ImageColorizer initialization."""
        with patch('src.image_colorizer.pipeline') as mock_pipeline:
            mock_pipeline.return_value = Mock()
            colorizer = ImageColorizer()
            assert colorizer.model_name == "timbrooks/instruct-pix2pix"
            assert colorizer.device in ["cpu", "cuda", "mps"]
    
    def test_get_device(self):
        """Test device detection."""
        with patch('src.image_colorizer.pipeline') as mock_pipeline:
            mock_pipeline.return_value = Mock()
            
            # Test explicit device
            colorizer = ImageColorizer(device="cpu")
            assert colorizer.device == "cpu"
            
            # Test auto-detection
            colorizer = ImageColorizer(device=None)
            assert colorizer.device in ["cpu", "cuda", "mps"]
    
    def test_load_image_from_file(self):
        """Test loading image from file."""
        with patch('src.image_colorizer.pipeline') as mock_pipeline:
            mock_pipeline.return_value = Mock()
            colorizer = ImageColorizer()
            
            # Create a temporary image file
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                # Create a simple test image
                test_image = Image.new('RGB', (100, 100), color='red')
                test_image.save(tmp_file.name)
                
                try:
                    loaded_image = colorizer.load_image(tmp_file.name)
                    assert isinstance(loaded_image, Image.Image)
                    assert loaded_image.size == (100, 100)
                finally:
                    os.unlink(tmp_file.name)
    
    def test_preprocess_image(self):
        """Test image preprocessing."""
        with patch('src.image_colorizer.pipeline') as mock_pipeline:
            mock_pipeline.return_value = Mock()
            colorizer = ImageColorizer()
            
            # Test RGB image
            rgb_image = Image.new('RGB', (100, 100), color='red')
            processed = colorizer.preprocess_image(rgb_image)
            assert processed.mode == 'RGB'
            
            # Test grayscale image
            gray_image = Image.new('L', (100, 100), color=128)
            processed = colorizer.preprocess_image(gray_image)
            assert processed.mode == 'RGB'
    
    def test_save_image(self):
        """Test saving image to file."""
        with patch('src.image_colorizer.pipeline') as mock_pipeline:
            mock_pipeline.return_value = Mock()
            colorizer = ImageColorizer()
            
            test_image = Image.new('RGB', (100, 100), color='blue')
            
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                try:
                    colorizer.save_image(test_image, tmp_file.name)
                    # Verify file was created and can be loaded
                    saved_image = Image.open(tmp_file.name)
                    assert saved_image.size == (100, 100)
                finally:
                    os.unlink(tmp_file.name)
    
    @patch('src.image_colorizer.pipeline')
    def test_colorize_with_pipeline(self, mock_pipeline):
        """Test colorization using pipeline."""
        # Mock the pipeline
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.return_value = [{'image': Image.new('RGB', (100, 100), color='green')}]
        mock_pipeline.return_value = mock_pipeline_instance
        
        colorizer = ImageColorizer()
        test_image = Image.new('RGB', (100, 100), color='red')
        
        result = colorizer.colorize(test_image)
        assert isinstance(result, Image.Image)
        assert result.size == (100, 100)
    
    def test_batch_colorize(self):
        """Test batch colorization."""
        with patch('src.image_colorizer.pipeline') as mock_pipeline:
            mock_pipeline_instance = Mock()
            mock_pipeline_instance.return_value = [{'image': Image.new('RGB', (100, 100), color='green')}]
            mock_pipeline.return_value = mock_pipeline_instance
            
            colorizer = ImageColorizer()
            
            # Create test images
            test_images = [
                Image.new('RGB', (100, 100), color='red'),
                Image.new('RGB', (100, 100), color='blue'),
                Image.new('RGB', (100, 100), color='yellow')
            ]
            
            results = colorizer.batch_colorize(test_images)
            assert len(results) == 3
            assert all(isinstance(img, Image.Image) for img in results)


class TestSampleImageCreation:
    """Test cases for sample image creation."""
    
    def test_create_sample_images(self):
        """Test creating sample images."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            sample_files = create_sample_images(tmp_dir)
            
            assert len(sample_files) == 1
            assert Path(sample_files[0]).exists()
            
            # Verify the image can be loaded
            image = Image.open(sample_files[0])
            assert isinstance(image, Image.Image)
            assert image.mode == 'L'  # Grayscale


class TestConfigManager:
    """Test cases for configuration management."""
    
    def test_config_creation(self):
        """Test configuration creation."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "test_config.yaml"
            config_manager = ConfigManager(str(config_path))
            
            config = config_manager.get_config()
            assert isinstance(config, AppConfig)
            assert config.model.name == "timbrooks/instruct-pix2pix"
            assert config.ui.port == 8501
    
    def test_env_override(self):
        """Test environment variable override."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "test_config.yaml"
            
            # Set environment variable
            os.environ['MODEL_NAME'] = 'test-model'
            os.environ['UI_PORT'] = '8080'
            
            try:
                config_manager = ConfigManager(str(config_path))
                config = config_manager.get_config()
                
                assert config.model.name == 'test-model'
                assert config.ui.port == 8080
            finally:
                # Clean up environment variables
                if 'MODEL_NAME' in os.environ:
                    del os.environ['MODEL_NAME']
                if 'UI_PORT' in os.environ:
                    del os.environ['UI_PORT']


class TestIntegration:
    """Integration tests."""
    
    @patch('src.image_colorizer.pipeline')
    def test_end_to_end_colorization(self, mock_pipeline):
        """Test complete colorization workflow."""
        # Mock the pipeline
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.return_value = [{'image': Image.new('RGB', (100, 100), color='green')}]
        mock_pipeline.return_value = mock_pipeline_instance
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create input image
            input_path = Path(tmp_dir) / "input.png"
            test_image = Image.new('RGB', (100, 100), color='red')
            test_image.save(input_path)
            
            # Initialize colorizer
            colorizer = ImageColorizer()
            
            # Colorize image
            result = colorizer.colorize(input_path)
            
            # Save result
            output_path = Path(tmp_dir) / "output.png"
            colorizer.save_image(result, output_path)
            
            # Verify output exists
            assert output_path.exists()
            
            # Verify output can be loaded
            output_image = Image.open(output_path)
            assert isinstance(output_image, Image.Image)


if __name__ == "__main__":
    pytest.main([__file__])
