#!/usr/bin/env python3
"""
Setup script for Image Colorization project.

This script helps users set up the project environment and dependencies.
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(command: str, description: str) -> bool:
    """Run a command and return success status."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stdout:
            print(f"Output: {e.stdout}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python version {version.major}.{version.minor}.{version.micro} is compatible")
    return True


def create_directories():
    """Create necessary directories."""
    directories = ["data/samples", "models", "outputs", "logs", "config"]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"📁 Created directory: {directory}")


def install_dependencies():
    """Install project dependencies."""
    if not Path("requirements.txt").exists():
        print("❌ requirements.txt not found")
        return False
    
    return run_command(
        f"{sys.executable} -m pip install -r requirements.txt",
        "Installing dependencies"
    )


def create_sample_data():
    """Create sample data for testing."""
    try:
        from src.image_colorizer import create_sample_images
        sample_dir = Path("data/samples")
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        if not list(sample_dir.glob("*.png")):
            print("🔄 Creating sample images...")
            create_sample_images(sample_dir)
            print("✅ Sample images created successfully")
        else:
            print("✅ Sample images already exist")
        return True
    except Exception as e:
        print(f"❌ Failed to create sample data: {e}")
        return False


def test_installation():
    """Test if the installation works."""
    try:
        print("🔄 Testing installation...")
        
        # Test imports
        from src.image_colorizer import ImageColorizer
        from src.config import ConfigManager
        
        print("✅ All imports successful")
        
        # Test configuration
        config_manager = ConfigManager()
        config = config_manager.get_config()
        print(f"✅ Configuration loaded: {config.model.name}")
        
        return True
    except Exception as e:
        print(f"❌ Installation test failed: {e}")
        return False


def main():
    """Main setup function."""
    print("🎨 Image Colorization Setup")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    print("\n📁 Creating directories...")
    create_directories()
    
    # Install dependencies
    print("\n📦 Installing dependencies...")
    if not install_dependencies():
        print("❌ Setup failed during dependency installation")
        sys.exit(1)
    
    # Create sample data
    print("\n🖼️ Creating sample data...")
    if not create_sample_data():
        print("⚠️ Warning: Could not create sample data")
    
    # Test installation
    print("\n🧪 Testing installation...")
    if not test_installation():
        print("❌ Setup failed during testing")
        sys.exit(1)
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Run the web interface: streamlit run web_app/app.py")
    print("2. Or use the CLI: python src/cli.py --help")
    print("3. Or run the example: python example.py")


if __name__ == "__main__":
    main()
