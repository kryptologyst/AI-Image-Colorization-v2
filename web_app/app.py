"""
Streamlit Web Interface for Image Colorization

This module provides a modern web interface for the image colorization
application using Streamlit.
"""

import streamlit as st
import logging
from pathlib import Path
from typing import Optional, List
import tempfile
import os

from image_colorizer import ImageColorizer, create_sample_images
from config import config

# Configure logging
logging.basicConfig(level=getattr(logging, config.logging.level))
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="AI Image Colorization",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #667eea;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
    .error-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_colorizer():
    """Load the colorization model (cached)."""
    try:
        return ImageColorizer(
            model_name=config.model.name,
            device=config.model.device,
            cache_dir=config.model.cache_dir
        )
    except Exception as e:
        st.error(f"Failed to load colorization model: {e}")
        return None


def create_sample_data():
    """Create sample images if they don't exist."""
    sample_dir = Path(config.data_dir) / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    if not list(sample_dir.glob("*.png")):
        with st.spinner("Creating sample images..."):
            create_sample_images(sample_dir)
        st.success("Sample images created successfully!")


def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🎨 AI Image Colorization</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Model settings
        st.subheader("Model Configuration")
        model_name = st.selectbox(
            "Model",
            ["timbrooks/instruct-pix2pix", "runwayml/stable-diffusion-v1-5"],
            index=0,
            help="Select the colorization model to use"
        )
        
        num_steps = st.slider(
            "Inference Steps",
            min_value=5,
            max_value=50,
            value=config.model.num_inference_steps,
            help="Number of denoising steps (higher = better quality, slower)"
        )
        
        guidance_scale = st.slider(
            "Guidance Scale",
            min_value=1.0,
            max_value=20.0,
            value=config.model.guidance_scale,
            step=0.5,
            help="How closely to follow the prompt"
        )
        
        prompt = st.text_area(
            "Prompt",
            value=config.model.prompt,
            help="Text prompt for colorization"
        )
        
        # UI settings
        st.subheader("Display Settings")
        show_comparison = st.checkbox("Show Comparison", value=True)
        save_results = st.checkbox("Save Results", value=False)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Image")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose a grayscale image",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            help="Upload a grayscale image to colorize"
        )
        
        # Sample images
        st.subheader("📁 Sample Images")
        if st.button("Create Sample Images"):
            create_sample_data()
        
        sample_dir = Path(config.data_dir) / "samples"
        if sample_dir.exists():
            sample_files = list(sample_dir.glob("*.png"))
            if sample_files:
                selected_sample = st.selectbox(
                    "Choose a sample image",
                    [f.name for f in sample_files]
                )
                if st.button("Use Sample Image"):
                    sample_path = sample_dir / selected_sample
                    uploaded_file = open(sample_path, 'rb')
    
    with col2:
        st.header("🎨 Colorized Result")
        
        if uploaded_file is not None:
            # Load colorizer
            colorizer = load_colorizer()
            
            if colorizer is not None:
                try:
                    # Display original image
                    original_image = colorizer.load_image(uploaded_file)
                    
                    st.subheader("Original Image")
                    st.image(original_image, caption="Original Grayscale", use_column_width=True)
                    
                    # Colorize button
                    if st.button("🎨 Colorize Image", type="primary"):
                        with st.spinner("Colorizing image... This may take a few minutes."):
                            try:
                                # Create temporary file for processing
                                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                                    original_image.save(tmp_file.name)
                                
                                # Colorize
                                colorized_image = colorizer.colorize(
                                    tmp_file.name,
                                    prompt=prompt,
                                    num_inference_steps=num_steps,
                                    guidance_scale=guidance_scale
                                )
                                
                                # Clean up temp file
                                os.unlink(tmp_file.name)
                                
                                # Display result
                                st.subheader("Colorized Image")
                                st.image(colorized_image, caption="Colorized Result", use_column_width=True)
                                
                                # Show comparison if requested
                                if show_comparison:
                                    st.subheader("📊 Comparison")
                                    col_a, col_b = st.columns(2)
                                    with col_a:
                                        st.image(original_image, caption="Original", use_column_width=True)
                                    with col_b:
                                        st.image(colorized_image, caption="Colorized", use_column_width=True)
                                
                                # Save results if requested
                                if save_results:
                                    output_dir = Path(config.output_dir)
                                    output_dir.mkdir(parents=True, exist_ok=True)
                                    
                                    original_path = output_dir / f"original_{uploaded_file.name}"
                                    colorized_path = output_dir / f"colorized_{uploaded_file.name}"
                                    
                                    colorizer.save_image(original_image, original_path)
                                    colorizer.save_image(colorized_image, colorized_path)
                                    
                                    st.success(f"Results saved to {output_dir}")
                                
                                # Download button
                                st.download_button(
                                    label="📥 Download Colorized Image",
                                    data=colorized_image.tobytes(),
                                    file_name=f"colorized_{uploaded_file.name}",
                                    mime="image/png"
                                )
                                
                            except Exception as e:
                                st.error(f"Colorization failed: {e}")
                                logger.error(f"Colorization error: {e}")
                    else:
                        st.info("Click the 'Colorize Image' button to start processing.")
                        
                except Exception as e:
                    st.error(f"Failed to load image: {e}")
                    logger.error(f"Image loading error: {e}")
            else:
                st.error("Failed to load colorization model. Please check the configuration.")
        else:
            st.info("👆 Please upload an image or select a sample image to get started.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div class="info-box">
        <h4>ℹ️ About This Application</h4>
        <p>This application uses state-of-the-art AI models to automatically colorize grayscale images. 
        The colorization process uses deep learning to predict realistic colors based on the image content.</p>
        
        <h4>🔧 Tips for Best Results</h4>
        <ul>
            <li>Use high-quality grayscale images</li>
            <li>Images with clear subjects work best</li>
            <li>Adjust inference steps for quality vs speed trade-off</li>
            <li>Try different prompts for different colorization styles</li>
        </ul>
        
        <h4>⚡ Performance Notes</h4>
        <ul>
            <li>First run may take longer due to model downloading</li>
            <li>GPU acceleration is used when available</li>
            <li>Larger images take more time to process</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()