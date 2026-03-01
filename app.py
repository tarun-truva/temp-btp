"""
Alzheimer's Disease MRI Classification App
Main application file - refactored for modularity

Author: Your Name
Version: 1.0.0
"""

import streamlit as st
from PIL import Image

# Import custom modules
from src.config import PAGE_CONFIG
from src.model import load_model, predict_image
from src.ui_components import (
    apply_custom_css,
    render_sidebar,
    render_header,
    render_warning_box,
    render_image_details,
    render_prediction_result,
    render_probability_bars,
    render_interpretation,
    render_usage_instructions,
    render_file_uploader
)
from src.utils import validate_image, preprocess_image


def main():
    """Main application logic"""
    
    # Configure page
    st.set_page_config(**PAGE_CONFIG)
    
    # Apply custom styling
    apply_custom_css()
    
    # Render header
    render_header()
    
    # Render sidebar
    render_sidebar()
    
    # Render warning box
    render_warning_box()
    
    # Load model with spinner
    with st.spinner("Loading AI model... This may take a moment on first run."):
        processor, model = load_model()
    
    # Check if model loaded successfully
    if processor is None or model is None:
        st.error("Failed to load the model. Please check your internet connection and try again.")
        return
    
    st.success("✅ Model loaded successfully!")
    
    # File upload section
    st.markdown("---")
    st.header("📤 Upload MRI Scan")
    
    uploaded_file = render_file_uploader()
    
    if uploaded_file is not None:
        # Create two columns for layout
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📷 Uploaded Image")
            
            # Load and display image
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True, caption="MRI Scan")
            
            # Validate image
            is_valid, error_msg = validate_image(image)
            if not is_valid:
                st.error(f"Invalid image: {error_msg}")
                return
            
            # Display image details
            render_image_details(image)
        
        with col2:
            st.subheader("🔍 Analysis Results")
            
            # Analyze button
            if st.button("🚀 Analyze MRI Scan", type="primary", use_container_width=True):
                with st.spinner("Analyzing image..."):
                    # Preprocess image
                    processed_image = preprocess_image(image)
                    
                    # Make prediction
                    predicted_label, confidence, all_probs = predict_image(
                        processed_image, processor, model
                    )
                    
                    if predicted_label:
                        # Display prediction result
                        render_prediction_result(predicted_label, confidence)
                        
                        # Display probability bars
                        render_probability_bars(all_probs)
                        
                        # Display interpretation
                        render_interpretation(predicted_label)
    else:
        # Show usage instructions when no file is uploaded
        render_usage_instructions()


if __name__ == "__main__":
    main()
