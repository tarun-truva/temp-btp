"""
UI Components and Styling
Contains all UI elements, custom CSS, and display functions
"""

import streamlit as st
from PIL import Image
from typing import Dict

from .config import (
    SIDEBAR_ABOUT, MEDICAL_DISCLAIMER, WARNING_MESSAGE,
    INTERPRETATIONS, USAGE_INSTRUCTIONS, LABEL_COLORS
)


def apply_custom_css():
    """Apply custom CSS styling to the Streamlit app"""
    st.markdown("""
        <style>
        .main {
            background-color: #f5f7fa;
        }
        .stTitle {
            color: #2c3e50;
            font-size: 3rem !important;
            font-weight: 700;
            text-align: center;
            margin-bottom: 0.5rem;
        }
        .stMarkdown h1 {
            color: #2c3e50;
        }
        .stMarkdown h2 {
            color: #34495e;
        }
        .stMarkdown h3 {
            color: #7f8c8d;
        }
        .upload-text {
            text-align: center;
            color: #7f8c8d;
            font-size: 1.2rem;
            margin: 2rem 0;
        }
        .result-box {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 15px;
            color: white;
            text-align: center;
            margin: 2rem 0;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        .prediction-label {
            font-size: 1.5rem;
            font-weight: 600;
            margin-bottom: 1rem;
        }
        .prediction-value {
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }
        .confidence-text {
            font-size: 1.2rem;
            opacity: 0.9;
        }
        .info-box {
            background-color: #e8f4f8;
            padding: 1.5rem;
            border-radius: 10px;
            border-left: 5px solid #3498db;
            margin: 1rem 0;
        }
        .warning-box {
            background-color: #fff3cd;
            padding: 1.5rem;
            border-radius: 10px;
            border-left: 5px solid #ffc107;
            margin: 1rem 0;
        }
        </style>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Render the sidebar with information and disclaimers"""
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown(SIDEBAR_ABOUT)
        
        st.markdown("---")
        st.markdown(MEDICAL_DISCLAIMER)


def render_header():
    """Render the main header and title"""
    st.title("🧠 Alzheimer's Disease Classification")
    st.markdown(
        "<p class='upload-text'>Advanced MRI-based Analysis using Deep Learning</p>",
        unsafe_allow_html=True
    )


def render_warning_box():
    """Render the warning/disclaimer box"""
    st.markdown(WARNING_MESSAGE, unsafe_allow_html=True)


def render_image_details(image: Image.Image):
    """
    Display image details in an info box.
    
    Args:
        image: PIL Image object
    """
    st.markdown(f"""
        <div class='info-box'>
            <strong>Image Details:</strong><br>
            Size: {image.size[0]} x {image.size[1]} pixels<br>
            Format: {image.format}<br>
            Mode: {image.mode}
        </div>
    """, unsafe_allow_html=True)


def render_prediction_result(predicted_label: str, confidence: float):
    """
    Display the prediction result in a styled box.
    
    Args:
        predicted_label: The predicted class label
        confidence: Confidence score (0-100)
    """
    st.markdown(f"""
        <div class='result-box'>
            <div class='prediction-label'>Diagnosis:</div>
            <div class='prediction-value'>{predicted_label}</div>
            <div class='confidence-text'>Confidence: {confidence:.2f}%</div>
        </div>
    """, unsafe_allow_html=True)


def render_probability_bars(all_probs: Dict[str, float]):
    """
    Display probability bars for all classes.
    
    Args:
        all_probs: Dictionary of class labels and their probabilities
    """
    st.markdown("### 📊 Detailed Probabilities")
    
    # Sort probabilities in descending order
    sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
    
    for label, prob in sorted_probs:
        color = LABEL_COLORS.get(label, "#95a5a6")
        
        st.markdown(f"**{label}**")
        st.progress(prob / 100)
        st.markdown(
            f"<span style='color: {color}; font-weight: bold;'>{prob:.2f}%</span>",
            unsafe_allow_html=True
        )
        st.markdown("")


def render_interpretation(predicted_label: str):
    """
    Display medical interpretation based on prediction.
    
    Args:
        predicted_label: The predicted class label
    """
    st.markdown("### 💡 Interpretation")
    
    interpretation = INTERPRETATIONS.get(predicted_label, "No interpretation available.")
    
    if predicted_label == "Non Demented":
        st.success(interpretation)
    elif predicted_label == "Very Mild Demented":
        st.warning(interpretation)
    elif predicted_label == "Mild Demented":
        st.warning(interpretation)
    else:  # Moderate Demented
        st.error(interpretation)


def render_usage_instructions():
    """Display usage instructions when no image is uploaded"""
    st.info("👆 Please upload a brain MRI image to begin analysis.")
    
    st.markdown("---")
    st.header("📝 How to Use")
    st.markdown(USAGE_INSTRUCTIONS)


def render_file_uploader():
    """
    Render the file uploader widget.
    
    Returns:
        Uploaded file object or None
    """
    return st.file_uploader(
        "Choose a brain MRI image...",
        type=["jpg", "jpeg", "png"],
        help="Upload a brain MRI scan in JPG or PNG format"
    )
