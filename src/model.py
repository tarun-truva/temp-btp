"""
Model loading and prediction logic
Handles all AI model operations
"""

import streamlit as st
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
from typing import Tuple, Optional, Dict

from .config import MODEL_NAME, CLASS_LABELS


@st.cache_resource
def load_model() -> Tuple[Optional[AutoImageProcessor], Optional[AutoModelForImageClassification]]:
    """
    Load the pre-trained model and processor from Hugging Face.
    Uses Streamlit caching to avoid reloading on every interaction.
    
    Returns:
        Tuple of (processor, model) or (None, None) if loading fails
    """
    try:
        processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
        model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
        return processor, model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None


def predict_image(
    image: Image.Image,
    processor: AutoImageProcessor,
    model: AutoModelForImageClassification
) -> Tuple[Optional[str], Optional[float], Optional[Dict[str, float]]]:
    """
    Process image and make prediction.
    
    Args:
        image: PIL Image object
        processor: Hugging Face image processor
        model: Pre-trained classification model
        
    Returns:
        Tuple of (predicted_label, confidence, all_probabilities)
        Returns (None, None, None) if prediction fails
    """
    try:
        # Prepare image for model
        inputs = processor(images=image, return_tensors="pt")
        
        # Make prediction (no gradient computation needed for inference)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            
        # Calculate probabilities using softmax
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        predicted_class_idx = probabilities.argmax(-1).item()
        confidence = probabilities[0][predicted_class_idx].item()
        
        # Get probabilities for all classes
        all_probs = {
            "Non Demented": probabilities[0][2].item() * 100,
            "Very Mild Demented": probabilities[0][3].item() * 100,
            "Mild Demented": probabilities[0][0].item() * 100,
            "Moderate Demented": probabilities[0][1].item() * 100
        }
        
        # Map class index to human-readable label
        predicted_label = CLASS_LABELS[predicted_class_idx]
        
        return predicted_label, confidence * 100, all_probs
        
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None, None, None


def get_model_info() -> Dict[str, str]:
    """
    Get model information for display.
    
    Returns:
        Dictionary containing model metadata
    """
    return {
        "Model Name": MODEL_NAME,
        "Base Architecture": "ResNet-34",
        "Dataset": "Falah/Alzheimer_MRI",
        "Accuracy": "92.19%",
        "Loss": "0.1983",
        "Framework": "PyTorch + Transformers"
    }
