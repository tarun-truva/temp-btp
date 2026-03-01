"""
Utility functions
Helper functions for various operations
"""

from PIL import Image
from typing import Tuple
import io


def validate_image(image: Image.Image) -> Tuple[bool, str]:
    """
    Validate uploaded image.
    
    Args:
        image: PIL Image object
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Check if image can be processed
        if image.mode not in ['RGB', 'RGBA', 'L']:
            return False, "Image must be in RGB, RGBA, or grayscale format"
        
        # Check minimum dimensions
        if image.size[0] < 50 or image.size[1] < 50:
            return False, "Image dimensions too small (minimum 50x50 pixels)"
        
        # Check maximum dimensions (to prevent memory issues)
        if image.size[0] > 5000 or image.size[1] > 5000:
            return False, "Image dimensions too large (maximum 5000x5000 pixels)"
        
        return True, ""
        
    except Exception as e:
        return False, f"Error validating image: {str(e)}"


def preprocess_image(image: Image.Image) -> Image.Image:
    """
    Preprocess image before prediction.
    Convert to RGB if needed.
    
    Args:
        image: PIL Image object
        
    Returns:
        Preprocessed PIL Image
    """
    # Convert to RGB if image is in RGBA or other modes
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    return image


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: File size in bytes
        
    Returns:
        Formatted string (e.g., "2.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def get_confidence_level(confidence: float) -> str:
    """
    Get confidence level description.
    
    Args:
        confidence: Confidence score (0-100)
        
    Returns:
        Confidence level string
    """
    if confidence >= 90:
        return "Very High"
    elif confidence >= 75:
        return "High"
    elif confidence >= 60:
        return "Moderate"
    elif confidence >= 45:
        return "Low"
    else:
        return "Very Low"


def get_severity_emoji(label: str) -> str:
    """
    Get emoji based on severity level.
    
    Args:
        label: Classification label
        
    Returns:
        Emoji string
    """
    emoji_map = {
        "Non Demented": "🟢",
        "Very Mild Demented": "🟡",
        "Mild Demented": "🟠",
        "Moderate Demented": "🔴"
    }
    return emoji_map.get(label, "⚪")
