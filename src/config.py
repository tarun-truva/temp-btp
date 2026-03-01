"""
Configuration file for the Alzheimer's Disease Classification App
Contains all constants, settings, and configuration parameters
"""

import os

# Model Configuration
MODEL_NAME = "Thamer/resnet-fine_tuned"
MODEL_CACHE_DIR = "./hf_cache"

# Set Hugging Face cache directory
os.environ['HF_HOME'] = MODEL_CACHE_DIR
os.environ['TRANSFORMERS_CACHE'] = MODEL_CACHE_DIR

# Model Performance Metrics
MODEL_ACCURACY = 0.9219
MODEL_LOSS = 0.1983

# Classification Labels
CLASS_LABELS = {
    0: "Mild Demented",
    1: "Moderate Demented",
    2: "Non Demented",
    3: "Very Mild Demented"
}

# Label Colors for UI (severity-based)
LABEL_COLORS = {
    "Non Demented": "#27ae60",           # Green
    "Very Mild Demented": "#f39c12",     # Yellow/Orange
    "Mild Demented": "#e67e22",          # Orange
    "Moderate Demented": "#e74c3c"       # Red
}

# Page Configuration
PAGE_CONFIG = {
    "page_title": "Alzheimer's Disease Classification",
    "page_icon": "🧠",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# File Upload Settings
ALLOWED_EXTENSIONS = ["jpg", "jpeg", "png"]
MAX_FILE_SIZE = 10  # MB

# UI Text Content
SIDEBAR_ABOUT = """
This application uses a fine-tuned ResNet-34 model to classify Alzheimer's disease 
stages from brain MRI scans.

**Model Information:**
- Base Model: microsoft/resnet-34
- Dataset: Falah/Alzheimer_MRI
- Accuracy: 92.19%
- Loss: 0.1983

**Classification Categories:**
- 🟢 Non Demented
- 🟡 Very Mild Demented
- 🟠 Mild Demented
- 🔴 Moderate Demented
"""

MEDICAL_DISCLAIMER = """
**⚠️ Medical Disclaimer:**
This tool is for research and educational purposes only. 
Always consult healthcare professionals for medical diagnosis.
"""

WARNING_MESSAGE = """
<div class='warning-box'>
    <strong>⚠️ Important Notice:</strong> This AI model is designed for research and educational purposes only. 
    It should not be used as a substitute for professional medical advice, diagnosis, or treatment.
</div>
"""

# Interpretation Messages
INTERPRETATIONS = {
    "Non Demented": """
    The MRI scan shows no signs of dementia. However, regular check-ups 
    are recommended for early detection of any changes.
    """,
    "Very Mild Demented": """
    The MRI scan indicates very mild signs of dementia. Early intervention 
    and lifestyle modifications may help slow progression. Consult with a 
    neurologist for proper evaluation.
    """,
    "Mild Demented": """
    The MRI scan shows mild dementia. Medical evaluation and treatment 
    planning are recommended. Support from healthcare professionals and 
    family is important.
    """,
    "Moderate Demented": """
    The MRI scan indicates moderate dementia. Comprehensive medical care 
    and support are essential. Please consult with healthcare professionals 
    for a detailed treatment plan.
    """
}

# Usage Instructions
USAGE_INSTRUCTIONS = """
1. **Upload** a brain MRI scan image (JPG or PNG format)
2. **Click** the "Analyze MRI Scan" button
3. **Review** the classification results and probabilities
4. **Consult** with healthcare professionals for proper medical guidance

**Tips for Best Results:**
- Use high-quality MRI scans
- Ensure the image shows a clear brain cross-section
- Axial view MRI scans work best with this model
"""
