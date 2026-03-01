# Quick Reference Guide

## 📁 File Overview

| File | Purpose | When to Edit |
|------|---------|-------------|
| `app.py` | Main entry point | Change app flow or add new pages |
| `src/config.py` | Constants & settings | Change text, colors, or parameters |
| `src/model.py` | AI model logic | Change model or prediction logic |
| `src/ui_components.py` | UI elements | Change styling or add UI components |
| `src/utils.py` | Helper functions | Add utility functions |

## 🔧 Common Tasks

### Change Model Parameters
Edit `src/config.py`:
```python
MODEL_NAME = "your-model-name"
MODEL_CACHE_DIR = "./your-cache-path"
```

### Modify UI Text
Edit `src/config.py`:
- `SIDEBAR_ABOUT` - Sidebar information
- `INTERPRETATIONS` - Result interpretations
- `USAGE_INSTRUCTIONS` - How-to guide

### Change Colors/Styling
Edit `src/ui_components.py` in the `apply_custom_css()` function:
```python
def apply_custom_css():
    st.markdown("""
        <style>
        /* Your custom CSS here */
        </style>
    """, unsafe_allow_html=True)
```

### Add New Classification Category
1. Update `CLASS_LABELS` in `src/config.py`
2. Add color to `LABEL_COLORS` in `src/config.py`
3. Add interpretation to `INTERPRETATIONS` in `src/config.py`

### Add Image Preprocessing
Edit `src/utils.py`:
```python
def preprocess_image(image: Image.Image) -> Image.Image:
    # Add your preprocessing steps
    return processed_image
```

## 🚀 Running the App

### Standard Run
```bash
streamlit run app.py
```

### With UV (if using uv package manager)
```bash
uv run streamlit run app.py
```

### Debug Mode
```bash
streamlit run app.py --logger.level=debug
```

## 🧪 Testing Individual Modules

### Test Model Loading
```python
from src.model import load_model
processor, model = load_model()
print(f"Model loaded: {model is not None}")
```

### Test Image Validation
```python
from PIL import Image
from src.utils import validate_image

image = Image.open("test.jpg")
is_valid, error = validate_image(image)
print(f"Valid: {is_valid}, Error: {error}")
```

### Test Prediction
```python
from PIL import Image
from src.model import load_model, predict_image
from src.utils import preprocess_image

processor, model = load_model()
image = Image.open("test_mri.jpg")
processed = preprocess_image(image)
label, conf, probs = predict_image(processed, processor, model)
print(f"Prediction: {label} ({conf:.2f}%)")
```

## 📦 Adding New Dependencies

1. Add to `requirements.txt`
2. Install: `pip install -r requirements.txt`
3. If using uv: `uv add package-name`

## 🐛 Debugging Tips

### Model Won't Load
- Check internet connection
- Verify `MODEL_NAME` in config.py
- Check cache directory permissions

### UI Looks Wrong
- Clear Streamlit cache: `streamlit cache clear`
- Check browser console for errors
- Verify CSS in `ui_components.py`

### Import Errors
- Ensure you're in the project root directory
- Check that `src/__init__.py` exists
- Verify Python path includes project root

## 📊 Performance Optimization

### Reduce Model Load Time
- Model is automatically cached after first load
- Cache is stored in `hf_cache/` directory
- Delete cache to re-download: `rm -rf hf_cache/`

### Improve UI Responsiveness
- Images are processed in-memory
- Use `@st.cache_resource` for expensive operations
- Consider image size limits in `utils.py`

## 🔒 Security Best Practices

- ✅ Input validation in `utils.py`
- ✅ File size limits configured
- ✅ Medical disclaimer prominently displayed
- ✅ No user data stored or logged

## 📝 Code Style Guidelines

- Use type hints for all functions
- Add docstrings to all public functions
- Keep functions focused and small
- Follow PEP 8 naming conventions
- Group related functions in modules

## 🎯 Next Steps / Extensions

Ideas for extending the app:
1. Add batch processing for multiple images
2. Export results to PDF report
3. Add confidence threshold settings
4. Include explainability (Grad-CAM visualization)
5. Add user authentication
6. Store prediction history
7. Add model comparison feature
