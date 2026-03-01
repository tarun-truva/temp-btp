# Project Structure

```
temp btp/
├── app.py                      # Main Streamlit application entry point
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── .gitignore                 # Git ignore rules
│
├── src/                       # Source code modules
│   ├── __init__.py           # Package initialization
│   ├── config.py             # Configuration and constants
│   ├── model.py              # AI model loading and prediction logic
│   ├── ui_components.py      # UI elements and styling
│   └── utils.py              # Utility functions
│
├── hf_cache/                  # Hugging Face model cache (auto-generated)
│   └── (model files)
│
└── .venv/                     # Virtual environment (if using venv)
```

## Module Descriptions

### `app.py`
- Main application entry point
- Orchestrates the entire application flow
- Clean, minimal code that imports from modules
- Easy to understand and maintain

### `src/config.py`
- Central configuration file
- Contains all constants, settings, and text content
- Model parameters and paths
- UI text and messages
- Easy to modify settings without touching logic

### `src/model.py`
- AI/ML model operations
- Model loading with caching
- Prediction logic
- Model information retrieval
- Fully typed with type hints

### `src/ui_components.py`
- All UI rendering functions
- Custom CSS styling
- Reusable UI components
- Consistent look and feel
- Separated from business logic

### `src/utils.py`
- Helper and utility functions
- Image validation and preprocessing
- Formatting functions
- Reusable across the application

## Design Principles

1. **Separation of Concerns**: Each module has a single, well-defined purpose
2. **DRY (Don't Repeat Yourself)**: Reusable components and functions
3. **Type Safety**: Type hints throughout for better IDE support
4. **Maintainability**: Easy to find and modify specific functionality
5. **Scalability**: Easy to add new features without breaking existing code
6. **Documentation**: Comprehensive docstrings for all functions

## Benefits of This Structure

- ✅ **Easy to Test**: Each module can be tested independently
- ✅ **Easy to Extend**: Add new features by creating new modules
- ✅ **Easy to Debug**: Issues are isolated to specific modules
- ✅ **Easy to Read**: Clear separation of concerns
- ✅ **Team Friendly**: Multiple developers can work on different modules
- ✅ **Professional**: Follows industry best practices
