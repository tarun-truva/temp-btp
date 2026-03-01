# Architecture Overview

## Application Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                          app.py (Main)                          │
│                    Orchestrates Everything                       │
└────────────┬────────────────────────────────────────────────────┘
             │
             ├──────────────┬──────────────┬──────────────────┬────────────┐
             │              │              │                  │            │
             ▼              ▼              ▼                  ▼            ▼
      ┌──────────┐   ┌────────────┐  ┌──────────┐    ┌────────────┐  ┌────────┐
      │ config.py│   │  model.py  │  │ui_comp.. │    │  utils.py  │  │ HF Hub │
      │          │   │            │  │          │    │            │  │        │
      │Constants │   │Load Model  │  │CSS Style │    │Validate    │  │ Model  │
      │Settings  │   │Predictions │  │Render UI │    │Preprocess  │  │ Cache  │
      │Labels    │   │Cache Model │  │Components│    │Format      │  │        │
      └──────────┘   └────────────┘  └──────────┘    └────────────┘  └────────┘
```

## Data Flow

```
1. User uploads MRI image
         ↓
2. Image validated (utils.py)
         ↓
3. Image preprocessed (utils.py)
         ↓
4. Model loads/cached (model.py)
         ↓
5. Prediction made (model.py)
         ↓
6. Results formatted (ui_components.py)
         ↓
7. Results displayed to user
```

## Module Dependencies

```
app.py
  ├─→ config.py (configuration)
  ├─→ model.py
  │     └─→ config.py (model settings)
  ├─→ ui_components.py
  │     └─→ config.py (UI text, colors)
  └─→ utils.py (helpers)
```

## Key Design Patterns

### 1. **Separation of Concerns**
- Each module has a single responsibility
- Business logic separated from presentation
- Configuration separated from code

### 2. **Dependency Injection**
- Model and processor passed as parameters
- Easy to test with mock objects
- Flexible and maintainable

### 3. **Caching Strategy**
- `@st.cache_resource` for model loading
- HF models cached on disk
- Fast subsequent runs

### 4. **Type Safety**
- Type hints on all functions
- Better IDE autocomplete
- Catch errors early

## Function Organization

### config.py
```
- Constants (MODEL_NAME, CLASS_LABELS, etc.)
- Environment setup (cache directories)
- UI text content (messages, instructions)
```

### model.py
```
- load_model() → Load from HuggingFace
- predict_image() → Make predictions
- get_model_info() → Model metadata
```

### ui_components.py
```
- apply_custom_css() → Styling
- render_*() functions → UI components
  - render_header()
  - render_sidebar()
  - render_prediction_result()
  - render_probability_bars()
  - etc.
```

### utils.py
```
- validate_image() → Check image validity
- preprocess_image() → Prepare for model
- format_file_size() → Human-readable sizes
- get_confidence_level() → Interpret scores
```

## Benefits

✅ **Maintainability**: Easy to find and modify code
✅ **Testability**: Each module can be unit tested
✅ **Scalability**: Add features without breaking existing code
✅ **Readability**: Clear structure, well-documented
✅ **Collaboration**: Team members can work on different modules
✅ **Debugging**: Issues isolated to specific modules
