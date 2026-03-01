# Alzheimer's Disease MRI Classification App

This Streamlit application uses a fine-tuned ResNet-34 model to classify Alzheimer's disease stages from brain MRI scans.

**✨ Features:**
- 🏗️ **Modular Architecture**: Clean, maintainable code structure
- 🎨 **Modern UI**: Beautiful, responsive interface
- 🤖 **High Accuracy**: 92.19% classification accuracy
- ⚡ **Fast**: Optimized model loading and caching
- 📊 **Detailed Analysis**: Comprehensive probability breakdowns

## Model Information

- **Model**: [Thamer/resnet-fine_tuned](https://huggingface.co/Thamer/resnet-fine_tuned)
- **Base Model**: microsoft/resnet-34
- **Dataset**: Falah/Alzheimer_MRI
- **Accuracy**: 92.19%
- **License**: Apache 2.0

## Classification Categories

The model classifies MRI scans into four categories:
- 🟢 **Non Demented**: No signs of dementia
- 🟡 **Very Mild Demented**: Very mild cognitive impairment
- 🟠 **Mild Demented**: Mild dementia symptoms
- 🔴 **Moderate Demented**: Moderate dementia symptoms

## Installation

1. Clone or download this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit app:

```bash
streamlit run app.py
```

The app will open in your default web browser at `http://localhost:8501`.

**Note**: On first run, the app will download the pre-trained model from Hugging Face (~85MB). The model is cached in the `hf_cache/` directory in your project folder, so subsequent runs will be instant.

## Project Structure

```
temp btp/
├── app.py                    # Main application entry point
├── src/                      # Source code modules
│   ├── config.py            # Configuration and constants
│   ├── model.py             # AI model operations
│   ├── ui_components.py     # UI rendering functions
│   └── utils.py             # Utility functions
├── requirements.txt          # Dependencies
└── hf_cache/                # Model cache (auto-generated)
```

For detailed structure documentation, see [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md).

## How to Use the App

1. **Upload** a brain MRI scan image (JPG or PNG format)
2. **Click** the "Analyze MRI Scan" button
3. **Review** the classification results and confidence scores
4. **Check** the detailed probabilities for all categories

## Features

- 🎨 **Modern UI**: Beautiful gradient designs and intuitive interface
- 📊 **Detailed Analysis**: Shows probabilities for all classification categories
- 🎯 **High Accuracy**: 92.19% accuracy on the validation set
- ⚡ **Fast Predictions**: Optimized model loading with caching
- 📱 **Responsive Design**: Works on desktop and mobile devices

## Medical Disclaimer

⚠️ **Important**: This application is designed for **research and educational purposes only**. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult qualified healthcare professionals for medical concerns.

## Model Performance

| Metric | Value |
|--------|-------|
| Validation Loss | 0.1983 |
| Validation Accuracy | 92.19% |
| Training Epochs | 15 |
| Learning Rate | 0.0002 |
| Batch Size | 64 |

## Technical Details

### Training Hyperparameters
- Learning rate: 0.0002
- Train batch size: 16
- Eval batch size: 16
- Optimizer: Adam (betas=0.9,0.999, epsilon=1e-08)
- LR scheduler: Linear with 0.1 warmup ratio
- Number of epochs: 15

### Framework Versions
- Transformers: 4.31.0
- PyTorch: 2.0.1
- Streamlit: 1.31.0

## Dataset

The [Falah/Alzheimer_MRI](https://huggingface.co/datasets/Falah/Alzheimer_MRI) dataset contains brain MRI images labeled into four categories representing different stages of Alzheimer's disease.

## Credits

- Model trained by [Thamer](https://huggingface.co/Thamer)
- Based on Microsoft's ResNet-34 architecture
- Dataset provided by Falah on Hugging Face

## License

This project uses the Apache 2.0 license, following the original model's licensing.
