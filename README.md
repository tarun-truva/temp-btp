# Alzheimer's Disease Classification

Multimodal Neural + Clinical Analysis for Alzheimer's Disease Classification.

## Project Structure

```
.
├── app/                    # Streamlit application files
│   ├── app.py             # Main app (dark theme)
│   ├── app1.py            # Alternative version
│   ├── app2.py            # Latest version (light theme)
│   └── lib.py             # Shared library functions
├── data/                   # Data files
│   ├── balanced_diagnosis_3.csv
│   └── processed_data_apoeres_3dec25.csv
├── models/                 # Trained model files
│   ├── Resnet-34-alz-trained.pkl
│   ├── best_random_forest_model_FULL.pkl
│   └── model_metadata.pkl
├── notebooks/              # Jupyter notebooks for experiments
│   ├── clinical-data.ipynb
│   └── clinical_genotype_experiments.ipynb
├── src/                    # Source modules
│   ├── config.py
│   ├── model.py
│   ├── ui_components.py
│   └── utils.py
├── config.toml             # Configuration file
├── main.py                 # Entry point
├── requirements.txt        # Dependencies
└── pyproject.toml          # Project metadata
```

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the App

```bash
streamlit run app/app2.py
```

## Features

- **MRI Image Analysis**: Upload brain MRI scans for classification using a fine-tuned ResNet model
- **Clinical Rule Engine**: CDR (Clinical Dementia Rating) based assessment with APOE genotype modifiers
- **Multimodal Fusion**: Combines image and clinical predictions for final diagnosis

## Classification Categories

- Non Demented
- Very Mild Demented
- Mild Demented
- Moderate Demented
