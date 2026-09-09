# AG News Classification - Machine Learning Project

## Project Overview
This project implements text classification on the AG News dataset using multiple machine learning algorithms. The dataset contains news articles from 4 categories: World, Sports, Business, and Science/Technology.

## Project Structure
```
btl/
├── data/                      # Dataset storage
├── notebooks/                 # Jupyter notebooks for experiments
├── src/                       # Source code
├── models/                    # Saved models
├── results/                   # Experiment results
└── experiments/               # Advanced experiments
```

## Setup Instructions

### Prerequisites
- Python 3.8+
- pip or conda

### Installation
1. Clone or extract the project
2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Download Dataset
The AG News dataset will be automatically downloaded when running the notebooks, or you can manually download from:
- Kaggle: https://www.kaggle.com/amananandrai/ag-news-classification-dataset
- Hugging Face: https://huggingface.co/datasets/ag_news

## Usage

### 1. Data Exploration
```bash
jupyter notebook notebooks/01_data_analysis.ipynb
```

### 2. Preprocessing
```bash
jupyter notebook notebooks/02_preprocessing.ipynb
```

### 3. Model Training
```bash
jupyter notebook notebooks/03_model_training.ipynb
```

### 4. Evaluation
```bash
jupyter notebook notebooks/04_evaluation.ipynb
```

## Models Implemented
- Naive Bayes (Baseline)
- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest
- Neural Network (Optional)

## Evaluation Metrics
- Accuracy
- Precision, Recall, F1-Score (per class)
- Confusion Matrix
- Training Time

## Results
Results and visualizations are saved in the `results/` directory.

## Notes
- Large model files are stored separately and linked in `models/model_links.txt`
- Experiments are reproducible using the provided random seeds
