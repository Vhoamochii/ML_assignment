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

### Method 1: Using Docker (Recommended for Reproducibility)

```bash
# Build the Docker image
docker build -t ag-news-ml-project .

# Run Jupyter notebook server
docker run -p 8888:8888 -v ${PWD}:/app ag-news-ml-project

# Access Jupyter at http://localhost:8888 (check terminal for token)
```

### Method 2: Local Installation

**Prerequisites:**
- Python 3.8+
- pip or conda

**Installation:**
1. Clone or extract the project
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download NLTK data:
```python
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

### Download Dataset
Download the AG News dataset from:
- Kaggle: https://www.kaggle.com/amananandrai/ag-news-classification-dataset

Place `train.csv` and `test.csv` in the `data/raw/` folder.

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

## Reproducibility

This project includes both `Dockerfile` and `requirements.txt` for easy reproduction:

### Using Docker:
```bash
docker build -t ag-news-ml-project .
docker run -p 8888:8888 -v ${PWD}:/app ag-news-ml-project
```

### Using pip:
```bash
pip install -r requirements.txt
jupyter notebook
```

Both methods ensure the same environment and dependencies.

## Notes
- Large model files are stored separately and linked in `models/model_links.txt`
- Experiments are reproducible using the provided random seeds
