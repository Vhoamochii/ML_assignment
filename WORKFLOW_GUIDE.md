# Complete Workflow Guide - AG News Classification Assignment

## Overview
This guide walks you through the entire assignment from downloading data to final submission.

---

## Phase 1: Setup and Data Acquisition

### Step 1.1: Install Dependencies
```bash
pip install -r requirements.txt
```

This installs:
- numpy, pandas, scikit-learn (ML libraries)
- nltk (text processing)
- matplotlib, seaborn (visualization)
- jupyter (notebooks)

### Step 1.2: Download AG News Dataset from Kaggle

1. Go to: https://www.kaggle.com/amananandrai/ag-news-classification-dataset
2. Click "Download" (you'll need a Kaggle account)
3. Extract the ZIP file
4. You'll get two files:
   - `train.csv` (120,000 samples)
   - `test.csv` (7,600 samples)
5. Place both files in `data/raw/` folder

```

### Step 1.3: Verify Data Structure
After downloading, your `data/raw/` folder should contain:
```
data/raw/
├── train.csv    (120,000 rows)
└── test.csv     (7,600 rows)
```

**CSV Format:**
```
class,title,description
3,"Wall St. Bears Claw Back Into the Black","Reuters - Short-sellers, Wall Street's dwindling..."
```
- Column 1: Class (1=World, 2=Sports, 3=Business, 4=Sci/Tech)
- Column 2: Title
- Column 3: Description

---

## Phase 2: Data Exploration

### Step 2.1: Run Notebook 01 - Data Analysis
```bash
jupyter notebook notebooks/01_data_analysis.ipynb
```

**What happens:**
- Loads `train.csv` and `test.csv`
- Combines title + description into single text
- Analyzes class distribution (balanced: 30,000 per class)
- Calculates text length statistics
- Generates visualizations

**Outputs created:**
```
data/processed/
├── train_explored.csv      # Train data with combined text
└── test_explored.csv       # Test data with combined text

results/figures/
├── class_distribution.png
└── text_length_distribution.png
```

**Key findings to note for report:**
- Dataset is balanced (equal samples per class)
- Average text length: ~200 characters
- Average word count: ~40 words

---

## Phase 3: Text Preprocessing

### Step 3.1: Run Notebook 02 - Preprocessing
```bash
jupyter notebook notebooks/02_preprocessing.ipynb
```

**What happens:**
1. **Text Cleaning:**
   - Convert to lowercase
   - Remove URLs, special characters, numbers
   - Remove extra whitespace

2. **Tokenization:**
   - Split text into words

3. **Stopword Removal:**
   - Remove common words (the, is, at, etc.)

4. **Lemmatization:**
   - Convert words to base form (running → run)

5. **Feature Extraction:**
   - Create TF-IDF features (5,000 features)
   - Create Count features (Bag of Words, 5,000 features)

**Outputs created:**
```
data/processed/
├── train_preprocessed.csv      # Cleaned text
├── test_preprocessed.csv       # Cleaned text
├── X_train_tfidf.pkl          # TF-IDF features (120,000 × 5,000)
├── X_test_tfidf.pkl           # TF-IDF features (7,600 × 5,000)
├── X_train_count.pkl          # BoW features (120,000 × 5,000)
├── X_test_count.pkl           # BoW features (7,600 × 5,000)
├── y_train.pkl                # Labels (120,000)
└── y_test.pkl                 # Labels (7,600)

models/
├── tfidf_vectorizer.pkl       # Fitted TF-IDF vectorizer
└── count_vectorizer.pkl       # Fitted Count vectorizer
```

---

## Phase 4: Model Training

### Step 4.1: Run Notebook 03 - Model Training
```bash
jupyter notebook notebooks/03_model_training.ipynb
```

**Models to train (as per your choice):**
1. **Naive Bayes** (Baseline)
2. **Logistic Regression**
3. **Linear Support Vector Machine (LinearSVC)**

**What happens for each model:**
1. Load preprocessed TF-IDF features
2. Train model on X_train, y_train
3. Predict on X_test
4. Calculate metrics:
   - Accuracy
   - Precision (macro & weighted)
   - Recall (macro & weighted)
   - F1-Score (macro & weighted)
   - Training time
   - Prediction time
5. Save trained model

**Outputs created:**
```
models/
├── naive_bayes.pkl            # Trained Naive Bayes
├── logistic_regression.pkl    # Trained Logistic Regression
└── svm.pkl                    # Trained Linear SVM

results/
└── comparison.csv             # All models comparison table
```

**Expected results (approximate):**
| Model | Accuracy | F1-Score | Train Time |
|-------|----------|----------|------------|
| Naive Bayes | 0.90 | 0.90 | 2s |
| Logistic Regression | 0.91 | 0.91 | 30s |
| Linear SVM | 0.91 | 0.91 | 45s |

---

## Phase 5: Evaluation and Analysis

### Step 5.1: Run Notebook 04 - Evaluation
```bash
jupyter notebook notebooks/04_evaluation.ipynb
```

**What happens:**
1. Load all trained models
2. Generate detailed metrics for each
3. Create visualizations:
   - Model comparison bar charts
   - Confusion matrices
   - Training time comparison
4. Print classification reports (per-class metrics)

**Outputs created:**
```
results/figures/
├── model_comparison.png           # Accuracy/F1/Precision/Recall comparison
├── confusion_matrix_naive_bayes.png
├── confusion_matrix_logistic_regression.png
├── confusion_matrix_svm.png
└── training_time.png             # Training time bar chart

results/metrics/
└── final_results.json            # All metrics in JSON format
```

**Important for report:**
- Which model performed best? (likely Logistic Regression or SVM)
- Which classes are confused? (check confusion matrix)
- Trade-off between accuracy and training time
- Explain WHY you chose F1-score as main metric (balanced dataset)

---

## Phase 6: Advanced Experiments (Optional)

### Step 6.1: Feature Selection Comparison
Create a new notebook or Python script:

```python
from experiments.feature_selection import compare_bow_vs_tfidf

# Compare BoW vs TF-IDF
results = compare_bow_vs_tfidf(
    texts_train=train_df['cleaned_text'],
    texts_test=test_df['cleaned_text'],
    y_train=y_train,
    y_test=y_test,
    max_features=5000,
    selection_method='chi2',
    k_best=1000
)
```

**Analysis to include in report:**
- Does BoW or TF-IDF perform better?
- What are the most important features (words)?

### Step 6.2: Hyperparameter Tuning (Optional)
```python
from experiments.hyperparameter_tuning import HyperparameterTuner, PARAM_GRIDS

tuner = HyperparameterTuner()
best_model, best_params = tuner.grid_search(
    LogisticRegression(),
    PARAM_GRIDS['logistic_regression'],
    X_train,
    y_train
)
```

---

## Phase 7: Report Writing

### Structure (as per assignment guide):

**1. Problem Introduction**
- AG News: 4-class news classification
- 120K training samples, 7.6K test samples
- Challenge: Distinguish similar news categories

**2. Data & Preprocessing**
- Dataset description and statistics
- Text cleaning pipeline
- Feature extraction: TF-IDF (explain why)
- Include figures from `results/figures/`

**3. Models & Methods**
- **Naive Bayes:** Probabilistic baseline, fast training
- **Logistic Regression:** Linear classifier with regularization
- **Linear SVM:** Maximum margin classifier
- Justify why you chose these 3 models

**4. Experiments & Results**
- Include `comparison.csv` table
- Show confusion matrices
- Per-class performance analysis
- Training time comparison

**5. Discussion**
- Best model: [Your best model] with accuracy X%
- Why it performed best (explain)
- Error analysis: Which classes are confused?
- Limitations: Linear models, no semantic understanding

**6. Conclusion & Future Work**
- Successfully classified news with 90%+ accuracy
- Future: Try deep learning (BERT), ensemble methods
- Potential deployment as API

---

## Phase 8: Prepare Submission

### Step 8.1: Clean Up Large Files
```bash
# Check file sizes
du -sh data/raw/*
du -sh models/*
```

If files are too large (>50MB), upload to Google Drive:
1. Upload large files to Google Drive
2. Get shareable links
3. Add links to `models/model_links.txt`

### Step 8.2: Update README.md
Edit the README to include:
- Your team member names and Student IDs
- Actual results from your experiments
- Any special instructions

### Step 8.3: Create Submission Package
```bash
# Create ZIP file named with your Student IDs
zip -r 2010001-2010002-2010003.zip \
  src/ \
  notebooks/ \
  data/ \
  models/ \
  results/ \
  experiments/ \
  README.md \
  requirements.txt \
  Dockerfile \
  .gitignore
```

**What to include:**
```
your-submission.zip
├── README.md                  # How to run
├── requirements.txt           # Dependencies
├── Dockerfile                 # Optional but recommended
├── src/                       # All Python code
├── notebooks/                 # All 4 notebooks (with outputs)
├── data/
│   ├── raw/                   # Empty (link in README)
│   └── processed/             # Processed data OR link
├── models/                    # Trained models OR links
├── results/                   # All figures and metrics
└── experiments/               # Feature selection code
```

### Step 8.4: Verify Everything Works
Before submission, test that someone else can run your code:

```bash
# Extract your ZIP
unzip 2010001-2010002-2010003.zip -d test-submission/
cd test-submission/

# Install dependencies
pip install -r requirements.txt

# Download data (from your provided link)
# Place in data/raw/

# Run notebooks in order
jupyter notebook notebooks/01_data_analysis.ipynb
jupyter notebook notebooks/02_preprocessing.ipynb
jupyter notebook notebooks/03_model_training.ipynb
jupyter notebook notebooks/04_evaluation.ipynb
```

---

## Quick Reference: What Goes Where

### data/raw/
- Original CSV files from Kaggle
- **DON'T commit to Git** (too large)
- Provide download link in README

### data/processed/
- Cleaned CSV files
- Pickled features (X_train, X_test)
- Pickled labels (y_train, y_test)
- Can be regenerated by running notebooks

### models/
- Trained model files (.pkl)
- Vectorizer files (.pkl)
- If too large: upload to Drive, add link to `model_links.txt`

### results/figures/
- All PNG images for report
- Confusion matrices
- Comparison charts
- Distribution plots

### results/metrics/
- JSON files with detailed metrics
- comparison.csv table

### notebooks/
- 01: Exploration → creates figures + explored CSVs
- 02: Preprocessing → creates features + vectorizers
- 03: Training → creates models + comparison.csv
- 04: Evaluation → creates detailed figures + analysis

### src/
- Reusable Python modules
- Don't modify during experiments
- Import into notebooks

### experiments/
- Advanced experiments
- Feature selection comparisons
- Hyperparameter tuning
- Optional for basic submission

---

## Timeline Suggestion

**Day 1-2:** Setup + Data Exploration
- Download data
- Run notebook 01
- Understand dataset

**Day 3-4:** Preprocessing
- Run notebook 02
- Verify features look correct
- Try different max_features values

**Day 5-6:** Model Training
- Run notebook 03
- Train all 3 models
- Compare results

**Day 7-8:** Analysis
- Run notebook 04
- Create all visualizations
- Analyze errors

**Day 9-10:** Report Writing
- Write report sections
- Include figures and tables
- Proofread

**Day 11:** Submission Prep
- Clean up code
- Test reproducibility
- Create ZIP file

---

## Troubleshooting

**Problem: Kaggle download requires authentication**
- Solution: Create Kaggle account, get API token from kaggle.com/account

**Problem: Out of memory when training**
- Solution: Reduce max_features from 5000 to 3000 in preprocessing

**Problem: NLTK data not found**
- Solution: Run in Python:
  ```python
  import nltk
  nltk.download('punkt')
  nltk.download('stopwords')
  nltk.download('wordnet')
  ```

**Problem: Notebooks don't show output**
- Solution: Make sure to save notebooks after running all cells

**Problem: ZIP file too large**
- Solution: Exclude data/raw/ and models/, upload to Drive instead

---

## Checklist Before Submission

- [ ] All 4 notebooks run without errors
- [ ] All models trained and saved
- [ ] Results folder contains all figures
- [ ] README.md updated with team info
- [ ] requirements.txt is complete
- [ ] Large files uploaded to Drive with links provided
- [ ] ZIP file named correctly (StudentID1-StudentID2-StudentID3.zip)
- [ ] Tested that ZIP extracts correctly
- [ ] Report completed with all sections
- [ ] Presentation slides prepared (if required)

---

## Expected File Structure at Submission

```
data/
├── raw/
│   └── .gitkeep                    # Empty, provide download link
├── processed/
│   ├── train_explored.csv          # ✓
│   ├── test_explored.csv           # ✓
│   ├── train_preprocessed.csv      # ✓
│   ├── test_preprocessed.csv       # ✓
│   ├── X_train_tfidf.pkl          # ✓
│   ├── X_test_tfidf.pkl           # ✓
│   ├── y_train.pkl                # ✓
│   └── y_test.pkl                 # ✓

models/
├── naive_bayes.pkl                # ✓
├── logistic_regression.pkl        # ✓
├── svm.pkl                        # ✓
├── tfidf_vectorizer.pkl          # ✓
└── model_links.txt               # If models on Drive

results/
├── comparison.csv                 # ✓
├── figures/
│   ├── class_distribution.png     # ✓
│   ├── text_length_distribution.png # ✓
│   ├── model_comparison.png       # ✓
│   ├── confusion_matrix_*.png     # ✓ (3 files)
│   └── training_time.png          # ✓
└── metrics/
    └── final_results.json         # Optional

notebooks/
├── 01_data_analysis.ipynb         # ✓ With outputs
├── 02_preprocessing.ipynb         # ✓ With outputs
├── 03_model_training.ipynb        # ✓ With outputs
└── 04_evaluation.ipynb            # ✓ With outputs

src/                               # ✓ All Python modules
experiments/                       # ✓ Advanced experiments
README.md                          # ✓ Updated
requirements.txt                   # ✓
Dockerfile                         # ✓ Optional
```

---

## Good Luck!

Follow this workflow step by step, and you'll have a complete, reproducible machine learning project that meets all assignment requirements.
