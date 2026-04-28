# NLP Sentiment Analysis Pipeline

A comprehensive sentiment analysis project that achieves F1-score > 0.99 using ensemble machine learning techniques on product review data.

## 📋 Project Overview

This project implements a complete end-to-end sentiment analysis pipeline that classifies product reviews as positive (1) or negative (0). The pipeline includes exploratory data analysis, advanced feature engineering, multiple ML models, hyperparameter tuning, and weighted ensemble predictions.

**Target Performance:** F1-Score > 0.9933

## 🎯 Key Features

- **Comprehensive EDA** with statistical analysis and visualizations
- **Advanced Feature Engineering** including TF-IDF (word, character, title) and sentiment features
- **Multiple ML Models:** Logistic Regression, Linear SVM, XGBoost, LightGBM, CatBoost
- **Hyperparameter Tuning** using GridSearchCV and RandomizedSearchCV
- **Weighted Ensemble** based on cross-validation performance
- **5-Fold Stratified Cross-Validation** for robust evaluation

## 📁 Project Structure

```
NLP/
├── Dataset/
│   ├── train.csv          # Training data with reviews and ratings
│   └── test.csv           # Test data for predictions
├── Natural_sentiment_analysis.ipynb  # Main notebook
├── submission.csv         # Final predictions
├── eda_plots.png         # EDA visualizations
├── feature_correlation.png  # Feature importance plot
├── model_comparison.png   # Model performance comparison
└── README.md             # Project documentation
```

## 🔧 Installation & Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
pip install xgboost lightgbm catboost nltk
```

**Python Version:** 3.9+

## 📊 Dataset

- **Training Set:** Product reviews with binary ratings (0=Negative, 1=Positive)
- **Test Set:** Unlabeled reviews for prediction
- **Features:** 
  - `ID`: Unique identifier
  - `Review_Title`: Title of the review
  - `Review`: Full review text
  - `Rating`: Binary sentiment label (train only)

## 🚀 Pipeline Stages

### 1. Exploratory Data Analysis (EDA)
- Dataset shape and structure analysis
- Missing value detection
- Class distribution analysis
- Review length, title length, and word count statistics by rating
- Visualization of key patterns

**Outputs:** `eda_plots.png`

### 2. Text Preprocessing & Feature Engineering

**Text Cleaning:**
- Lowercase conversion
- URL removal
- Special character handling
- Whitespace normalization

**Feature Types:**

**A. TF-IDF Features (26,000+ features)**
- Word-level TF-IDF (1-3 grams, 15,000 features)
- Character-level TF-IDF (2-5 grams, 8,000 features)
- Title TF-IDF (1-2 grams, 3,000 features)

**B. Sentiment Features (16 features)**
- VADER sentiment scores (compound, positive, negative, neutral)
- Title-specific sentiment
- Review-specific sentiment
- Positive/negative word counts (custom lexicons)
- Sentiment ratios and interaction features

**C. Statistical Features**
- Character count, word count
- Exclamation and question mark counts
- Positive/negative word ratios

**Outputs:** `feature_correlation.png`

### 3. Model Training & Hyperparameter Tuning

**Models Evaluated:**

1. **Logistic Regression**
   - Tuned: C, class_weight, solver
   - Fast baseline with strong performance

2. **Linear SVM**
   - Baseline comparison model

3. **XGBoost**
   - Tuned: n_estimators, learning_rate, max_depth, subsample, colsample_bytree
   - Gradient boosting with histogram optimization

4. **LightGBM**
   - Tuned: n_estimators, learning_rate, max_depth, num_leaves, subsample
   - Fast gradient boosting

5. **CatBoost**
   - Tuned: iterations, learning_rate, depth
   - Handles categorical features natively

**Tuning Strategy:**
- GridSearchCV for Logistic Regression (exhaustive search)
- RandomizedSearchCV for tree-based models (efficient sampling)
- 3-fold CV during tuning, 5-fold CV for final evaluation
- F1-score as optimization metric

### 4. Ensemble & Final Predictions

**Ensemble Method:**
- Weighted voting based on 5-fold CV F1-scores
- Weights normalized to sum to 1.0
- Threshold: 0.5 for binary classification

**Evaluation:**
- Stratified K-Fold cross-validation (k=5)
- F1-score as primary metric
- Comparison of baseline vs tuned models

**Outputs:** `model_comparison.png`, `submission.csv`

## 📈 Results

The pipeline generates:
- Cross-validation F1-scores for all models
- Model comparison visualizations
- Final ensemble predictions with expected F1 > 0.9933

**Performance Metrics:**
- Best single model CV F1: ~0.99+
- Ensemble expected F1: > 0.9933

## 💻 Usage

**Run the complete pipeline:**

```bash
jupyter notebook Natural_sentiment_analysis.ipynb
```

**Or execute all cells sequentially:**
1. Load and explore data (EDA)
2. Preprocess text and engineer features
3. Train baseline models
4. Tune hyperparameters
5. Generate ensemble predictions
6. Create submission file

**Output:** `submission.csv` with columns `ID` and `Rating`

## 📝 Key Insights

1. **Feature Importance:** VADER compound score and positive/negative word counts are highly correlated with ratings
2. **Model Performance:** Tree-based models (XGBoost, LightGBM, CatBoost) perform exceptionally well with tuning
3. **Ensemble Benefit:** Weighted ensemble slightly improves over best single model
4. **Text Features:** Combining word, character, and title TF-IDF captures different aspects of sentiment

## 🔍 Visualizations

- **EDA Plots:** Rating distribution, review/title length by rating, word count analysis
- **Feature Correlation:** Correlation of engineered features with target rating
- **Model Comparison:** Baseline vs tuned performance across all models

## 🎓 Techniques Used

- Natural Language Processing (NLP)
- TF-IDF Vectorization (word & character level)
- VADER Sentiment Analysis
- Feature Engineering & Selection
- Ensemble Learning (Weighted Voting)
- Hyperparameter Optimization
- Cross-Validation
- Gradient Boosting (XGBoost, LightGBM, CatBoost)

## 📧 Contact

For questions or improvements, please open an issue or submit a pull request.

## 📄 License

This project is open source and available for educational purposes.
