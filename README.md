# Amazon ML Challenge 2026 – Practice Task

## Objective

Predict customer ratings from Amazon product reviews using machine learning and optimize Macro-F1.

## Pipeline

1. Data preprocessing
2. Exploratory Data Analysis
3. TF-IDF feature engineering
4. Logistic Regression / Linear SVM / Multinomial Naive Bayes comparison
5. Hyperparameter tuning
6. Final training
7. Test prediction
8. Submission generation

## Best Validation Result

Review + Title TF-IDF + Logistic Regression

Macro-F1: 0.8759

## Project Structure

- `src/preprocessing.py` – preprocessing
- `src/feature_engineering.py` – TF-IDF feature pipeline
- `src/train.py` – model training/comparison
- `src/predict.py` – final prediction
- `tests/` – unit tests
- `notebooks/02_eda.py` – EDA
- `submissions/predictions.csv` – final predictions

## Run

```bash
python -m pytest
python notebooks/02_eda.py
python -m src.train
python -m src.predict
