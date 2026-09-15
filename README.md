# QADSS - Quantitative & AI-Based Stock Trading Decision Support System

## Overview

QADSS is a quantitative and machine-learning research project designed to support market analysis, probability estimation, risk assessment, and data-driven decision-making.

The project uses historical market data from TSETMC, with Bank Mellat (وبملت) as the case study.

## Project Objectives

- Data exploration and quality assessment
- Feature engineering
- Forward-looking target definition
- Baseline modeling
- Machine-learning model development
- Walk-forward validation
- Probability calibration
- SHAP-based model explainability
- Threshold and risk analysis
- Economic evaluation
- Decision support

## Project Architecture

Market Data
->
Data Quality
->
Feature Engineering
->
Target Definition
->
Baseline
->
Logistic Regression / XGBoost
->
Walk-Forward Validation
->
Probability Calibration
->
SHAP Explainability
->
Threshold Analysis
->
Economic Evaluation
->
Decision Support

## Data

- Data source: TSETMC
- Case study: Bank Mellat (وبملت)
- Instrument code: 778253364357513
- Data type: Historical daily market data
- Prediction horizon: Five future trading days
- Target: Whether the future five-day return is positive

Raw market data is intentionally excluded from the public repository through .gitignore.

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- SHAP
- Matplotlib
- JupyterLab
- Git and GitHub

## Project Status

Prototype / Research Validation

The project is intended to evaluate whether machine-learning models can provide useful predictive signals for quantitative decision support.

## Limitations

The current study focuses on a single stock as a case study. Predictive performance is modest and does not establish a guaranteed or consistently profitable trading strategy.

Further validation across multiple stocks, market regimes, and longer periods is required.

## Repository Structure

QADSS/
|
+-- README.md
+-- .gitignore
|
+-- notebooks/
    +-- 01_data_exploration.ipynb

## Author

Dr. Ghafar Eskandari

PhD in Biotechnology

Data Science, Machine Learning, and AI-Based Decision Support

## Disclaimer

This project is intended for research, education, and decision-support purposes only. It is not financial advice and does not guarantee future investment performance.
