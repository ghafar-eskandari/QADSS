# QADSS — Quantitative & AI-Based Stock Trading Decision Support System

## Overview

**QADSS** is a research-oriented prototype for quantitative market analysis and AI-based decision support.

The project explores how a structured machine-learning pipeline can be used to transform historical market data into:

* Predictive probabilities
* Risk indicators
* Explainable model outputs
* Threshold-based decision signals
* Data-driven decision support

The initial case study uses historical market data for **Bank Mellat (وبملت)** from the **TSETMC** market data platform.

> **Important:** QADSS is a research and methodological prototype.
> It is **not** presented as a guaranteed profitable trading strategy, investment recommendation, or production-ready automated trading system.

---

## Research Objective

The primary objective of QADSS is to develop and examine a reproducible end-to-end machine-learning pipeline for financial decision support.

The project focuses on methodological questions such as:

* How should the prediction target be defined?
* How can temporal data leakage and look-ahead bias be controlled?
* Which market features provide useful information?
* How should models be validated on time-dependent data?
* How should predicted probabilities be calibrated?
* How can model outputs be explained using SHAP?
* How should probability thresholds be translated into decision rules?
* How can risk and economic implications be evaluated?

The current stage emphasizes **methodological validation rather than claims of predictive superiority or profitability**.

---

# End-to-End Pipeline

The current QADSS architecture follows this general workflow:

```text
Market Data
     ↓
Data Quality & Integrity
     ↓
Feature Engineering
     ↓
Target Definition
     ↓
Model-Ready Dataset
     ↓
Baseline Model
     ↓
Logistic Regression
     ↓
XGBoost
     ↓
Walk-Forward Validation
     ↓
Model Stability
     ↓
Probability Calibration
     ↓
SHAP Explainability
     ↓
Threshold Analysis
     ↓
Economic & Risk Evaluation
     ↓
Decision Support
```

The pipeline is designed so that each stage can be examined independently.

---

# Case Study

### Asset

**Bank Mellat — وبملت**

### Data Source

Historical market data obtained from **TSETMC**.

### Prediction Horizon

The current target formulation focuses on the **next five trading days**.

The objective is not simply to predict the exact future price. Instead, the project investigates whether available historical and technical information can provide useful information for a defined future market outcome.

---

# Main Features

The current feature set includes variables representing:

### Return

* `return_1d`
* `return_5d`
* `return_10d`
* `return_20d`

### Moving Average Structure

* `close_to_ma5`
* `close_to_ma20`
* `ma5_to_ma20`

### Trend

* `ma5_slope_5d`
* `ma20_slope_5d`

### Volatility

* `volatility_20`

### Volume

* `volume_ratio_20`

### Trading Activity

* `no_trade`

These features are intended to provide a compact representation of recent price, trend, volatility, and trading activity.

---

# Target Definition

The target is defined according to the project's five-trading-day prediction framework.

The target construction is treated as an important methodological component because incorrect target construction can introduce:

* Look-ahead bias
* Target contamination
* Incorrect temporal alignment
* Artificial model performance

Therefore, target integrity is explicitly examined in the later stages of the project.

---

# Modeling

The project currently evaluates several modeling stages.

## 1. Baseline

A baseline model is used as a reference point before applying more complex machine-learning methods.

The purpose is to determine whether more sophisticated models provide meaningful additional information beyond a simple reference approach.

---

## 2. Logistic Regression

Logistic Regression provides a relatively interpretable probabilistic baseline.

It is useful for examining whether the relationship between engineered features and the target can be captured using a comparatively simple model.

---

## 3. XGBoost

XGBoost is used to investigate nonlinear relationships and interactions between features.

The current implementation uses a binary classification framework to estimate the probability of the target class.

---

# Temporal Validation

Financial data are inherently time-dependent.

Therefore, conventional random train/test splitting is not considered sufficient for the main validation framework.

QADSS uses **walk-forward validation** to examine model behavior across sequential time periods.

Conceptually:

```text
Train → Validate/Test
       ↓
Move Forward
       ↓
Train → Validate/Test
       ↓
Move Forward
       ↓
...
```

This approach is intended to provide a more realistic assessment of how a model behaves when trained on historical information and evaluated on subsequent observations.

---

# Model Stability

Model performance should not depend entirely on one particular time period.

QADSS therefore includes additional analyses addressing:

* Model stability
* Common test periods
* Different training sizes
* Date-aligned sensitivity
* Integrity sensitivity

These analyses are intended to determine whether conclusions remain reasonably stable under different methodological conditions.

---

# Probability Calibration

Classification models can produce probabilities that are not necessarily well calibrated.

QADSS therefore includes a dedicated probability calibration stage.

The objective is to distinguish between:

```text
Model prediction
        ↓
Probability estimate
        ↓
Calibrated probability
        ↓
Decision threshold
```

This distinction is important because a decision-support system should not automatically interpret an uncalibrated model output as a reliable probability.

---

# Explainability with SHAP

QADSS uses **SHAP (SHapley Additive exPlanations)** to investigate the contribution of individual features to model predictions.

The explainability stage examines:

* Feature importance
* SHAP direction
* Feature dependence
* SHAP stability
* Threshold-related feature behavior

The purpose is not only to ask:

> "What did the model predict?"

but also:

> "Which features contributed to that prediction?"

---

# Threshold Analysis

A model probability does not automatically constitute a decision.

QADSS therefore separates:

```text
Prediction
     ↓
Probability
     ↓
Decision Threshold
     ↓
Decision Signal
```

Threshold analysis is used to investigate how different probability thresholds affect the resulting decisions.

The current decision-support prototype uses a configurable threshold and should not be interpreted as an independently validated optimal trading threshold unless supported by the corresponding out-of-sample analysis.

---

# Risk & Economic Evaluation

The project also investigates the relationship between model outputs and decision consequences.

Relevant analyses include:

* Economic backtesting
* Risk performance
* Threshold behavior
* Sensitivity analysis
* Data integrity effects
* Extreme-event behavior

These analyses are intended to help distinguish **statistical model performance** from **practical decision usefulness**.

---

# Data Integrity & Contamination Analysis

A significant part of the project is devoted to checking whether data issues could affect model conclusions.

The pipeline includes analyses related to:

* Corporate-action windows
* Corporate-action residuals
* Target contamination
* Feature contamination
* Event-aware data cleaning
* Clean dataset construction
* Extreme event classification
* Integrity sensitivity

The purpose is to reduce the possibility that apparent model behavior is caused by data artifacts rather than meaningful predictive information.

---

# Decision Support Layer

The final stage of QADSS translates model outputs into a structured decision-support representation.

Conceptually:

```text
Market Data
     ↓
Features
     ↓
ML Model
     ↓
Probability
     ↓
Threshold
     ↓
Signal
     ↓
Confidence
     ↓
Risk Assessment
     ↓
Decision Support
     ↓
SHAP Explanation
```

The decision-support layer is intentionally separated from the underlying predictive model.

This distinction is important:

**The ML model estimates an outcome probability.**

**The decision-support layer applies explicit rules to that probability and other risk information.**

Therefore, elements such as confidence category, risk level, and position-sizing rules should not be interpreted as direct outputs of the machine-learning model.

---

# Current Decision Support Prototype

The current prototype generates a structured output containing elements such as:

* Prediction probability
* Decision threshold
* Signal
* Confidence category
* Volatility-based risk level
* Position-size rule
* Data-quality status
* Data-quality warnings
* Decision status
* Top SHAP contributing features

The current implementation is intended primarily to demonstrate the **architecture of a decision-support layer**.

It should not yet be interpreted as a production-grade real-time trading engine.

---

# Project Structure

```text
QADSS/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│
├── notebooks/
│
├── reports/
│
├── src/
│   ├── 01_create_processed_data.py
│   ├── 02_feature_engineering.py
│   ├── 03_target_definition.py
│   ├── 04_prepare_model_data.py
│   ├── 05_baseline.py
│   ├── 06_logistic_regression.py
│   ├── 07_xgboost.py
│   ├── ...
│   ├── 49_final_dataset_validation.py
│   ├── 50_final_walkforward_calibration.py
│   ├── 51_shap_threshold.py
│   ├── 52_economic_risk_evaluation.py
│   └── 53_qadss_decision_support.py
│
└── README.md
```

The numbered scripts reflect the progressive development of the research pipeline.

The later stages consolidate the major methodological checks and final decision-support components.

---

# Technology Stack

The project is implemented in Python using tools including:

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* SHAP
* Matplotlib
* JupyterLab

---

# Methodological Principles

The project follows several principles:

### 1. Time Awareness

Future information must not be used to construct historical features or training information.

### 2. Separation of Stages

Data preparation, feature engineering, target construction, modeling, validation, calibration, explainability, and decision support are treated as separate stages.

### 3. Baseline Before Complexity

More complex models should be compared against simpler reference models.

### 4. Out-of-Sample Thinking

Model evaluation should focus on information that would genuinely have been available at the prediction time.

### 5. Explainability

A useful decision-support system should provide not only a prediction but also interpretable information about the factors contributing to that prediction.

### 6. Sensitivity Analysis

Important conclusions should be examined under alternative methodological conditions.

### 7. No Guaranteed Profitability Claims

Statistical modeling results do not automatically imply profitable trading performance.

---

# Current Research Status

**Status: Research Prototype / Methodological Validation**

The current project should be viewed as an experimental framework for studying:

```text
Data
→ Features
→ Target
→ Models
→ Validation
→ Calibration
→ Explainability
→ Risk
→ Decision Support
```

The primary goal at this stage is to make the methodology transparent and open to expert review.

---

# Limitations

Several limitations remain before the framework could be considered a production-grade trading system.

These include:

* The current case study focuses on a single security.
* Historical market behavior may not represent future market regimes.
* Model probabilities require careful out-of-sample validation and calibration.
* Decision thresholds require independent validation.
* Risk and position-sizing rules are currently rule-based.
* Economic backtesting requires careful treatment of transaction costs, liquidity, execution assumptions, and other market frictions.
* A decision-support prototype should not be interpreted as investment advice.

In particular, the current implementation of the final decision-support script is primarily intended to demonstrate the decision-support architecture. A production implementation would need to ensure that the probability presented for a current observation is generated strictly from a model trained only on information available before that observation.

---

# Planned Development

Future development may include:

1. Extending the analysis to multiple securities.
2. Testing longer historical periods.
3. Evaluating different market regimes.
4. Strengthening out-of-sample probability generation.
5. Improving calibration and threshold selection.
6. Incorporating transaction costs and liquidity assumptions.
7. Improving risk modeling.
8. Comparing alternative feature sets.
9. Developing a more rigorous real-time decision-support architecture.

---

# Research Questions for Expert Review

I welcome methodological feedback from researchers and practitioners in:

* Machine Learning
* Data Science
* Quantitative Finance
* Financial Econometrics
* AI-based Decision Support
* Time-Series Modeling

In particular, feedback is welcome on:

1. **Target construction**
2. **Data leakage and look-ahead bias**
3. **Feature engineering**
4. **Temporal data splitting**
5. **Walk-forward validation**
6. **Probability calibration**
7. **SHAP-based explainability**
8. **Threshold selection**
9. **Risk evaluation**
10. **Economic backtesting**
11. **Decision-support architecture**

The key question is:

> **What methodological changes would be necessary before extending this framework to multiple securities, longer periods, and different market regimes?**

---

# Research Philosophy

The purpose of this project is not to demonstrate that an AI model can "beat the market."

Instead, the objective is to investigate whether a carefully structured and transparent machine-learning pipeline can support quantitative decision-making while making its assumptions, limitations, uncertainty, and potential sources of bias explicit.

The same methodological framework may eventually be transferable to other domains where decisions depend on:

```text
Historical Data
      ↓
Feature Engineering
      ↓
Prediction
      ↓
Uncertainty / Risk
      ↓
Explainability
      ↓
Decision Support
```

This is particularly relevant to future applications in data-driven agricultural and livestock decision support.

---

# Repository

GitHub:

**QADSS — Quantitative & AI-Based Stock Trading Decision Support System**

https://github.com/ghafar-eskandari/QADSS

---

# Disclaimer

This repository is intended for research, educational, and methodological purposes.

It does not constitute financial, investment, or trading advice.

No guarantee is made regarding future predictive performance or profitability.

Any practical use of model outputs should involve independent validation, appropriate risk management, and consideration of market conditions and execution constraints.

---

## Author

**Dr. Ghafar Eskandari**

PhD in Biotechnology

Research interests:

* Data Science
* Artificial Intelligence
* Quantitative Decision Support
* Agricultural Data Analytics
* Poultry Genetics & Breeding
* AI and Digital Transformation in Agriculture
