import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score
)

from xgboost import XGBClassifier


# --------------------------------------------------
# Configuration
# --------------------------------------------------

INPUT_PATH = "data/processed/webmelat_model_ready.csv"

FEATURES = [
    "return_1d",
    "return_5d",
    "ma_5",
    "ma_20",
    "price_to_ma20",
    "volatility_20",
    "volume",
    "no_trade",
]

TARGET = "target"

PURGE_DAYS = 5


# --------------------------------------------------
# Load data
# --------------------------------------------------

df = pd.read_csv(INPUT_PATH)

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values("date").reset_index(drop=True)


# --------------------------------------------------
# Same temporal split used in previous models
# --------------------------------------------------

split_index = int(len(df) * 0.60)

train_end = split_index - PURGE_DAYS

train = df.iloc[:train_end].copy()

test = df.iloc[split_index:].copy()


X_train = train[FEATURES]
y_train = train[TARGET]

X_test = test[FEATURES]
y_test = test[TARGET]


# --------------------------------------------------
# Baseline
# --------------------------------------------------

majority_class = y_train.mode()[0]

baseline_pred = [majority_class] * len(test)

baseline_accuracy = accuracy_score(
    y_test,
    baseline_pred
)

baseline_balanced_accuracy = balanced_accuracy_score(
    y_test,
    baseline_pred
)


# --------------------------------------------------
# Logistic Regression
# --------------------------------------------------

logistic_model = Pipeline([
    (
        "scaler",
        StandardScaler()
    ),
    (
        "logistic",
        LogisticRegression(
            max_iter=1000,
            random_state=42
        )
    )
])

logistic_model.fit(
    X_train,
    y_train
)

logistic_pred = logistic_model.predict(X_test)

logistic_prob = logistic_model.predict_proba(
    X_test
)[:, 1]


logistic_accuracy = accuracy_score(
    y_test,
    logistic_pred
)

logistic_balanced_accuracy = balanced_accuracy_score(
    y_test,
    logistic_pred
)

logistic_auc = roc_auc_score(
    y_test,
    logistic_prob
)


# --------------------------------------------------
# XGBoost
# --------------------------------------------------

xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=3,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(
    X_train,
    y_train
)

xgb_pred = xgb_model.predict(X_test)

xgb_prob = xgb_model.predict_proba(
    X_test
)[:, 1]


xgb_accuracy = accuracy_score(
    y_test,
    xgb_pred
)

xgb_balanced_accuracy = balanced_accuracy_score(
    y_test,
    xgb_pred
)

xgb_auc = roc_auc_score(
    y_test,
    xgb_prob
)


# --------------------------------------------------
# Comparison table
# --------------------------------------------------

results = pd.DataFrame([
    {
        "Model": "Baseline",
        "Accuracy": baseline_accuracy,
        "Balanced Accuracy": baseline_balanced_accuracy,
        "ROC-AUC": None
    },
    {
        "Model": "Logistic Regression",
        "Accuracy": logistic_accuracy,
        "Balanced Accuracy": logistic_balanced_accuracy,
        "ROC-AUC": logistic_auc
    },
    {
        "Model": "XGBoost",
        "Accuracy": xgb_accuracy,
        "Balanced Accuracy": xgb_balanced_accuracy,
        "ROC-AUC": xgb_auc
    }
])


# --------------------------------------------------
# Output
# --------------------------------------------------

print("QADSS Model Comparison")
print("=" * 60)
print()

print("Train rows:", len(train))
print("Purged rows:", PURGE_DAYS)
print("Test rows:", len(test))
print()

print("Train period:")
print(
    train["date"].min().date(),
    "to",
    train["date"].max().date()
)

print()

print("Test period:")
print(
    test["date"].min().date(),
    "to",
    test["date"].max().date()
)

print()

print("Comparison:")
print(
    results.round(4).to_string(index=False)
)

print()

print("Interpretation:")
print(
    "ROC-AUC is not available for the majority-class baseline."
)