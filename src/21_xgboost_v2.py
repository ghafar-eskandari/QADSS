from pathlib import Path

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from xgboost import XGBClassifier


# --------------------------------------------------
# 1. Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "webmelat_model_ready_v2.csv"
)


# --------------------------------------------------
# 2. Configuration
# --------------------------------------------------

TARGET = "target"

TRAIN_RATIO = 0.60
PURGE_DAYS = 5

RANDOM_STATE = 42


# --------------------------------------------------
# 3. Features
# --------------------------------------------------

FEATURES_V2 = [
    "return_1d",
    "return_5d",
    "return_10d",
    "return_20d",
    "close_to_ma5",
    "close_to_ma20",
    "ma5_to_ma20",
    "ma5_slope_5d",
    "ma20_slope_5d",
    "volatility_20",
    "volume_ratio_20",
    "no_trade",
]


# --------------------------------------------------
# 4. Load data
# --------------------------------------------------

df = pd.read_csv(INPUT_PATH)


# --------------------------------------------------
# 5. Validation
# --------------------------------------------------

required_columns = FEATURES_V2 + [TARGET]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        "Missing required columns: "
        + ", ".join(missing_columns)
    )

if df[required_columns].isna().any().any():
    raise ValueError(
        "Missing values detected in model data."
    )


# --------------------------------------------------
# 6. Preserve chronological order
# --------------------------------------------------

if "dEven" in df.columns:
    df = df.sort_values(
        "dEven"
    ).reset_index(drop=True)


# --------------------------------------------------
# 7. Prepare X and y
# --------------------------------------------------

X = df[FEATURES_V2].copy()

y = df[TARGET].astype(int)


# --------------------------------------------------
# 8. Purged time split
# --------------------------------------------------

n = len(df)

split_index = int(
    TRAIN_RATIO * n
)

train_end = (
    split_index - PURGE_DAYS
)

X_train = X.iloc[
    :train_end
].copy()

y_train = y.iloc[
    :train_end
].copy()

X_test = X.iloc[
    split_index:
].copy()

y_test = y.iloc[
    split_index:
].copy()


# --------------------------------------------------
# 9. XGBoost
# --------------------------------------------------

model = XGBClassifier(
    n_estimators=300,
    max_depth=3,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)


# --------------------------------------------------
# 10. Train
# --------------------------------------------------

model.fit(
    X_train,
    y_train
)


# --------------------------------------------------
# 11. Predictions
# --------------------------------------------------

y_pred = model.predict(
    X_test
)

y_prob = model.predict_proba(
    X_test
)[:, 1]


# --------------------------------------------------
# 12. Metrics
# --------------------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

balanced_accuracy = (
    balanced_accuracy_score(
        y_test,
        y_pred
    )
)

roc_auc = roc_auc_score(
    y_test,
    y_prob
)

cm = confusion_matrix(
    y_test,
    y_pred
)


# --------------------------------------------------
# 13. Report
# --------------------------------------------------

print("=" * 80)
print("QADSS - XGBOOST V2")
print("=" * 80)

print(
    f"Total observations: {n}"
)

print(
    f"Split index: {split_index}"
)

print(
    f"Purged rows: {PURGE_DAYS}"
)

print(
    f"Training observations: "
    f"{len(X_train)}"
)

print(
    f"Test observations: "
    f"{len(X_test)}"
)


# --------------------------------------------------
# 14. Dates
# --------------------------------------------------

if "dEven" in df.columns:

    print("\n")

    print(
        f"Train period: "
        f"{df.iloc[0]['dEven']} "
        f"to "
        f"{df.iloc[train_end - 1]['dEven']}"
    )

    print(
        f"Test period: "
        f"{df.iloc[split_index]['dEven']} "
        f"to "
        f"{df.iloc[-1]['dEven']}"
    )


# --------------------------------------------------
# 15. Metrics
# --------------------------------------------------

print("\n")

print(
    f"Accuracy: "
    f"{accuracy:.4f}"
)

print(
    f"Balanced Accuracy: "
    f"{balanced_accuracy:.4f}"
)

print(
    f"ROC-AUC: "
    f"{roc_auc:.4f}"
)


# --------------------------------------------------
# 16. Confusion Matrix
# --------------------------------------------------

print("\n")
print("Confusion Matrix:")
print(cm)


# --------------------------------------------------
# 17. Classification Report
# --------------------------------------------------

print("\n")
print("Classification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        digits=4
    )
)


# --------------------------------------------------
# 18. Feature Importance
# --------------------------------------------------

print("\n")
print("=" * 80)
print("XGBOOST FEATURE IMPORTANCE")
print("=" * 80)

importance_df = pd.DataFrame(
    {
        "feature": FEATURES_V2,
        "importance": model.feature_importances_,
    }
)

importance_df = importance_df.sort_values(
    "importance",
    ascending=False
)

print(
    importance_df.to_string(
        index=False
    )
)

print("=" * 80)