import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
)

from xgboost import XGBClassifier


# ============================================================
# QADSS - WALK-FORWARD VALIDATION V2
# ============================================================

INPUT_FILE = "data/processed/webmelat_model_ready_v2.csv"

FEATURES = [
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

TARGET = "target"

INITIAL_TRAIN_RATIO = 0.40
TEST_RATIO = 0.10
PURGE_DAYS = 5


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

X = df[FEATURES]
y = df[TARGET].astype(int)

n = len(df)

print("=" * 80)
print("QADSS - WALK-FORWARD VALIDATION V2")
print("=" * 80)

print(f"Total observations: {n}")
print(f"Number of features: {len(FEATURES)}")
print(f"Initial train ratio: {INITIAL_TRAIN_RATIO}")
print(f"Test ratio per fold: {TEST_RATIO}")
print(f"Purged rows: {PURGE_DAYS}")


# ============================================================
# MODEL DEFINITIONS
# ============================================================

logistic_model = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        (
            "model",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        ),
    ]
)


xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=3,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
)


# ============================================================
# WALK-FORWARD PARAMETERS
# ============================================================

initial_train_size = int(n * INITIAL_TRAIN_RATIO)
test_size = int(n * TEST_RATIO)

results = []

fold = 1
train_end = initial_train_size


# ============================================================
# WALK-FORWARD LOOP
# ============================================================

while True:

    test_start = train_end + PURGE_DAYS
    test_end = test_start + test_size

    if test_end > n:
        break

    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]

    X_test = X.iloc[test_start:test_end]
    y_test = y.iloc[test_start:test_end]

    print("\n" + "=" * 80)
    print(f"FOLD {fold}")
    print("=" * 80)

    print(f"Train: 0 -> {train_end - 1}")
    print(f"Purge: {train_end} -> {test_start - 1}")
    print(f"Test: {test_start} -> {test_end - 1}")

    print(f"Training observations: {len(X_train)}")
    print(f"Test observations: {len(X_test)}")


    # --------------------------------------------------------
    # LOGISTIC REGRESSION
    # --------------------------------------------------------

    logistic_model.fit(X_train, y_train)

    logistic_pred = logistic_model.predict(X_test)
    logistic_prob = logistic_model.predict_proba(X_test)[:, 1]

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


    # --------------------------------------------------------
    # XGBOOST
    # --------------------------------------------------------

    xgb_model.fit(X_train, y_train)

    xgb_pred = xgb_model.predict(X_test)
    xgb_prob = xgb_model.predict_proba(X_test)[:, 1]

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


    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print("\nLogistic Regression:")
    print(f"Accuracy:          {logistic_accuracy:.4f}")
    print(f"Balanced Accuracy: {logistic_balanced_accuracy:.4f}")
    print(f"ROC-AUC:           {logistic_auc:.4f}")

    print("\nXGBoost:")
    print(f"Accuracy:          {xgb_accuracy:.4f}")
    print(f"Balanced Accuracy: {xgb_balanced_accuracy:.4f}")
    print(f"ROC-AUC:           {xgb_auc:.4f}")


    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    results.append(
        {
            "fold": fold,

            "logistic_accuracy":
                logistic_accuracy,

            "logistic_balanced_accuracy":
                logistic_balanced_accuracy,

            "logistic_auc":
                logistic_auc,

            "xgb_accuracy":
                xgb_accuracy,

            "xgb_balanced_accuracy":
                xgb_balanced_accuracy,

            "xgb_auc":
                xgb_auc,
        }
    )


    # --------------------------------------------------------
    # MOVE FORWARD
    # --------------------------------------------------------

    train_end = test_end
    fold += 1


# ============================================================
# RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(results)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("WALK-FORWARD SUMMARY V2")
print("=" * 80)

print("\nFold-by-fold results:")
print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


print("\n" + "=" * 80)
print("MEAN PERFORMANCE")
print("=" * 80)

print("\nLogistic Regression:")
print(
    f"Mean Accuracy:          "
    f"{results_df['logistic_accuracy'].mean():.4f}"
)

print(
    f"Mean Balanced Accuracy: "
    f"{results_df['logistic_balanced_accuracy'].mean():.4f}"
)

print(
    f"Mean ROC-AUC:            "
    f"{results_df['logistic_auc'].mean():.4f}"
)


print("\nXGBoost:")
print(
    f"Mean Accuracy:          "
    f"{results_df['xgb_accuracy'].mean():.4f}"
)

print(
    f"Mean Balanced Accuracy: "
    f"{results_df['xgb_balanced_accuracy'].mean():.4f}"
)

print(
    f"Mean ROC-AUC:            "
    f"{results_df['xgb_auc'].mean():.4f}"
)


# ============================================================
# SAVE RESULTS
# ============================================================

OUTPUT_FILE = (
    "data/processed/"
    "walk_forward_results_v2.csv"
)

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nResults saved to:")
print(OUTPUT_FILE)

print("=" * 80)