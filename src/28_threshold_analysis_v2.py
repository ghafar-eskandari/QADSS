import pandas as pd
import numpy as np

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# ============================================================
# QADSS - THRESHOLD ANALYSIS - V2 XGBOOST
# ============================================================

INPUT_FILE = (
    "data/processed/"
    "webmelat_model_ready_v2.csv"
)

OUTPUT_FILE = (
    "data/processed/"
    "v2_threshold_analysis.csv"
)

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

PURGE = 5


print("=" * 80)
print("QADSS - THRESHOLD ANALYSIS - V2 XGBOOST")
print("=" * 80)


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv(
    INPUT_FILE
)

print(
    f"\nTotal observations: {len(df)}"
)


# ============================================================
# 2. PREPARE DATA
# ============================================================

X = df[FEATURES].copy()

y = df[TARGET].astype(int)


# ============================================================
# 3. TIME-BASED SPLIT
# ============================================================

n = len(df)

split_index = int(n * 0.60)

train_end = split_index - PURGE

X_train = X.iloc[
    :train_end
]

y_train = y.iloc[
    :train_end
]

X_test = X.iloc[
    split_index:
]

y_test = y.iloc[
    split_index:
]


print("\n")
print("=" * 80)
print("TIME-BASED SPLIT")
print("=" * 80)

print(
    f"Training observations: {len(X_train)}"
)

print(
    f"Test observations    : {len(X_test)}"
)

print(
    f"Purged observations  : {PURGE}"
)


# ============================================================
# 4. TRAIN XGBOOST
# ============================================================

model = XGBClassifier(
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


print("\n")
print("=" * 80)
print("TRAINING XGBOOST")
print("=" * 80)

model.fit(
    X_train,
    y_train
)


# ============================================================
# 5. PREDICT PROBABILITIES
# ============================================================

probabilities = model.predict_proba(
    X_test
)[:, 1]


auc = roc_auc_score(
    y_test,
    probabilities
)


print("\n")
print("=" * 80)
print("BASE MODEL")
print("=" * 80)

print(
    f"\nROC-AUC: {auc:.4f}"
)


# ============================================================
# 6. THRESHOLD GRID
# ============================================================

thresholds = np.arange(
    0.30,
    0.71,
    0.05
)


results = []


# ============================================================
# 7. EVALUATE EACH THRESHOLD
# ============================================================

for threshold in thresholds:

    predictions = (
        probabilities >= threshold
    ).astype(int)

    signal_count = predictions.sum()

    signal_rate = (
        signal_count
        / len(predictions)
    )

    results.append(
        {
            "Threshold": round(
                float(threshold),
                2
            ),

            "Accuracy": accuracy_score(
                y_test,
                predictions
            ),

            "Balanced_Accuracy":
                balanced_accuracy_score(
                    y_test,
                    predictions
                ),

            "Precision":
                precision_score(
                    y_test,
                    predictions,
                    zero_division=0
                ),

            "Recall":
                recall_score(
                    y_test,
                    predictions,
                    zero_division=0
                ),

            "F1":
                f1_score(
                    y_test,
                    predictions,
                    zero_division=0
                ),

            "Signal_Count":
                int(signal_count),

            "Signal_Rate":
                signal_rate
        }
    )


results_df = pd.DataFrame(
    results
)


# ============================================================
# 8. PRINT RESULTS
# ============================================================

print("\n")
print("=" * 80)
print("THRESHOLD RESULTS")
print("=" * 80)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# 9. BEST THRESHOLD BY BALANCED ACCURACY
# ============================================================

best_balanced_idx = (
    results_df[
        "Balanced_Accuracy"
    ].idxmax()
)

best_balanced = results_df.loc[
    best_balanced_idx
]


print("\n")
print("=" * 80)
print("BEST THRESHOLD BY BALANCED ACCURACY")
print("=" * 80)

print(
    f"\nThreshold: "
    f"{best_balanced['Threshold']:.2f}"
)

print(
    f"Balanced Accuracy: "
    f"{best_balanced['Balanced_Accuracy']:.4f}"
)

print(
    f"Precision: "
    f"{best_balanced['Precision']:.4f}"
)

print(
    f"Recall: "
    f"{best_balanced['Recall']:.4f}"
)

print(
    f"F1: "
    f"{best_balanced['F1']:.4f}"
)

print(
    f"Signal Count: "
    f"{int(best_balanced['Signal_Count'])}"
)

print(
    f"Signal Rate: "
    f"{best_balanced['Signal_Rate']:.4f}"
)


# ============================================================
# 10. IMPORTANT WARNING
# ============================================================

print("\n")
print("=" * 80)
print("IMPORTANT")
print("=" * 80)

print(
    "\nThe threshold with the highest "
    "Balanced Accuracy is NOT automatically "
    "the final trading threshold."
)

print(
    "Final threshold selection requires "
    "economic evaluation / backtesting."
)

print(
    "A threshold that produces fewer signals "
    "may have different economic consequences "
    "from one that produces many signals."
)


# ============================================================
# 11. SAVE RESULTS
# ============================================================

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\n")
print("=" * 80)
print("RESULTS SAVED")
print("=" * 80)

print(
    f"\nSaved to:\n{OUTPUT_FILE}"
)

print("=" * 80)