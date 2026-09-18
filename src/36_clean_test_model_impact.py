import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    confusion_matrix
)

MODEL = r".\data\processed\webmelat_model_ready_v2.csv"
IMPACT = r".\data\processed\contaminated_target_impact.csv"
OUTPUT = r".\data\processed\clean_test_model_impact.csv"

# Load model-ready data
df = pd.read_csv(
    MODEL,
    dtype={"dEven": str}
)

df["date"] = pd.to_datetime(
    df["dEven"],
    format="%Y-%m-%d"
)

df = df.sort_values("date").reset_index(drop=True)

# Stage 19 split
TRAIN_SIZE = 2387
PURGE_SIZE = 5

df["split"] = "TEST"

df.loc[
    :TRAIN_SIZE - 1,
    "split"
] = "TRAIN"

df.loc[
    TRAIN_SIZE:TRAIN_SIZE + PURGE_SIZE - 1,
    "split"
] = "PURGE"

# Load contamination flags
impact = pd.read_csv(
    IMPACT,
    dtype={"dEven": str}
)

impact = impact[
    [
        "dEven",
        "target_contaminated"
    ]
].copy()

impact["target_contaminated"] = (
    impact["target_contaminated"]
    .astype(bool)
)

# Merge contamination information
df = df.merge(
    impact,
    on="dEven",
    how="left"
)

df["target_contaminated"] = (
    df["target_contaminated"]
    .fillna(False)
)

# --------------------------------------------------
# IMPORTANT:
# We use the existing XGBoost predictions saved/
# reconstructed from the Stage 21 model procedure.
# --------------------------------------------------

from xgboost import XGBClassifier

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
    "no_trade"
]

X = df[FEATURES]
y = df["target"].astype(int)

train_idx = df["split"] == "TRAIN"
test_idx = df["split"] == "TEST"

X_train = X.loc[train_idx]
y_train = y.loc[train_idx]

X_test = X.loc[test_idx]
y_test = y.loc[test_idx]

# Same XGBoost configuration used in the V2 model
model = XGBClassifier(
    n_estimators=300,
    max_depth=3,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42
)

model.fit(
    X_train,
    y_train
)

test_prob = model.predict_proba(
    X_test
)[:, 1]

test_pred = (
    test_prob >= 0.50
).astype(int)

test_result = df.loc[
    test_idx
].copy()

test_result["pred_prob"] = test_prob
test_result["prediction"] = test_pred

# --------------------------------------------------
# Evaluation function
# --------------------------------------------------

def evaluate(data, name):

    y_true = data["target"].astype(int)
    y_pred = data["prediction"].astype(int)
    prob = data["pred_prob"]

    auc = roc_auc_score(
        y_true,
        prob
    )

    acc = accuracy_score(
        y_true,
        y_pred
    )

    bal = balanced_accuracy_score(
        y_true,
        y_pred
    )

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    return {
        "dataset": name,
        "n": len(data),
        "target_0": int(
            (y_true == 0).sum()
        ),
        "target_1": int(
            (y_true == 1).sum()
        ),
        "accuracy": acc,
        "balanced_accuracy": bal,
        "roc_auc": auc,
        "TN": int(cm[0, 0]),
        "FP": int(cm[0, 1]),
        "FN": int(cm[1, 0]),
        "TP": int(cm[1, 1])
    }

# Full Test
full_result = evaluate(
    test_result,
    "FULL_TEST"
)

# Clean Test
clean_result = evaluate(
    test_result[
        ~test_result["target_contaminated"]
    ],
    "CLEAN_TEST"
)

# Contaminated Test
contaminated_result = evaluate(
    test_result[
        test_result["target_contaminated"]
    ],
    "CONTAMINATED_TEST"
)

results = pd.DataFrame([
    full_result,
    clean_result,
    contaminated_result
])

# Calculate changes
full = results[
    results["dataset"] == "FULL_TEST"
].iloc[0]

clean = results[
    results["dataset"] == "CLEAN_TEST"
].iloc[0]

print()
print("=== STAGE 20.6: CLEAN-TEST MODEL IMPACT ===")
print()

print(results.to_string(index=False))

print()
print("=== FULL TEST -> CLEAN TEST CHANGE ===")

print(
    "AUC change:",
    round(
        clean["roc_auc"]
        - full["roc_auc"],
        4
    )
)

print(
    "Accuracy change:",
    round(
        clean["accuracy"]
        - full["accuracy"],
        4
    )
)

print(
    "Balanced Accuracy change:",
    round(
        clean["balanced_accuracy"]
        - full["balanced_accuracy"],
        4
    )
)

results.to_csv(
    OUTPUT,
    index=False,
    encoding="utf-8-sig"
)

print()
print("Saved:", OUTPUT)
