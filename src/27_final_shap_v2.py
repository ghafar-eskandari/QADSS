import pandas as pd
import numpy as np
import shap

from xgboost import XGBClassifier


# ============================================================
# QADSS - FINAL SHAP & EXPLAINABILITY - V2 XGBOOST
# ============================================================

INPUT_FILE = (
    "data/processed/"
    "webmelat_model_ready_v2.csv"
)

OUTPUT_FILE = (
    "data/processed/"
    "v2_final_shap_results.csv"
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


print("=" * 80)
print("QADSS - FINAL SHAP & EXPLAINABILITY - V2 XGBOOST")
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
# 3. TIME-BASED TRAIN / TEST SPLIT
# ============================================================

n = len(df)

split_index = int(n * 0.60)

purge = 5

train_end = split_index - purge

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
    f"Purged observations  : {purge}"
)


# ============================================================
# 4. TRAIN FINAL XGBOOST STRUCTURE
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
# 5. SHAP EXPLAINER
# ============================================================

print("\n")
print("=" * 80)
print("CALCULATING SHAP VALUES")
print("=" * 80)

explainer = shap.TreeExplainer(
    model
)

shap_values = explainer.shap_values(
    X_test
)


# ============================================================
# 6. MEAN ABSOLUTE SHAP
# ============================================================

mean_abs_shap = np.abs(
    shap_values
).mean(
    axis=0
)

importance_df = pd.DataFrame(
    {
        "Feature": FEATURES,
        "Mean_Abs_SHAP": mean_abs_shap
    }
)

importance_df = importance_df.sort_values(
    "Mean_Abs_SHAP",
    ascending=False
).reset_index(
    drop=True
)


# ============================================================
# 7. MEAN SIGNED SHAP
# ============================================================

mean_signed_shap = shap_values.mean(
    axis=0
)

direction_df = pd.DataFrame(
    {
        "Feature": FEATURES,
        "Mean_SHAP": mean_signed_shap
    }
)

direction_df = direction_df.sort_values(
    "Mean_SHAP",
    ascending=False
).reset_index(
    drop=True
)


# ============================================================
# 8. COMBINE RESULTS
# ============================================================

results_df = importance_df.merge(
    direction_df,
    on="Feature"
)


results_df["Direction"] = np.where(
    results_df["Mean_SHAP"] > 0,
    "Positive",
    np.where(
        results_df["Mean_SHAP"] < 0,
        "Negative",
        "Neutral"
    )
)


# ============================================================
# 9. PRINT FEATURE IMPORTANCE
# ============================================================

print("\n")
print("=" * 80)
print("FEATURE IMPORTANCE - MEAN ABSOLUTE SHAP")
print("=" * 80)

print(
    importance_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# ============================================================
# 10. PRINT SHAP DIRECTION
# ============================================================

print("\n")
print("=" * 80)
print("SHAP DIRECTION")
print("=" * 80)

print(
    direction_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# ============================================================
# 11. FINAL EXPLAINABILITY TABLE
# ============================================================

print("\n")
print("=" * 80)
print("FINAL EXPLAINABILITY TABLE")
print("=" * 80)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# ============================================================
# 12. IMPORTANT INTERPRETATION
# ============================================================

print("\n")
print("=" * 80)
print("INTERPRETATION NOTE")
print("=" * 80)

print(
    "\nMean Absolute SHAP measures how strongly "
    "a feature contributes to model output."
)

print(
    "Mean SHAP indicates the average direction "
    "of contribution."
)

print(
    "\nSHAP importance does NOT prove causality."
)

print(
    "A positive SHAP value means the feature "
    "pushes the model output toward class 1 "
    "for that observation."
)

print(
    "A negative SHAP value means the feature "
    "pushes the model output toward class 0 "
    "for that observation."
)


# ============================================================
# 13. SAVE RESULTS
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