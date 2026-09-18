import pandas as pd


# ============================================================
# QADSS - FINAL MODEL SELECTION - V2
# ============================================================

COMMON_RESULTS = (
    "data/processed/"
    "v1_v2_common_period_results.csv"
)

STABILITY_RESULTS = (
    "data/processed/"
    "v2_model_stability_results.csv"
)

OUTPUT_FILE = (
    "data/processed/"
    "v2_final_model_selection.csv"
)


print("=" * 80)
print("QADSS - FINAL MODEL SELECTION - V2")
print("=" * 80)


# ============================================================
# 1. LOAD RESULTS
# ============================================================

common_df = pd.read_csv(
    COMMON_RESULTS
)

stability_df = pd.read_csv(
    STABILITY_RESULTS
)


print("\nResults loaded successfully.")


# ============================================================
# 2. EXTRACT MEAN PERFORMANCE
# ============================================================

performance = {
    "V2 Logistic": {
        "AUC Mean": common_df[
            "v2_logistic_auc"
        ].mean(),

        "Balanced Accuracy Mean": common_df[
            "v2_logistic_balanced"
        ].mean(),
    },

    "V2 XGBoost": {
        "AUC Mean": common_df[
            "v2_xgb_auc"
        ].mean(),

        "Balanced Accuracy Mean": common_df[
            "v2_xgb_balanced"
        ].mean(),
    },
}


# ============================================================
# 3. EXTRACT STABILITY
# ============================================================

for model in performance:

    row_auc = stability_df[
        (stability_df["Model"] == model)
        & (stability_df["Metric"] == "AUC")
    ]

    row_balanced = stability_df[
        (stability_df["Model"] == model)
        & (
            stability_df["Metric"]
            == "Balanced Accuracy"
        )
    ]

    performance[model][
        "AUC Std"
    ] = row_auc["Std"].iloc[0]

    performance[model][
        "Balanced Accuracy Std"
    ] = row_balanced["Std"].iloc[0]


# ============================================================
# 4. CREATE COMPARISON TABLE
# ============================================================

comparison_df = pd.DataFrame(
    performance
).T


print("\n")
print("=" * 80)
print("MODEL COMPARISON")
print("=" * 80)

print(
    comparison_df.to_string(
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# 5. DIRECT COMPARISON
# ============================================================

logistic_auc = performance[
    "V2 Logistic"
]["AUC Mean"]

xgb_auc = performance[
    "V2 XGBoost"
]["AUC Mean"]

logistic_balanced = performance[
    "V2 Logistic"
]["Balanced Accuracy Mean"]

xgb_balanced = performance[
    "V2 XGBoost"
]["Balanced Accuracy Mean"]

logistic_auc_std = performance[
    "V2 Logistic"
]["AUC Std"]

xgb_auc_std = performance[
    "V2 XGBoost"
]["AUC Std"]


print("\n")
print("=" * 80)
print("DIRECT COMPARISON")
print("=" * 80)


print(
    f"\nAUC Mean:"
)

print(
    f"  V2 Logistic : {logistic_auc:.4f}"
)

print(
    f"  V2 XGBoost  : {xgb_auc:.4f}"
)


print(
    f"\nBalanced Accuracy Mean:"
)

print(
    f"  V2 Logistic : {logistic_balanced:.4f}"
)

print(
    f"  V2 XGBoost  : {xgb_balanced:.4f}"
)


print(
    f"\nAUC Stability (Std):"
)

print(
    f"  V2 Logistic : {logistic_auc_std:.4f}"
)

print(
    f"  V2 XGBoost  : {xgb_auc_std:.4f}"
)


# ============================================================
# 6. PROVISIONAL REFERENCE MODEL
# ============================================================

if (
    xgb_auc > logistic_auc
    and
    xgb_balanced > logistic_balanced
):

    reference_model = "V2 XGBoost"

elif (
    logistic_auc > xgb_auc
    and
    logistic_balanced > xgb_balanced
):

    reference_model = "V2 Logistic"

else:

    reference_model = (
        "No unique reference model"
    )


print("\n")
print("=" * 80)
print("PROVISIONAL MODEL SELECTION")
print("=" * 80)

print(
    f"\nReference model: "
    f"{reference_model}"
)


# ============================================================
# 7. IMPORTANT LIMITATION
# ============================================================

print("\n")
print("=" * 80)
print("SELECTION STATUS")
print("=" * 80)

print(
    "\nThis is a provisional model selection."
)

print(
    "Final confirmation requires:"
)

print(
    "1. Probability calibration"
)

print(
    "2. Threshold analysis"
)

print(
    "3. Economic evaluation / backtest"
)

print(
    "4. Final SHAP explainability"
)


# ============================================================
# 8. SAVE RESULTS
# ============================================================

comparison_df.to_csv(
    OUTPUT_FILE
)

print("\n")
print("=" * 80)
print("RESULTS SAVED")
print("=" * 80)

print(
    f"\nSaved to:"
)

print(
    OUTPUT_FILE
)

print("\n")
print("=" * 80)