import pandas as pd


# ============================================================
# QADSS - V2 MODEL STABILITY / REGIME ANALYSIS
# ============================================================

INPUT_FILE = (
    "data/processed/"
    "v1_v2_common_period_results.csv"
)


# ============================================================
# LOAD RESULTS
# ============================================================

df = pd.read_csv(INPUT_FILE)


print("=" * 80)
print("QADSS - V2 MODEL STABILITY / REGIME ANALYSIS")
print("=" * 80)

print(
    f"Number of folds: {len(df)}"
)


# ============================================================
# V2 METRICS
# ============================================================

metrics = {
    "V2 Logistic": {
        "AUC": "v2_logistic_auc",
        "Balanced Accuracy": "v2_logistic_balanced",
    },

    "V2 XGBoost": {
        "AUC": "v2_xgb_auc",
        "Balanced Accuracy": "v2_xgb_balanced",
    },
}


# ============================================================
# FOLD-BY-FOLD PERFORMANCE
# ============================================================

print("\n")
print("=" * 80)
print("FOLD-BY-FOLD PERFORMANCE")
print("=" * 80)

for model_name, model_metrics in metrics.items():

    print("\n" + "-" * 80)
    print(model_name)
    print("-" * 80)

    for metric_name, column in model_metrics.items():

        print(f"\n{metric_name}:")

        for _, row in df.iterrows():

            print(
                f"Fold {int(row['fold'])}: "
                f"{row[column]:.4f}"
            )


# ============================================================
# STABILITY SUMMARY
# ============================================================

print("\n")
print("=" * 80)
print("STABILITY SUMMARY")
print("=" * 80)

stability_results = []


for model_name, model_metrics in metrics.items():

    for metric_name, column in model_metrics.items():

        values = df[column]

        stability_results.append(
            {
                "Model": model_name,
                "Metric": metric_name,
                "Mean": values.mean(),
                "Std": values.std(ddof=1),
                "Min": values.min(),
                "Max": values.max(),
                "Range": values.max() - values.min(),
            }
        )


stability_df = pd.DataFrame(
    stability_results
)


print(
    stability_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# FOLD VARIABILITY
# ============================================================

print("\n")
print("=" * 80)
print("FOLD VARIABILITY")
print("=" * 80)

for model_name, model_metrics in metrics.items():

    for metric_name, column in model_metrics.items():

        values = df[column]

        print(
            f"{model_name} - {metric_name}: "
            f"Mean={values.mean():.4f}, "
            f"Std={values.std(ddof=1):.4f}, "
            f"Min={values.min():.4f}, "
            f"Max={values.max():.4f}"
        )


# ============================================================
# BEST AND WORST FOLD
# ============================================================

print("\n")
print("=" * 80)
print("BEST / WORST FOLD")
print("=" * 80)

for model_name, model_metrics in metrics.items():

    for metric_name, column in model_metrics.items():

        best_idx = df[column].idxmax()
        worst_idx = df[column].idxmin()

        best_fold = int(
            df.loc[best_idx, "fold"]
        )

        worst_fold = int(
            df.loc[worst_idx, "fold"]
        )

        best_value = df.loc[
            best_idx,
            column
        ]

        worst_value = df.loc[
            worst_idx,
            column
        ]

        print(
            f"{model_name} - {metric_name}:"
        )

        print(
            f"  Best  -> Fold {best_fold}: "
            f"{best_value:.4f}"
        )

        print(
            f"  Worst -> Fold {worst_fold}: "
            f"{worst_value:.4f}"
        )


# ============================================================
# SAVE RESULTS
# ============================================================

OUTPUT_FILE = (
    "data/processed/"
    "v2_model_stability_results.csv"
)


stability_df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\n")
print("=" * 80)
print("RESULTS SAVED")
print("=" * 80)

print(
    f"Saved to:\n{OUTPUT_FILE}"
)

print("=" * 80)