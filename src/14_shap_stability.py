from pathlib import Path

import pandas as pd
import shap
from xgboost import XGBClassifier


# --------------------------------------------------
# 1. Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "webmelat_model_ready.csv"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "shap_stability.csv"
)


# --------------------------------------------------
# 2. Features
# --------------------------------------------------

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


# --------------------------------------------------
# 3. Load data
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

df["dEven"] = pd.to_datetime(
    df["dEven"].astype(str),
    format="%Y%m%d"
)

df = df.dropna(
    subset=FEATURES + [TARGET]
).copy()

X = df[FEATURES]
y = df[TARGET].astype(int)


# --------------------------------------------------
# 4. Time split
# --------------------------------------------------

n = len(df)

split_index = int(0.60 * n)
purge_days = 5

train_end = split_index - purge_days

X_train = X.iloc[:train_end]
y_train = y.iloc[:train_end]

X_test = X.iloc[split_index:]
y_test = y.iloc[split_index:]

dates_test = df["dEven"].iloc[split_index:]


# --------------------------------------------------
# 5. Train XGBoost
# --------------------------------------------------

model = XGBClassifier(
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

model.fit(X_train, y_train)


# --------------------------------------------------
# 6. SHAP
# --------------------------------------------------

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X_test)

shap_df = pd.DataFrame(
    shap_values,
    columns=FEATURES,
    index=X_test.index
)


# --------------------------------------------------
# 7. Add dates
# --------------------------------------------------

analysis_df = X_test.copy()

analysis_df["date"] = dates_test.values

analysis_df["year"] = (
    analysis_df["date"]
    .dt.year
)


# --------------------------------------------------
# 8. Main features
# --------------------------------------------------

MAIN_FEATURES = [
    "ma_5",
    "volume",
    "volatility_20",
    "ma_20",
]


# --------------------------------------------------
# 9. Temporal SHAP stability
# --------------------------------------------------

results = []

for year in sorted(
    analysis_df["year"].unique()
):

    year_indices = (
        analysis_df["year"] == year
    )

    for feature in MAIN_FEATURES:

        feature_shap = shap_df.loc[
            year_indices,
            feature
        ]

        feature_values = analysis_df.loc[
            year_indices,
            feature
        ]

        results.append(
            {
                "year": int(year),
                "feature": feature,
                "mean_shap": feature_shap.mean(),
                "mean_abs_shap": (
                    feature_shap.abs().mean()
                ),
                "positive_shap_count": (
                    feature_shap > 0
                ).sum(),
                "negative_shap_count": (
                    feature_shap < 0
                ).sum(),
                "observations": len(
                    feature_shap
                ),
                "mean_feature_value": (
                    feature_values.mean()
                ),
            }
        )


# --------------------------------------------------
# 10. Create report
# --------------------------------------------------

stability_df = pd.DataFrame(
    results
)

stability_df = stability_df[
    [
        "year",
        "feature",
        "mean_shap",
        "mean_abs_shap",
        "positive_shap_count",
        "negative_shap_count",
        "observations",
        "mean_feature_value",
    ]
]


# --------------------------------------------------
# 11. Save report
# --------------------------------------------------

REPORT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

stability_df.to_csv(
    REPORT_PATH,
    index=False
)


# --------------------------------------------------
# 12. Display
# --------------------------------------------------

pd.set_option(
    "display.max_rows",
    200
)

pd.set_option(
    "display.float_format",
    lambda x: f"{x:.6f}"
)

print("=" * 80)
print("SHAP TEMPORAL STABILITY ANALYSIS")
print("=" * 80)

print(
    f"Test period: "
    f"{dates_test.iloc[0].date()} "
    f"to "
    f"{dates_test.iloc[-1].date()}"
)

print(
    f"Test observations: "
    f"{len(X_test)}"
)

print("\n")
print("=" * 80)
print("YEARLY SHAP STABILITY")
print("=" * 80)

print(
    stability_df.to_string(
        index=False
    )
)

print("\n")
print("=" * 80)
print("REPORT SAVED")
print("=" * 80)

print(REPORT_PATH)