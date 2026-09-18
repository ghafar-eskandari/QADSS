from pathlib import Path

import pandas as pd
import shap
from xgboost import XGBClassifier


# --------------------------------------------------
# 1. Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "webmelat_model_ready.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "shap_dependence.csv"


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

PURGE_DAYS = 5


# --------------------------------------------------
# 3. Load data
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

df["dEven"] = pd.to_datetime(
    df["dEven"].astype(str),
    format="%Y%m%d"
)

df = df.dropna(subset=FEATURES + [TARGET]).copy()

X = df[FEATURES]
y = df[TARGET].astype(int)


# --------------------------------------------------
# 4. Purged time split
# --------------------------------------------------

n = len(df)

split_index = int(0.60 * n)

train_end = split_index - PURGE_DAYS

X_train = X.iloc[:train_end]
y_train = y.iloc[:train_end]

X_test = X.iloc[split_index:]
y_test = y.iloc[split_index:]

print("=" * 70)
print("SHAP DEPENDENCE ANALYSIS")
print("=" * 70)

print(f"Total rows: {n}")
print(f"Train rows: {len(X_train)}")
print(f"Purged rows: {PURGE_DAYS}")
print(f"Test rows: {len(X_test)}")

print(
    f"Train period: "
    f"{df['dEven'].iloc[0].date()} "
    f"to "
    f"{df['dEven'].iloc[train_end - 1].date()}"
)

print(
    f"Test period: "
    f"{df['dEven'].iloc[split_index].date()} "
    f"to "
    f"{df['dEven'].iloc[-1].date()}"
)


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
# 6. SHAP values
# --------------------------------------------------

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X_test)

shap_df = pd.DataFrame(
    shap_values,
    columns=FEATURES,
    index=X_test.index
)


# --------------------------------------------------
# 7. Main features
# --------------------------------------------------

MAIN_FEATURES = [
    "ma_5",
    "volume",
    "volatility_20",
    "ma_20",
]


# --------------------------------------------------
# 8. Quantile-based dependence analysis
# --------------------------------------------------

results = []

for feature in MAIN_FEATURES:

    temp = pd.DataFrame({
        "feature_value": X_test[feature],
        "shap_value": shap_df[feature],
    }).copy()

    temp["bin"] = pd.qcut(
        temp["feature_value"],
        q=5,
        duplicates="drop"
    )

    grouped = (
        temp
        .groupby("bin", observed=True)
        .agg(
            mean_feature_value=("feature_value", "mean"),
            mean_shap=("shap_value", "mean"),
            mean_abs_shap=("shap_value", lambda x: x.abs().mean()),
            observations=("shap_value", "size"),
        )
        .reset_index()
    )

    grouped["feature"] = feature
    grouped["bin"] = grouped["bin"].astype(str)

    results.append(grouped)


# --------------------------------------------------
# 9. Combine results
# --------------------------------------------------

dependence_df = pd.concat(
    results,
    ignore_index=True
)

dependence_df = dependence_df[
    [
        "feature",
        "bin",
        "mean_feature_value",
        "mean_shap",
        "mean_abs_shap",
        "observations",
    ]
]


# --------------------------------------------------
# 10. Save report
# --------------------------------------------------

REPORT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

dependence_df.to_csv(
    REPORT_PATH,
    index=False
)


# --------------------------------------------------
# 11. Display results
# --------------------------------------------------

pd.set_option(
    "display.max_rows",
    100
)

pd.set_option(
    "display.float_format",
    lambda x: f"{x:.6f}"
)

print("\n")
print("=" * 70)
print("SHAP DEPENDENCE RESULTS")
print("=" * 70)

print(dependence_df.to_string(index=False))

print("\n")
print("=" * 70)
print(f"Report saved to:")
print(REPORT_PATH)
print("=" * 70)