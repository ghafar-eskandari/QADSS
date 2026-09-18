import pandas as pd
import numpy as np
from pathlib import Path

from xgboost import XGBClassifier
import shap


# =========================================================
# PATHS
# =========================================================

BASE = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    BASE / "data" / "processed" /
    "webmelat_model_ready_v2.csv"
)

OUTPUT_FILE = (
    BASE / "data" / "processed" /
    "stage25_qadss_decision_support.csv"
)


# =========================================================
# FEATURES
# =========================================================

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


# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv(
    INPUT_FILE
)


# =========================================================
# DATE
# =========================================================

df["date"] = pd.to_datetime(
    df["dEven"],
    errors="coerce"
)

df = (
    df
    .sort_values("date")
    .reset_index(drop=True)
)


# =========================================================
# FORCE NUMERIC FEATURES
# =========================================================

for feature in FEATURES:

    df[feature] = pd.to_numeric(
        df[feature],
        errors="coerce"
    )


# =========================================================
# TARGET
# =========================================================

df["target"] = pd.to_numeric(
    df["target"],
    errors="coerce"
)


# =========================================================
# DATA QUALITY CHECK
# =========================================================

quality_warning = []


missing_features = [
    feature
    for feature in FEATURES
    if feature not in df.columns
]

if missing_features:

    raise ValueError(
        "Missing features: "
        + ", ".join(missing_features)
    )


missing_count = int(
    df[FEATURES].isna().sum().sum()
)

inf_count = int(
    np.isinf(
        df[FEATURES]
        .to_numpy(dtype=float)
    ).sum()
)

duplicate_dates = int(
    df["date"].duplicated().sum()
)


if missing_count > 0:

    quality_warning.append(
        f"MISSING_VALUES:{missing_count}"
    )


if inf_count > 0:

    quality_warning.append(
        f"INFINITE_VALUES:{inf_count}"
    )


if duplicate_dates > 0:

    quality_warning.append(
        f"DUPLICATE_DATES:{duplicate_dates}"
    )


# =========================================================
# REMOVE INVALID TRAINING ROWS
# =========================================================

valid_rows = (
    df[FEATURES + ["target"]]
    .notna()
    .all(axis=1)
)

df = (
    df.loc[valid_rows]
    .copy()
    .reset_index(drop=True)
)


# =========================================================
# TRAINING DATA
# =========================================================

X = (
    df[FEATURES]
    .astype(float)
)

y = (
    df["target"]
    .astype(int)
)


# =========================================================
# MODEL
# =========================================================

model = XGBClassifier(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=1
)


model.fit(
    X,
    y
)


# =========================================================
# CURRENT OBSERVATION
# =========================================================

current = (
    df
    .iloc[-1]
    .copy()
)


X_current = pd.DataFrame(
    [[
        current[feature]
        for feature in FEATURES
    ]],
    columns=FEATURES
)


# ---------------------------------------------------------
# IMPORTANT:
# Force the current observation to numeric
# ---------------------------------------------------------

X_current = (
    X_current
    .apply(
        pd.to_numeric,
        errors="coerce"
    )
    .astype(float)
)


# =========================================================
# FINAL INPUT QUALITY CHECK
# =========================================================

if X_current.isna().any().any():

    bad_features = (
        X_current.columns[
            X_current.isna().iloc[0]
        ]
        .tolist()
    )

    raise ValueError(
        "Current observation contains "
        "invalid numeric values: "
        + ", ".join(bad_features)
    )


# =========================================================
# PROBABILITY
# =========================================================

probability = float(
    model
    .predict_proba(
        X_current
    )[0, 1]
)


# =========================================================
# DECISION THRESHOLD
# =========================================================

THRESHOLD = 0.65


if probability >= THRESHOLD:

    signal = "SIGNAL"

else:

    signal = "NO_SIGNAL"


# =========================================================
# CONFIDENCE
# =========================================================

distance = abs(
    probability - 0.50
)


if distance >= 0.20:

    confidence = "HIGH"

elif distance >= 0.10:

    confidence = "MEDIUM"

else:

    confidence = "LOW"


# =========================================================
# RISK
# =========================================================

volatility = float(
    current["volatility_20"]
)


if volatility >= 0.05:

    risk_level = "HIGH"

elif volatility >= 0.025:

    risk_level = "MEDIUM"

else:

    risk_level = "LOW"


# =========================================================
# POSITION SIZE
# =========================================================

if signal == "NO_SIGNAL":

    position_size = 0.0

else:

    if risk_level == "LOW":

        position_size = 1.00

    elif risk_level == "MEDIUM":

        position_size = 0.50

    else:

        position_size = 0.25


# =========================================================
# DATA QUALITY / STRUCTURAL WARNING
# =========================================================

if int(current["no_trade"]) == 1:

    quality_warning.append(
        "NO_TRADE"
    )


if abs(
    float(current["return_1d"])
) >= 0.20:

    quality_warning.append(
        "EXTREME_1D_RETURN"
    )


if abs(
    float(current["return_20d"])
) >= 0.50:

    quality_warning.append(
        "EXTREME_20D_RETURN"
    )


if len(quality_warning) == 0:

    data_quality_status = "OK"

else:

    data_quality_status = "WARNING"


# =========================================================
# SHAP
# =========================================================

explainer = shap.TreeExplainer(
    model
)

shap_values = explainer.shap_values(
    X_current
)

shap_values = np.asarray(
    shap_values
)


if shap_values.ndim == 3:

    shap_values = (
        shap_values[:, :, 1]
    )


shap_row = (
    shap_values[0]
)


shap_df = pd.DataFrame({

    "feature":
        FEATURES,

    "shap_value":
        shap_row,

    "abs_shap":
        np.abs(shap_row)

})


shap_df = (
    shap_df
    .sort_values(
        "abs_shap",
        ascending=False
    )
    .reset_index(drop=True)
)


top_features = (
    shap_df
    .head(5)
    ["feature"]
    .tolist()
)


# =========================================================
# DECISION STATUS
# =========================================================

if data_quality_status == "WARNING":

    decision_status = (
        "LIMITED_DECISION_SUPPORT"
    )

elif signal == "SIGNAL":

    decision_status = (
        "POSITIVE_DECISION_SIGNAL"
    )

else:

    decision_status = (
        "NO_DECISION_SIGNAL"
    )


# =========================================================
# OUTPUT
# =========================================================

result = pd.DataFrame([{

    "date":
        current["date"],

    "probability":
        probability,

    "threshold":
        THRESHOLD,

    "signal":
        signal,

    "confidence":
        confidence,

    "volatility_20":
        volatility,

    "risk_level":
        risk_level,

    "position_size":
        position_size,

    "data_quality_status":
        data_quality_status,

    "data_quality_warning":
        ";".join(
            quality_warning
        ),

    "decision_status":
        decision_status,

    "top_shap_feature_1":
        top_features[0],

    "top_shap_feature_2":
        top_features[1],

    "top_shap_feature_3":
        top_features[2],

    "top_shap_feature_4":
        top_features[3],

    "top_shap_feature_5":
        top_features[4]

}])


# =========================================================
# PRINT
# =========================================================

print()
print("=" * 70)
print("STAGE 25: QADSS DECISION SUPPORT")
print("=" * 70)

print()

print(
    f"Date: {current['date'].date()}"
)

print(
    f"Probability: {probability:.4f}"
)

print(
    f"Threshold: {THRESHOLD:.2f}"
)

print(
    f"Signal: {signal}"
)

print(
    f"Confidence: {confidence}"
)

print(
    f"Risk: {risk_level}"
)

print(
    f"Position size: {position_size:.2f}"
)

print(
    f"Data quality: {data_quality_status}"
)

print(
    f"Decision status: {decision_status}"
)

print()

print(
    "Top SHAP features:"
)

for i, feature in enumerate(
    top_features,
    start=1
):

    print(
        f"{i}. {feature}"
    )


if quality_warning:

    print()

    print(
        "Warnings:"
    )

    for warning in quality_warning:

        print(
            f"- {warning}"
        )


# =========================================================
# SAVE
# =========================================================

result.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


print()
print("=" * 70)
print("STAGE 25 COMPLETED")
print("=" * 70)

print()

print(
    f"Saved: {OUTPUT_FILE}"
)