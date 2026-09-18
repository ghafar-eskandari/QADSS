import pandas as pd
import shap

from xgboost import XGBClassifier


# --------------------------------------------------
# Configuration
# --------------------------------------------------

INPUT_PATH = "data/processed/webmelat_model_ready.csv"

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
# Load data
# --------------------------------------------------

df = pd.read_csv(INPUT_PATH)

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values("date").reset_index(drop=True)


# --------------------------------------------------
# Temporal split
# --------------------------------------------------

split_index = int(len(df) * 0.60)

train_end = split_index - PURGE_DAYS

train = df.iloc[:train_end].copy()

test = df.iloc[split_index:].copy()


X_train = train[FEATURES]

y_train = train[TARGET]

X_test = test[FEATURES]


# --------------------------------------------------
# XGBoost model
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
    n_jobs=-1
)

model.fit(
    X_train,
    y_train
)


# --------------------------------------------------
# SHAP values
# --------------------------------------------------

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X_test)


# --------------------------------------------------
# Direction analysis
# --------------------------------------------------

direction_results = []

for feature in FEATURES:

    feature_index = FEATURES.index(feature)

    feature_values = X_test[feature].values

    feature_shap = shap_values[:, feature_index]

    positive_mask = feature_shap > 0

    negative_mask = feature_shap < 0

    positive_count = positive_mask.sum()

    negative_count = negative_mask.sum()

    mean_shap = feature_shap.mean()

    mean_abs_shap = abs(feature_shap).mean()

    if mean_shap > 0:
        overall_direction = "Positive"
    elif mean_shap < 0:
        overall_direction = "Negative"
    else:
        overall_direction = "Neutral"

    direction_results.append({
        "Feature": feature,
        "Mean_SHAP": mean_shap,
        "Mean_Abs_SHAP": mean_abs_shap,
        "Positive_SHAP_Count": positive_count,
        "Negative_SHAP_Count": negative_count,
        "Mean_Feature_Value": feature_values.mean(),
        "Overall_Direction": overall_direction
    })


# --------------------------------------------------
# Results
# --------------------------------------------------

results = pd.DataFrame(direction_results)

results = results.sort_values(
    "Mean_Abs_SHAP",
    ascending=False
).reset_index(drop=True)


print("SHAP direction analysis completed.")
print()

print("Train rows:", len(train))

print("Purged rows:", PURGE_DAYS)

print("Test rows:", len(test))

print()

print("Train period:")
print(
    train["date"].min().date(),
    "to",
    train["date"].max().date()
)

print()

print("Test period:")
print(
    test["date"].min().date(),
    "to",
    test["date"].max().date()
)

print()

print("SHAP Direction Analysis:")
print(
    results.round(6).to_string(index=False)
)