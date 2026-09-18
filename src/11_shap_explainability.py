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

y_test = test[TARGET]


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
# SHAP explainer
# --------------------------------------------------

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X_test)


# --------------------------------------------------
# Global SHAP importance
# --------------------------------------------------

importance = pd.DataFrame({
    "Feature": FEATURES,
    "Mean_Abs_SHAP": abs(shap_values).mean(axis=0)
})

importance = importance.sort_values(
    "Mean_Abs_SHAP",
    ascending=False
).reset_index(drop=True)


# --------------------------------------------------
# Output
# --------------------------------------------------

print("SHAP explainability completed.")
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

print("Global SHAP Feature Importance:")
print(
    importance.round(6).to_string(index=False)
)