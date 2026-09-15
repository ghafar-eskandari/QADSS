import json
import pandas as pd

RAW_PATH = "data/raw/webmelat_tsetmc_raw.json"
OUT_PATH = "data/processed/webmelat_clean.csv"

with open(RAW_PATH, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

df = pd.DataFrame(raw_data["closingPriceDaily"])

df["date"] = pd.to_datetime(
    df["dEven"].astype(str),
    format="%Y%m%d"
)

df = df.sort_values("date").reset_index(drop=True)

df["no_trade"] = (df["zTotTran"] == 0).astype(int)

df["close"] = df["pClosing"]

df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

print("Processed dataset created successfully.")
print("Rows:", len(df))
print("Columns:", len(df.columns))
print("No-trade rows:", df["no_trade"].sum())
print("Output:", OUT_PATH)
