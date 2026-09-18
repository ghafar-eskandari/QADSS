import pandas as pd
import requests

AUDIT = r".\data\processed\corporate_action_audit_candidates.csv"
OUTPUT = r".\data\processed\corporate_action_window_match.csv"

a = pd.read_csv(AUDIT, dtype={"event_date": str})

url = "https://cdn.tsetmc.com/api/ClosingPrice/GetPriceAdjustList/778253364357513"
headers = {"User-Agent": "Mozilla/5.0"}

j = requests.get(url, headers=headers, timeout=20).json()

p = pd.DataFrame(j["priceAdjust"])
p["dEven"] = pd.to_datetime(p["dEven"].astype(str), format="%Y%m%d")
p["adjustment_factor"] = p["pClosing"] / p["pClosingNotAdjusted"]

rows = []

for _, e in a.iterrows():
    event_date = pd.to_datetime(e["event_date"])

    q = p[
        (p["dEven"] >= event_date - pd.Timedelta(days=60))
        & (p["dEven"] <= event_date + pd.Timedelta(days=60))
    ].copy()

    q["distance"] = (q["dEven"] - event_date).abs()
    q = q.sort_values("distance")

    row = {
        "event_date": e["event_date"],
        "close_before": e["close_before"],
        "close_after": e["close_after"],
        "raw_return": e["raw_return"],
    }

    if len(q) > 0:
        x = q.iloc[0]

        row.update({
            "adjustment_date": x["dEven"].strftime("%Y-%m-%d"),
            "tsetmc_not_adjusted_close": x["pClosingNotAdjusted"],
            "tsetmc_adjusted_close": x["pClosing"],
            "adjustment_factor": x["adjustment_factor"],
            "days_from_event": x["distance"].days,
            "verification_status": "WINDOW_MATCH",
        })
    else:
        row.update({
            "adjustment_date": None,
            "tsetmc_not_adjusted_close": None,
            "tsetmc_adjusted_close": None,
            "adjustment_factor": None,
            "days_from_event": None,
            "verification_status": "NO_ADJUSTMENT_IN_WINDOW",
        })

    rows.append(row)

out = pd.DataFrame(rows)

out.to_csv(
    OUTPUT,
    index=False,
    encoding="utf-8-sig"
)

print(out.to_string(index=False))
print()
print("Saved:", OUTPUT)
