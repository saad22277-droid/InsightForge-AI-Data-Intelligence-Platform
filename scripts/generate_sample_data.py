"""Generate a realistic demo dataset with deliberate data-quality problems.

Usage:  python scripts/generate_sample_data.py
Output: data/samples/sales_demo.csv  (deterministic, seed 42)
"""
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 800

region = rng.choice(["North", "South", "East", "West"], n, p=[0.3, 0.25, 0.25, 0.2])
category = rng.choice(["Technology", "Furniture", "Office Supplies"], n, p=[0.4, 0.25, 0.35])
segment = rng.choice(["Consumer", "Corporate", "Home Office"], n, p=[0.5, 0.3, 0.2])
units = rng.integers(1, 15, n)
base_price = {"Technology": 220, "Furniture": 160, "Office Supplies": 35}
unit_price = np.array([base_price[c] for c in category]) * rng.normal(1, 0.15, n)
discount = rng.choice([0, 0.05, 0.1, 0.2], n, p=[0.5, 0.2, 0.2, 0.1])
customer_age = np.clip(rng.normal(41, 12, n), 18, 80).round()
sales = units * unit_price * (1 - discount) * rng.normal(1, 0.03, n)
profit = sales * (0.18 - discount * 0.6 + rng.normal(0, 0.05, n))
order_date = pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 365, n), unit="D")

df = pd.DataFrame({
    "Order ID": [f"ORD-{i:05d}" for i in range(n)],
    "Order Date": order_date.strftime("%Y-%m-%d"),
    "Region": region,
    "Category": category,
    "Segment": segment,
    "Units": units,
    "Unit Price": unit_price.round(2),
    "Discount": discount,
    "Customer Age": customer_age,
    "Sales": sales.round(2),
    "Profit": profit.round(2),
})

# --- deliberate problems for the cleaning / quality / AI features to find ---
df.loc[rng.choice(n, 45, replace=False), "Customer Age"] = np.nan          # missing values
df.loc[rng.choice(n, 30, replace=False), "Segment"] = np.nan
df.loc[rng.choice(n, 8, replace=False), "Region"] = "  north "              # dirty text
df.loc[rng.choice(n, 6, replace=False), "Category"] = "N/A"                 # placeholders
df.loc[[11, 222, 505], "Sales"] = [48000.0, 52000.0, 61000.0]              # outliers
df.loc[[70, 71], "Customer Age"] = [-4, 240]                                # invalid ages
df = pd.concat([df, df.sample(12, random_state=1)], ignore_index=True)      # duplicates

out = Path(__file__).resolve().parents[1] / "data" / "samples" / "sales_demo.csv"
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False)
print(f"Wrote {out} ({len(df)} rows, {len(df.columns)} columns)")
