# -*- coding: utf-8 -*-
"""
Statistical table for the 580 Tier-1 (highest-priority retrofitting) stops,
grouped by district. Safety-score indicators: Mean / Min / Max / Std.
"""
import os
import pandas as pd

TBL_DIR = r"E:\HKBusStopResearch\20260821forB\results\tables"
GIS = pd.read_csv(os.path.join(TBL_DIR, "esdi_spatial_distribution.csv"))

tier1 = GIS[GIS["Priority"] == "Tier 1 (Highest)"].copy()
n_total = len(tier1)

rows = []
for dist, sub in tier1.groupby("NAME_EN"):
    s = sub["safety_0_100"]
    rows.append({
        "District": dist,
        "Count": int(len(sub)),
        "Percentage": round(len(sub) / n_total * 100, 1),
        "Mean": round(s.mean(), 2),
        "Min": round(s.min(), 2),
        "Max": round(s.max(), 2),
        "Std": round(s.std(), 2),
    })

df_out = pd.DataFrame(rows).sort_values("Count", ascending=False).reset_index(drop=True)

# Save CSV
csv_path = os.path.join(TBL_DIR, "tier1_district_stats.csv")
df_out.to_csv(csv_path, index=False, encoding="utf-8-sig")
print(f"[Saved] {csv_path}\n")

# Print aligned table
print(f"Tier-1 (highest-priority retrofitting) stops: N = {n_total}\n")
header = f"{'District':<28s} {'Count':>5s} {'Pct%':>6s} {'Mean':>6s} {'Min':>6s} {'Max':>6s} {'Std':>6s}"
print(header)
print("-" * len(header))
for _, r in df_out.iterrows():
    print(f"{r['District']:<28s} {r['Count']:>5d} {r['Percentage']:>6.1f} "
          f"{r['Mean']:>6.2f} {r['Min']:>6.2f} {r['Max']:>6.2f} {r['Std']:>6.2f}")

print("\nNote: Percentage = share of the 580 Tier-1 stops; Mean/Min/Max/Std refer to safety scores (0-100).")
