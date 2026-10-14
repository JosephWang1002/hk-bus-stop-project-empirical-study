# -*- coding: utf-8 -*-
"""
GWR local coefficient descriptive statistics table.
Reads gwr_local_params.csv (standardized coefficients) and computes
mean/std/min/25%/median/75%/max for each coefficient column.
"""
import os
import pandas as pd

BASE = r"E:\HKBusStopResearch\20260821forB\results\tables"
gwr = pd.read_csv(os.path.join(BASE, "gwr_local_params.csv"))

# Column display order (display label -> csv column)
ORDER = [
    ("Intercept", "Intercept"),
    ("Information board", "information_board"),
    ("Shelter", "shelter"),
    ("Step", "step"),
    ("Light pole", "light_pole"),
    ("Pillar", "pillar"),
    ("Seat", "seat"),
    ("Bin", "bin"),
    ("Ads board", "ads_board"),
    ("Fence", "fence"),
    ("Power Box", "box"),
    ("Handrail", "handrail"),
    ("Traffic sign", "traffic_sign"),
    ("Bollard", "Bollard"),
    ("Screen", "screen"),
    ("Camera", "camera"),
    ("Vending machine", "vending_machine"),
    ("Distance to CBD", "to_cbd_km"),
    ("Elderly density", "Old_Density"),
]

rows = []
for disp, col in ORDER:
    s = gwr[col]
    rows.append({
        "Variable": disp,
        "Mean": round(s.mean(), 4),
        "Std": round(s.std(), 4),
        "Min": round(s.min(), 4),
        "25%": round(s.quantile(0.25), 4),
        "Median": round(s.median(), 4),
        "75%": round(s.quantile(0.75), 4),
        "Max": round(s.max(), 4),
    })

df_out = pd.DataFrame(rows)

# Save CSV
csv_path = os.path.join(BASE, "gwr_coef_descriptive_stats.csv")
df_out.to_csv(csv_path, index=False, encoding="utf-8-sig")
print(f"[Saved] {csv_path}\n")

# Print aligned table
print("GWR local coefficient descriptive statistics (standardized coefficients, N = 3080)")
print("=" * 95)
header = (f"{'Variable':<18s} {'Mean':>8s} {'Std':>8s} {'Min':>8s} "
          f"{'25%':>8s} {'Median':>8s} {'75%':>8s} {'Max':>8s}")
print(header)
print("-" * 95)
for r in rows:
    print(f"{r['Variable']:<18s} {r['Mean']:>8.4f} {r['Std']:>8.4f} {r['Min']:>8.4f} "
          f"{r['25%']:>8.4f} {r['Median']:>8.4f} {r['75%']:>8.4f} {r['Max']:>8.4f}")

print("\nNote: coefficients are standardized (z-scored X and y before GWR).")
