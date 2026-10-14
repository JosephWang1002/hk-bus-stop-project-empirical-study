# -*- coding: utf-8 -*-
"""
Cross-validate ESDI Tier 1 (Jenks) against Bivariate LISA "Low-High"
(low safety + high elderly density) — all in merged_data.csv (image-level).
"""
import numpy as np
import pandas as pd

M = pd.read_csv(r"E:\HKBusStopResearch\20260821forB\results\merged_data.csv")
GIS = pd.read_csv(r"E:\HKBusStopResearch\20260821forB\results\tables\esdi_spatial_distribution.csv")

# Merge Jenks Priority onto merged_data via image_name
M = M.merge(GIS[["image_name", "Priority"]], on="image_name", how="left")

# Bivariate LISA quadrants (from merged_data): 2 = Low-High (low safety, high density)
lisa_sig = M["Significant"] == 1
lh = (M["Quadrant"] == 2) & lisa_sig          # Low-High
hh = (M["Quadrant"] == 1) & lisa_sig          # High-High
ll = (M["Quadrant"] == 3) & lisa_sig
hl = (M["Quadrant"] == 4) & lisa_sig

tier1 = M["Priority"] == "Tier 1 (Highest)"

print("=" * 72)
print("BIVARIATE LISA — significant quadrants (p<0.05)")
print("=" * 72)
for name, m in [("High-High (safe+high density)", hh), ("Low-High (unsafe+high density)", lh),
                ("Low-Low (unsafe+low density)", ll), ("High-Low (safe+low density)", hl)]:
    print(f"  {name:<36s}: n={m.sum()}")

print("\n" + "=" * 72)
print("CROSS-VALIDATION: ESDI Tier 1 (Jenks) vs Bivariate LISA Low-High")
print("=" * 72)
print(f"  ESDI Tier 1 (Jenks):          {tier1.sum()}")
print(f"  LISA Low-High (significant):  {lh.sum()}")
print(f"  Overlap (both):               {(tier1 & lh).sum()}")
print(f"  Tier1 also in LISA LH:        {(tier1 & lh).sum()/tier1.sum()*100:.1f}%")
print(f"  LISA LH also in Tier1:        {(tier1 & lh).sum()/lh.sum()*100:.1f}%")

print("\n  Profile:")
for name, m in [("Tier 1", tier1), ("LISA Low-High", lh), ("Overlap", tier1 & lh),
                ("Tier1 only (not LISA LH)", tier1 & ~lh)]:
    print(f"    {name:<26s}: n={m.sum():4d}, safety mean={M.loc[m,'safety_0_100'].mean():.2f}, "
          f"Old mean={M.loc[m,'Old_Density'].mean():.0f}")

# Tier1 not in LISA LH: safety distribution
t1_only = tier1 & ~lh
print(f"\n  Tier1-only safety: <42.67 = {(M.loc[t1_only,'safety_0_100']<42.67).sum()}, "
      f">=42.67 = {(M.loc[t1_only,'safety_0_100']>=42.67).sum()}")
