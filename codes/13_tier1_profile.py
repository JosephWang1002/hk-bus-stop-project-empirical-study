# -*- coding: utf-8 -*-
"""Profile Tier 1 under Jenks classification: district distribution and
comparison with the ad-hoc low-safety + high-density criterion."""
import numpy as np
import pandas as pd

gis = pd.read_csv(r"E:\HKBusStopResearch\20260821forB\results\tables\esdi_spatial_distribution.csv")
safety = gis["safety_0_100"].values
old = gis["Old_Density"].values

t1 = gis["Priority"] == "Tier 1 (Highest)"
print("=" * 70)
print("TIER 1 (Jenks) PROFILE")
print("=" * 70)
print(f"  n = {t1.sum()} ({t1.mean()*100:.1f}%)")
print(f"  safety: mean={safety[t1].mean():.2f}, range=[{safety[t1].min():.2f},{safety[t1].max():.2f}]")
print(f"  Old_Density: mean={old[t1].mean():.0f}, median={np.median(old[t1]):.0f}")
print(f"  ESDI: mean={gis.loc[t1,'ESDI'].mean():.4f}")

print("\n  Tier 1 by district (top 10):")
dist = gis[t1].groupby("NAME_EN").size().sort_values(ascending=False)
for d, n in dist.head(10).items():
    pct = n / gis[gis["NAME_EN"] == d].shape[0] * 100
    print(f"    {d:<30s} {n:4d}  ({pct:.0f}% of district stops)")

# Ad-hoc comparison: safety < 42.67 (sufficiency threshold) AND old > median
T_STAR = 42.67
adhoc = (safety < T_STAR) & (old > np.median(old))
print(f"\n  Ad-hoc criterion (safety<{T_STAR} & old>median): {adhoc.sum()} stops")
overlap = (t1 & adhoc).sum()
print(f"  Tier1 ∩ ad-hoc: {overlap}")
print(f"  Tier1 not in ad-hoc: {t1.sum() - overlap}")
print(f"  Ad-hoc not in Tier1: {adhoc.sum() - overlap}")

# Tier1 safety composition
print(f"\n  Tier1 stops by safety level:")
print(f"    safety < 42.67 (insufficient): {(safety[t1] < T_STAR).sum()} ({(safety[t1]<T_STAR).mean()*100:.0f}%)")
print(f"    safety >= 42.67: {(safety[t1] >= T_STAR).sum()}")
