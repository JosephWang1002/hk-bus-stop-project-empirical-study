# -*- coding: utf-8 -*-
"""
Re-classify ESDI into 4 priority tiers using Fisher-Jenks Natural Breaks
(data-driven, minimizes within-class variance — Jenks 1967), replacing the
arbitrary percentile cutoffs.

Also:
  - rebuild retrofitting_priority_stops.csv with ALL 3080 stops (with tiers + coords)
  - rebuild esdi_spatial_distribution.csv with Jenks-based Priority
"""
import os
import numpy as np
import pandas as pd
from libpysal.weights import KNN
from esda.moran import Moran_Local

OUT_DIR = r"E:\HKBusStopResearch\20260821forB\results"
TBL_DIR = os.path.join(OUT_DIR, "tables")

df = pd.read_csv(os.path.join(OUT_DIR, "merged_data.csv"))
safety = df["safety_0_100"].values
old = df["Old_Density"].values
coords = df[["longitude", "latitude"]].values

# ---- ESDI (equal weight) ----
safety_norm = (safety - safety.min()) / (safety.max() - safety.min())
old_norm = (old - old.min()) / (old.max() - old.min())
esdi = 0.5 * (1 - safety_norm) + 0.5 * old_norm

# ============================================================
# Fisher-Jenks Natural Breaks (1D dynamic programming, vectorized)
# ============================================================
def jenks_breaks(data, n_classes):
    x = np.sort(data).astype(float)
    n = len(x)
    ps = np.concatenate([[0.0], np.cumsum(x)])
    psq = np.concatenate([[0.0], np.cumsum(x * x)])

    def ssd(i, j):
        s = ps[j + 1] - ps[i]
        sq = psq[j + 1] - psq[i]
        m = j - i + 1
        return sq - s * s / m

    dp = np.full((n_classes, n), np.inf)
    boundary = np.zeros((n_classes, n), dtype=int)
    for i in range(n):
        dp[0, i] = ssd(0, i)

    for k in range(1, n_classes):
        for i in range(k, n):
            j_range = np.arange(k - 1, i)          # candidate split index j
            js = j_range + 1                       # start of second segment
            s = ps[i + 1] - ps[js]
            sq = psq[i + 1] - psq[js]
            m = i - js + 1
            seg_ssd = sq - s * s / m
            vals = dp[k - 1, j_range] + seg_ssd
            best = int(np.argmin(vals))
            dp[k, i] = vals[best]
            boundary[k, i] = j_range[best]

    # backtrack break values
    breaks = []
    i = n - 1
    for k in range(n_classes - 1, 0, -1):
        b = boundary[k, i]
        breaks.append(x[b + 1])
        i = b
    breaks.reverse()
    ssd_total = ssd(0, n - 1)
    ssd_within = dp[n_classes - 1, n - 1]
    gvf = 1.0 - ssd_within / ssd_total
    return np.array(breaks), gvf

print("=" * 70)
print("FISHER-JENKS NATURAL BREAKS (4 classes)")
print("=" * 70)
breaks, gvf = jenks_breaks(esdi, 4)
print(f"  Break values (3 cutoffs): {np.round(breaks, 4)}")
print(f"  GVF (Goodness of Variance Fit) = {gvf:.4f}")

# Class boundaries: [min, b0], (b0, b1], (b1, b2], (b2, max]
bounds = np.concatenate([[esdi.min()], breaks, [esdi.max()]])
print(f"  Class ranges:")
for c in range(4):
    mask = (esdi >= bounds[c]) & (esdi <= bounds[c + 1])
    # handle lower boundary exclusivity for c>0
    if c > 0:
        mask = (esdi > bounds[c]) & (esdi <= bounds[c + 1])
    print(f"    class {c}: ({bounds[c]:.4f}, {bounds[c+1]:.4f}], n={mask.sum()}")

# ---- Assign tiers (highest ESDI class = Tier 1) ----
def assign_tier(v):
    # Tier1 = top class (v > breaks[2]), Tier2 = breaks[1]<v<=breaks[2], etc.
    if v > breaks[2]: return "Tier 1 (Highest)"
    if v > breaks[1]: return "Tier 2 (High)"
    if v > breaks[0]: return "Tier 3 (Medium)"
    return "Low priority"

priority = np.vectorize(assign_tier)(esdi)

# ---- ESDI hotspots (Local Moran's I HH, k=15) ----
w = KNN(coords, k=15)
lisa = Moran_Local(esdi, w, permutations=999, seed=42)
hotspot = (lisa.p_sim <= 0.05) & (lisa.q == 1)

print("\n" + "=" * 70)
print("PRIORITY DISTRIBUTION (Jenks-based)")
print("=" * 70)
for t in ["Tier 1 (Highest)", "Tier 2 (High)", "Tier 3 (Medium)", "Low priority"]:
    m = priority == t
    print(f"  {t:<20s}: n={m.sum():4d} ({m.mean()*100:.1f}%), "
          f"safety mean={safety[m].mean():.2f}, safety max={safety[m].max():.2f}")

# ---- Rebuild retrofitting_priority_stops.csv (ALL 3080) ----
rp = pd.DataFrame({
    "image_name": df["image_name"].values,
    "panorama_id": df["panorama_id"].values,
    "latitude": df["latitude"].values,
    "longitude": df["longitude"].values,
    "safety_0_100": safety,
    "Old_Density": old,
    "to_cbd_km": df["to_cbd_km"].values,
    "NAME_EN": df["NAME_EN"].values,
    "ESDI": np.round(esdi, 4),
    "Priority": priority,
    "ESDI_hotspot": hotspot,
}).sort_values("ESDI", ascending=False)

rp_path = os.path.join(TBL_DIR, "retrofitting_priority_stops.csv")
rp.to_csv(rp_path, index=False, encoding="utf-8-sig")
print(f"\n[Saved] {rp_path} (rows={len(rp)})")

# ---- Rebuild esdi_spatial_distribution.csv (Jenks Priority) ----
gis = pd.DataFrame({
    "image_name": df["image_name"].values,
    "panorama_id": df["panorama_id"].values,
    "longitude": df["longitude"].values,
    "latitude": df["latitude"].values,
    "safety_0_100": safety,
    "Old_Density": old,
    "ESDI": np.round(esdi, 4),
    "NAME_EN": df["NAME_EN"].values,
    "Priority": priority,
    "ESDI_hotspot": hotspot,
})
gis_path = os.path.join(TBL_DIR, "esdi_spatial_distribution.csv")
gis.to_csv(gis_path, index=False, encoding="utf-8-sig")
print(f"[Saved] {gis_path} (rows={len(gis)})")

# ---- Save break metadata (Tier order high->low, bounds matched correctly) ----
tier_names = ["Tier 1 (Highest)", "Tier 2 (High)", "Tier 3 (Medium)", "Low priority"]
meta = pd.DataFrame({
    "Class": tier_names,
    "Lower_bound": [round(bounds[3], 4), round(bounds[2], 4), round(bounds[1], 4), round(bounds[0], 4)],
    "Upper_bound": [round(bounds[4], 4), round(bounds[3], 4), round(bounds[2], 4), round(bounds[1], 4)],
    "n": [int((priority == t).sum()) for t in tier_names],
})
meta.to_csv(os.path.join(TBL_DIR, "esdi_jenks_breaks.csv"), index=False, encoding="utf-8-sig")
print(f"[Saved] esdi_jenks_breaks.csv")
print(meta.to_string(index=False))
print(f"\nGVF = {gvf:.4f} (closer to 1 = better separation between classes)")
