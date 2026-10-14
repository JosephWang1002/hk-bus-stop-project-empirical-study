# -*- coding: utf-8 -*-
"""
Export ESDI spatial distribution for GIS mapping, and add coordinates
to retrofitting_priority_stops.csv.

1) Recompute ESDI = 0.5*(1 - safety_norm) + 0.5*old_density_norm
2) Priority tiers (p90/p75/p50) and ESDI hotspots (Local Moran's I HH)
3) Save esdi_spatial_distribution.csv (all 3080 stops, GIS-ready)
4) Patch retrofitting_priority_stops.csv with latitude/longitude
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

# ---- 1. ESDI (equal weights, min-max normalized) ----
safety_norm = (safety - safety.min()) / (safety.max() - safety.min())
old_norm = (old - old.min()) / (old.max() - old.min())
esdi = 0.5 * (1 - safety_norm) + 0.5 * old_norm

# ---- 2. Priority tiers ----
p90, p75, p50 = np.percentile(esdi, [90, 75, 50])
def tier(v):
    if v >= p90: return "Tier 1 (Highest)"
    if v >= p75: return "Tier 2 (High)"
    if v >= p50: return "Tier 3 (Medium)"
    return "Low priority"
priority = np.vectorize(tier)(esdi)

# ---- 3. ESDI hotspots (Local Moran's I, HH, k=15) ----
w = KNN(coords, k=15)
lisa = Moran_Local(esdi, w, permutations=999, seed=42)
hotspot = (lisa.p_sim <= 0.05) & (lisa.q == 1)

# ---- Build GIS-ready table ----
gis = pd.DataFrame({
    "image_name": df["image_name"].values,
    "panorama_id": df["panorama_id"].values,
    "longitude": df["longitude"].values,
    "latitude": df["latitude"].values,
    "safety_0_100": safety,
    "Old_Density": old,
    "ESDI": esdi.round(4),
    "NAME_EN": df["NAME_EN"].values,
    "Priority": priority,
    "ESDI_hotspot": hotspot,
})

gis_path = os.path.join(TBL_DIR, "esdi_spatial_distribution.csv")
gis.to_csv(gis_path, index=False, encoding="utf-8-sig")
print(f"[Saved] {gis_path}")
print(f"  rows={len(gis)}, cols={list(gis.columns)}")
print(f"  ESDI range: [{esdi.min():.4f}, {esdi.max():.4f}], mean={esdi.mean():.4f}")
print(f"  Priority counts:\n{gis['Priority'].value_counts().to_string()}")
print(f"  ESDI hotspots: {hotspot.sum()}")

# ---- 4. Patch retrofitting_priority_stops.csv with coordinates ----
rp_path = os.path.join(TBL_DIR, "retrofitting_priority_stops.csv")
rp = pd.read_csv(rp_path)
coords_lookup = df[["image_name", "latitude", "longitude"]].drop_duplicates(subset="image_name")
rp = rp.merge(coords_lookup, on="image_name", how="left")
# Reorder: put lat/lon after panorama_id
cols = list(rp.columns)
id_cols = ["image_name", "panorama_id", "latitude", "longitude"]
other = [c for c in cols if c not in id_cols]
rp = rp[id_cols + other]
rp.to_csv(rp_path, index=False, encoding="utf-8-sig")
print(f"\n[Saved] {rp_path}")
print(f"  rows={len(rp)}, cols={list(rp.columns)}")
print(f"  lat/lon matched: {rp['latitude'].notna().sum()}/{len(rp)}")
print(rp.head(3).to_string())
