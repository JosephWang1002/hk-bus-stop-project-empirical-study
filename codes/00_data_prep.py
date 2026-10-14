# -*- coding: utf-8 -*-
"""
Data Preparation: Merge safety + detection + old_density + control variables
"""
import os, re
import numpy as np
import pandas as pd
from math import radians, sin, cos, asin, sqrt

OUT_DIR = r"E:\HKBusStopResearch\20260821forB\results"
os.makedirs(OUT_DIR, exist_ok=True)

CLASSES = [
    "shelter", "ads_board", "seat", "light_pole", "pillar", "box",
    "information_board", "step", "screen", "fence", "handrail",
    "traffic_sign", "bin", "camera", "Bollard", "vending_machine"
]
CBD_LAT, CBD_LON = 22.2817, 114.1588

def haversine_km(lat1, lon1, lat2, lon2):
    rlat1, rlon1, rlat2, rlon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = rlat2 - rlat1, rlon2 - rlon1
    a = sin(dlat/2)**2 + cos(rlat1)*cos(rlat2)*sin(dlon/2)**2
    return 6371.0 * 2 * asin(sqrt(a))

def extract_pid(filename):
    m = re.match(r"panorama_(\d+)_", filename)
    return int(m.group(1)) if m else None

print("=" * 70)
print("  DATA PREPARATION")
print("=" * 70)

# 1. Load safety_ranking (3080 rows, unique image_file)
print("\n[1] Loading safety_ranking.csv...")
df = pd.read_csv(r"E:\HKBusStopResearch\population_inference\06_population_results\safety_ranking.csv")
df = df.rename(columns={"image_file": "image_name"})
df["panorama_id"] = df["image_name"].apply(extract_pid)
# Extract lat/lon from filename for matching
df["lat_from_name"] = df["image_name"].str.extract(r"_(\d+\.\d+)_").astype(float)
df["lon_from_name"] = df["image_name"].str.extract(r"_\d+\.\d+_(\d+\.\d+)_").astype(float)
print(f"  Rows: {len(df)}")

# 2. Load detection results, deduplicate on image_name
print("\n[2] Loading detection_results.xlsx...")
df_det = pd.read_excel(r"E:\HKBusStopResearch\yolo_detection\detection_results.xlsx")
print(f"  Raw rows: {len(df_det)}, unique image_name: {df_det['image_name'].nunique()}")
df_det = df_det.drop_duplicates(subset="image_name", keep="first")
print(f"  After dedup: {len(df_det)}")

# 3. Merge detection (left join on image_name)
print("\n[3] Merging detection...")
det_cols = ["image_name"] + CLASSES + ["total_detections", "latitude", "longitude"]
df = df.merge(df_det[det_cols], on="image_name", how="left")
# Use detection coords if available, else from filename
df["latitude"] = df["latitude"].fillna(df["lat_from_name"])
df["longitude"] = df["longitude"].fillna(df["lon_from_name"])
df = df.drop(columns=["lat_from_name", "lon_from_name"])
print(f"  Matched: {df['total_detections'].notna().sum()} / {len(df)}")

# 4. Merge Old_Density using (panorama_id, latitude, longitude)
print("\n[4] Merging Old_Density...")
df_lisa = pd.read_csv(r"E:\HKBusStopResearch\yolo_detection\spatial_output\lisa_output\bivariate_lisa_results.csv")
df_lisa["panorama_id"] = df_lisa["panorama_id"].astype(int)
# Round coords for matching (avoid float precision issues)
df["match_key"] = df["panorama_id"].astype(str) + "_" + df["latitude"].round(5).astype(str) + "_" + df["longitude"].round(5).astype(str)
df_lisa["match_key"] = df_lisa["panorama_id"].astype(str) + "_" + df_lisa["latitude"].round(5).astype(str) + "_" + df_lisa["longitude"].round(5).astype(str)
lisa_cols = ["match_key", "Old_Density", "LISA_I", "LISA_p", "Quadrant", "Significant", "Cluster"]
df_lisa_dedup = df_lisa[lisa_cols].drop_duplicates(subset="match_key", keep="first")
df = df.merge(df_lisa_dedup, on="match_key", how="left")
df = df.drop(columns=["match_key"])
print(f"  Old_Density matched: {df['Old_Density'].notna().sum()} / {len(df)}")

# 5. Merge district info
print("\n[5] Merging district info...")
df_dist = pd.read_csv(r"E:\HKBusStopResearch\district.csv")
df_dist["panorama_id"] = df_dist["panorama_id"].astype(float).astype(int)
df_dist["match_key"] = df_dist["panorama_id"].astype(str) + "_" + df_dist["latitude"].round(5).astype(str) + "_" + df_dist["longitude"].round(5).astype(str)
dist_cols = ["match_key", "AREA_CODE", "NAME_EN"]
df_dist_dedup = df_dist[dist_cols].drop_duplicates(subset="match_key", keep="first")
df["match_key"] = df["panorama_id"].astype(str) + "_" + df["latitude"].round(5).astype(str) + "_" + df["longitude"].round(5).astype(str)
df = df.merge(df_dist_dedup, on="match_key", how="left")
df = df.drop(columns=["match_key"])
print(f"  District matched: {df['NAME_EN'].notna().sum()} / {len(df)}")

# 6. Derive control variables
print("\n[6] Deriving control variables...")
df["to_cbd_km"] = df.apply(lambda r: haversine_km(r["latitude"], r["longitude"], CBD_LAT, CBD_LON), axis=1)
df["district_code"] = pd.Categorical(df["NAME_EN"]).codes

# Fill missing detection counts with 0
for c in CLASSES + ["total_detections"]:
    df[c] = df[c].fillna(0).astype(int)

# 7. Save
out_path = os.path.join(OUT_DIR, "merged_data.csv")
df.to_csv(out_path, index=False, encoding="utf-8-sig")
print(f"\n[7] Saved: {out_path}")
print(f"  Final shape: {df.shape}")

print("\n" + "=" * 70)
print("  DATA SUMMARY")
print("=" * 70)
print(f"  N = {len(df)}")
print(f"  Safety: mean={df['safety_0_100'].mean():.2f}, std={df['safety_0_100'].std():.2f}")
print(f"  Old_Density: mean={df['Old_Density'].mean():.1f}, n_missing={df['Old_Density'].isna().sum()}")
print(f"  to_cbd_km: mean={df['to_cbd_km'].mean():.2f}")
print(f"  Districts: {df['NAME_EN'].nunique()}")
print("=" * 70)
