# -*- coding: utf-8 -*-
"""
================================================================================
ESDI: Elderly Safety Deprivation Index & Retrofitting Priority
================================================================================
Constructs a simplified ESDI combining:
  (1) Safety level (lower = more deprived) — inverted and normalized
  (2) Elderly population density (higher = more need) — normalized
  (3) Safety gap (distance from city median) — penalizes far-below-average

ESDI = w1 * (1 - safety_norm) + w2 * old_density_norm
Equal weights (0.5/0.5) for transparency and replicability.

Then identifies retrofitting priority hotspots and compares with
the previous ad-hoc 516-stop selection.
================================================================================
"""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from libpysal.weights import KNN
from esda.moran import Moran_Local

OUT_DIR = r"E:\HKBusStopResearch\20260821forB\results"
FIG_DIR = os.path.join(OUT_DIR, "figures")
TBL_DIR = os.path.join(OUT_DIR, "tables")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TBL_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "Arial", "font.size": 10,
    "axes.labelsize": 11, "axes.titlesize": 12,
    "figure.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42,
})

df = pd.read_csv(os.path.join(OUT_DIR, "merged_data.csv"))
print(f"Loaded {len(df)} rows")

coords = df[["longitude", "latitude"]].values
safety = df["safety_0_100"].values
old_density = df["Old_Density"].values

# ============================================================
# 1. Normalize components (0-1 scale)
# ============================================================
safety_norm = (safety - safety.min()) / (safety.max() - safety.min())
old_norm = (old_density - old_density.min()) / (old_density.max() - old_density.min())

# Invert safety: higher deprivation = lower safety
deprivation_safety = 1.0 - safety_norm

# ============================================================
# 2. Compute ESDI (equal weights)
# ============================================================
w1, w2 = 0.5, 0.5
esdi = w1 * deprivation_safety + w2 * old_norm
df["ESDI"] = esdi

print(f"\n[ESDI] mean={esdi.mean():.4f}, std={esdi.std():.4f}, range=[{esdi.min():.4f}, {esdi.max():.4f}]")

# ============================================================
# 3. Classify retrofitting priority
# ============================================================
# Tier 1 (highest priority): ESDI top 10%
# Tier 2: ESDI top 10-25%
# Tier 3: ESDI top 25-50%
# Others: below median ESDI

p90, p75, p50 = np.percentile(esdi, [90, 75, 50])

def priority_tier(val):
    if val >= p90: return "Tier 1 (Highest)"
    if val >= p75: return "Tier 2 (High)"
    if val >= p50: return "Tier 3 (Medium)"
    return "Low priority"

df["Priority"] = np.vectorize(priority_tier)(esdi)
tier_counts = df["Priority"].value_counts()
print(f"\n[Priority distribution]")
for tier in ["Tier 1 (Highest)", "Tier 2 (High)", "Tier 3 (Medium)", "Low priority"]:
    print(f"  {tier}: {tier_counts.get(tier, 0)} ({tier_counts.get(tier, 0)/len(df)*100:.1f}%)")

# ============================================================
# 4. Identify ESDI hotspots (spatial clustering of high ESDI)
# ============================================================
w_knn = KNN(coords, k=15)
lisa_esdi = Moran_Local(esdi, w_knn, permutations=999, seed=42)

# HH = high ESDI cluster = deprivation hotspot
sig = lisa_esdi.p_sim <= 0.05
q = lisa_esdi.q
hh_mask = sig & (q == 1)  # High-High = deprivation hotspot
df["ESDI_hotspot"] = hh_mask

print(f"\n[ESDI Hotspots] HH clusters (p<0.05): {hh_mask.sum()} stops")

# ============================================================
# 5. Compare with previous ad-hoc selection (low safety + high old density)
# ============================================================
# Previous method: safety < 33rd percentile (roughly)
safety_p33 = np.percentile(safety, 33)
old_median = np.median(old_density)
prev_criteria = (safety < safety_p33) & (old_density > old_median)
print(f"\n[Comparison] Previous ad-hoc method: {prev_criteria.sum()} stops")
print(f"[Comparison] ESDI Tier 1: {tier_counts.get('Tier 1 (Highest)', 0)} stops")
print(f"[Comparison] ESDI HH hotspots: {hh_mask.sum()} stops")

# Overlap
overlap = (prev_criteria & (df["Priority"] == "Tier 1 (Highest)")).sum()
print(f"[Comparison] Overlap (prev ∩ ESDI Tier 1): {overlap}")

# ============================================================
# 6. District-level ESDI summary
# ============================================================
district_esdi = df.groupby("NAME_EN").agg(
    n=("ESDI", "count"),
    esdi_mean=("ESDI", "mean"),
    esdi_std=("ESDI", "std"),
    safety_mean=("safety_0_100", "mean"),
    old_density_mean=("Old_Density", "mean"),
    n_hotspot=("ESDI_hotspot", "sum"),
).round(4).sort_values("esdi_mean", ascending=False)
district_esdi["hotspot_pct"] = (district_esdi["n_hotspot"] / district_esdi["n"] * 100).round(1)
district_esdi.to_csv(os.path.join(TBL_DIR, "district_esdi_summary.csv"), encoding="utf-8-sig")
print(f"\n[Saved] district_esdi_summary.csv")
print(district_esdi.head(10).to_string())

# ============================================================
# 7. Save retrofitting priority list
# ============================================================
priority_stops = df[df["Priority"].isin(["Tier 1 (Highest)", "Tier 2 (High)"])].copy()
priority_stops = priority_stops.sort_values("ESDI", ascending=False)
priority_stops[["image_name", "panorama_id", "safety_0_100", "Old_Density",
                 "to_cbd_km", "NAME_EN", "ESDI", "Priority", "ESDI_hotspot"]].to_csv(
    os.path.join(TBL_DIR, "retrofitting_priority_stops.csv"), index=False, encoding="utf-8-sig")
print(f"\n[Saved] retrofitting_priority_stops.csv ({len(priority_stops)} stops)")

# ============================================================
# 8. Save results JSON
# ============================================================
results = {
    "ESDI_stats": {
        "mean": round(float(esdi.mean()), 4),
        "std": round(float(esdi.std()), 4),
        "min": round(float(esdi.min()), 4),
        "max": round(float(esdi.max()), 4),
    },
    "priority_tiers": {
        "Tier1_highest": int(tier_counts.get("Tier 1 (Highest)", 0)),
        "Tier2_high": int(tier_counts.get("Tier 2 (High)", 0)),
        "Tier3_medium": int(tier_counts.get("Tier 3 (Medium)", 0)),
        "Low": int(tier_counts.get("Low priority", 0)),
    },
    "ESDI_hotspots_HH": int(hh_mask.sum()),
    "comparison": {
        "previous_adhoc": int(prev_criteria.sum()),
        "ESDI_Tier1": int(tier_counts.get("Tier 1 (Highest)", 0)),
        "overlap": int(overlap),
    },
}
with open(os.path.join(TBL_DIR, "esdi_results.json"), "w") as f:
    json.dump(results, f, indent=2)

# ============================================================
# 9. Figures
# ============================================================
# (a) ESDI spatial map with hotspots
fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

ax = axes[0]
sc = ax.scatter(coords[:, 0], coords[:, 1], c=esdi, cmap="YlOrRd",
                alpha=0.7, s=5, edgecolors="none")
ax.scatter(coords[hh_mask, 0], coords[hh_mask, 1], c="#B2182B",
           s=12, edgecolors="black", linewidth=0.3, label=f"Deprivation hotspot (HH, p<0.05)")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("(a) ESDI spatial distribution", fontweight="bold")
ax.legend(fontsize=8, framealpha=0.9)
plt.colorbar(sc, ax=ax, shrink=0.8, label="ESDI")

# (b) Priority tiers map
ax = axes[1]
tier_colors = {"Tier 1 (Highest)": "#B2182B", "Tier 2 (High)": "#EF8A62",
               "Tier 3 (Medium)": "#FDDBC7", "Low priority": "#D0D0D0"}
for tier, color in tier_colors.items():
    mask = df["Priority"] == tier
    if mask.sum() > 0:
        ax.scatter(coords[mask, 0], coords[mask, 1], c=color, s=5 if "Low" not in tier else 3,
                   alpha=0.7, edgecolors="none", label=f"{tier} (n={mask.sum()})")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("(b) Retrofitting priority tiers", fontweight="bold")
ax.legend(fontsize=8, framealpha=0.9, markerscale=2)

fig.suptitle("Elderly Safety Deprivation Index (ESDI) and retrofitting priority",
             fontsize=13, fontweight="bold", y=1.02)
fig.savefig(os.path.join(FIG_DIR, "fig_esdi_map.png"), dpi=300, bbox_inches="tight")
fig.savefig(os.path.join(FIG_DIR, "fig_esdi_map.pdf"), bbox_inches="tight")
plt.close()
print(f"[Saved] fig_esdi_map.png/pdf")

# (c) ESDI vs Safety scatter (shows the trade-off)
fig2, ax2 = plt.subplots(figsize=(8, 5.5))
sc = ax2.scatter(safety, old_density, c=esdi, cmap="YlOrRd", s=8, alpha=0.7, edgecolors="none")
ax2.axvline(np.median(safety), color="gray", ls="--", lw=0.8, label="Median safety")
ax2.axhline(np.median(old_density), color="gray", ls=":", lw=0.8, label="Median old density")
ax2.set_xlabel("Safety score (0-100)")
ax2.set_ylabel("Elderly population density")
ax2.set_title("ESDI: safety vs elderly density trade-off", fontweight="bold")
ax2.legend(fontsize=8)
plt.colorbar(sc, ax=ax2, shrink=0.8, label="ESDI")
fig2.tight_layout()
fig2.savefig(os.path.join(FIG_DIR, "fig_esdi_scatter.png"), dpi=300, bbox_inches="tight")
fig2.savefig(os.path.join(FIG_DIR, "fig_esdi_scatter.pdf"), bbox_inches="tight")
plt.close()
print(f"[Saved] fig_esdi_scatter.png/pdf")

# Save updated merged data with ESDI
df.to_csv(os.path.join(OUT_DIR, "merged_data_with_esdi.csv"), index=False, encoding="utf-8-sig")

print("\n" + "=" * 70)
print("  ESDI ANALYSIS COMPLETE")
print("=" * 70)
print(json.dumps(results, indent=2))
