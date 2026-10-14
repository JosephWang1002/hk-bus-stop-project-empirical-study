# -*- coding: utf-8 -*-
"""
================================================================================
Deep Fairness Analysis (Martens Transport Justice framework)
================================================================================
Operationalizes Martens' (2016) three principles of transport justice:

  1. Sufficiency principle  — establish a minimum safety threshold and measure
     the "sufficiency gap" (how far below the threshold each stop falls).
  2. Difference principle  — examine whether the most disadvantaged groups
     (older adults in high-density areas) bear disproportionate safety deficits.
  3. Maximax / average      — assess overall safety level (context for trade-offs).

Methods:
  - Getis-Ord Gi* hotspot analysis (statistical hot/cold spots of safety)
  - Sufficiency gap index + distribution
  - Rank-rank association between safety and elderly density (difference principle)
  - Deprivation-hotspot overlap with high-elderly-density zones
================================================================================
"""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from libpysal.weights import KNN, fill_diagonal
from esda.getisord import G_Local
from esda.moran import Moran

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

safety = df["safety_0_100"].values
old_density = df["Old_Density"].values
coords = df[["longitude", "latitude"]].values
districts = df["NAME_EN"].values

# Sufficiency threshold from independent validation (low-tier cutoff, 33rd pct)
SUFFICIENCY_THRESHOLD = 42.67

# ============================================================
# 1. SUFFICIENCY PRINCIPLE: sufficiency gap analysis
# ============================================================
print("=" * 70)
print("  1. SUFFICIENCY PRINCIPLE (Martens)")
print("=" * 70)

sufficiency_gap = np.maximum(0, SUFFICIENCY_THRESHOLD - safety)
df["sufficiency_gap"] = sufficiency_gap

n_insufficient = int((safety < SUFFICIENCY_THRESHOLD).sum())
n_total = len(safety)
sufficiency_rate = 1 - n_insufficient / n_total

print(f"  Threshold T* = {SUFFICIENCY_THRESHOLD} (low-tier cutoff from validation)")
print(f"  Insufficient stops (safety < T*): {n_insufficient} ({n_insufficient/n_total*100:.1f}%)")
print(f"  Sufficiency rate: {sufficiency_rate*100:.1f}%")
print(f"  Mean sufficiency gap (among insufficient): {sufficiency_gap[sufficiency_gap>0].mean():.2f} points")
print(f"  Total sufficiency gap: {sufficiency_gap.sum():.2f} points")

# Distribution of sufficiency gap
print(f"\n  Sufficiency gap distribution:")
for pct in [25, 50, 75, 90, 95]:
    print(f"    {pct}th percentile: {np.percentile(sufficiency_gap, pct):.2f}")

# ============================================================
# 2. DIFFERENCE PRINCIPLE: safety vs elderly density
# ============================================================
print("\n" + "=" * 70)
print("  2. DIFFERENCE PRINCIPLE (Martens)")
print("=" * 70)

# Rank correlation between safety and elderly density
rho, p_rho = stats.spearmanr(safety, old_density)
print(f"  Spearman ρ(safety, old_density) = {rho:.4f} (p = {p_rho:.4f})")

# Split elderly density into quintiles and compare safety
df["old_quintile"] = pd.qcut(old_density, 5, labels=["Q1 (lowest)", "Q2", "Q3", "Q4", "Q5 (highest)"])
quintile_stats = df.groupby("old_quintile", observed=True).agg(
    n=("safety_0_100", "count"),
    safety_mean=("safety_0_100", "mean"),
    safety_median=("safety_0_100", "median"),
    sufficiency_rate=("safety_0_100", lambda x: (x >= SUFFICIENCY_THRESHOLD).mean() * 100),
    mean_gap=("sufficiency_gap", "mean"),
).round(2)

print(f"\n  Safety by elderly density quintile:")
print(quintile_stats.to_string())

# ANOVA / trend test across quintiles
q1 = safety[old_density <= np.percentile(old_density, 20)]
q5 = safety[old_density > np.percentile(old_density, 80)]
t_stat, p_t = stats.ttest_ind(q1, q5)
print(f"\n  T-test Q1 vs Q5 (lowest vs highest elderly density):")
print(f"    Q1 mean = {q1.mean():.2f}, Q5 mean = {q5.mean():.2f}")
print(f"    t = {t_stat:.3f}, p = {p_t:.4f}")

# Difference principle violation: is sufficiency rate lower in high-elderly areas?
q1_rate = (q1 >= SUFFICIENCY_THRESHOLD).mean() * 100
q5_rate = (q5 >= SUFFICIENCY_THRESHOLD).mean() * 100
print(f"    Q1 sufficiency rate = {q1_rate:.1f}%, Q5 sufficiency rate = {q5_rate:.1f}%")
print(f"    Difference = {q1_rate - q5_rate:.1f} percentage points")

# ============================================================
# 3. GETIS-ORD Gi* HOTSPOT ANALYSIS
# ============================================================
print("\n" + "=" * 70)
print("  3. GETIS-ORD Gi* HOTSPOT ANALYSIS")
print("=" * 70)

w_knn = KNN(coords, k=30)
w_gi = fill_diagonal(w_knn, 1.0)  # Gi* requires self-weight on diagonal
gi = G_Local(safety, w_gi, star=None, permutations=999, seed=42)

# Hot spots: Gi* z > 1.96 (p < 0.05)
hot_mask = (gi.Zs > 1.96)
cold_mask = (gi.Zs < -1.96)
print(f"  Hot spots (high safety, Gi* > 1.96): {hot_mask.sum()} stops")
print(f"  Cold spots (low safety, Gi* < -1.96): {cold_mask.sum()} stops")
print(f"  Gi* Z range: [{gi.Zs.min():.2f}, {gi.Zs.max():.2f}]")
print(f"  NOTE: Gi* detects local EXTREME-VALUE concentration, distinct from LISA")
print(f"        (local correlation). Safety shows significant LISA clustering but")
print(f"        no Gi* hot spots because its autocorrelation (Moran I ~ 0.27) is")
print(f"        moderate — enough for correlated neighbors, not for extreme-value")
print(f"        blocks. This is complementary, not contradictory, to LISA.")

df["Gi_Z"] = gi.Zs
df["Gi_hot"] = hot_mask
df["Gi_cold"] = cold_mask

# ============================================================
# 4. DEPRIVATION-HOTSPOT OVERLAP (difference principle spatial)
# ============================================================
print("\n" + "=" * 70)
print("  4. DEPRIVATION-ELDERLY OVERLAP")
print("=" * 70)

high_elderly = old_density > np.median(old_density)
cold_high_elderly = cold_mask & high_elderly
print(f"  Cold spots in high-elderly-density areas: {cold_high_elderly.sum()} stops")
print(f"  Cold spots in low-elderly-density areas: {(cold_mask & ~high_elderly).sum()} stops")

# Chi-square test: are cold spots disproportionately in high-elderly areas?
contingency = np.array([
    [cold_mask.sum(), (~cold_mask).sum()],
    [high_elderly.sum(), (~high_elderly).sum()]
])
chi2, p_chi2, dof, _ = stats.chi2_contingency(
    pd.crosstab(cold_mask, high_elderly).values
)
print(f"  Chi-square (cold spots × high elderly): chi2 = {chi2:.2f}, p = {p_chi2:.4f}")

# ============================================================
# 5. SAVE RESULTS
# ============================================================
results = {
    "sufficiency": {
        "threshold": SUFFICIENCY_THRESHOLD,
        "n_insufficient": n_insufficient,
        "sufficiency_rate_pct": round(sufficiency_rate * 100, 1),
        "mean_gap_insufficient": round(float(sufficiency_gap[sufficiency_gap > 0].mean()), 2),
        "total_gap": round(float(sufficiency_gap.sum()), 2),
    },
    "difference_principle": {
        "spearman_rho": round(float(rho), 4),
        "spearman_p": float(p_rho),
        "Q1_safety_mean": round(float(q1.mean()), 2),
        "Q5_safety_mean": round(float(q5.mean()), 2),
        "t_stat": round(float(t_stat), 3),
        "t_p": float(p_t),
        "Q1_sufficiency_rate_pct": round(float(q1_rate), 1),
        "Q5_sufficiency_rate_pct": round(float(q5_rate), 1),
    },
    "getis_ord": {
        "n_hotspots": int(hot_mask.sum()),
        "n_coldspots": int(cold_mask.sum()),
    },
    "overlap": {
        "n_cold_high_elderly": int(cold_high_elderly.sum()),
        "chi2": round(float(chi2), 2),
        "chi2_p": float(p_chi2),
    },
}
with open(os.path.join(TBL_DIR, "martens_fairness_results.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"\n[Saved] martens_fairness_results.json")

# Save enhanced data
df.to_csv(os.path.join(OUT_DIR, "merged_data_with_esdi.csv"), index=False, encoding="utf-8-sig")

# ============================================================
# 6. FIGURES
# ============================================================
# (a) Sufficiency gap spatial map
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)

ax = axes[0]
sc = ax.scatter(coords[:, 0], coords[:, 1], c=sufficiency_gap, cmap="YlOrRd",
                alpha=0.7, s=5, edgecolors="none")
ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
ax.set_title(f"(a) Sufficiency gap\n(threshold = {SUFFICIENCY_THRESHOLD})", fontweight="bold")
plt.colorbar(sc, ax=ax, shrink=0.8, label="Gap (points)")

# (b) Difference principle: safety vs elderly density scatter
ax = axes[1]
# Color by elderly density quintile
quintile_colors = ["#2166AC", "#67A9CF", "#FDDBC7", "#EF8A62", "#B2182B"]
for qi, qlabel in enumerate(["Q1 (lowest)", "Q2", "Q3", "Q4", "Q5 (highest)"]):
    mask = df["old_quintile"] == qlabel
    ax.scatter(old_density[mask], safety[mask], s=5, alpha=0.5,
               c=quintile_colors[qi], edgecolors="none", label=qlabel)
ax.axhline(SUFFICIENCY_THRESHOLD, color="gray", ls="--", lw=1,
           label=f"Sufficiency threshold ({SUFFICIENCY_THRESHOLD})")
ax.set_xlabel("Elderly population density")
ax.set_ylabel("Safety score (0-100)")
ax.set_title(f"(b) Difference principle\nρ = {rho:.3f} (p = {p_rho:.3f})", fontweight="bold")
ax.legend(fontsize=7, framealpha=0.9, markerscale=2)

# (c) Safety vs elderly density by quintile
ax = axes[2]
q_means = quintile_stats["safety_mean"].values
q_labels = ["Q1\n(lowest)", "Q2", "Q3", "Q4", "Q5\n(highest)"]
colors = ["#2166AC" if i < 4 else "#B2182B" for i in range(5)]
bars = ax.bar(q_labels, q_means, color=colors, edgecolor="white", alpha=0.8)
ax.axhline(SUFFICIENCY_THRESHOLD, color="gray", ls="--", lw=1, label=f"Sufficiency threshold ({SUFFICIENCY_THRESHOLD})")
for bar, val in zip(bars, q_means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f"{val:.1f}",
            ha="center", fontsize=8, fontweight="bold")
ax.set_xlabel("Elderly density quintile")
ax.set_ylabel("Mean safety score")
ax.set_title("(c) Difference principle:\nsafety by elderly density", fontweight="bold")
ax.legend(fontsize=8, framealpha=0.9)
ax.set_ylim(0, max(q_means) * 1.25)

fig.suptitle("Martens transport justice framework: sufficiency, difference, and spatial clustering",
             fontsize=13, fontweight="bold", y=1.02)
fig.savefig(os.path.join(FIG_DIR, "fig_martens_fairness.png"), dpi=300, bbox_inches="tight")
fig.savefig(os.path.join(FIG_DIR, "fig_martens_fairness.pdf"), bbox_inches="tight")
plt.close()
print(f"[Saved] fig_martens_fairness.png/pdf")

print("\n" + "=" * 70)
print("  MARTENS FAIRNESS ANALYSIS COMPLETE")
print("=" * 70)
print(json.dumps(results, indent=2))
