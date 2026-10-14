# -*- coding: utf-8 -*-
"""
================================================================================
Fairness Analysis: Gini + Theil Index + Theil Decomposition
================================================================================
Measures spatial inequality of bus stop safety for older adults.
Theory anchor: Mobility Justice — safety as basic capability for elderly mobility.
================================================================================
"""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

# Load data
df = pd.read_csv(os.path.join(OUT_DIR, "merged_data.csv"))
print(f"Loaded {len(df)} rows")

y = df["safety_0_100"].values
old_density = df["Old_Density"].values
districts = df["NAME_EN"].values

# ============================================================
# 1. Gini Coefficient
# ============================================================
def gini(x):
    x = np.sort(x[x > 0])  # safety is positive
    n = len(x)
    cum = np.cumsum(x)
    return (2 * np.sum((np.arange(1, n+1)) * x) - (n + 1) * cum[-1]) / (n * cum[-1])

gini_safety = gini(y)
print(f"\n[Gini] Safety score Gini = {gini_safety:.4f}")

# ============================================================
# 2. Theil Index (T = overall inequality)
# ============================================================
def theil_index(x):
    x = x[x > 0]
    mu = np.mean(x)
    return np.mean((x / mu) * np.log(x / mu))

theil_safety = theil_index(y)
print(f"[Theil] Safety score Theil T = {theil_safety:.6f}")

# ============================================================
# 3. Theil Decomposition (between-district vs within-district)
# ============================================================
def theil_decomposition(x, groups):
    """Decompose Theil T into between-group and within-group components.
    Returns: (T_total, T_between, T_within, share_between, share_within)
    """
    x = np.asarray(x, dtype=float)
    groups = np.asarray(groups)
    n = len(x)
    mu = np.mean(x)

    unique_groups = np.unique(groups)
    T_between = 0.0
    T_within = 0.0

    for g in unique_groups:
        mask = groups == g
        x_g = x[mask]
        n_g = len(x_g)
        mu_g = np.mean(x_g)
        share = n_g / n

        # Between-group component
        if mu_g > 0 and mu > 0:
            T_between += share * (mu_g / mu) * np.log(mu_g / mu)

        # Within-group component (Theil of this group, weighted)
        if len(x_g) > 1 and mu_g > 0:
            T_g = np.mean((x_g / mu_g) * np.log(x_g / mu_g))
            T_within += share * (mu_g / mu) * T_g

    T_total = T_between + T_within
    return T_total, T_between, T_within, T_between / T_total, T_within / T_total

T_total, T_between, T_within, share_between, share_within = theil_decomposition(y, districts)
print(f"\n[Theil Decomposition]")
print(f"  T_total   = {T_total:.6f}")
print(f"  T_between = {T_between:.6f} ({share_between*100:.1f}%)")
print(f"  T_within  = {T_within:.6f} ({share_within*100:.1f}%)")

# ============================================================
# 4. Per-district statistics
# ============================================================
district_stats = df.groupby("NAME_EN").agg(
    n=("safety_0_100", "count"),
    safety_mean=("safety_0_100", "mean"),
    safety_std=("safety_0_100", "std"),
    safety_min=("safety_0_100", "min"),
    safety_max=("safety_0_100", "max"),
    old_density_mean=("Old_Density", "mean"),
).round(2).sort_values("safety_mean")
district_stats.to_csv(os.path.join(TBL_DIR, "district_safety_stats.csv"), encoding="utf-8-sig")
print(f"\n[District stats] saved")
print(district_stats.to_string())

# ============================================================
# 5. Concentration Index for elderly safety
# ============================================================
# Measures whether higher elderly density areas have lower safety
# CI < 0 means disadvantaged (higher old density → lower safety)
def concentration_index(health, ses):
    """Health Achievement Index / Concentration Index.
    health: safety scores (higher = better)
    ses: socioeconomic status (old_density, higher = more need)
    """
    # Rank by SES
    order = np.argsort(ses)
    h = health[order]
    s = ses[order]
    n = len(h)
    mu_h = np.mean(h)
    # Weighted by rank
    ranks = np.arange(1, n + 1) / n
    ci = (2 / (n * mu_h)) * np.sum((ranks - 0.5) * h)
    return ci

ci = concentration_index(y, old_density)
print(f"\n[Concentration Index] CI = {ci:.6f}")
print(f"  (negative = pro-rich, i.e., higher elderly density → lower safety)")

# ============================================================
# 6. Save results JSON
# ============================================================
results = {
    "N": int(len(df)),
    "Gini_safety": round(float(gini_safety), 4),
    "Theil_T": round(float(theil_safety), 6),
    "Theil_decomposition": {
        "T_total": round(float(T_total), 6),
        "T_between": round(float(T_between), 6),
        "T_within": round(float(T_within), 6),
        "share_between_pct": round(float(share_between * 100), 1),
        "share_within_pct": round(float(share_within * 100), 1),
    },
    "Concentration_Index": round(float(ci), 6),
    "n_districts": int(df["NAME_EN"].nunique()),
}
with open(os.path.join(TBL_DIR, "fairness_results.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"\n[Saved] fairness_results.json")

# ============================================================
# 7. Figure: Lorenz curve + Theil decomposition
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)

# (a) Lorenz curve
ax = axes[0]
y_sorted = np.sort(y)
cum_share = np.cumsum(y_sorted) / np.sum(y_sorted)
pop_share = np.arange(1, len(y) + 1) / len(y)
ax.plot(np.concatenate([[0], pop_share]), np.concatenate([[0], cum_share]),
        color="#2166AC", lw=2, label=f"Lorenz curve\nGini = {gini_safety:.4f}")
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect equality")
ax.fill_between(np.concatenate([[0], pop_share]), np.concatenate([[0], cum_share]),
                np.concatenate([[0], pop_share]), alpha=0.15, color="#2166AC")
ax.set_xlabel("Cumulative share of bus stops (ranked by safety)")
ax.set_ylabel("Cumulative share of total safety scores")
ax.set_title("(a) Lorenz curve of safety distribution", fontweight="bold")
ax.legend(loc="upper left", framealpha=0.9)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

# (b) Theil decomposition bar chart
ax = axes[1]
components = ["Between-district\n(T_between)", "Within-district\n(T_within)"]
values = [T_between, T_within]
shares = [share_between * 100, share_within * 100]
colors = ["#B2182B", "#2166AC"]
bars = ax.bar(components, values, color=colors, edgecolor="white", width=0.5)
for bar, val, share in zip(bars, values, shares):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
            f"T = {val:.4f}\n({share:.1f}%)", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.set_ylabel("Theil T component")
ax.set_title(f"(b) Theil decomposition\n(Total T = {T_total:.4f})", fontweight="bold")
ax.set_ylim(0, max(values) * 1.35)

fig.suptitle("Spatial inequality of bus stop safety for older adults",
             fontsize=12.5, fontweight="bold", y=1.02)
fig.savefig(os.path.join(FIG_DIR, "fig_fairness_lorenz_theil.png"), dpi=300, bbox_inches="tight")
fig.savefig(os.path.join(FIG_DIR, "fig_fairness_lorenz_theil.pdf"), bbox_inches="tight")
plt.close()
print(f"[Saved] fig_fairness_lorenz_theil.png/pdf")

# ============================================================
# 8. Figure: District-level safety map (bar chart)
# ============================================================
fig2, ax2 = plt.subplots(figsize=(10, 6))
district_stats_sorted = district_stats.sort_values("safety_mean", ascending=True)
bars = ax2.barh(district_stats_sorted.index, district_stats_sorted["safety_mean"],
                color="#2166AC", edgecolor="white", alpha=0.8)
# Color bars by above/below mean
mean_safety = df["safety_0_100"].mean()
for bar, val in zip(bars, district_stats_sorted["safety_mean"]):
    bar.set_color("#B2182B" if val < mean_safety else "#2166AC")
ax2.axvline(mean_safety, color="gray", ls="--", lw=1, label=f"City mean = {mean_safety:.1f}")
ax2.set_xlabel("Mean safety score (0-100)")
ax2.set_title("District-level mean safety scores", fontweight="bold")
ax2.legend(framealpha=0.9)
fig2.tight_layout()
fig2.savefig(os.path.join(FIG_DIR, "fig_district_safety_bar.png"), dpi=300, bbox_inches="tight")
fig2.savefig(os.path.join(FIG_DIR, "fig_district_safety_bar.pdf"), bbox_inches="tight")
plt.close()
print(f"[Saved] fig_district_safety_bar.png/pdf")

print("\n" + "=" * 70)
print("  FAIRNESS ANALYSIS COMPLETE")
print("=" * 70)
print(json.dumps(results, indent=2))
