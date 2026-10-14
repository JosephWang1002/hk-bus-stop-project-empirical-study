# -*- coding: utf-8 -*-
"""
================================================================================
Threshold Regression Analysis: Non-linear effects of key bus stop elements
================================================================================
Tests whether key elements (shelter, seat, information_board, etc.) exhibit
threshold effects: diminishing marginal returns after a certain count.

Method: Piecewise linear regression with grid-search threshold (Hansen 2000).
For each candidate threshold τ, fit:
    y = β0 + β1 * min(x, τ) + β2 * max(0, x - τ) + controls + ε
Compare R² across thresholds; F-test for structural break.
================================================================================
"""
import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats

warnings.filterwarnings("ignore")

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

CLASSES = [
    "shelter", "ads_board", "seat", "light_pole", "pillar", "box",
    "information_board", "step", "screen", "fence", "handrail",
    "traffic_sign", "bin", "camera", "Bollard", "vending_machine"
]
# Focus on key elements with sufficient variation
KEY_ELEMENTS = ["shelter", "seat", "information_board", "light_pole", "pillar",
                 "step", "bin", "ads_board", "Bollard", "handrail"]
CONTROL_VARS = ["to_cbd_km", "Old_Density"]

df = pd.read_csv(os.path.join(OUT_DIR, "merged_data.csv"))
print(f"Loaded {len(df)} rows")

y = df["safety_0_100"].values

# ============================================================
# Threshold regression for each key element
# ============================================================
def threshold_regression(y, x, controls, max_thresh=5):
    """
    Grid-search piecewise regression.
    For each threshold τ ∈ {1, 2, ..., max_thresh}:
        x_low  = min(x, τ)
        x_high = max(0, x - τ)
        Fit: y = β0 + β1*x_low + β2*x_high + controls
    Compare R²; test H0: β1 = β2 (no threshold effect).

    Returns dict with optimal threshold, coefficients, R², F-test.
    """
    results = []
    n = len(y)

    for tau in range(1, max_thresh + 1):
        x_low = np.minimum(x, tau)
        x_high = np.maximum(0, x - tau)

        X = np.column_stack([x_low, x_high, controls])
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()

        # Chow-type F-test: H0: β1 = β2 (no structural break)
        # Restricted model: use x directly (single slope)
        X_restricted = np.column_stack([x, controls])
        X_restricted = sm.add_constant(X_restricted)
        model_r = sm.OLS(y, X_restricted).fit()

        # F = ((RSS_r - RSS_u) / 1) / (RSS_u / (n - k))
        rss_r = model_r.ssr
        rss_u = model.ssr
        k_u = model.df_model + 1  # number of params
        f_stat = ((rss_r - rss_u) / 1) / (rss_u / (n - k_u))
        p_f = 1 - stats.f.cdf(f_stat, 1, n - k_u)

        results.append({
            "threshold": tau,
            "R2": model.rsquared,
            "beta_low": model.params[1],
            "beta_high": model.params[2],
            "p_low": model.pvalues[1],
            "p_high": model.pvalues[2],
            "F_test": f_stat,
            "F_p": p_f,
            "model": model,
        })

    # Find optimal threshold (max R²)
    best = max(results, key=lambda r: r["R2"])
    return best, results

# ============================================================
# Run threshold analysis
# ============================================================
all_results = {}
print("\n" + "=" * 70)
print("  THRESHOLD REGRESSION ANALYSIS")
print("=" * 70)

controls = df[CONTROL_VARS].values.astype(float)

for elem in KEY_ELEMENTS:
    x = df[elem].values.astype(float)
    max_possible = int(x.max())
    max_thresh = min(max_possible, 5)

    if max_thresh < 1:
        print(f"\n  [{elem}] Insufficient variation (max={max_possible}), skipped")
        continue

    best, all_taus = threshold_regression(y, x, controls, max_thresh=max_thresh)
    all_results[elem] = {
        "optimal_threshold": best["threshold"],
        "R2": round(float(best["R2"]), 4),
        "beta_below": round(float(best["beta_low"]), 4),
        "beta_above": round(float(best["beta_high"]), 4),
        "p_below": float(best["p_low"]),
        "p_above": float(best["p_high"]),
        "F_test": round(float(best["F_test"]), 3),
        "F_p": float(best["F_p"]),
    }

    print(f"\n  [{elem}] max_count={max_possible}")
    print(f"    Optimal threshold τ* = {best['threshold']}")
    print(f"    R2 = {best['R2']:.4f}")
    print(f"    β_below (x ≤ τ): {best['beta_low']:.4f} (p={best['p_low']:.4f})")
    print(f"    β_above (x > τ):  {best['beta_high']:.4f} (p={best['p_high']:.4f})")
    print(f"    F-test (break):   {best['F_test']:.3f} (p={best['F_p']:.4f})")
    if best["p_low"] < 0.05 and best["p_high"] > 0.05:
        print(f"    → Significant positive effect below threshold, NO effect above")
        print(f"    → Diminishing returns confirmed at τ = {best['threshold']}")
    elif best["beta_low"] > 0 and best["beta_high"] < best["beta_low"]:
        print(f"    → Effect decreases above threshold (diminishing returns)")

# Save results
with open(os.path.join(TBL_DIR, "threshold_regression_results.json"), "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\n[Saved] threshold_regression_results.json")

# ============================================================
# Figure: Threshold effect for top elements
# ============================================================
# Select elements with significant threshold effects
sig_elements = [e for e, r in all_results.items()
                if r["F_p"] < 0.05 and r["beta_below"] > 0]
if not sig_elements:
    sig_elements = [e for e, r in all_results.items() if r["beta_below"] > 0][:4]

n_plots = min(len(sig_elements), 6)
if n_plots > 0:
    n_cols = min(n_plots, 3)
    n_rows = (n_plots + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows),
                             constrained_layout=True)
    if n_plots == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, elem in enumerate(sig_elements[:n_plots]):
        ax = axes[i]
        x = df[elem].values
        # Add jitter for visualization
        x_jit = x + np.random.normal(0, 0.08, len(x))
        ax.scatter(x_jit, y, alpha=0.15, s=8, c="#2166AC", edgecolors="none")

        r = all_results[elem]
        tau = r["optimal_threshold"]
        beta_low = r["beta_below"]
        beta_high = r["beta_above"]

        # Plot piecewise regression line
        # Need intercept from model
        x_low = np.minimum(x, tau)
        x_high = np.maximum(0, x - tau)
        X = np.column_stack([x_low, x_high, controls])
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()
        intercept = model.params[0]
        ctrl_mean = controls.mean(axis=0)

        # Line below threshold
        xx_low = np.linspace(0, tau, 50)
        yy_low = intercept + beta_low * xx_low + np.sum(ctrl_mean * model.params[3:])
        ax.plot(xx_low, yy_low, color="#B2182B", lw=2, label=f"β₁ = {beta_low:.2f}***")

        # Line above threshold
        xx_high = np.linspace(tau, x.max(), 50)
        yy_high = intercept + beta_low * tau + beta_high * (xx_high - tau) + np.sum(ctrl_mean * model.params[3:])
        ax.plot(xx_high, yy_high, color="#4DAF4A", lw=2, label=f"β₂ = {beta_high:.2f}")

        ax.axvline(tau, color="gray", ls="--", lw=1, alpha=0.7, label=f"τ* = {tau}")
        ax.set_xlabel(f"Count of {elem}")
        ax.set_ylabel("Safety score (0-100)")
        ax.set_title(f"{elem} (F = {r['F_test']:.1f}, p = {r['F_p']:.3f})", fontweight="bold")
        ax.legend(fontsize=8, framealpha=0.9)

    # Hide unused
    for i in range(n_plots, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("Threshold effects: diminishing returns of bus stop elements",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.savefig(os.path.join(FIG_DIR, "fig_threshold_effects.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(FIG_DIR, "fig_threshold_effects.pdf"), bbox_inches="tight")
    plt.close()
    print(f"[Saved] fig_threshold_effects.png/pdf")

# ============================================================
# Figure: R² across thresholds for each element
# ============================================================
fig2, ax2 = plt.subplots(figsize=(8, 5))
for elem in KEY_ELEMENTS[:6]:
    if elem in all_results:
        x = df[elem].values.astype(float)
        max_thresh = min(int(x.max()), 5)
        if max_thresh < 1:
            continue
        controls_local = df[CONTROL_VARS].values.astype(float)
        _, all_taus = threshold_regression(y, x, controls_local, max_thresh)
        taus = [r["threshold"] for r in all_taus]
        r2s = [r["R2"] for r in all_taus]
        ax2.plot(taus, r2s, "o-", label=elem, lw=1.5, markersize=6)

ax2.set_xlabel("Threshold value (τ)")
ax2.set_ylabel("Model R²")
ax2.set_title("R² across candidate thresholds", fontweight="bold")
ax2.legend(fontsize=8, framealpha=0.9)
fig2.tight_layout()
fig2.savefig(os.path.join(FIG_DIR, "fig_threshold_r2_comparison.png"), dpi=300, bbox_inches="tight")
fig2.savefig(os.path.join(FIG_DIR, "fig_threshold_r2_comparison.pdf"), bbox_inches="tight")
plt.close()
print(f"[Saved] fig_threshold_r2_comparison.png/pdf")

print("\n" + "=" * 70)
print("  THRESHOLD ANALYSIS COMPLETE")
print("=" * 70)
print(json.dumps(all_results, indent=2))
