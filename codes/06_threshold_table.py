# -*- coding: utf-8 -*-
"""
================================================================================
Threshold Regression: Full table for all 16 elements
================================================================================
Piecewise threshold regression (Hansen 2000) for each of the 16 detected
elements. Model: y = β0 + β1*min(x,τ) + β2*max(0,x-τ) + controls
Controls: to_cbd_km, Old_Density.

Outputs:
  - threshold_regression_16_elements.csv  (full table)
  - printed summary
================================================================================
"""
import os, warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

warnings.filterwarnings("ignore")

OUT_DIR = r"E:\HKBusStopResearch\20260821forB\results"
TBL_DIR = os.path.join(OUT_DIR, "tables")
os.makedirs(TBL_DIR, exist_ok=True)

# Element display order (column name, display label)
ELEMENT_ORDER = [
    ("information_board", "Information board"),
    ("shelter", "Shelter"),
    ("step", "Step"),
    ("light_pole", "Light pole"),
    ("pillar", "Pillar"),
    ("seat", "Seat"),
    ("bin", "Bin"),
    ("ads_board", "Ads board"),
    ("fence", "Fence"),
    ("box", "Power Box"),
    ("handrail", "Handrail"),
    ("traffic_sign", "Traffic sign"),
    ("Bollard", "Bollard"),
    ("screen", "Screen"),
    ("camera", "Camera"),
    ("vending_machine", "Vending machine"),
]
CONTROL_VARS = ["to_cbd_km", "Old_Density"]

df = pd.read_csv(os.path.join(OUT_DIR, "merged_data.csv"))
y = df["safety_0_100"].values
controls = df[CONTROL_VARS].values.astype(float)

def sig_stars(p):
    if p is None or np.isnan(p):
        return ""
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    if p < 0.1:   return "."
    return ""

def threshold_regression(y, x, controls, max_thresh):
    """Grid-search piecewise regression; returns best dict."""
    results = []
    n = len(y)
    for tau in range(1, max_thresh + 1):
        x_low = np.minimum(x, tau)
        x_high = np.maximum(0, x - tau)
        X = sm.add_constant(np.column_stack([x_low, x_high, controls]))
        model = sm.OLS(y, X).fit()
        # Restricted model (no break): single slope
        Xr = sm.add_constant(np.column_stack([x, controls]))
        model_r = sm.OLS(y, Xr).fit()
        k_u = model.df_model + 1
        f_stat = ((model_r.ssr - model.ssr) / 1) / (model.ssr / (n - k_u))
        p_f = 1 - stats.f.cdf(f_stat, 1, n - k_u)
        results.append({
            "threshold": tau, "R2": model.rsquared,
            "beta_low": model.params[1], "beta_high": model.params[2],
            "p_low": model.pvalues[1], "p_high": model.pvalues[2],
            "F_test": f_stat, "F_p": p_f,
        })
    return max(results, key=lambda r: r["R2"])

def interpret(max_cnt, best):
    """Assign a concise interpretation label."""
    if max_cnt < 1:
        return "No variation (absent)"
    b1, b2 = best["beta_low"], best["beta_high"]
    p1, p2 = best["p_low"], best["p_high"]
    fp = best["F_p"]
    if fp < 0.05:
        if b1 > 0 and p1 < 0.05 and p2 >= 0.05:
            return "Presence-only (diminishing to zero)"
        if b1 > 0 and b2 > 0 and p1 < 0.05 and b2 < b1:
            return "Diminishing but positive"
        if b1 < 0 and p1 < 0.05:
            return "Negative effect"
        return "Significant threshold"
    return "No significant threshold"

rows = []
print("=" * 110)
print("  THRESHOLD REGRESSION — ALL 16 ELEMENTS")
print("=" * 110)

for col, disp in ELEMENT_ORDER:
    x = df[col].values.astype(float)
    max_cnt = int(x.max())
    max_thresh = min(max_cnt, 5)

    if max_thresh < 1:
        # No variation at all (max count = 0)
        rows.append({
            "Element": disp, "Max_Count": max_cnt, "tau*": None,
            "beta_below": None, "beta_above": None,
            "p_below": None, "p_above": None,
            "F_stat": None, "F_p": None, "R2": None,
            "Interpretation": "No variation (absent)",
        })
        print(f"  {disp:<18s} max=0 -> no variation, skipped")
        continue

    best = threshold_regression(y, x, controls, max_thresh)
    interp = interpret(max_cnt, best)
    rows.append({
        "Element": disp,
        "Max_Count": max_cnt,
        "tau*": best["threshold"],
        "beta_below": round(float(best["beta_low"]), 3),
        "beta_above": round(float(best["beta_high"]), 3),
        "p_below": float(best["p_low"]),
        "p_above": float(best["p_high"]),
        "F_stat": round(float(best["F_test"]), 2),
        "F_p": float(best["F_p"]),
        "R2": round(float(best["R2"]), 4),
        "Interpretation": interp,
    })
    print(f"  {disp:<18s} max={max_cnt}, tau*={best['threshold']}, "
          f"beta1={best['beta_low']:.2f}{sig_stars(best['p_low']):<3s} "
          f"beta2={best['beta_high']:.2f}{sig_stars(best['p_high']):<3s} "
          f"F={best['F_test']:.1f}{sig_stars(best['F_p']):<3s} R2={best['R2']:.3f}")

df_out = pd.DataFrame(rows)

# Save CSV
csv_path = os.path.join(TBL_DIR, "threshold_regression_16_elements.csv")
df_out.to_csv(csv_path, index=False, encoding="utf-8-sig")
print(f"\n[Saved] {csv_path}")

# ============================================================
# Print formatted table
# ============================================================
print("\n" + "=" * 110)
print("  FINAL TABLE")
print("=" * 110)

def fmt(v, nd=2):
    return "—" if v is None else f"{v:.{nd}f}"

header = (f"{'Element':<18s} {'Max':>4s} {'tau*':>5s} "
          f"{'beta1(≤τ)':>10s} {'beta2(>τ)':>10s} "
          f"{'F_stat':>8s} {'R2':>6s}  Interpretation")
print(header)
print("-" * 110)
for r in rows:
    e = r["Element"]
    mx = r["Max_Count"]
    tau = "—" if r["tau*"] is None else str(r["tau*"])
    b1 = "—" if r["beta_below"] is None else f"{r['beta_below']:.2f}{sig_stars(r['p_below'])}"
    b2 = "—" if r["beta_above"] is None else f"{r['beta_above']:.2f}{sig_stars(r['p_above'])}"
    fv = "—" if r["F_stat"] is None else f"{r['F_stat']:.1f}{sig_stars(r['F_p'])}"
    r2 = "—" if r["R2"] is None else f"{r['R2']:.3f}"
    print(f"{e:<18s} {mx:>4d} {tau:>5s} {b1:>10s} {b2:>10s} {fv:>8s} {r2:>6s}  {r['Interpretation']}")

print("\nNote: beta1 = slope below threshold tau*; beta2 = slope above tau*.")
print("      Sig: *** p<0.001, ** p<0.01, * p<0.05, . p<0.1")
print("      Controls: to_cbd_km, Old_Density (included in all models).")
