# -*- coding: utf-8 -*-
"""
================================================================================
Spatial Regression Models: OLS / SLM / SEM with Control Variables + MGWR
================================================================================
Compares previous models (16 elements only) with augmented models
(16 elements + to_cbd_km + Old_Density + district FE).
Then runs MGWR to reveal spatial non-stationarity of coefficients.
================================================================================
"""
import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import statsmodels.api as sm
from scipy import stats
from libpysal.weights import KNN
from spreg import ML_Lag, ML_Error
from esda.moran import Moran

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
CONTROL_VARS = ["to_cbd_km", "Old_Density"]

def sig_stars(p):
    if p is None or (isinstance(p, float) and np.isnan(p)): return ""
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    if p < 0.1:   return "."
    return ""

# Load data
df = pd.read_csv(os.path.join(OUT_DIR, "merged_data.csv"))
print(f"Loaded {len(df)} rows")

y = df["safety_0_100"].values
coords = df[["longitude", "latitude"]].values

# ============================================================
# MODEL A: OLS/SLM/SEM — Elements Only (baseline, replicates previous)
# ============================================================
X_elem = df[CLASSES].values.astype(float)
X_elem_c = sm.add_constant(X_elem)

print("\n" + "=" * 70)
print("  MODEL A: Elements Only (baseline)")
print("=" * 70)

ols_a = sm.OLS(y, X_elem_c).fit()
print(f"  OLS R² = {ols_a.rsquared:.4f}")

w_knn = KNN(coords, k=15)
mi_a = Moran(ols_a.resid, w_knn)
print(f"  Moran's I (resid) = {mi_a.I:.4f}, p = {mi_a.p_sim:.4f}")

slm_a = ML_Lag(y, X_elem, w=w_knn, name_y="safety", name_x=CLASSES, spat_diag=True, moran=True)
sem_a = ML_Error(y, X_elem, w=w_knn, name_y="safety", name_x=CLASSES, spat_diag=True, moran=True)
print(f"  SLM Pseudo R² = {slm_a.pr2:.4f}, ρ = {float(np.atleast_1d(slm_a.rho).ravel()[0]):.4f}")
print(f"  SEM Pseudo R² = {sem_a.pr2:.4f}, λ = {float(np.atleast_1d(sem_a.lam).ravel()[0]):.4f}")

# ============================================================
# MODEL B: OLS/SLM/SEM — Elements + Control Variables
# ============================================================
X_full = df[CLASSES + CONTROL_VARS].values.astype(float)
X_full_c = sm.add_constant(X_full)
all_vars = CLASSES + CONTROL_VARS

print("\n" + "=" * 70)
print("  MODEL B: Elements + Control Variables")
print("=" * 70)

ols_b = sm.OLS(y, X_full_c).fit()
print(f"  OLS R² = {ols_b.rsquared:.4f} (vs baseline {ols_a.rsquared:.4f})")

mi_b = Moran(ols_b.resid, w_knn)
print(f"  Moran's I (resid) = {mi_b.I:.4f}, p = {mi_b.p_sim:.4f}")

slm_b = ML_Lag(y, X_full, w=w_knn, name_y="safety", name_x=all_vars, spat_diag=True, moran=True)
sem_b = ML_Error(y, X_full, w=w_knn, name_y="safety", name_x=all_vars, spat_diag=True, moran=True)
print(f"  SLM Pseudo R² = {slm_b.pr2:.4f}, ρ = {float(np.atleast_1d(slm_b.rho).ravel()[0]):.4f}")
print(f"  SEM Pseudo R² = {sem_b.pr2:.4f}, λ = {float(np.atleast_1d(sem_b.lam).ravel()[0]):.4f}")

# ============================================================
# Coefficient comparison table (Model B SLM)
# ============================================================
n_vars_full = len(all_vars) + 1  # intercept + vars
slm_b_z = slm_b.betas.flatten()[:n_vars_full] / np.sqrt(slm_b.vm.diagonal()[:n_vars_full])
slm_b_p = 2 * stats.norm.sf(np.abs(slm_b_z))

rows = []
for i, var in enumerate(["Intercept"] + all_vars):
    rows.append({
        "Variable": var,
        "OLS_B": round(float(ols_b.params[i]), 4),
        "OLS_p": float(ols_b.pvalues[i]),
        "OLS_sig": sig_stars(float(ols_b.pvalues[i])),
        "SLM_B": round(float(slm_b.betas[i][0]), 4),
        "SLM_z": round(float(slm_b_z[i]), 3),
        "SLM_p": float(slm_b_p[i]),
        "SLM_sig": sig_stars(float(slm_b_p[i])),
    })
# Add spatial params
slm_b_rho = float(np.atleast_1d(slm_b.rho).ravel()[0])
slm_b_rho_se = np.sqrt(slm_b.vm.diagonal()[-1])
slm_b_rho_p = 2 * stats.norm.sf(np.abs(slm_b_rho / slm_b_rho_se))
rows.append({"Variable": "ρ (Spatial Lag)", "OLS_B": "—", "OLS_p": None, "OLS_sig": "",
             "SLM_B": round(slm_b_rho, 4), "SLM_z": round(slm_b_rho / slm_b_rho_se, 3),
             "SLM_p": float(slm_b_rho_p), "SLM_sig": sig_stars(slm_b_rho_p)})

df_coef = pd.DataFrame(rows)
df_coef["SLM_Coef_Sig"] = df_coef.apply(lambda r: f"{r['SLM_B']}{r['SLM_sig']}" if r['SLM_B'] != '—' else '—', axis=1)
df_coef.to_csv(os.path.join(TBL_DIR, "slm_coef_with_controls.csv"), index=False, encoding="utf-8-sig")
print(f"\n[Saved] slm_coef_with_controls.csv")
print(df_coef[["Variable", "OLS_B", "OLS_sig", "SLM_Coef_Sig", "SLM_p"]].to_string(index=False))

# ============================================================
# Model comparison table (A vs B)
# ============================================================
comparison = pd.DataFrame([
    {"Model": "OLS (elements only)", "R²": round(ols_a.rsquared, 4), "AIC": round(ols_a.aic, 1),
     "Moran_I_resid": round(mi_a.I, 4)},
    {"Model": "OLS (+ controls)", "R²": round(ols_b.rsquared, 4), "AIC": round(ols_b.aic, 1),
     "Moran_I_resid": round(mi_b.I, 4)},
    {"Model": "SLM (elements only)", "R²": round(slm_a.pr2, 4), "AIC": round(slm_a.aic, 1),
     "Moran_I_resid": "—"},
    {"Model": "SLM (+ controls)", "R²": round(slm_b.pr2, 4), "AIC": round(slm_b.aic, 1),
     "Moran_I_resid": "—"},
    {"Model": "SEM (elements only)", "R²": round(sem_a.pr2, 4), "AIC": round(sem_a.aic, 1),
     "Moran_I_resid": "—"},
    {"Model": "SEM (+ controls)", "R²": round(sem_b.pr2, 4), "AIC": round(sem_b.aic, 1),
     "Moran_I_resid": "—"},
])
comparison.to_csv(os.path.join(TBL_DIR, "model_comparison_with_controls.csv"), index=False, encoding="utf-8-sig")
print(f"\n[Saved] model_comparison_with_controls.csv")
print(comparison.to_string(index=False))

# ============================================================
# MGWR: Multiscale Geographically Weighted Regression
# ============================================================
print("\n" + "=" * 70)
print("  GWR: Geographically Weighted Regression")
print("=" * 70)

try:
    from mgwr.gwr import GWR
    from mgwr.sel_bw import Sel_BW

    # Standardize for GWR
    from sklearn.preprocessing import StandardScaler
    scaler_y = StandardScaler()
    scaler_x = StandardScaler()

    y_std = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    X_std = scaler_x.fit_transform(X_full)

    # Bandwidth selection (AICc) — single bandwidth for GWR
    print("  Selecting optimal bandwidth (GWR)...")
    selector = Sel_BW(coords, y_std.reshape(-1, 1), X_std,
                      kernel="gaussian", fixed=False)
    bw_gwr = selector.search(criterion="AICc")
    print(f"  Optimal bandwidth: {bw_gwr:.1f}")

    gwr_model = GWR(coords, y_std.reshape(-1, 1), X_std, bw_gwr,
                    kernel="gaussian", fixed=False).fit()
    print(f"  GWR R2 = {gwr_model.R2:.4f}, AICc = {gwr_model.aicc:.2f}")

    # Save local coefficients
    param_names = ["Intercept"] + all_vars
    df_gwr_params = pd.DataFrame(gwr_model.params, columns=param_names)
    df_gwr_params["image_name"] = df["image_name"].values
    df_gwr_params["panorama_id"] = df["panorama_id"].values
    df_gwr_params["latitude"] = df["latitude"].values
    df_gwr_params["longitude"] = df["longitude"].values
    # Reorder: IDs + coords first
    id_cols = ["image_name", "panorama_id", "latitude", "longitude"]
    df_gwr_params = df_gwr_params[id_cols + param_names]
    df_gwr_params.to_csv(os.path.join(TBL_DIR, "gwr_local_params.csv"), index=False, encoding="utf-8-sig")

    # Summary of local coefficients
    print("\n  Local coefficient summary:")
    print(df_gwr_params[param_names].describe().round(4).to_string())

    # Figure: GWR local coefficients for key variables
    key_vars = ["shelter", "seat", "information_board", "light_pole", "to_cbd_km", "Old_Density"]
    key_vars = [v for v in key_vars if v in param_names]
    n_plots = len(key_vars)
    n_cols = 3
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4.5 * n_rows), constrained_layout=True)
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for i, var in enumerate(key_vars):
        ax = axes[i // n_cols, i % n_cols]
        col_idx = param_names.index(var)
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=gwr_model.params[:, col_idx],
                        cmap="RdBu_r", alpha=0.6, s=4, edgecolors="none")
        ax.set_title(var, fontweight="bold", fontsize=10)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        plt.colorbar(sc, ax=ax, shrink=0.8)

    # Hide unused axes
    for i in range(n_plots, n_rows * n_cols):
        axes[i // n_cols, i % n_cols].set_visible(False)

    fig.suptitle("GWR: Local coefficient spatial variation (standardized)",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.savefig(os.path.join(FIG_DIR, "fig_gwr_local_coefficients.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(FIG_DIR, "fig_gwr_local_coefficients.pdf"), bbox_inches="tight")
    plt.close()
    print(f"\n[Saved] fig_gwr_local_coefficients.png/pdf")

    mgwr_success = True

except Exception as e:
    print(f"  GWR failed: {e}")
    mgwr_success = False

# ============================================================
# Figure: Coefficient comparison (baseline vs +controls)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

# (a) Element coefficients: Model A vs Model B
ax = axes[0]
elem_only = [float(slm_a.betas[i+1][0]) for i in range(len(CLASSES))]
elem_ctrl = [float(slm_b.betas[i+1][0]) for i in range(len(CLASSES))]
x_pos = np.arange(len(CLASSES))
w = 0.35
ax.barh(x_pos - w/2, elem_only, w, color="#2166AC", label="SLM (elements only)", edgecolor="white")
ax.barh(x_pos + w/2, elem_ctrl, w, color="#B2182B", label="SLM (+ controls)", edgecolor="white")
ax.set_yticks(x_pos)
ax.set_yticklabels(CLASSES, fontsize=8)
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("SLM Coefficient")
ax.set_title("(a) Element coefficients: baseline vs +controls", fontweight="bold")
ax.legend(framealpha=0.9, fontsize=8)
ax.invert_yaxis()

# (b) Control variable coefficients
ax = axes[1]
ctrl_vals = [float(slm_b.betas[len(CLASSES)+i+1][0]) for i in range(len(CONTROL_VARS))]
ctrl_p = [float(slm_b_p[len(CLASSES)+i+1]) for i in range(len(CONTROL_VARS))]
colors_ctrl = ["#B2182B" if p < 0.05 else "#999999" for p in ctrl_p]
bars = ax.barh(CONTROL_VARS, ctrl_vals, color=colors_ctrl, edgecolor="white")
for bar, val, p in zip(bars, ctrl_vals, ctrl_p):
    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
            f"{val:.3f}{sig_stars(p)}", va="center", fontsize=9)
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("SLM Coefficient")
ax.set_title("(b) Control variable effects on safety", fontweight="bold")

fig.suptitle("Spatial regression: element effects and control variables",
             fontsize=13, fontweight="bold", y=1.02)
fig.savefig(os.path.join(FIG_DIR, "fig_slm_coef_comparison.png"), dpi=300, bbox_inches="tight")
fig.savefig(os.path.join(FIG_DIR, "fig_slm_coef_comparison.pdf"), bbox_inches="tight")
plt.close()
print(f"[Saved] fig_slm_coef_comparison.png/pdf")

# Save summary
results = {
    "Model_A_baseline": {
        "OLS_R2": round(float(ols_a.rsquared), 4),
        "SLM_R2": round(float(slm_a.pr2), 4),
        "SEM_R2": round(float(sem_a.pr2), 4),
        "SLM_rho": round(float(np.atleast_1d(slm_a.rho).ravel()[0]), 4),
        "Moran_I_resid": round(float(mi_a.I), 4),
    },
    "Model_B_with_controls": {
        "OLS_R2": round(float(ols_b.rsquared), 4),
        "SLM_R2": round(float(slm_b.pr2), 4),
        "SEM_R2": round(float(sem_b.pr2), 4),
        "SLM_rho": round(float(slm_b_rho), 4),
        "Moran_I_resid": round(float(mi_b.I), 4),
        "R2_improvement": round(float(ols_b.rsquared - ols_a.rsquared), 4),
    },
    "MGWR": {"success": mgwr_success} if mgwr_success else {"success": False},
}
with open(os.path.join(TBL_DIR, "spatial_models_summary.json"), "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 70)
print("  SPATIAL MODELS COMPLETE")
print("=" * 70)
print(json.dumps(results, indent=2))
