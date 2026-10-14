# -*- coding: utf-8 -*-
"""
Compare GWR with vs without control variables (to_cbd_km, Old_Density).
Quantifies how much the 16 element coefficients change, and how R2/AICc shift.
"""
import warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")

from mgwr.gwr import GWR
from mgwr.sel_bw import Sel_BW
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(r"E:\HKBusStopResearch\20260821forB\results\merged_data.csv")
coords = df[["longitude", "latitude"]].values
y = df["safety_0_100"].values

CLASSES = [
    "shelter", "ads_board", "seat", "light_pole", "pillar", "box",
    "information_board", "step", "screen", "fence", "handrail",
    "traffic_sign", "bin", "camera", "Bollard", "vending_machine"
]
CONTROLS = ["to_cbd_km", "Old_Density"]

scaler_y = StandardScaler()
y_std = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

def run_gwr(X, name):
    scaler_x = StandardScaler()
    X_std = scaler_x.fit_transform(X)
    sel = Sel_BW(coords, y_std.reshape(-1, 1), X_std, kernel="gaussian", fixed=False)
    bw = sel.search(criterion="AICc")
    m = GWR(coords, y_std.reshape(-1, 1), X_std, bw, kernel="gaussian", fixed=False).fit()
    print(f"  [{name}] bw={bw:.1f}, R2={m.R2:.4f}, AICc={m.aicc:.2f}")
    return m

# 1. With controls
print("=" * 70)
print("GWR WITH controls (16 elements + to_cbd_km + Old_Density)")
print("=" * 70)
X_with = df[CLASSES + CONTROLS].values.astype(float)
m_with = run_gwr(X_with, "with controls")

# 2. Without controls
print("\n" + "=" * 70)
print("GWR WITHOUT controls (16 elements only)")
print("=" * 70)
X_without = df[CLASSES].values.astype(float)
m_without = run_gwr(X_without, "without controls")

# 3. Compare element coefficients (16 elements = columns 1..16; col 0 = intercept)
# GWR params shape: (n, k+1) where k = n_vars, col 0 = intercept
coef_with = m_with.params[:, 1:17]      # 16 element coefficients
coef_without = m_without.params[:, 1:17]

print("\n" + "=" * 70)
print("ELEMENT COEFFICIENT COMPARISON (with vs without controls)")
print("=" * 70)
print(f"{'Element':<18s} {'corr':>6s} {'mean_abs_diff':>14s} {'max_abs_diff':>13s}")
for i, c in enumerate(CLASSES):
    a = coef_with[:, i]
    b = coef_without[:, i]
    corr = np.corrcoef(a, b)[0, 1]
    mad = np.mean(np.abs(a - b))
    maxd = np.max(np.abs(a - b))
    print(f"{c:<18s} {corr:>6.3f} {mad:>14.4f} {maxd:>13.4f}")

# Overall
overall_corr = np.corrcoef(coef_with.flatten(), coef_without.flatten())[0, 1]
overall_mad = np.mean(np.abs(coef_with - coef_without))
print("-" * 70)
print(f"{'OVERALL':<18s} {overall_corr:>6.3f} {overall_mad:>14.4f} {np.max(np.abs(coef_with - coef_without)):>13.4f}")

print("\nR2 change:", round(m_without.R2 - m_with.R2, 4))
print("AICc change:", round(m_without.aicc - m_with.aicc, 2))
