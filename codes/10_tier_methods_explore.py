# -*- coding: utf-8 -*-
"""
Explore alternative Tier-1 (highest priority) classification methods for ESDI.
Compares: quantile (current), Martens-sufficiency threshold, Jenks/K-means breaks.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

df = pd.read_csv(r"E:\HKBusStopResearch\20260821forB\results\merged_data.csv")
safety = df["safety_0_100"].values
old = df["Old_Density"].values

# ESDI (equal weight)
safety_norm = (safety - safety.min()) / (safety.max() - safety.min())
old_norm = (old - old.min()) / (old.max() - old.min())
esdi = 0.5 * (1 - safety_norm) + 0.5 * old_norm

T_STAR = 42.67  # sufficiency threshold from validation

print("=" * 70)
print("METHOD 1: Quantile (current) — Tier1 = top 10% of ESDI")
print("=" * 70)
p90 = np.percentile(esdi, 90)
tier1_q = esdi >= p90
print(f"  ESDI >= {p90:.4f}  ->  {tier1_q.sum()} stops (10%)")

print("\n" + "=" * 70)
print("METHOD 2: Martens sufficiency × demand (theory-driven)")
print("=" * 70)
insufficient = safety < T_STAR
high_need = old > np.median(old)
print(f"  Safety < {T_STAR} (insufficient):  {insufficient.sum()} stops ({insufficient.mean()*100:.1f}%)")
print(f"  Old_Density > median (high need):  {high_need.sum()} stops ({high_need.mean()*100:.1f}%)")
tier1_intersect = insufficient & high_need
print(f"  Tier1 = insufficient AND high need: {tier1_intersect.sum()} stops ({tier1_intersect.mean()*100:.1f}%)")

print("\n" + "=" * 70)
print("METHOD 3: Jenks natural breaks (K-means 1D, 4 classes)")
print("=" * 70)
km = KMeans(n_clusters=4, random_state=42, n_init=20).fit(esdi.reshape(-1, 1))
labels = km.labels_
# identify highest-ESDI cluster
centers = km.cluster_centers_.flatten()
highest_cluster = centers.argmax()
tier1_jenks = labels == highest_cluster
print(f"  Cluster centers: {np.sort(centers).round(4)}")
print(f"  Tier1 (highest cluster): {tier1_jenks.sum()} stops ({tier1_jenks.mean()*100:.1f}%)")
# boundaries
for i in range(4):
    c = centers[i]
    mask = labels == i
    print(f"    cluster {i}: center={c:.4f}, n={mask.sum()}, esdi range=[{esdi[mask].min():.4f},{esdi[mask].max():.4f}]")

print("\n" + "=" * 70)
print("METHOD 4: Mean + k*std (statistical outlier)")
print("=" * 70)
for k in [1.0, 1.5]:
    thr = esdi.mean() + k * esdi.std()
    n = (esdi >= thr).sum()
    print(f"  ESDI >= mean + {k}*std = {thr:.4f}  ->  {n} stops ({n/len(esdi)*100:.1f}%)")

# Overlap summary
print("\n" + "=" * 70)
print("OVERLAP AMONG METHODS (Tier1)")
print("=" * 70)
methods = {"Quantile10%": tier1_q, "Sufficiency×Demand": tier1_intersect, "Jenks4": tier1_jenks}
for a in methods:
    for b in methods:
        if a < b:
            n_both = (methods[a] & methods[b]).sum()
            print(f"  {a} ∩ {b}: {n_both}")

# Sanity: what safety/Old_Density do Tier1 look like under each method
print("\n" + "=" * 70)
print("PROFILE OF Tier1 UNDER EACH METHOD")
print("=" * 70)
for name, mask in methods.items():
    print(f"  {name}: n={mask.sum()}, safety mean={safety[mask].mean():.2f}, "
          f"safety max={safety[mask].max():.2f}, Old mean={old[mask].mean():.0f}")
