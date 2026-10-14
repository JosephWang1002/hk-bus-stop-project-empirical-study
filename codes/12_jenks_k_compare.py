# -*- coding: utf-8 -*-
"""
Compare Jenks natural breaks for k = 3,4,5,6 classes.
GVF increases monotonically with k, so we look at the MARGINAL GVF gain
(elbow criterion) and resulting class sizes to judge the best k.
"""
import numpy as np
import pandas as pd

df = pd.read_csv(r"E:\HKBusStopResearch\20260821forB\results\merged_data.csv")
safety = df["safety_0_100"].values
old = df["Old_Density"].values
safety_norm = (safety - safety.min()) / (safety.max() - safety.min())
old_norm = (old - old.min()) / (old.max() - old.min())
esdi = 0.5 * (1 - safety_norm) + 0.5 * old_norm


def jenks_breaks(data, n_classes):
    x = np.sort(data).astype(float)
    n = len(x)
    ps = np.concatenate([[0.0], np.cumsum(x)])
    psq = np.concatenate([[0.0], np.cumsum(x * x)])

    def ssd(i, j):
        s = ps[j + 1] - ps[i]
        sq = psq[j + 1] - psq[i]
        m = j - i + 1
        return sq - s * s / m

    dp = np.full((n_classes, n), np.inf)
    boundary = np.zeros((n_classes, n), dtype=int)
    for i in range(n):
        dp[0, i] = ssd(0, i)
    for k in range(1, n_classes):
        for i in range(k, n):
            j_range = np.arange(k - 1, i)
            js = j_range + 1
            s = ps[i + 1] - ps[js]
            sq = psq[i + 1] - psq[js]
            m = i - js + 1
            seg = sq - s * s / m
            vals = dp[k - 1, j_range] + seg
            best = int(np.argmin(vals))
            dp[k, i] = vals[best]
            boundary[k, i] = j_range[best]
    breaks = []
    i = n - 1
    for k in range(n_classes - 1, 0, -1):
        b = boundary[k, i]
        breaks.append(x[b + 1])
        i = b
    breaks.reverse()
    gvf = 1.0 - dp[n_classes - 1, n - 1] / ssd(0, n - 1)
    return np.array(breaks), gvf


print("=" * 80)
print("  JENKS NATURAL BREAKS: k = 3, 4, 5, 6")
print("=" * 80)

prev_gvf = None
for k in [3, 4, 5, 6]:
    breaks, gvf = jenks_breaks(esdi, k)
    gain = gvf - prev_gvf if prev_gvf is not None else float("nan")
    # class sizes
    bounds = np.concatenate([[esdi.min()], breaks, [esdi.max()]])
    sizes = []
    for c in range(k):
        m = (esdi > bounds[c]) & (esdi <= bounds[c + 1]) if c > 0 else (esdi >= bounds[c]) & (esdi <= bounds[c + 1])
        sizes.append(int(m.sum()))
    print(f"\n  k={k}: GVF={gvf:.4f},  GVF gain from k-1 = {gain:.4f}")
    print(f"    breaks = {np.round(breaks, 4)}")
    print(f"    class sizes (low->high ESDI) = {sizes}")
    prev_gvf = gvf

print("\n" + "=" * 80)
print("  INTERPRETATION")
print("=" * 80)
print("  GVF gain shrinks as k grows; the 'elbow' is where the gain")
print("  drops sharply — that k is the efficient number of classes.")
