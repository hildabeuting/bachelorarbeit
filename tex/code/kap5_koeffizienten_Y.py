# Beispiel aus Kapitel 5: Detailkoeffizienten d_5 bis d_1 der Messwerte Y mit Schwelle T (Abb. 5.7).
import numpy as np
import pywt
import matplotlib.pyplot as plt

N = 2048
J = 5
sigma = 0.3
zentren = [1.3, 3.7, 6.2]

t = 8 * np.arange(N) / N
u = np.sin(2 * np.pi * t)

s = sum(3 * np.maximum(0, 1 - np.abs(t - c) / 0.03) for c in zentren)
f = u + s

Z = np.random.default_rng(93).standard_normal(N)
Y = f + sigma * Z

c = pywt.wavedec(Y, "db2", mode="periodization", level=J)

sigma_dach = np.median(np.abs(c[-1])) / 0.6745
T = sigma_dach * np.sqrt(2 * np.log(N))
AZ = np.concatenate(pywt.wavedec(Z, "db2", mode="periodization", level=J))

print(f"sigma_dach = {sigma_dach:.3f}")
print(f"T = {T:.2f}")
print(f"max|AZ| = {np.max(np.abs(AZ)):.2f}, max|sigma AZ| = {sigma * np.max(np.abs(AZ)):.2f}")
print(f"behalten (hart): {np.sum(np.abs(np.concatenate(c)) > T)} von {N}")

plt.rcParams.update({"font.size": 11, "mathtext.fontset": "cm"})
fig, ax = plt.subplots(J, 1, figsize=(7.0, 6.0), sharex=True)

for i in range(J):
    j = J - i
    d = c[i + 1]
    pos = (np.arange(len(d)) + 0.5) * 2**j / 256
    ax[i].vlines(pos, 0, np.clip(d, -2.5, 2.5), color="royalblue", lw=1.0)
    ax[i].axhline(0, color="black", lw=0.4)
    ax[i].axhline(T, color="gray", ls="--", lw=0.8)
    ax[i].axhline(-T, color="gray", ls="--", lw=0.8)
    ax[i].set_ylim(-2.5, 2.5)
    ax[i].set_yticks([-2, 0, 2])
    ax[i].set_ylabel(f"$d_{j}$", rotation=0, labelpad=12, va="center")
    ax[i].spines[["top", "right"]].set_visible(False)

ax[-1].set_xlim(0, 8)
ax[-1].set_xlabel("$t$")
fig.tight_layout(h_pad=0.3)

plt.show()
