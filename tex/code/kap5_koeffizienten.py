# Beispiel aus Kapitel 5: Koeffizienten a_5 und d_5 bis d_1 des Testsignals f (Abb. 5.6).
import numpy as np
import pywt
import matplotlib.pyplot as plt

N = 2048
J = 5
zentren = [1.3, 3.7, 6.2]

t = 8 * np.arange(N) / N
u = np.sin(2 * np.pi * t)

s = sum(3 * np.maximum(0, 1 - np.abs(t - c) / 0.03) for c in zentren)
f = u + s

c = pywt.wavedec(f, "db2", mode="periodization", level=J)

plt.rcParams.update({"font.size": 11, "mathtext.fontset": "cm"})
fig, ax = plt.subplots(J + 1, 1, figsize=(7.0, 7.0), sharex=True)

for i in range(J + 1):
    j = J - i + 1 if i > 0 else J
    d = c[i]
    grenze = 6.0 if i == 0 else 2.5
    pos = (np.arange(len(d)) + 0.5) * 2**j / 256
    ax[i].vlines(pos, 0, np.clip(d, -grenze, grenze), color="royalblue", lw=1.1)
    ax[i].axhline(0, color="black", lw=0.4)
    ax[i].set_ylim(-grenze, grenze)
    ax[i].set_yticks([-5, 0, 5] if i == 0 else [-2, 0, 2])
    ax[i].set_ylabel("$a_5$" if i == 0 else f"$d_{j}$", rotation=0, labelpad=12, va="center")
    ax[i].spines[["top", "right"]].set_visible(False)

ax[-1].set_xlim(0, 8)
ax[-1].set_xlabel("$t$")
fig.tight_layout(h_pad=0.3)

plt.show()
