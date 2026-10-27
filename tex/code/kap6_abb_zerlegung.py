# Abbildungen 6.7 bis 6.9: DWT-Zerlegung (d_1 bis d_8, a_8) von f, Y und f_hat im Ausschnitt 60-64 s
import numpy as np
import matplotlib.pyplot as plt
import pywt
from kap6_verfahren import fs, wavelet, J, JS, entrauschen
from kap6_test import testsignal, gestoert, SEED

plt.rcParams.update({"font.size": 9, "font.family": "serif"})
f = testsignal()
y = gestoert(f, SEED)
fh = entrauschen(y)

def zerlegung(x):
    # pywt liefert [a_J, d_J, ..., d_1]; umsortiert zu [d_1, ..., d_J, a_J]
    c = pywt.wavedec(x, wavelet, mode="periodization", level=J)
    return c[:0:-1] + [c[0]]

namen = [f"$d_{{{j}}}$" for j in range(1, J + 1)] + [f"$a_{{{J}}}$"]
stufen = list(range(1, J + 1)) + [J]                    # Skala 2^j jeder Zeile
Z = {"f": zerlegung(f), "Y": zerlegung(y), "fhat": zerlegung(fh)}
sigma = np.median(np.abs(Z["Y"][0])) / 0.6745
T = sigma * np.sqrt(2 * np.log(len(y)))                 # universelle Schwelle wie im Verfahren
t0, t1 = 60, 64

def ausschnitt(c, j):
    t = np.arange(len(c)) * 2 ** j / fs                 # Koeffizient k gehört ungefähr zur Zeit k 2^j / f_s
    m = (t >= t0) & (t < t1)
    return t[m], c[m]

# gleicher Maßstab je Zeile in allen drei Zerlegungen
grenze = [1.1 * max([np.max(np.abs(ausschnitt(Z[s][i], stufen[i])[1])) for s in Z]
                   + ([T] if i < JS else []))   # Schwelle T auf d_1 bis d_JS sichtbar
          for i in range(J + 1)]

for s in Z:
    fig, ax = plt.subplots(J + 1, 1, figsize=(6.3, 7.4), sharex=True)
    for i in range(J + 1):
        t, c = ausschnitt(Z[s][i], stufen[i])
        ax[i].vlines(t, 0, c, color="royalblue", lw=0.6 if i < 2 else 0.9)
        ax[i].axhline(0, color="black", lw=0.4)
        if i < JS:
            ax[i].axhline(T, color="gray", lw=0.7, ls="--")
            ax[i].axhline(-T, color="gray", lw=0.7, ls="--")
        ax[i].set_ylim(-grenze[i], grenze[i])
        ax[i].set_ylabel(namen[i], rotation=0, labelpad=12, va="center")
        ax[i].tick_params(axis="y", labelsize=7)
    ax[-1].set_xlim(t0, t1)
    ax[-1].set_xlabel("Zeit $t$ in s")
    fig.tight_layout(h_pad=0.2)
    fig.savefig(f"../abbildungen/abb6_7_zerlegung_{s}.pdf")
print(f"T = {T:.3f}")
plt.show()
