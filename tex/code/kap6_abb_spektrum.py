# Abbildung 6.9: Energiespektrum von Datensatz 103 (MLII) vor und nach dem Entrauschen
import numpy as np
import matplotlib.pyplot as plt
import wfdb
from kap6_verfahren import entrauschen, spektrum, fs, N, J, JS

plt.rcParams.update({"font.size": 9, "font.family": "serif"})
x = wfdb.rdrecord("../daten/mitdb/103").p_signal[:N, 0]
y = x - np.mean(x)
f = entrauschen(y)
fr, P = spektrum(y)
_, Q = spektrum(f)
fig, ax = plt.subplots(1, 2, figsize=(6.3, 2.6))
for achse, hi in zip(ax, [2, 180]):
    m = (fr > 0) & (fr <= hi)
    achse.semilogy(fr[m], P[m], color="gray", lw=0.7)
    achse.semilogy(fr[m], Q[m], color="royalblue", lw=0.7)
    achse.set_xlim(0, hi)
    achse.set_xlabel("Frequenz in Hz")
ax[0].axvline(fs / 2 ** (J + 1), color="black", lw=0.6, ls="--")
ax[0].axvline(fs / 2 ** J, color="black", lw=0.6, ls=":")
ax[1].axvline(fs / 2 ** (JS + 1), color="black", lw=0.6, ls="--")
ax[0].set_ylabel(r"Energiespektrum")
fig.tight_layout()
fig.savefig("../abbildungen/abb6_5_spektrum.pdf")
plt.show()
