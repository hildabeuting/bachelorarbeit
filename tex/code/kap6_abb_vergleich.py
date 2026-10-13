# Abbildung 6.10: 8-s-Ausschnitte aus den Datensätzen 100 und 103 (MLII) nach und vor dem Entrauschen
import numpy as np
import matplotlib.pyplot as plt
import wfdb
from kap6_verfahren import entrauschen, fs, N

plt.rcParams.update({"font.size": 9, "font.family": "serif"})
ausschnitte = {"100": 393, "103": 1088}
dauer = 8
fig, ax = plt.subplots(4, 1, figsize=(6.3, 5.0))
for k, (name, von) in enumerate(ausschnitte.items()):
    x = wfdb.rdrecord("../daten/mitdb/" + name).p_signal[:N, 0]
    y = x - np.mean(x)
    f = entrauschen(y)
    t = np.arange(N) / fs
    m = (t >= von) & (t <= von + dauer)
    unten, oben = min(y[m].min(), f[m].min()) - 0.2, max(y[m].max(), f[m].max()) + 0.2
    ax[2 * k].plot(t[m], f[m], color="royalblue", lw=0.8)
    ax[2 * k].set_ylabel(f"{name}: $\\hat f$ in mV")
    ax[2 * k + 1].plot(t[m], y[m], color="gray", lw=0.8)
    ax[2 * k + 1].set_ylabel(f"{name}: $Y$ in mV")
    for achse in ax[2 * k:2 * k + 2]:
        achse.set_xlim(von, von + dauer)
        achse.set_ylim(unten, oben)
        achse.axhline(0, color="gray", lw=0.5, ls="--")
    ax[2 * k + 1].set_xlabel("Zeit $t$ in s")
fig.tight_layout()
fig.savefig("../abbildungen/abb6_6_vergleich.pdf")
plt.show()
