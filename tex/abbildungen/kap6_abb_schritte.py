# Abbildung 6.4: Wirkung der drei Schritte auf die 8-s-Ausschnitte aus den Datensätzen 100 und 103 (MLII)
import numpy as np
import matplotlib.pyplot as plt
import wfdb
from kap6_verfahren import entrauschen, fs, N

plt.rcParams.update({"font.size": 9, "font.family": "serif"})
ausschnitte = {"100": 393, "103": 1088}
dauer = 8
fig, ax = plt.subplots(3, 2, figsize=(6.3, 5.0), sharex="col")
for spalte, (name, von) in enumerate(ausschnitte.items()):
    x = wfdb.rdrecord("../daten/mitdb/" + name).p_signal[:N, 0]
    y = x - np.mean(x)
    nur_schwelle = entrauschen(y, a_null=False)
    f = entrauschen(y)
    t = np.arange(N) / fs
    m = (t >= von) & (t <= von + dauer)
    ax[0, spalte].plot(t[m], y[m], color="gray", lw=0.7)
    ax[0, spalte].plot(t[m], (nur_schwelle - f)[m], color="royalblue", lw=1.0)
    ax[1, spalte].plot(t[m], 1000 * (y - nur_schwelle)[m], color="royalblue", lw=0.5)
    ax[2, spalte].plot(t[m], f[m], color="royalblue", lw=0.8)
    ax[0, spalte].set_title(f"Datensatz {name}")
    ax[-1, spalte].set_xlabel("Zeit $t$ in s")
    ax[-1, spalte].set_xlim(von, von + dauer)
ax[0, 0].set_ylabel("$y$ und $a_8$-Anteil\nin mV")
ax[1, 0].set_ylabel("durch Schwellen\nentfernt in $\\mu$V")
ax[2, 0].set_ylabel(r"Schätzung $\hat f$" + "\nin mV")
fig.tight_layout()
fig.savefig("../abbildungen/abb6_4_schritte.pdf")
plt.show()
