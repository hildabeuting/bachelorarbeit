# Abbildung 6.1: Datensatz 103 (MLII), Ausschnitt um die als verrauscht markierte Stelle, vor und nach dem Entrauschen
import numpy as np
import matplotlib.pyplot as plt
import wfdb
from kap6_verfahren import entrauschen, fs, N

plt.rcParams.update({"font.size": 9, "font.family": "serif"})
von, bis = 1084, 1102
verrauscht = (1089.8, 1098.4)
x = wfdb.rdrecord("../daten/mitdb/103").p_signal[:N, 0]
y = x - np.mean(x)
f = entrauschen(y)
t = np.arange(N) / fs
m = (t >= von) & (t <= bis)
fig, ax = plt.subplots(3, 1, figsize=(6.3, 4.4), sharex=True)
for achse in ax:
    achse.axvspan(*verrauscht, color="gray", alpha=0.15, lw=0)
ax[0].plot(t[m], y[m], color="gray", lw=0.7)
ax[0].set_ylabel("Messung $y$ in mV")
ax[1].plot(t[m], f[m], color="royalblue", lw=0.8)
ax[1].set_ylabel(r"Schätzung $\hat f$ in mV")
ax[2].plot(t[m], y[m] - f[m], color="gray", lw=0.8)
ax[2].set_ylabel(r"$y - \hat f$ in mV")
ax[2].set_xlabel("Zeit $t$ in s")
ax[2].set_xlim(von, bis)
fig.tight_layout()
fig.savefig("../abbildungen/abb6_1_zeit.pdf")
plt.show()
