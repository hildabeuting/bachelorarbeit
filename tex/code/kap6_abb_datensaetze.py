# Abbildung 6.2: je ein Ausschnitt von 8 s aus den Datensätzen 100 und 103 (MLII)
import numpy as np
import matplotlib.pyplot as plt
import wfdb
from kap6_verfahren import fs, N

plt.rcParams.update({"font.size": 9, "font.family": "serif"})
ausschnitte = {"100": 393, "103": 1088}
dauer = 8
verrauscht = (1089.8, 1098.4)
fig, ax = plt.subplots(2, 1, figsize=(6.3, 3.6))
for achse, (name, von) in zip(ax, ausschnitte.items()):
    x = wfdb.rdrecord("../daten/mitdb/" + name).p_signal[:N, 0]
    y = x - np.mean(x)
    t = np.arange(N) / fs
    m = (t >= von) & (t <= von + dauer)
    if name == "103":
        achse.axvspan(verrauscht[0], von + dauer, color="gray", alpha=0.15, lw=0)
    achse.plot(t[m], y[m], color="royalblue", lw=0.8)
    achse.set_xlim(von, von + dauer)
    achse.set_ylabel(f"Datensatz {name}\nin mV")
ax[1].set_xlabel("Zeit $t$ in s")
fig.tight_layout()
fig.savefig("../abbildungen/abb6_2_datensaetze.pdf")
plt.show()
