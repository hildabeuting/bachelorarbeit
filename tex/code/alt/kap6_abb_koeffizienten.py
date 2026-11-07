# Abbildung 6.7: Koeffizienten a8 und d8 bis d1 der 8-s-Ausschnitte aus den Datensätzen 100 und 103 (MLII)
import numpy as np
import matplotlib.pyplot as plt
import pywt
import wfdb
from kap6_verfahren import fs, N, J, JS, wavelet

plt.rcParams.update({"font.size": 9, "font.family": "serif"})
ausschnitte = {"100": 393, "103": 1088}
dauer = 8
namen = ["$a_8$"] + [f"$d_{j}$" for j in range(J, 0, -1)]
fig, ax = plt.subplots(J + 1, 2, figsize=(6.3, 7.0), sharex="col")
for spalte, (name, von) in enumerate(ausschnitte.items()):
    x = wfdb.rdrecord("../daten/mitdb/" + name).p_signal[:N, 0]
    c = pywt.wavedec(x - np.mean(x), wavelet, mode="periodization", level=J)
    T = np.median(np.abs(c[-1])) / 0.6745 * np.sqrt(2 * np.log(N))
    for zeile, (d, beschriftung) in enumerate(zip(c, namen)):
        stufe = J if zeile == 0 else J + 1 - zeile
        t = (np.arange(len(d)) + 0.5) * 2 ** stufe / fs
        m = (t >= von) & (t <= von + dauer)
        achse = ax[zeile, spalte]
        achse.vlines(t[m], 0, d[m], color="royalblue", lw=0.6)
        if zeile > J - JS:
            achse.axhline(T, color="gray", lw=0.6, ls="--")
            achse.axhline(-T, color="gray", lw=0.6, ls="--")
        achse.set_xlim(von, von + dauer)
        if spalte == 0:
            achse.set_ylabel(beschriftung)
    ax[0, spalte].set_title(f"Datensatz {name}")
    ax[-1, spalte].set_xlabel("Zeit $t$ in s")
fig.tight_layout()
fig.savefig("../abbildungen/abb6_3_koeffizienten.pdf")
plt.show()
