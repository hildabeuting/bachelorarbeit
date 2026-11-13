# Abschnitt 6.6: Ausschnitte vor und nach dem Entrauschen, Abbildung 6.10
import os
import numpy as np
import pywt
import wfdb
from kap6_verfahren import fs, N, wavelet, entrauschen

hier = os.path.dirname(os.path.abspath(__file__))
ordner = os.path.join(hier, "..", "abbildungen")
daten = os.path.join(hier, "..", "..", "data")
ausschnitte = {"100": 393, "103": 1088}
dauer = 8

spalten = []
for name, von in ausschnitte.items():
    x = wfdb.rdrecord(os.path.join(daten, name)).p_signal[:N, 0]
    y = x - np.mean(x)
    f_dach = entrauschen(y)
    d1 = pywt.dwt(y, wavelet, mode="periodization")[1]
    print(f"{name}: sigma_dach = {np.median(np.abs(d1)) / 0.6745:.4f} mV")
    n = np.arange(von * fs, (von + dauer) * fs + 1)
    spalten += [n / fs, y[n], f_dach[n]]

kopf = "t100 y100 f100 t103 y103 f103"
np.savetxt(os.path.join(ordner, "kap6_vergleich.dat"), np.column_stack(spalten), header=kopf, comments="", fmt="%.6f")
