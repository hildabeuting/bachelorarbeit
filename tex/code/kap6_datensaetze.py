# Abschnitt 6.2: Ausschnitte aus den Datensätzen 100 und 103, Abbildung 6.2
import os
import numpy as np
import wfdb
from kap6_verfahren import fs, N

hier = os.path.dirname(os.path.abspath(__file__))
ordner = os.path.join(hier, "..", "abbildungen")
daten = os.path.join(hier, "..", "..", "data")
ausschnitte = {"100": 393, "103": 1088}
dauer = 8

spalten = []
for name, von in ausschnitte.items():
    x = wfdb.rdrecord(os.path.join(daten, name)).p_signal[:N, 0]
    y = x - np.mean(x)
    n = np.arange(von * fs, (von + dauer) * fs + 1)
    spalten += [n / fs, y[n]]

kopf = "t100 y100 t103 y103"
np.savetxt(os.path.join(ordner, "kap6_datensaetze.dat"), np.column_stack(spalten), header=kopf, comments="", fmt="%.6f")