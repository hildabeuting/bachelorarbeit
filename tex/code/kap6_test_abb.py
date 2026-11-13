# Abschnitt 6.5: Daten der Abbildungen 6.3 bis 6.10 zum Test-EKG
import os
import numpy as np
import pywt
from kap6_verfahren import fs, wavelet, J, entrauschen
from kap6_test import testsignal, gestoert, drift, SEED

ordner = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "abbildungen")
von, bis = 60, 64

f = testsignal()
y = gestoert(f, SEED)
f_dach = entrauschen(y)
f_dach3 = entrauschen(y, JH=1, faktor=0.2)
t = np.arange(len(f)) / fs
m = (t >= von) & (t < bis)

def zerlegung(x):
    c = pywt.wavedec(x, wavelet, mode="periodization", level=J)
    return c[:0:-1] + [c[0]]

stufe = list(range(1, J + 1)) + [J]

tabelle = np.column_stack([t, f, y, drift(len(f)), f_dach, f_dach3])[m]
kopf = "t f Y b fdach fdach3"
np.savetxt(os.path.join(ordner, "kap6_test.dat"), tabelle, header=kopf, comments="", fmt="%.6f")
namen = ["d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "a8"]
kopf = " ".join(f"t{n} {n}" for n in namen)
for name, x in [("f", f), ("Y", y), ("fdach", f_dach)]:
    tabelle = np.full((int((bis - von) * fs) // 2, 18), np.nan)
    for i, (d, j) in enumerate(zip(zerlegung(x), stufe)):
        tk = np.arange(len(d)) * 2**j / fs
        mk = (tk >= von) & (tk < bis)
        tabelle[:mk.sum(), 2 * i], tabelle[:mk.sum(), 2 * i + 1] = tk[mk], d[mk]
    np.savetxt(os.path.join(ordner, f"kap6_zerlegung_{name}.dat"), tabelle, header=kopf, comments="", fmt="%.6f")
