# Satz 4.14 und Beispiel 4.15: Daubechies-Wavelets mit p = 2 und p = 6
import os
import numpy as np
import pywt

ordner = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "abbildungen")

for p in [2, 6]:
    phi, psi, t = pywt.Wavelet(f"db{p}").wavefun(level=8)
    tabelle = np.column_stack([t, phi, t + 1 - p, psi])
    np.savetxt(os.path.join(ordner, f"kap4_db{p}.dat"), tabelle, header="t phi s psi", comments="", fmt="%.6f")