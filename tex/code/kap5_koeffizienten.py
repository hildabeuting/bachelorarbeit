# Beispiel 5.11: Koeffizienten des Testsignals, Abbildung 5.6
import os
import numpy as np
import pywt

ordner = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "abbildungen")
N = 2048
J = 5
zentren = [1.3, 3.7, 6.2]

t = 8 * np.arange(N) / N
u = np.sin(2 * np.pi * t)
s = sum(3 * np.maximum(0, 1 - np.abs(t - c) / 0.03) for c in zentren)
f = u + s
c = pywt.wavedec(f, "db2", mode="periodization", level=J)
stufe = [J] + list(range(J, 0, -1))
zeit = [(np.arange(len(d)) + 0.5) * 2**j / 256 for d, j in zip(c, stufe)]

tabelle = np.full((N // 2, 12), np.nan)
for i, (z, d) in enumerate(zip(zeit, c)):
    tabelle[:len(d), 2 * i], tabelle[:len(d), 2 * i + 1] = z, d
kopf = "ta5 a5 td5 d5 td4 d4 td3 d3 td2 d2 td1 d1"
np.savetxt(os.path.join(ordner, "kap5_koeffizienten.dat"), tabelle, header=kopf, comments="", fmt="%.6f")