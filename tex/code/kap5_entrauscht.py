# Beispiel 5.14: Entrauschtes Testsignal, Abbildung 5.8
import os
import numpy as np
import pywt

ordner = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "abbildungen")
N = 2048
J = 5
sigma = 0.3
zentren = [1.3, 3.7, 6.2]

t = 8 * np.arange(N) / N
u = np.sin(2 * np.pi * t)
s = sum(3 * np.maximum(0, 1 - np.abs(t - c) / 0.03) for c in zentren)
f = u + s
Z = np.random.default_rng(93).standard_normal(N)
Y = f + sigma * Z
c = pywt.wavedec(Y, "db2", mode="periodization", level=J)
sigma_dach = np.median(np.abs(c[-1])) / 0.6745
T = sigma_dach * np.sqrt(2 * np.log(N))
c_hart = [pywt.threshold(d, T, "hard") for d in c]
c_weich = [pywt.threshold(d, T, "soft") for d in c]
hart = pywt.waverec(c_hart, "db2", mode="periodization")
weich = pywt.waverec(c_weich, "db2", mode="periodization")

print(f"||Y - f||^2 = {np.sum((Y - f) ** 2):.1f}")
print(f"hart:  ||f_hart - f||^2 = {np.sum((hart - f) ** 2):.1f}")
print(f"weich: ||f_weich - f||^2 = {np.sum((weich - f) ** 2):.1f}")

tabelle = np.column_stack([t, f, Y, hart, weich])
np.savetxt(os.path.join(ordner, "kap5_entrauscht.dat"), tabelle, header="t f Y hart weich", comments="", fmt="%.6f")