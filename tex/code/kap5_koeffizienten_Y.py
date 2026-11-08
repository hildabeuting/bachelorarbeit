# Beispiel 5.12: Koeffizienten der Messwerte, Abbildung 5.7
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
stufe = [J] + list(range(J, 0, -1))
zeit = [(np.arange(len(d)) + 0.5) * 2**j / 256 for d, j in zip(c, stufe)]

sigma_dach = np.median(np.abs(c[-1])) / 0.6745
T = sigma_dach * np.sqrt(2 * np.log(N))
koeff = pywt.wavedec(Z, "db2", mode="periodization", level=J)
Z_tilde = np.concatenate(koeff)
print(f"sigma_dach = {sigma_dach:.3f}")
print(f"T = {T:.2f}")
print(f"max |sigma Z_tilde| = {sigma * np.max(np.abs(Z_tilde)):.2f}")
print(f"behalten: {np.sum(np.abs(np.concatenate(c)) > T)} von {N}")

tabelle = np.full((N // 2, 12), np.nan)
for i, (z, d) in enumerate(zip(zeit, c)):
    tabelle[:len(d), 2 * i], tabelle[:len(d), 2 * i + 1] = z, d
kopf = "ta5 a5 td5 d5 td4 d4 td3 d3 td2 d2 td1 d1"
np.savetxt(os.path.join(ordner, "kap5_koeffizienten_Y.dat"), tabelle, header=kopf, comments="", fmt="%.6f")