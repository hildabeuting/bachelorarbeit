# Beispiel 5.11: Sinus mit Spitzen, Abbildung 5.5
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
cu = pywt.wavedec(u, "db2", mode="periodization", level=J)
cf = pywt.wavedec(f, "db2", mode="periodization", level=J)

print("Stufe, Anzahl, max|u|, >= sigma (u), >= sigma (f)")
for name, a, b in zip(["a5", "d5", "d4", "d3", "d2", "d1"], cu, cf):
    print(name, len(a), f"{np.max(np.abs(a)):.2g}",
          np.sum(np.abs(a) >= sigma), np.sum(np.abs(b) >= sigma))
for name, c in [("u", cu), ("f", cf)]:
    theta = np.concatenate(c)
    M = np.sum(np.abs(theta) >= sigma)
    geloescht = np.sum(theta[np.abs(theta) < sigma] ** 2)
    r_id = np.sum(np.minimum(theta**2, sigma**2))
    print(f"{name}: M = {M}, geloescht = {geloescht:.1f}, r_id = {r_id:.1f}")

tabelle = np.column_stack([t, f, Y])
np.savetxt(os.path.join(ordner, "kap5_signal.dat"), tabelle, header="t f Y", comments="", fmt="%.6f")