# Beispiel 3.7: Fourier-Transformierte und Haar-CWT des Testsignals (Abbildungen 3.5 und 3.6)
import os
import numpy as np
import matplotlib.pyplot as plt

ordner = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "abbildungen")
pi = np.pi
omega = np.linspace(-50, 50, 2001)
a = np.logspace(np.log10(0.02), np.log10(3), 600)[:, None]
b = np.linspace(-0.6, 5.6, 1000)[None, :]


def F(t):
    return np.select([t < 0, t <= 3, t <= 4, t <= 5],
                     [0, t - np.sin(2 * pi * t) / (2 * pi), 3,
                      3 + (t - 4) / 2 - np.sin(10 * pi * t) / (20 * pi)], 3.5)


def f_dach(w):
    w = np.where(np.isin(w, [0, 2 * pi, -2 * pi, 10 * pi, -10 * pi]), w + 1e-9, w)
    langsam = 4 * pi**2 * (1 - np.exp(-3j * w)) / (w**2 - 4 * pi**2)
    schnell = 50 * pi**2 * (np.exp(-4j * w) - np.exp(-5j * w)) / (w**2 - 100 * pi**2)
    return 1j * (langsam + schnell) / (np.sqrt(2 * pi) * w)


W = np.abs(2 * F(b + a / 2) - F(b) - F(b + a)) / np.sqrt(a)

np.savetxt(os.path.join(ordner, "kap3_fourier.dat"), np.column_stack([omega, np.abs(f_dach(omega))]),
           header="omega betrag", comments="", fmt="%.6f")
plt.imsave(os.path.join(ordner, "kap3_cwt.png"), W, cmap="viridis", origin="lower", vmin=0)
print(f"max |W f| = {W.max():.4f}")
