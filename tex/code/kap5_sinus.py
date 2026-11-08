# Beispiel 5.5: Verrauschter Sinus, Abbildung 5.2
import os
import numpy as np
import pywt

ordner = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "abbildungen")
N = 2048
J = 5
sigma = 0.3

t = 8 * np.arange(N) / N
u = np.sin(2 * np.pi * t)
Z = np.random.default_rng(93).standard_normal(N)
Y = u + sigma * Z
koeff = pywt.wavedec(Z, "db2", mode="periodization", level=J)
Z_tilde = np.concatenate(koeff)

print(f"||Y - u||^2 = {np.sum((Y - u) ** 2):.1f}")
print(f"N sigma^2 = {N * sigma**2:.1f}")
print(f"Mittelwert Z_tilde = {np.mean(Z_tilde):.2f}")
print(f"Varianz Z_tilde = {np.var(Z_tilde):.2f}")
print(f"||AZ|| - ||Z|| = {np.linalg.norm(Z_tilde) - np.linalg.norm(Z):.1e}")

tabelle = np.column_stack([t, u, Y])
np.savetxt(os.path.join(ordner, "kap5_sinus.dat"), tabelle, header="t u Y", comments="", fmt="%.6f")