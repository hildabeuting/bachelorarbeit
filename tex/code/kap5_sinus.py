# Beispiel aus Kapitel 5: verrauschter Sinus, Fehlerenergie und Statistik der Rauschkoeffizienten (Abb. 5.2).
import numpy as np
import pywt
import matplotlib.pyplot as plt

N = 2048
J = 5
sigma = 0.3

t = 8 * np.arange(N) / N
u = np.sin(2 * np.pi * t)

Z = np.random.default_rng(93).standard_normal(N)
Y = u + sigma * Z

Z_tilde = np.concatenate(pywt.wavedec(Z, "db2", mode="periodization", level=J))

print(f"||Y - u||^2 = {np.sum((Y - u) ** 2):.1f}")
print(f"N sigma^2 = {N * sigma**2:.1f}")
print(f"Mittelwert Z_tilde = {np.mean(Z_tilde):.2f}")
print(f"Varianz Z_tilde = {np.var(Z_tilde):.2f}")
print(f"||AZ|| - ||Z|| = {np.linalg.norm(Z_tilde) - np.linalg.norm(Z):.1e}")

plt.rcParams.update({"font.size": 11, "mathtext.fontset": "cm"})
fig, ax = plt.subplots(2, 1, figsize=(7.0, 3.6), sharex=True)

ax[0].plot(t, u, color="royalblue", lw=0.9)
ax[0].set_ylabel("$u$")
ax[1].plot(t, Y, color="gray", lw=0.5)
ax[1].set_ylabel("$Y$")
ax[1].set_xlabel("$t$")
ax[1].set_xlim(0, 8)

for a in ax:
    a.set_ylim(-2.2, 2.2)
    a.spines[["top", "right"]].set_visible(False)

plt.show()