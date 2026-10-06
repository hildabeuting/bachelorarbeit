# Beispiel aus Kapitel 5: Sinus mit drei Spitzen, Koeffizienten pro Stufe und Risiko der idealen Auswahl (Abb. 5.1).
import numpy as np
import pywt
import matplotlib.pyplot as plt

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

namen = ["a5", "d5", "d4", "d3", "d2", "d1"]
print("Stufe  Anzahl  max|u|   >=sigma(u)  >=sigma(f)")
for name, a, b in zip(namen, cu, cf):
    print(f"{name:5}  {len(a):6}  {np.max(np.abs(a)):.2g}  {np.sum(np.abs(a) >= sigma):10}  {np.sum(np.abs(b) >= sigma):10}")

for name, c in [("u", cu), ("f", cf)]:
    theta = np.concatenate(c)
    M = np.sum(np.abs(theta) >= sigma)
    r = np.sum(np.minimum(theta**2, sigma**2))
    print(f"{name}: M = {M}, M sigma^2 = {M * sigma**2:.1f}, r_id = {r:.1f}")

print(f"N sigma^2 = {N * sigma**2:.1f}")

plt.rcParams.update({"font.size": 11, "mathtext.fontset": "cm"})
fig, ax = plt.subplots(2, 1, figsize=(7.0, 3.6), sharex=True)

ax[0].plot(t, f, color="royalblue", lw=0.9)
ax[0].set_ylabel("$f$")
ax[1].plot(t, Y, color="gray", lw=0.5)
ax[1].set_ylabel("$Y$")
ax[1].set_xlabel("$t$")
ax[1].set_xlim(0, 8)

for a in ax:
    a.set_ylim(-2.2, 4.0)
    a.spines[["top", "right"]].set_visible(False)

plt.show()