# Testsignal aus Kapitel 5: Sinus mit drei Spitzen und verrauschte Messwerte (Abb. 5.1).
import numpy as np
import matplotlib.pyplot as plt

N = 2048
sigma = 0.3
zentren = [1.3, 3.7, 6.2]

t = 8 * np.arange(N) / N      
u = np.sin(2 * np.pi * t)

s = sum(3 * np.maximum(0, 1 - np.abs(t - c) / 0.03) for c in zentren)
f = u + s

Z = np.random.default_rng(93).standard_normal(N)
Y = f + sigma * Z

plt.rcParams.update({"font.size": 9, "font.family": "serif"})
fig, ax = plt.subplots(2, 1, figsize=(6.0, 3.2), sharex=True)

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
