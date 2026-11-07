# Abbildung 6.5: schematisches Muskelrauschen
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 9, "font.family": "serif"})
fs = 360
t = np.arange(0, 10, 1 / fs)
rng = np.random.default_rng(5)
spektrum = np.fft.rfft(rng.standard_normal(len(t)))
f = np.fft.rfftfreq(len(t), 1 / fs)
spektrum[(f < 20) | (f > 80)] = 0
s = np.fft.irfft(spektrum, len(t))
s = s / np.max(np.abs(s)) * (0.15 + np.exp(-((t - 6) / 0.8) ** 2))
fig, ax = plt.subplots(figsize=(6.3, 1.4))
ax.plot(t, s, color="royalblue", lw=0.6)
ax.axhline(0, color="gray", lw=0.5, ls="--")
ax.set_xlim(0, 10)
ax.set_yticks([])
ax.set_xlabel("Zeit $t$ in s")
fig.tight_layout()
fig.savefig("../abbildungen/abb6_1e_muskel.pdf")
plt.show()
