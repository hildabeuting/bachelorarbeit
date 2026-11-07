# Abbildung 6.3: schematische Elektrodenbewegung
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 9, "font.family": "serif"})
fs = 360
t = np.arange(0, 10, 1 / fs)
rng = np.random.default_rng(3)
spektrum = np.fft.rfft(rng.standard_normal(len(t)))
f = np.fft.rfftfreq(len(t), 1 / fs)
spektrum[(f < 1) | (f > 10)] = 0
s = np.fft.irfft(spektrum, len(t))
s = s / np.max(np.abs(s)) * np.exp(-((t - 5.5) / 1.5) ** 2)
fig, ax = plt.subplots(figsize=(6.3, 1.4))
ax.plot(t, s, color="royalblue", lw=1.0)
ax.axhline(0, color="gray", lw=0.5, ls="--")
ax.set_xlim(0, 10)
ax.set_yticks([])
ax.set_xlabel("Zeit $t$ in s")
fig.tight_layout()
fig.savefig("../abbildungen/abb6_1c_elektrode.pdf")
plt.show()
