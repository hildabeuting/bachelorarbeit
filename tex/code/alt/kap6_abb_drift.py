# Abbildung 6.2: schematische Basisliniendrift
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 9, "font.family": "serif"})
fs = 360
t = np.arange(0, 10, 1 / fs)
b = np.sin(2 * np.pi * 0.15 * t) + 0.5 * np.sin(2 * np.pi * 0.27 * t + 1.0)
fig, ax = plt.subplots(figsize=(6.3, 1.4))
ax.plot(t, b, color="royalblue", lw=1.0)
ax.axhline(0, color="gray", lw=0.5, ls="--")
ax.set_xlim(0, 10)
ax.set_yticks([])
ax.set_xlabel("Zeit $t$ in s")
fig.tight_layout()
fig.savefig("../abbildungen/abb6_1b_drift.pdf")
plt.show()
