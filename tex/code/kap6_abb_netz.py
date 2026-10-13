# Abbildung 6.4: schematisches Netzbrummen bei 60 Hz
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 9, "font.family": "serif"})
fs = 360
t = np.arange(0, 0.2, 1 / (20 * fs))
fig, ax = plt.subplots(figsize=(6.3, 1.4))
ax.plot(1000 * t, np.sin(2 * np.pi * 60 * t), color="royalblue", lw=1.0)
ax.axhline(0, color="gray", lw=0.5, ls="--")
ax.set_xlim(0, 200)
ax.set_yticks([])
ax.set_xlabel("Zeit $t$ in ms")
fig.tight_layout()
fig.savefig("../abbildungen/abb6_1d_netz.pdf")
plt.show()
