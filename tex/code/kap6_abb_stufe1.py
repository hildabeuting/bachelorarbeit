# Abbildung 6.2: synthetisches EKG, Messung bei 5 dB und Schätzung (db2, hart, J = 4)
import numpy as np
import matplotlib.pyplot as plt
from kap6_modell import ekg_modell, entrauschen, rauschen_fuer_snr

plt.rcParams.update({"font.size": 9, "font.family": "serif"})
t, x = ekg_modell()
rng = np.random.default_rng(93)
e, sigma = rauschen_fuer_snr(x, 5, rng)
y = x + e
f = entrauschen(y, "db2", 4, "hard")
fig, ax = plt.subplots(2, 1, figsize=(6.3, 3.6), sharex=True)
ax[0].plot(t, y, color="gray", lw=0.6)
ax[0].plot(t, x, color="royalblue", lw=1.0)
ax[0].set_ylabel("Messung $Y$")
ax[1].plot(t, x, color="gray", lw=0.8, ls="--")
ax[1].plot(t, f, color="royalblue", lw=1.0)
ax[1].set_ylabel(r"Schätzung $\hat f$")
ax[1].set_xlabel("Zeit $t$ in s")
ax[1].set_xlim(0, t[-1])
fig.tight_layout()
fig.savefig("../abbildungen/abb6_2_stufe1.pdf")
plt.show()
