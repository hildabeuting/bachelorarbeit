# Abbildung 6.3: Ausschnitt von 4 s aus dem Test-EKG f und der gestörten Messung Y (sigma = 0,05 mV)
import numpy as np
import matplotlib.pyplot as plt
from kap6_verfahren import fs
from kap6_test import testsignal, gestoert, DRIFT

plt.rcParams.update({"font.size": 9, "font.family": "serif"})
f = testsignal()
y = gestoert(f, 0)
t = np.arange(len(f)) / fs
b = DRIFT[0] * np.sin(2 * np.pi * DRIFT[1] * t)
m = (t >= 60) & (t < 64)
fig, ax = plt.subplots(2, 1, figsize=(6.3, 3.6), sharex=True)
ax[0].plot(t[m], f[m], color="black", lw=0.8)
ax[0].set_ylabel("$f$ in mV")
ax[1].plot(t[m], y[m], color="gray", lw=0.6, label="$Y$")
ax[1].plot(t[m], b[m], color="crimson", lw=1.0, ls="--", label="Drift $b$")
ax[1].legend(loc="upper right", frameon=False, ncol=2)
ax[1].set_ylabel("$Y$ in mV")
ax[1].set_xlabel("Zeit $t$ in s")
ax[1].set_xlim(60, 64)
fig.tight_layout()
fig.savefig("../abbildungen/abb6_3_testsignal.pdf")
plt.show()
