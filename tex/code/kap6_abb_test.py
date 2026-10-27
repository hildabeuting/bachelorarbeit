# Abbildung 6.4: Ausschnitt von 4 s aus dem Test-EKG f und dem Ergebnis f_hat (sigma = 0,025 mV)
import numpy as np
import matplotlib.pyplot as plt
from kap6_verfahren import fs, entrauschen
from kap6_test import testsignal, gestoert, SEED

plt.rcParams.update({"font.size": 9, "font.family": "serif"})
f = testsignal()
fh = entrauschen(gestoert(f, SEED))
t = np.arange(len(f)) / fs
m = (t >= 60) & (t < 64)
fig, ax = plt.subplots(figsize=(6.3, 2.2))
ax.plot(t[m], f[m], color="black", lw=1.4, alpha=0.35, label="$f$")
ax.plot(t[m], fh[m], color="royalblue", lw=0.7, label=r"$\hat f$")
ax.set_ylabel("in mV")
ax.set_xlabel("Zeit $t$ in s")
ax.legend(loc="upper right", frameon=False, ncol=2)
ax.set_xlim(60, 64)
ax.set_ylim(-0.6, 1.6)
fig.tight_layout()
fig.savefig("../abbildungen/abb6_4_test.pdf")
plt.show()
