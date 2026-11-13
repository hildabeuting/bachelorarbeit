# Abbildung 6.5: ein Herzschlag aus Abbildung 6.4 vergrößert, Test-EKG f und Ergebnis f_hat
import numpy as np
import matplotlib.pyplot as plt
from kap6_verfahren import fs, entrauschen
from kap6_test import testsignal, gestoert, SEED

plt.rcParams.update({"font.size": 9, "font.family": "serif"})
f = testsignal()
fh = entrauschen(gestoert(f, SEED))
t = np.arange(len(f)) / fs
m = (t >= 60.45) & (t < 61.35)
fig, ax = plt.subplots(figsize=(6.3, 2.6))
ax.plot(t[m], f[m], color="black", lw=2.0, alpha=0.35, label="$f$")
ax.plot(t[m], fh[m], color="royalblue", lw=0.9, label=r"$\hat f$")
ax.set_ylabel("in mV")
ax.set_xlabel("Zeit $t$ in s")
ax.legend(loc="upper right", frameon=False, ncol=2)
ax.set_xlim(60.45, 61.35)
fig.tight_layout()
fig.savefig("../abbildungen/abb6_5_test_schlag.pdf")
plt.show()
