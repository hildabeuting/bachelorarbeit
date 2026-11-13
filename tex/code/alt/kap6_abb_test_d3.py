# Abbildung 6.10: Herzschlag aus Abbildung 6.5 mit f, f_hat und f_hat_3 (Schwelle 0,2 T auf d_3) und die Fehler beider Schätzungen
import numpy as np
import matplotlib.pyplot as plt
from kap6_verfahren import fs, entrauschen
from kap6_test import testsignal, gestoert, SEED

plt.rcParams.update({"font.size": 9, "font.family": "serif"})
f = testsignal()
y = gestoert(f, SEED)
fh = entrauschen(y)
fh3 = entrauschen(y, JH=1, faktor=0.2)
t = np.arange(len(f)) / fs
m = (t >= 60.45) & (t < 61.35)
fig, ax = plt.subplots(2, 1, figsize=(6.3, 4.6), sharex=True)
ax[0].plot(t[m], f[m], color="black", lw=2.0, alpha=0.35, label="$f$")
ax[0].plot(t[m], fh[m], color="royalblue", lw=0.9, label=r"$\hat f$")
ax[0].plot(t[m], fh3[m], color="crimson", lw=0.9, ls="--", label=r"$\hat f_3$")
ax[0].set_ylabel("in mV")
ax[0].legend(loc="upper right", frameon=False, ncol=3)
ax[1].axhline(0, color="black", lw=0.5)
ax[1].plot(t[m], (fh - f)[m], color="royalblue", lw=0.9, label=r"$\hat f - f$")
ax[1].plot(t[m], (fh3 - f)[m], color="crimson", lw=0.9, ls="--", label=r"$\hat f_3 - f$")
ax[1].set_ylabel("Fehler in mV")
ax[1].legend(loc="upper right", frameon=False, ncol=2)
ax[1].set_xlabel("Zeit $t$ in s")
ax[1].set_xlim(60.45, 61.35)
fig.tight_layout()
fig.savefig("../abbildungen/abb6_10_test_d3.pdf")
plt.show()
