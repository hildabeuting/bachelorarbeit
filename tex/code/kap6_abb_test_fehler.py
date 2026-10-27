# Abbildung 6.6: Fehler f_hat - f für denselben Herzschlag wie in Abbildung 6.5 (Realisierung Z^(1), Seed 0)
import numpy as np
import matplotlib.pyplot as plt
from kap6_verfahren import fs, entrauschen
from kap6_test import testsignal, gestoert, SEED

plt.rcParams.update({"font.size": 9, "font.family": "serif"})
f = testsignal()
fh = entrauschen(gestoert(f, SEED))
t = np.arange(len(f)) / fs
m = (t >= 60.45) & (t < 61.35)
fig, ax = plt.subplots(figsize=(6.3, 2.2))
ax.axhline(0, color="black", lw=0.5)
ax.plot(t[m], (fh - f)[m], color="crimson", lw=0.9)
ax.set_ylabel(r"$\hat f - f$ in mV")
ax.set_xlabel("Zeit $t$ in s")
ax.set_xlim(60.45, 61.35)
fig.tight_layout()
fig.savefig("../abbildungen/abb6_6_test_fehler.pdf")
plt.show()
