# Abbildung 6.4: Datensatz 103 mit Muskelartefakten (NSTDB ma) bei 5 dB; Schätzung ohne und mit a_8 = 0
import numpy as np
import matplotlib.pyplot as plt
import wfdb
from kap6_modell import entrauschen

plt.rcParams.update({"font.size": 9, "font.family": "serif"})
N = 2048
start = 10 * 360
x = wfdb.rdrecord("../daten/mitdb/103").p_signal[start:start + N, 0]
x = x - np.mean(x)
r = wfdb.rdrecord("../daten/nstdb/ma").p_signal[start:start + N, 0]
r = r - np.mean(r)
r = r * np.sqrt(np.sum(x ** 2) / np.sum(r ** 2) / 10 ** 0.5)
y = x + r
t = np.arange(N) / 360
f1 = entrauschen(y, "db2", 4, "hard")
f2 = entrauschen(y, "db2", 4, "hard", a_null=8)
fig, ax = plt.subplots(3, 1, figsize=(6.3, 3.9), sharex=True, sharey=True)
ax[0].plot(t, y, color="gray", lw=0.6)
ax[0].plot(t, x, color="royalblue", lw=1.0)
ax[0].set_ylabel("Messung $Y$")
ax[1].plot(t, x, color="gray", lw=0.8, ls="--")
ax[1].plot(t, f1, color="royalblue", lw=1.0)
ax[1].set_ylabel(r"$\hat f$, $a_4$ erhalten")
ax[2].plot(t, x, color="gray", lw=0.8, ls="--")
ax[2].plot(t, f2, color="royalblue", lw=1.0)
ax[2].set_ylabel(r"$\hat f$, $a_8 = 0$")
ax[2].set_xlabel("Zeit $t$ in s (ab Sekunde 10 der Aufzeichnung)")
ax[2].set_xlim(0, t[-1])
fig.tight_layout()
fig.savefig("../abbildungen/abb6_4_stufe3.pdf")
plt.show()
