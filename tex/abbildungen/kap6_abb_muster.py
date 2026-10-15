# Abbildung 6.1: schematischer Herzschlag mit P-, Q-, R-, S-, T- und U-Welle
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 9, "font.family": "serif"})
t = np.linspace(0, 850, 2000)
wellen = {"P": (0.15, 130, 22), "Q": (-0.12, 275, 8), "R": (1.2, 300, 9),
          "S": (-0.25, 322, 9), "T": (0.3, 600, 45), "U": (0.05, 760, 25)}
schlag = sum(a * np.exp(-((t - m) / s) ** 2 / 2) for a, m, s in wellen.values())
fig, ax = plt.subplots(figsize=(6.3, 2.6))
ax.axhline(0, color="gray", lw=0.6, ls="--")
ax.plot(t, schlag, color="royalblue", lw=1.2)
for name, (a, m, s) in wellen.items():
    i = np.argmin(np.abs(t - m))
    ax.annotate(name, (t[i], schlag[i]), textcoords="offset points",
                xytext=(0, 6 if schlag[i] > 0 else -12), ha="center")
ax.set_xlabel("Zeit $t$ in ms")
ax.set_ylabel("Spannung in mV")
ax.set_xlim(t[0], t[-1])
ax.set_ylim(-0.4, 1.45)
fig.tight_layout()
fig.savefig("../abbildungen/abb6_1_muster.pdf")
plt.show()
