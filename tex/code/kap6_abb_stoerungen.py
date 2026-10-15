# Abbildung 6.2: schematisches EKG mit Basisliniendrift, Elektrodenbewegung, Netzbrummen, Muskelrauschen und weißem Rauschen
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 9, "font.family": "serif"})
fs = 360
t = np.arange(0, 10, 1 / (10 * fs))
wellen = [(0.15, -170, 22), (-0.12, -25, 8), (1.2, 0, 9), (-0.25, 22, 9), (0.3, 300, 45)]
rng = np.random.default_rng(3)


def ekg(t):
    x = np.zeros_like(t)
    for r in np.arange(0.4, t[-1] + 1, 0.8):
        x += sum(a * np.exp(-((1000 * (t - r) - m) / s) ** 2 / 2) for a, m, s in wellen)
    return x


def band(von, bis):
    spektrum = np.fft.rfft(rng.standard_normal(len(t)))
    f = np.fft.rfftfreq(len(t), 1 / (10 * fs))
    spektrum[(f < von) | (f > bis)] = 0
    s = np.fft.irfft(spektrum, len(t))
    return s / np.max(np.abs(s))


stoerungen = {
    "(a) Basisliniendrift": 0.2 + 0.5 * np.sin(2 * np.pi * 0.25 * t),
    "(b) Elektrodenbewegung": 1.2 * band(1, 10) * np.exp(-((t - 5.5) / 1.2) ** 2),
    "(c) Netzbrummen": 0.1 * np.sin(2 * np.pi * 60 * t),
    "(d) Muskelrauschen": 0.4 * band(20, 80) * (0.15 + np.exp(-((t - 6) / 0.8) ** 2)),
    "(e) weißes Rauschen": 0.05 * rng.standard_normal(len(t))[::10].repeat(10),
}
fig, ax = plt.subplots(len(stoerungen), 1, figsize=(6.3, 7.0), sharex=True)
for achse, (name, s) in zip(ax, stoerungen.items()):
    achse.plot(t, ekg(t) + s, color="royalblue", lw=0.6)
    achse.axhline(0, color="gray", lw=0.5, ls="--")
    achse.set_title(name, loc="left", fontsize=9)
    achse.set_ylabel("mV")
ax[-1].set_xlim(0, 10)
ax[-1].set_xlabel("Zeit $t$ in s")
fig.tight_layout()
fig.savefig("../abbildungen/abb6_1b_stoerungen.pdf")
plt.show()
