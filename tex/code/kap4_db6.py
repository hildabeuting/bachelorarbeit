# Abbildung 4.x: Skalierungsfunktion und Wavelet des Daubechies-Wavelets mit p = 6
import matplotlib.pyplot as plt
import pywt

plt.rcParams.update({"font.size": 9, "font.family": "serif"})
p = 6
phi, psi, x = pywt.Wavelet(f"db{p}").wavefun(level=10)
fig, ax = plt.subplots(1, 2, figsize=(6.3, 2.2), sharey=True)
ax[0].plot(x, phi, color="royalblue", lw=1.0)
ax[1].plot(x + 1 - p, psi, color="royalblue", lw=1.0)
ax[0].set_title(r"(a) $\varphi_6$")
ax[1].set_title(r"(b) $\psi_6$")
ax[0].set_xlim(0, 2 * p - 1)
ax[1].set_xlim(1 - p, p)
for achse in ax:
    achse.axhline(0, color="gray", lw=0.5, ls="--")
    achse.set_xlabel("$t$")
fig.tight_layout()
fig.savefig("../abbildungen/abb4_db6.pdf")
plt.show()
