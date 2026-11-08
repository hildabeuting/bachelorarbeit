# Daubechies-Skalierungsfunktion phi_2 und Wavelet psi_2 aus Kapitel 4.
import pywt
import matplotlib.pyplot as plt

phi, psi, t = pywt.Wavelet("db2").wavefun(level=10)

plt.rcParams.update({"font.size": 11, "axes.titlesize": 11, "mathtext.fontset": "cm"})
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6))

ax[0].plot(t, phi, color="royalblue", lw=0.9)
ax[0].set_title(r"(a) $\varphi_2$")
ax[1].plot(t - 1, psi, color="royalblue", lw=0.9)
ax[1].set_title(r"(b) $\psi_2$")

for a in ax:
    a.axhline(0, color="black", lw=0.5)
    a.set_xlabel("$t$")
    a.spines[["top", "right"]].set_visible(False)

plt.show()
