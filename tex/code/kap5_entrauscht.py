# Beispiel aus Kapitel 5: Entrauschen des Testsignals mit Hard- und Soft-Thresholding (Abb. 5.8).
import numpy as np
import pywt
import matplotlib.pyplot as plt

N = 2048
J = 5
sigma = 0.3
zentren = [1.3, 3.7, 6.2]

t = 8 * np.arange(N) / N
u = np.sin(2 * np.pi * t)

s = sum(3 * np.maximum(0, 1 - np.abs(t - c) / 0.03) for c in zentren)
f = u + s

Z = np.random.default_rng(93).standard_normal(N)
Y = f + sigma * Z

c = pywt.wavedec(Y, "db2", mode="periodization", level=J)
sigma_dach = np.median(np.abs(c[-1])) / 0.6745
T = sigma_dach * np.sqrt(2 * np.log(N))

hart = pywt.waverec([pywt.threshold(d, T, "hard") for d in c], "db2", mode="periodization")
weich = pywt.waverec([pywt.threshold(d, T, "soft") for d in c], "db2", mode="periodization")

print(f"||Y - f||^2 = {np.sum((Y - f) ** 2):.1f}")
print(f"hart:  ||f_hart - f||^2 = {np.sum((hart - f) ** 2):.1f}")
print(f"weich: ||f_weich - f||^2 = {np.sum((weich - f) ** 2):.1f}")

plt.rcParams.update({"font.size": 11, "mathtext.fontset": "cm"})
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True, gridspec_kw={"width_ratios": [2.2, 1]})

ax[0].plot(t, hart, color="royalblue", lw=0.9, label="hart")
ax[0].plot(t, weich, color="red", lw=0.7, label="weich")
ax[0].set_xlim(0, 8)
ax[0].set_ylim(-2.2, 4.0)
ax[0].set_xlabel("$t$")

aus = (t > 3.45) & (t < 3.95)
ax[1].plot(t[aus], f[aus], color="black", ls=":", lw=0.9, label="$f$")
ax[1].plot(t[aus], hart[aus], color="royalblue", lw=0.9, label="hart")
ax[1].plot(t[aus], weich[aus], color="red", lw=0.7, label="weich")
ax[1].set_xlabel("$t$ (Ausschnitt)")
ax[1].legend(frameon=False, loc="upper right")

for a in ax:
    a.spines[["top", "right"]].set_visible(False)

fig.tight_layout()

plt.show()
