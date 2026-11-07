# Beispiel aus Abschnitt 3.1.2: Signal, Fourier-Transformierte und Haar-CWT.
import numpy as np
import matplotlib.pyplot as plt

pi = np.pi


def f(t):
    y = np.zeros_like(t)
    m = (t >= 0) & (t <= 3)
    y[m] += 1 - np.cos(2 * pi * t[m])
    m = (t >= 4) & (t <= 5)
    y[m] += 0.5 * (1 - np.cos(10 * pi * t[m]))
    return y


def F(t):
    y = np.zeros_like(t)
    m = (t > 0) & (t < 3)
    y[m] += t[m] - np.sin(2 * pi * t[m]) / (2 * pi)
    y[t >= 3] += 3
    m = (t > 4) & (t < 5)
    y[m] += 0.5 * ((t[m] - 4) - np.sin(10 * pi * t[m]) / (10 * pi))
    y[t >= 5] += 0.5
    return y


def fourier(w):
    w = w.astype(complex)
    for p in [0, 2 * pi, -2 * pi, 10 * pi, -10 * pi]:
        w = np.where(np.abs(w - p) < 1e-7, p + 1e-7, w)
    a = 4 * pi**2 * (1 - np.exp(-3j * w)) / (w**2 - 4 * pi**2)
    b = 50 * pi**2 * (np.exp(-4j * w) - np.exp(-5j * w)) / (w**2 - 100 * pi**2)
    return 1j * (a + b) / (np.sqrt(2 * pi) * w)


def cwt_haar(a, b):
    return (2 * F(b + a / 2) - F(b) - F(b + a)) / np.sqrt(a)


plt.rcParams.update({"font.size": 9, "font.family": "serif", "savefig.dpi": 300})

t = np.linspace(-0.6, 5.6, 40000)
fig, ax = plt.subplots(figsize=(7.0, 2.8))
ax.plot(t, f(t), color="royalblue", lw=0.9)
for x in [0, 3, 4, 5]:
    ax.axvline(x, color="gray", lw=0.6, alpha=0.3)
ax.set_xlim(-0.6, 5.6)
ax.set_ylim(-0.25, 2.45)
ax.set_xticks(range(6))
ax.set_xlabel("$t$")
ax.set_ylabel("$f(t)$")
ax.annotate("$f_1$", (1.5, 2.18), ha="center")
ax.annotate("$f_2$", (4.5, 1.22), ha="center")
ax.spines[["top", "right"]].set_visible(False)

w = np.linspace(-50, 50, 40000)
fig, ax = plt.subplots(figsize=(7.0, 2.8))
ax.plot(w, np.abs(fourier(w)), color="royalblue", lw=0.9)
for p in [-10 * pi, -2 * pi, 0, 2 * pi, 10 * pi]:
    ax.axvline(p, color="gray", ls="--", lw=0.6)
ax.set_xlim(-50, 50)
ax.set_ylim(0, 1.55)
ax.set_xticks([-10 * pi, -2 * pi, 0, 2 * pi, 10 * pi])
ax.set_xticklabels([r"$-10\pi$", r"$-2\pi$", "$0$", r"$2\pi$", r"$10\pi$"])
ax.set_xlabel(r"$\omega$")
ax.set_ylabel(r"$|\hat f(\omega)|$")
ax.spines[["top", "right"]].set_visible(False)

a = np.logspace(np.log10(0.02), np.log10(3.0), 900)[:, None]
b = np.linspace(-0.6, 5.6, 1400)[None, :]
fig, ax = plt.subplots(figsize=(9.0, 4.6))
pc = ax.pcolormesh(b.ravel(), a.ravel(), np.abs(cwt_haar(a, b)), shading="gouraud", cmap="magma")
ax.set_yscale("log")
ax.set_xticks(range(6))
ax.set_xlabel("$b$")
ax.set_ylabel("$a$")
fig.colorbar(pc, ax=ax).set_label(r"$|W_\psi f(a,b)|$")

plt.show()