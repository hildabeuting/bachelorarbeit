r"""
Beispiel zur kontinuierlichen Wavelet-Transformation (Abschnitt 3.1.2)

    f_1(t) = 1 - cos(2 pi t)         auf [0,3]    (Frequenz 1, 3 Perioden)
    f_2(t) = 1/2 (1 - cos(10 pi t))  auf [4,5]    (Frequenz 5, 5 Perioden)
    f      = f_1 + f_2

Fourier-Konvention:  hat f(w) = (2 pi)^{-1/2} int f(t) e^{-i w t} dt
"""

import numpy as np
import matplotlib.pyplot as plt

pi = np.pi
plt.rcParams.update({"font.size": 11, "mathtext.fontset": "cm"})


# ================================================================ Signal
def f(t):
    t = np.asarray(t, float)
    y = np.zeros_like(t)

    m = (t >= 0) & (t <= 3)
    y[m] += 1 - np.cos(2 * pi * t[m])

    m = (t >= 4) & (t <= 5)
    y[m] += 0.5 * (1 - np.cos(10 * pi * t[m]))

    return y


# ======================================================== Stammfunktion
def F(t):
    """F(t) = Integral von -unendlich bis t ueber f.

    Auf [0,3]:  Integral von 0 bis t ueber (1 - cos(2 pi s)) ds
                = t - sin(2 pi t)/(2 pi).
    Ab t = 3:   konstant gleich 3, denn sin(6 pi) = 0.
    Auf [4,5]:  0.5 * ( (t-4) - sin(10 pi t)/(10 pi) ),  da sin(40 pi) = 0.
    Ab t = 5:   zusaetzlich konstant 0.5, denn sin(50 pi) = 0.
    """
    t = np.asarray(t, float)
    out = np.zeros_like(t)

    m = (t > 0) & (t < 3)
    out[m] += t[m] - np.sin(2 * pi * t[m]) / (2 * pi)
    out[t >= 3] += 3.0

    m = (t > 4) & (t < 5)
    out[m] += 0.5 * ((t[m] - 4) - np.sin(10 * pi * t[m]) / (10 * pi))
    out[t >= 5] += 0.5

    return out


# ================================================ Fourier-Transformierte
def fourier(w):
    """Geschlossene Formel:

        hat f(w) =  i / (sqrt(2 pi) w)
                    * [ 4 pi^2 (1 - e^{-3iw}) / (w^2 - 4 pi^2)
                      + 50 pi^2 (e^{-4iw} - e^{-5iw}) / (w^2 - 100 pi^2) ]

    Die Zaehler merken sich nur die Traegerraender 0, 3 bzw. 4, 5.
    Bei w = 0, +-2 pi, +-10 pi verschwinden Zaehler und Nenner gemeinsam;
    diese Stellen sind hebbar und werden numerisch minimal umgangen.
    """
    w = np.asarray(w, complex)
    for pol in (0.0, 2 * pi, -2 * pi, 10 * pi, -10 * pi):
        w = np.where(np.abs(w - pol) < 1e-7, pol + 1e-7, w)

    langsam = 4 * pi**2 * (1 - np.exp(-3j * w)) / (w**2 - 4 * pi**2)
    schnell = 50 * pi**2 * (np.exp(-4j * w) - np.exp(-5j * w)) / (w**2 - 100 * pi**2)

    return 1j * (langsam + schnell) / (np.sqrt(2 * pi) * w)


# ============================================ Wavelet-Trafo, Haar-Wavelet
def cwt_haar(a, b):
    """psi_{a,b} ist +1/sqrt(a) auf [b, b+a/2) und -1/sqrt(a) auf [b+a/2, b+a),
    daher laesst sich das Integral exakt durch F ausdruecken."""
    return (2 * F(b + a / 2) - F(b) - F(b + a)) / np.sqrt(a)


# ---------------------------------------------------------------- Abb. 1
FIGSIZE = (7.0, 2.8)          # gemeinsame Groesse fuer Abb. 1 und Abb. 2
t = np.linspace(-0.6, 5.6, 40000)

fig1, ax = plt.subplots(figsize=FIGSIZE)
ax.plot(t, f(t), "k", lw=1.1)
ax.axhline(0, color="0.7", lw=0.6)
for x0 in (0, 3, 4, 5):
    ax.axvline(x0, color="0.9", lw=0.7, zorder=0)
ax.set_xlim(-0.6, 5.6)
ax.set_ylim(-0.25, 2.45)
ax.set_xticks(range(6))
ax.set_xlabel(r"$t$")
ax.set_ylabel(r"$f(t)$")
ax.annotate(r"$f_1$", (1.5, 2.18), ha="center")
ax.annotate(r"$f_2$", (4.5, 1.22), ha="center")
fig1.tight_layout()

# ---------------------------------------------------------------- Abb. 2
w = np.linspace(-50, 50, 40000)

fig2, ax = plt.subplots(figsize=FIGSIZE)
ax.plot(w, np.abs(fourier(w)), "k", lw=1.0)
ax.axhline(0, color="0.7", lw=0.6)
for x0 in (-10 * pi, -2 * pi, 0, 2 * pi, 10 * pi):
    ax.axvline(x0, color="0.75", ls="--", lw=0.7, zorder=0)
ax.set_xlim(-50, 50)
ax.set_ylim(0, 1.55)
ax.set_xticks([-10 * pi, -2 * pi, 0, 2 * pi, 10 * pi])
ax.set_xticklabels([r"$-10\pi$", r"$-2\pi$", r"$0$", r"$2\pi$", r"$10\pi$"])
ax.set_xlabel(r"$\omega$")
ax.set_ylabel(r"$|\hat f(\omega)|$")
fig2.tight_layout()

# ---------------------------------------------------------------- Abb. 3
a = np.logspace(np.log10(0.02), np.log10(3.0), 900)[:, None]
b = np.linspace(-0.6, 5.6, 1400)[None, :]

fig3, ax = plt.subplots(figsize=(9.0, 4.6))
pc = ax.pcolormesh(b.ravel(), a.ravel(), np.abs(cwt_haar(a, b)),
                   shading="gouraud", cmap="magma")
ax.set_yscale("log")
ax.set_xticks(range(6))
ax.set_xlabel(r"$b$")
ax.set_ylabel(r"$a$")
fig3.colorbar(pc, ax=ax, pad=0.015).set_label(r"$|W_\psi f(a,b)|$")
fig3.tight_layout()

plt.show()