"""Laufendes Beispiel aus Kapitel 5: Sinus + drei Spitzen + weißes Rauschen.
Erzeugt abb5_1_signal.pdf, abb5_2_koeffizienten.pdf, abb5_3_entrauscht.pdf
und gibt alle im Text verwendeten Zahlen aus."""
import os, math
import numpy as np, pywt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "abbildungen")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.6, "lines.linewidth": 0.9,
                     "savefig.bbox": "tight", "font.family": "serif"})
BLAU, ROT, GRAU = "#0271bb", "#e2001a", "0.55"

# ---------------- Signal und Rauschen ----------------
N, J, sigma, W = 2048, 5, 0.3, "db2"
t = 8 * np.arange(N) / N                       # 8 Zeiteinheiten, 8 Perioden
g = np.sin(2 * np.pi * t)                       # Sinus
zentren = [1.3, 3.7, 6.2]
s = sum(np.interp(t, [c - 0.03, c, c + 0.03], [0, 3, 0]) for c in zentren)
f = g + s
# Seed 93: typische Realisierung (SNR, sigma_hat und Spitzenfehler nahe am
# Median über 200 Läufe, geprüft mit seed 0..199)
Z = np.random.default_rng(93).standard_normal(N)
Y, Yg = f + sigma * Z, g + sigma * Z
lnN = math.log(N)
spitzen_idx = [int(np.argmax(f * (np.abs(t - c) < 0.05))) for c in zentren]

dwt = lambda x: pywt.wavedec(x, W, mode="periodization", level=J)
idwt = lambda c: pywt.waverec(c, W, mode="periodization")
vec = lambda c: np.concatenate(c)
snr = lambda ref, e: 10 * np.log10(np.sum(ref**2) / np.sum(e**2))

def fourier_reell(x):
    """Koeffizienten in der reellen orthonormalen Fourier-Basis
    {1, sqrt2 cos, sqrt2 sin, (-1)^n}/sqrt(N)."""
    F = np.fft.rfft(x) / math.sqrt(N)
    return np.concatenate([[F[0].real], math.sqrt(2) * F[1:-1].real,
                           -math.sqrt(2) * F[1:-1].imag, [F[-1].real]])

orakel = lambda th: np.sum(np.minimum(th**2, sigma**2))

def visushrink(y, modus):
    c = dwt(y)
    sh = np.median(np.abs(c[-1])) / 0.6745
    T = sh * math.sqrt(2 * lnN)
    return idwt([c[0]] + [pywt.threshold(d, T, modus) for d in c[1:]]), sh, T

# ---------------- Zahlen ----------------
print("== 5.1 Modell und Invarianz")
print(f"Nsigma^2 = {N*sigma**2:.2f}, ||Y-f||^2 = {np.sum((Y-f)**2):.2f}, SNR_in = {snr(f, Y-f):.2f} dB")
Zt = vec(dwt(Z))
print(f"AZ: Mittel {Zt.mean():.4f}, Varianz {Zt.var():.4f}; Orthogonalität: ||AZ||-||Z|| = {np.linalg.norm(Zt)-np.linalg.norm(Z):.2e}")
print(f"Approx.-Koeffizienten: {N//2**J}")

print("== 5.2 Dünnbesetztheit (Orakel-Risiko)")
for name, x in [("Sinus", g), ("Sinus+Spitzen", f), ("nur Spitzen", s)]:
    tw, tf = vec(dwt(x)), fourier_reell(x)
    print(f"{name:14s} Wavelet: r_or={orakel(tw):6.2f}, #>sigma={np.sum(np.abs(tw)>sigma):4d} | "
          f"Fourier: r_or={orakel(tf):6.2f}, #>sigma={np.sum(np.abs(tf)>sigma):4d}")
cs = dwt(s)
for j, d in zip(range(J, 0, -1), cs[1:]):
    print(f"  Spitzen, Skala j={j}: {np.sum(np.abs(d)>1e-10)} von {len(d)} Detailkoeff. != 0")
cg = dwt(g)
print("  Sinus, max|d_j| pro Skala:", [f"j={j}: {np.max(np.abs(d)):.1e}" for j, d in zip(range(J, 0, -1), cg[1:])])

print("== 5.3/5.4/5.5 Entrauschen")
print(f"T/sigma = {math.sqrt(2*lnN):.4f}, max|AZ| = {np.max(np.abs(Zt)):.4f}, max|AZ| nur Details = {np.max(np.abs(vec(dwt(Z)[1:]))):.4f}")
res = {}
for m in ["hard", "soft"]:
    fh, sh, T = visushrink(Y, m)
    res[m] = fh
    r = np.sum((fh - f)**2)
    print(f"{m}: sigma_hat={sh:.4f}, T={T:.4f}, Risiko ||fh-f||^2={r:.2f}, SNR={snr(f, fh-f):.2f} dB, "
          f"Spitzen f vs fh {[(round(float(f[i]),2), round(float(fh[i]),2)) for i in spitzen_idx]}")
r_or = orakel(vec(dwt(f)))
print(f"DJ-Schranke (2lnN+1)(sigma^2+r_or) = {(2*lnN+1)*(sigma**2+r_or):.1f}, Faktor {2*lnN+1:.2f}")
fhg, shg, _ = visushrink(Yg, "hard")
print(f"reiner Sinus: sigma_hat={shg:.4f}, SNR {snr(g, Yg-g):.2f} -> {snr(g, fhg-g):.2f} dB")

# ---------------- Abbildung 5.1: Signal ----------------
fig, ax = plt.subplots(2, 1, figsize=(6.0, 3.2), sharex=True)
ax[0].plot(t, f, color=BLAU); ax[0].set_ylabel("$f$")
ax[1].plot(t, Y, color=GRAU, lw=0.5); ax[1].set_ylabel("$Y$")
ax[1].set_xlabel("$t$"); ax[1].set_xlim(0, 8)
for a in ax: a.set_ylim(-2.2, 4.0); a.spines[["top", "right"]].set_visible(False)
fig.savefig(os.path.join(OUT, "abb5_1_signal.pdf"))

# ---------------- Abbildung 5.2: Koeffizienten pro Skala ----------------
SK = 0.13   # Zeilenhöhe 1 entspricht |d| = 7,7; größere Werte werden bei 0,47 abgeschnitten
def skalen(ax, c, titel, T=None):
    ax.set_title(titel, fontsize=9)
    for i, (j, d) in enumerate(zip(range(J, 0, -1), c[1:])):
        y0 = i
        pos = (np.arange(len(d)) + 0.5) * 8 / len(d)
        sk = SK                            # gemeinsamer Maßstab für alle Skalen und beide Bilder
        ax.vlines(pos, y0, y0 + np.clip(sk * d, -0.47, 0.47), color=BLAU, lw=0.6)
        ax.axhline(y0, color="0.8", lw=0.4)
        if T is not None:
            ax.axhline(y0 + sk * T, color=ROT, lw=0.4, ls=(0, (3, 2))); ax.axhline(y0 - sk * T, color=ROT, lw=0.4, ls=(0, (3, 2)))
    ax.set_yticks(range(J)); ax.set_yticklabels([f"$d_{j}$" for j in range(J, 0, -1)])
    ax.set_xlim(0, 8); ax.set_ylim(-0.7, J - 0.3); ax.set_xlabel("$t$")
    ax.spines[["top", "right", "left"]].set_visible(False); ax.tick_params(axis="y", length=0)
_, _, Th = visushrink(Y, "hard")
fig, ax = plt.subplots(1, 2, figsize=(6.0, 3.0), sharey=True)
skalen(ax[0], dwt(f), "(a) Koeffizienten von $f$")
skalen(ax[1], dwt(Y), "(b) Koeffizienten von $Y$", T=Th)
fig.savefig(os.path.join(OUT, "abb5_2_koeffizienten.pdf"))

# ---------------- Abbildung 5.3: entrauscht ----------------
fig, ax = plt.subplots(1, 2, figsize=(6.0, 2.4), sharey=True, gridspec_kw={"width_ratios": [2.2, 1]})
ax[0].plot(t, res["hard"], color=BLAU, label="hart")
ax[0].plot(t, res["soft"], color=ROT, lw=0.7, label="weich")
ax[0].set_xlim(0, 8); ax[0].set_xlabel("$t$"); sel = (t > 3.45) & (t < 3.95)
ax[1].plot(t[sel], f[sel], color="k", ls=":", lw=0.9, label="$f$")
ax[1].plot(t[sel], res["hard"][sel], color=BLAU, label="hart")
ax[1].plot(t[sel], res["soft"][sel], color=ROT, lw=0.7, label="weich")
ax[1].set_xlabel("$t$ (Ausschnitt)"); ax[1].legend(frameon=False, loc="upper right")
for a in ax: a.spines[["top", "right"]].set_visible(False)
ax[0].set_ylim(-2.2, 4.0)
fig.savefig(os.path.join(OUT, "abb5_3_entrauscht.pdf"))
print("Abbildungen in", OUT)
