# Gemeinsame Funktionen für Kapitel 6: synthetisches EKG, Entrauschen, Bewertungsmaße
import numpy as np
import pywt

fs = 360
N = 2048
rr = 60 / 72
wellen = {"P": (-0.139, 0.25, 0.033), "Q": (-0.035, -0.17, 0.013),
          "R": (0.0, 1.0, 0.013), "S": (0.035, -0.25, 0.013), "T": (0.208, 0.40, 0.053)}


def ekg_modell(n=N, fs=fs, rr=rr, t0=0.3):
    t = np.arange(n) / fs
    x = np.zeros(n)
    for r in np.arange(t0, t[-1] + rr, rr):
        for mu, a, b in wellen.values():
            x += a * np.exp(-((t - r - mu) ** 2) / (2 * b ** 2))
    return t, x


def entrauschen(y, wavelet="db2", J=5, modus="hard", sigma_je_stufe=False, a_schwellen=False, a_null=None):
    L = a_null if a_null else J
    c = pywt.wavedec(y, wavelet, mode="periodization", level=L)
    sigma = np.median(np.abs(c[-1])) / 0.6745
    n = len(y)
    if a_null:
        neu = [np.zeros_like(c[0])]
    else:
        neu = [pywt.threshold(c[0], sigma * np.sqrt(2 * np.log(n)), modus) if a_schwellen else c[0]]
    for i, d in enumerate(c[1:]):
        if i < L - J:
            neu.append(d)
            continue
        s = np.median(np.abs(d)) / 0.6745 if sigma_je_stufe else sigma
        neu.append(pywt.threshold(d, s * np.sqrt(2 * np.log(n)), modus))
    return pywt.waverec(neu, wavelet, mode="periodization")


def snr(x, y):
    return 10 * np.log10(np.sum(x ** 2) / np.sum((y - x) ** 2))


def rmse(x, y):
    return np.sqrt(np.mean((x - y) ** 2))


def prd(x, y):
    return 100 * np.sqrt(np.sum((x - y) ** 2) / np.sum(x ** 2))


def rauschen_fuer_snr(x, snr_db, rng):
    z = rng.standard_normal(len(x))
    sigma = np.sqrt(np.sum(x ** 2) / len(x) / 10 ** (snr_db / 10))
    return sigma * z, sigma
