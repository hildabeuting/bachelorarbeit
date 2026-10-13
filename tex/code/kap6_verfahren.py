# Gemeinsame Funktionen für Kapitel 6: Entrauschen nach Castillo et al. und referenzfreie Kennzahlen
import numpy as np
import pywt

fs = 360
N = 2 ** 19
wavelet = "db6"
J = 8
JS = 3


def entrauschen(y, modus="soft", a_null=True, J=J, JS=JS):
    c = pywt.wavedec(y, wavelet, mode="periodization", level=J)
    sigma = np.median(np.abs(c[-1])) / 0.6745
    T = sigma * np.sqrt(2 * np.log(len(y)))
    neu = [np.zeros_like(c[0]) if a_null else c[0]]
    for i, d in enumerate(c[1:]):
        if J - i <= JS:
            neu.append(pywt.threshold(d, T, modus))
        else:
            neu.append(d)
    return pywt.waverec(neu, wavelet, mode="periodization")


def isoelektrisch(y, r):
    a, b = int(0.08 * fs), int(0.06 * fs)
    return np.array([np.mean(y[p - a:p - b]) for p in r])


def amplitude(y, f0):
    t = np.arange(len(y)) / fs
    return 2 * np.abs(np.mean(y * np.exp(-2j * np.pi * f0 * t)))


def stufenenergie(y):
    c = pywt.wavedec(y, wavelet, mode="periodization", level=J)
    return np.array([np.sum(v ** 2) for v in c])


def spektrum(y, n=2 ** 16):
    teile = y[:len(y) // n * n].reshape(-1, n)
    return np.fft.rfftfreq(n, 1 / fs), np.mean(np.abs(np.fft.rfft(teile, axis=1)) ** 2, axis=0)
