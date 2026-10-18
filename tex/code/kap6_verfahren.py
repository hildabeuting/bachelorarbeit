# Gemeinsame Funktionen für Kapitel 6: Entrauschen nach Castillo et al.
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

