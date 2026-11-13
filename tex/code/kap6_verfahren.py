# Abschnitt 6.4: Ein-Schritt-Verfahren nach Castillo et al.
import numpy as np
import pywt

fs = 360
N = 2 ** 19
wavelet = "db6"
J = 8
JS = 2

def entrauschen(y, modus="soft", a_null=True, J=J, JS=JS, JH=0, faktor=1.0):
    c = pywt.wavedec(y, wavelet, mode="periodization", level=J)
    sigma_dach = np.median(np.abs(c[-1])) / 0.6745
    T = sigma_dach * np.sqrt(2 * np.log(len(y)))
    neu = [np.zeros_like(c[0]) if a_null else c[0]]
    for j, d in zip(range(J, 0, -1), c[1:]):
        if j <= JS:
            d = pywt.threshold(d, T, modus)
        elif j <= JS + JH:
            d = pywt.threshold(d, faktor * T, modus)
        neu.append(d)
    return pywt.waverec(neu, wavelet, mode="periodization")
