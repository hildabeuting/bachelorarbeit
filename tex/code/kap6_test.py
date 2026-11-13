# Abschnitt 6.5: Test an einem künstlichen EKG, Tabelle 6.3
import os
import numpy as np
import pywt
from kap6_verfahren import fs, entrauschen

daten = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")
DRIFT = (0.5, 0.25)
SIGMA = 0.025
SEED = 0

def testsignal():
    return np.loadtxt(os.path.join(daten, "testekg.dat"), skiprows=1)

def drift(n):
    return DRIFT[0] * np.sin(2 * np.pi * DRIFT[1] * np.arange(n) / fs)

def gestoert(f, seed, sigma=SIGMA):
    z = np.random.default_rng(seed).standard_normal(len(f))
    return f + drift(len(f)) + sigma * z

def fehler(f, g):
    return np.sum((g - f) ** 2)

varianten = {"Verfahren": {}, "Hard": dict(modus="hard"),
             "J_S = 1": dict(JS=1), "J_S = 3": dict(JS=3),
             "d_3 mit 0,2 T": dict(JH=1, faktor=0.2), "nur Drift": dict(JS=0),
             "J = 7": dict(J=7), "J = 9": dict(J=9)}

if __name__ == "__main__":
    f = testsignal()
    y = gestoert(f, SEED)
    print(f"||f||^2 = {np.sum(f ** 2):.0f}")
    print(f"vorher: {fehler(f, y):.1f}")
    for name, arg in varianten.items():
        print(f"{name}: {fehler(f, entrauschen(y, **arg)):.1f}")

    zerl = lambda x: pywt.wavedec(x, "db6", mode="periodization",
                                  level=8)[::-1]
    sigma_dach = np.median(np.abs(zerl(y)[0])) / 0.6745
    print(f"T = {sigma_dach * np.sqrt(2 * np.log(len(y))):.3f}")
    print("Stufe: Energie f, sigma^2 N/2^j, Fehler je Variante")
    c = zerl(f)
    auswahl = ["Verfahren", "J_S = 3", "d_3 mit 0,2 T"]
    fe = [zerl(entrauschen(y, **varianten[v]) - f) for v in auswahl]
    for j in range(1, 5):
        erwartet = SIGMA**2 * len(f) / 2**j
        print(f"d_{j}: {np.sum(c[j - 1] ** 2):.1f}, {erwartet:.1f},",
              ", ".join(f"{np.sum(e[j - 1] ** 2):.1f}" for e in fe))
    print("a_8:", ", ".join(f"{np.sum(e[-1] ** 2):.1f}" for e in [c] + fe))

    print("d_3 mit faktor * T:")
    for faktor in np.arange(0, 1.01, 0.05):
        e = fehler(f, entrauschen(y, JH=1, faktor=faktor))
        print(f"{faktor:.2f}: {e:.1f}")

    print("Kontrolle, 10 Realisierungen (min, max, Abweichung von Seed 0):")
    for name, arg in [("vorher", None)] + list(varianten.items()):
        y_s = [gestoert(f, s) for s in range(10)]
        e = [fehler(f, y if arg is None else entrauschen(y, **arg))
             for y in y_s]
        abw = max(abs(x - e[0]) for x in e) / e[0]
        print(f"{name}: {min(e):.1f}, {max(e):.1f}, {abw:.1%}")

    print("Rauschpegel sigma: J_S = 2, J_S = 3, d_3 mit 0,2 T")
    for s in [0.01, 0.025, 0.05, 0.1]:
        y_s = gestoert(f, SEED, sigma=s)
        werte = [fehler(f, entrauschen(y_s, **arg))
                 for arg in (dict(JS=2), dict(JS=3), dict(JH=1, faktor=0.2))]
        print(f"{s}:", ", ".join(f"{w:.1f}" for w in werte))
