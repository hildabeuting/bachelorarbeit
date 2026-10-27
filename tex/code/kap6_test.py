# Abschnitt 6.5: Test an einem künstlichen EKG (ECGSYN über NeuroKit2), Tabelle tab:test
import math
import sys
import numpy as np
import neurokit2 as nk
import pywt
from kap6_verfahren import fs, entrauschen

N_TEST = 2 ** 16
DRIFT = (0.5, 0.25)          # Amplitude in mV, Frequenz in Hz
SIGMA = 0.025                # Standardabweichung des Rauschens in mV (McSharry et al., Abschn. V)
SEED = 0                     # feste Realisierung von Z für Tabelle und Abbildungen
KONTROLLE = 10               # Realisierungen für die Kontrolle der Schwankung


def _ableitung_ohne_drift(t, x, rr, ti, sfint, ai, bi):
    # wie neurokit2 (_ecg_simulate_derivsecgsyn), aber ohne die eingebaute Atmungsdrift zbase
    ta = math.atan2(x[1], x[0])
    a0 = 1.0 - np.sqrt(x[0] ** 2 + x[1] ** 2)
    ip = np.floor(t * sfint).astype(int)
    w0 = 2 * np.pi / rr[min(ip, len(rr) - 1)]
    dti = (ta - ti) - np.round((ta - ti) / 2 / np.pi) * 2 * np.pi
    return np.array([a0 * x[0] - w0 * x[1], a0 * x[1] + w0 * x[0],
                     -np.sum(ai * dti * np.exp(-0.5 * (dti / bi) ** 2)) - x[2]])


def testsignal(n=N_TEST, seed=42):
    sys.modules["neurokit2.ecg.ecg_simulate"]._ecg_simulate_derivsecgsyn = _ableitung_ohne_drift
    f = nk.ecg_simulate(duration=n / fs + 2, sampling_rate=fs, heart_rate=70,
                        noise=0, method="ecgsyn", random_state=seed)
    f = np.asarray(f[:n])
    return f - np.mean(f)    # Test-EKG mittelwertfrei (eigene Festlegung)


def fehlerenergie(f, g):
    # Fehlerenergie ||g - f||^2 (Definition 5.2)
    return np.sum((g - f) ** 2)


def gestoert(f, seed):
    t = np.arange(len(f)) / fs
    b = DRIFT[0] * np.sin(2 * np.pi * DRIFT[1] * t)
    return f + b + SIGMA * np.random.default_rng(seed).standard_normal(len(f))


varianten = [("J = 8, J_S = 2, Soft", dict()),
             ("Hard", dict(modus="hard")),
             ("J_S = 1", dict(JS=1)),
             ("J_S = 3", dict(JS=3)),
             ("nur Drift (J_S = 0)", dict(JS=0)),
             ("J = 7", dict(J=7)),
             ("J = 9", dict(J=9))]

if __name__ == "__main__":
    f = testsignal()
    y = gestoert(f, SEED)
    print(f"||f||^2 = {np.sum(f ** 2):.0f}")
    print(f"  {'vorher (Messung Y)':22s} {fehlerenergie(f, y):8.1f}")
    for name, arg in varianten:
        print(f"  {name:22s} {fehlerenergie(f, entrauschen(y, **arg)):8.1f}")
    # Energie von f und erwartete Rauschenergie sigma^2 N / 2^j auf d_1 bis d_4
    c = pywt.wavedec(f, "db6", mode="periodization", level=8)[::-1]
    cy = pywt.wavedec(y, "db6", mode="periodization", level=8)
    T = np.median(np.abs(cy[-1])) / 0.6745 * np.sqrt(2 * np.log(len(y)))   # Schwelle wie im Verfahren
    print(f"T = {T:.3f}")
    print("Energie je Stufe (f, sigma^2 N/2^j, Verlust sum min(|f|,T)^2 beim Schwellen):")
    for j in range(1, 5):
        print(f"  d_{j}: {np.sum(c[j - 1] ** 2):7.1f} {SIGMA ** 2 * len(f) / 2 ** j:7.1f}"
              f" {np.sum(np.minimum(np.abs(c[j - 1]), T) ** 2):7.1f}")
    # Kontrolle: Schwankung der Fehlerenergie über KONTROLLE Realisierungen
    print("Kontrolle (min, max, max. relative Abweichung von Seed 0):")
    for name, arg in [("vorher (Messung Y)", None)] + varianten:
        e = [fehlerenergie(f, gestoert(f, s) if arg is None else entrauschen(gestoert(f, s), **arg))
             for s in range(KONTROLLE)]
        print(f"  {name:22s} {min(e):8.1f} {max(e):8.1f} {max(abs(x - e[0]) for x in e) / e[0]:6.1%}")
    # Abhängigkeit vom Rauschpegel: J_S = 2 gegen J_S = 3 bei derselben Realisierung Z
    print("Rauschpegel (sigma in mV: J_S = 2, J_S = 3):")
    t = np.arange(len(f)) / fs
    b = DRIFT[0] * np.sin(2 * np.pi * DRIFT[1] * t)
    z = np.random.default_rng(SEED).standard_normal(len(f))
    for s in [0.01, 0.025, 0.05, 0.1]:
        y_s = f + b + s * z
        print(f"  {s:5.3f}: " + "  ".join(f"{fehlerenergie(f, entrauschen(y_s, JS=js)):7.1f}" for js in (2, 3)))
