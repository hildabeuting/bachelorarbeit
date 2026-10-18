# Abschnitt 6.5: Test an einem künstlichen EKG (ECGSYN über NeuroKit2), Tabelle tab:test
import math
import sys
import numpy as np
import neurokit2 as nk
from kap6_verfahren import fs, entrauschen

N_TEST = 2 ** 16
DRIFT = (0.5, 0.25)          # Amplitude in mV, Frequenz in Hz
SIGMA = 0.05                 # Standardabweichung des Rauschens in mV
WIEDERHOLUNGEN = 10


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
    return np.asarray(f[:n])


def fehlerenergie(f, schaetzungen):
    # mittlere Fehlerenergie über die Realisierungen (Schätzung des Risikos r(f)),
    # Mittelwerte von f und den Schätzungen vorher abgezogen
    f = f - np.mean(f)
    return np.mean([np.sum((g - np.mean(g) - f) ** 2) for g in schaetzungen])


def gestoert(f, seed):
    t = np.arange(len(f)) / fs
    b = DRIFT[0] * np.sin(2 * np.pi * DRIFT[1] * t)
    return f + b + SIGMA * np.random.default_rng(seed).standard_normal(len(f))


varianten = [("J = 8, J_S = 3, Soft", dict()),
             ("Hard", dict(modus="hard")),
             ("J_S = 2", dict(JS=2)),
             ("J_S = 4", dict(JS=4)),
             ("nur Drift (J_S = 0)", dict(JS=0)),
             ("J = 7", dict(J=7)),
             ("J = 9", dict(J=9))]

if __name__ == "__main__":
    f = testsignal()
    messungen = [gestoert(f, s) for s in range(WIEDERHOLUNGEN)]
    print(f"||f||^2 = {np.sum((f - np.mean(f)) ** 2):.0f}")
    print(f"  {'vorher (Messung Y)':22s} {fehlerenergie(f, messungen):8.1f}")
    for name, arg in varianten:
        print(f"  {name:22s} {fehlerenergie(f, [entrauschen(y, **arg) for y in messungen]):8.1f}")
