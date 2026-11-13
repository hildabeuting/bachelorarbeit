# Abschnitt 6.5: Erzeugung des Test-EKG ohne Atmungsdrift
import functools
import math
import os
import sys
import numpy as np
import scipy.integrate
import neurokit2 as nk
from kap6_verfahren import fs

daten = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")
n = 2 ** 16

def ableitung_ohne_drift(t, x, rr, ti, sfint, ai, bi):
    ta = math.atan2(x[1], x[0])
    a0 = 1.0 - np.sqrt(x[0] ** 2 + x[1] ** 2)
    w0 = 2 * np.pi / rr[min(int(np.floor(t * sfint)), len(rr) - 1)]
    dti = (ta - ti) - np.round((ta - ti) / 2 / np.pi) * 2 * np.pi
    dz = -np.sum(ai * dti * np.exp(-0.5 * (dti / bi) ** 2)) - x[2]
    return np.array([a0 * x[0] - w0 * x[1], a0 * x[1] + w0 * x[0], dz])

scipy.integrate.solve_ivp = functools.partial(scipy.integrate.solve_ivp,
                                              rtol=1e-8, atol=1e-10)
modul = sys.modules["neurokit2.ecg.ecg_simulate"]
modul._ecg_simulate_derivsecgsyn = ableitung_ohne_drift
f = nk.ecg_simulate(duration=n / fs + 2, sampling_rate=fs, heart_rate=70,
                    noise=0, method="ecgsyn", random_state=42)
f = np.asarray(f[:n])
f = f - np.mean(f)
print(f"||f||^2 = {np.sum(f ** 2):.0f}")

np.savetxt(os.path.join(daten, "testekg.dat"), f, header="f", comments="", fmt="%.10f")
