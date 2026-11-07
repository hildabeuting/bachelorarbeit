# Tabellen 6.2 und 6.3: referenzfreie Kennzahlen für die Datensätze 100 und 103 (MLII)
import numpy as np
import pywt
import wfdb
from kap6_verfahren import entrauschen, isoelektrisch, amplitude, stufenenergie, spektrum, fs, N, J, JS, wavelet

datensaetze = ["100", "103"]
pfad = "../daten/mitdb/"
rand = 10 * fs
fenster = 2 ** 13
baender = [(0, 0.7), (0.7, 1.5), (1.5, 40), (59, 61)]
varianten = [(7, 3), (9, 3), (8, 2), (8, 4)]


def kennzahlen(y, r, fr, P):
    iso = isoelektrisch(y, r)
    _, Q = spektrum(y)
    return {"Basislinie uV": 1e3 * np.std(iso),
            "60 Hz uV": 1e3 * np.mean([amplitude(y[k:k + fenster], 60) for k in range(0, N, fenster)]),
            "R-Zacke mV": np.mean(y[r] - iso),
            **{f"{a}-{b} Hz %": 100 * Q[(fr >= a) & (fr < b)].sum() / P[(fr >= a) & (fr < b)].sum() for a, b in baender}}


for name in datensaetze:
    x = wfdb.rdrecord(pfad + name).p_signal[:N, 0]
    ann = wfdb.rdann(pfad + name, "atr")
    y = x - np.mean(x)
    r = np.array([p for p, z in zip(ann.sample, ann.symbol) if rand <= p < N - rand and z == "N"])
    c = pywt.wavedec(y, wavelet, mode="periodization", level=J)
    sigma = np.median(np.abs(c[-1])) / 0.6745
    T = sigma * np.sqrt(2 * np.log(N))
    fr, P = spektrum(y)
    print(f"{name}: {len(r)} Normalschläge, sigma {1e3 * sigma:.2f} uV, T {1e3 * T:.1f} uV")
    print("  behalten in d3 d2 d1 in %: " + " ".join(f"{100 * np.mean(np.abs(d) > T):.1f}" for d in c[-JS:]))
    alle = np.array([p for p, z in zip(ann.sample, ann.symbol) if z in "NLRAaJSVFejE/fQ"])
    nahe = []
    for j, d in zip(range(JS, 0, -1), c[-JS:]):
        lage = (np.where(np.abs(d) > T)[0] + 0.5) * 2 ** j
        nahe.append(100 * np.mean(np.min(np.abs(lage[:, None] - alle[None, :]), axis=1) < 0.1 * fs))
    print("  davon höchstens 100 ms von einer R-Zacke, d3 d2 d1 in %: " + " ".join(f"{v:.0f}" for v in nahe))
    werte = {"vorher": kennzahlen(y, r, fr, P)}
    for modus in ["soft", "hard"]:
        f = entrauschen(y, modus)
        werte[modus] = kennzahlen(f, r, fr, P)
        entfernt = np.sqrt(stufenenergie(y - f) / N)
        print(f"  {modus}: entfernt (Effektivwert in uV) a8 {1e3 * entfernt[0]:.1f}, d3..d1 "
              + " ".join(f"{1e3 * v:.1f}" for v in entfernt[-JS:]))
        d = y - entrauschen(y, modus, a_null=False)
        print(f"  {modus}: entfernter Detailanteil, um die R-Zacken gemittelt: max "
              f"{1e3 * np.max(np.abs(np.mean([d[p - 72:p + 72] for p in r], axis=0))):.1f} uV")
    for l, m in varianten:
        werte[f"J{l}JS{m}"] = kennzahlen(entrauschen(y, J=l, JS=m), r, fr, P)
    alle_rr = np.diff(alle[alle < N]) / fs
    print(f"  Herzfrequenz: Mittel {60 / np.mean(alle_rr):.1f}/min, kleinster Einzelwert {60 / np.max(alle_rr):.1f}/min")
    _, Q = spektrum(entrauschen(y))
    k60, umgebung = np.argmin(np.abs(fr - 60)), (np.abs(fr - 60) > 1) & (np.abs(fr - 60) < 5)
    print(f"  Spitze bei 60 Hz über der Umgebung: vorher {P[k60] / np.median(P[umgebung]):.0f}, "
          f"nachher {Q[k60] / np.median(Q[umgebung]):.0f}")
    for k in werte["vorher"]:
        print(f"  {k:14s} " + "  ".join(f"{m} {werte[m][k]:8.3f}" for m in werte))
