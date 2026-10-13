# Stufen 2 und 3: MIT-BIH-Datensätze mit weißem bzw. realem Rauschen (Tabelle 6.6)
import numpy as np
import pywt
import wfdb
from kap6_modell import entrauschen, snr, rauschen_fuer_snr

datensaetze = ["100", "103"]
mitdb = "../daten/mitdb/"
nstdb = "../daten/nstdb/"
segmente = 10
N = 2048
start = 10 * 360
eingangs_snr = 5
wiederholungen = 20
rng = np.random.default_rng(93)
stoerungen = {s: wfdb.rdrecord(nstdb + s).p_signal[:, 0] for s in ["bw", "em", "ma"]}
varianten = [("db2", "hard", False, None), ("db2", "soft", False, None), ("db4", "hard", False, None),
             ("db2", "hard", True, None), ("db2", "hard", False, 8)]


def segment(signal, k):
    s = signal[start + k * N:start + (k + 1) * N]
    return s - np.mean(s)


def r_stellen(ann, k):
    a, b = start + k * N, start + (k + 1) * N
    return np.array([p - a for p, z in zip(ann.sample, ann.symbol) if a <= p < b and z == "N"])


def reales_rauschen(x, quelle, k, snr_db):
    r = segment(stoerungen[quelle], k)
    return r * np.sqrt(np.sum(x ** 2) / np.sum(r ** 2) / 10 ** (snr_db / 10))


for q, r in stoerungen.items():
    anteil = []
    for k in range(segmente):
        c = pywt.wavedec(segment(r, k), "db2", mode="periodization", level=4)
        e = np.array([np.sum(v ** 2) for v in c])
        anteil.append(e / e.sum())
    print(q, "Energieanteil a4 d4 d3 d2 d1 in %:", np.round(100 * np.mean(anteil, axis=0), 1))

for name in datensaetze:
    ekg = wfdb.rdrecord(mitdb + name).p_signal[:, 0]
    ann = wfdb.rdann(mitdb + name, "atr")
    sigma0 = [np.median(np.abs(pywt.wavedec(segment(ekg, k), "db2", mode="periodization", level=1)[-1])) / 0.6745
              for k in range(segmente)]
    sigma_zu = np.mean([rauschen_fuer_snr(segment(ekg, k), eingangs_snr, rng)[1] for k in range(segmente)])
    print(name, f"Eigenrauschen sigma0 {np.mean(sigma0):.4f}, zugesetztes sigma {sigma_zu:.4f}")
    for w, m, je_stufe, a_null in varianten:
        weiss, r_zacke = [], []
        for k in range(segmente):
            x = segment(ekg, k)
            r = r_stellen(ann, k)
            for _ in range(wiederholungen):
                e, _ = rauschen_fuer_snr(x, eingangs_snr, rng)
                f = entrauschen(x + e, w, 4, m, sigma_je_stufe=je_stufe, a_null=a_null)
                weiss.append(snr(x, f))
                r_zacke.append(np.mean(f[r] / x[r]))
        real = {q: np.mean([snr(x, entrauschen(x + reales_rauschen(x, q, k, eingangs_snr), w, 4, m,
                                               sigma_je_stufe=je_stufe, a_null=a_null))
                            for k in range(segmente) for x in [segment(ekg, k)]])
                for q in stoerungen}
        ohne = np.mean([snr(x, entrauschen(x, w, 4, m, sigma_je_stufe=je_stufe, a_null=a_null))
                        for k in range(segmente) for x in [segment(ekg, k)]])
        print(name, w, m, "je Stufe" if je_stufe else "global", f"a_{a_null}=0" if a_null else "",
              f"weiß {np.mean(weiss):6.2f}  R-Zacke {np.mean(r_zacke):5.2f}",
              " ".join(f"{q} {v:6.2f}" for q, v in real.items()), f" ohne Rauschen {ohne:6.2f}")
