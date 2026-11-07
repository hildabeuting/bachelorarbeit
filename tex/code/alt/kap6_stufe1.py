# Stufe 1: Parameterstudie am synthetischen EKG (Tabellen 6.2 und 6.3)
import numpy as np
from kap6_modell import ekg_modell, entrauschen, snr, rmse, prd, rauschen_fuer_snr, fs, rr

wiederholungen = 100
t, x = ekg_modell()
r_index = np.round((0.3 + rr * np.arange(7)) * fs).astype(int)
r_index = r_index[r_index < len(x)]
rng = np.random.default_rng(93)

print("Tiefe J, SNR_out in dB, Eingangs-SNR 5 dB, hart:")
for w in ["db1", "db2", "db4"]:
    zeile = []
    for J in range(1, 9):
        werte = []
        for _ in range(wiederholungen):
            e, _ = rauschen_fuer_snr(x, 5, rng)
            werte.append(snr(x, entrauschen(x + e, w, J, "hard")))
        zeile.append(np.mean(werte))
    print(w, " ".join(f"{v:6.2f}" for v in zeile))

print("Wavelet und Modus bei J=4: SNR_out je Eingangs-SNR 0/5/10, dazu RMSE, PRD, R-Amplitude bei 5 dB:")
for w in ["db1", "db2", "db4"]:
    for m in ["hard", "soft"]:
        zeile = []
        for s in [0, 5, 10]:
            werte = []
            for _ in range(wiederholungen):
                e, _ = rauschen_fuer_snr(x, s, rng)
                werte.append(snr(x, entrauschen(x + e, w, 4, m)))
            zeile.append(np.mean(werte))
        r, p, a = [], [], []
        for _ in range(wiederholungen):
            e, _ = rauschen_fuer_snr(x, 5, rng)
            f = entrauschen(x + e, w, 4, m)
            r.append(rmse(x, f)); p.append(prd(x, f)); a.append(np.mean(f[r_index] / x[r_index]))
        print(w, m, " ".join(f"{v:6.2f}" for v in zeile), f"{np.mean(r):.3f} {np.mean(p):.1f} {np.mean(a):.3f}")

print("a_J schwellen (db2, J=4, 5 dB):")
for m in ["hard", "soft"]:
    for a_s in [False, True]:
        werte = [snr(x, entrauschen(x + rauschen_fuer_snr(x, 5, rng)[0], "db2", 4, m, a_schwellen=a_s)) for _ in range(wiederholungen)]
        print(m, a_s, f"{np.mean(werte):6.2f}")
print("sigma bei 5 dB:", rauschen_fuer_snr(x, 5, rng)[1])
