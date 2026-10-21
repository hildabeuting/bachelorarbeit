# Abschnitt 6.2 und 6.4.2: Auswertung der Annotationen der Datensätze 100 und 103
# (Herzfrequenz aus den Schlagmarkierungen, verrauschte Abschnitte aus der Signalqualität)
import numpy as np
import wfdb
from kap6_verfahren import fs, N

schlaege = "NLRAaJSVFejE/fQ"   # Annotationssymbole, die einen Herzschlag markieren
for name in ["100", "103"]:
    ann = wfdb.rdann("../daten/mitdb/" + name, "atr")
    r = np.array([p for p, z in zip(ann.sample, ann.symbol) if z in schlaege and p < N])
    rr = np.diff(r) / fs    # Abstände aufeinanderfolgender Schläge in s
    print(f"{name}: Mittel {60 / np.mean(rr):.1f}/min, langsamster Schlag {60 / np.max(rr):.1f}/min")
    # Signalqualität: Symbol "~", Subtyp 0 = beide sauber, 1 = oberes, 2 = unteres Signal verrauscht
    for p, z, s in zip(ann.sample, ann.symbol, ann.subtype):
        if z == "~" and p < N:
            print(f"  ab {p / fs:7.1f} s: Signalqualität {s}")
