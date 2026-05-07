"""
EKG Signal plotten
==================
Wähle die .hea Datei aus, der Rest wird automatisch geladen.
dann soll der code die Signale einlesen, die Annotationen im Zeitfenster von 10 Sekunden extrahieren und das Signal mit den Annotationen plotten.
"""

import tkinter as tk # phyton bibliothek für grafische fenster
from tkinter import filedialog # damit ich eine datei auswählen kann
import wfdb # bibliothek zum lesen von ekg daten
import matplotlib.pyplot as plt # bibliothek zum plotten von diagrammen
import numpy as np # bibliothek für numerische berechnungen und fourier transformation
from pathlib import Path # bibliothek zum arbeiten mit dateipfaden

# Matplotlib Backend zurücksetzen für GUI-Anzeige
# plt.switch_backend('Agg')  # Auskommentiert, damit Plots in GUI angezeigt werden


# ── 1. Datei auswählen ────────────────────────────────────────────────────

# Temporär: Feste Datei verwenden, um das Skript zu testen
record_pfad = "data/103"  # Beispiel: Record 100 aus dem data-Ordner

# Original: Dialog für Dateiauswahl (funktioniert nur in GUI-Umgebung)
# root = tk.Tk() # Hauptfenster erstellen
# root.withdraw()  # Hauptfenster verstecken

# print("Bitte wähle die .hea Datei aus...")

# hea_pfad = filedialog.askopenfilename( #der dateiauswahl dialog wird geöffnet und auswählter pfad wird in hea_pfad gespeichert
#     title="EKG Datei auswählen (.hea)", 
#     filetypes=[("EKG Header", "*.hea")]
# )

# if not hea_pfad:
#     print("Keine Datei ausgewählt. Programm beendet.")
#     exit()

# # Dateipfad ohne .hea Endung (wfdb braucht das so)
# record_pfad = hea_pfad.replace(".hea", "") #wfdb arbeitet mit basisnamen deswegen .hea entfernen
print(f"Geladen: {record_pfad}") #zeigt an welche datei man ausgewählt hat


# ── 2. Signal einlesen ────────────────────────────────────────────────────

record = wfdb.rdrecord(record_pfad) #lädt die ekg daten in record
ann    = wfdb.rdann(record_pfad, "atr") #lädt sich die zugehörige atr (= annotation datei) runter

print(record.sig_name) #zeigt welche kanäle in den ekg daten vorhanden sind

abtastrate = record.fs           # z.B. 360 Hz dh EKG wurde pro sek 360 mal gemessen ; fs = sampling frequency = abtastrate - wie oft pro sekunde gemessen wurde
signal     = record.p_signal[:, 0]   # Kanal MLII 


# ── 3. Zeitfenster festlegen ──────────────────────────────────────────────

start_sek = 0
end_sek   = 10

start_idx = start_sek * abtastrate 
end_idx   = end_sek   * abtastrate # 10 sekunden * 360 messungen pro sekunde = 3600 messungen im zeitraum von 10 sekunden

t   = [i / abtastrate for i in range(start_idx, end_idx)] # erstellt die zeitachse für die x-achse des diagramms, indem es die indices durch die abtastrate teilt (z.b. 0, 1/360, 2/360, ..., 9.997/360)
sig = signal[start_idx:end_idx] # extrahiert die signalwerte für die ersten 10 sekunden, indem es die entsprechenden indices aus dem signal array nimmt (z.b. signal[0:3600] für die ersten 10 sekunden)


# ── 4. Annotationen im Fenster ────────────────────────────────────────────

ann_zeiten  = [] #leere liste für die zeit der ann
ann_symbole = [] #leere liste für die symbole der ann

for i in range(len(ann.sample)): #ann.sample enthält die indices der messungen, bei denen eine annotation vorliegt, z.b. [360, 720, 1080, ...], also wo was wichtiges passiert 
    s = ann.sample[i] #Holt die Sample-Position der Annotation
    if start_idx <= s < end_idx: #liegt die annotation in sichtbarem zeitfenster?
        ann_zeiten.append(s / abtastrate) #wenn ja, dann wird die zeit der annotation zur liste ann_zeiten hinzugefügt (z.b. 360/360 = 1 sekunde, 720/360 = 2 sekunden, ...)
        ann_symbole.append(ann.symbol[i]) #und das zugehörige symbol (z.b. "N" für normaler herzschlag, "V" für ventrikuläre extrasystole, "+" für unklassifizierte annotation, ...) wird zur liste ann_symbole hinzugefügt


# ── 5. Plotten ────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(14, 5)) # fig = figure, ax = axes = zeichenfläche 14 zoll breit, 5 zoll hoch

ax.plot(t, sig, color="steelblue", linewidth=0.9, label="EKG Signal (MLII)") #zeichnet die ekg kurve in der zeichenfläche, x-achse = zeit t, y-achse = signal sig, farbe = steelblue, linienbreite = 0.9, label für legende = "EKG Signal (MLII)"

# Annotationslinien einzeichnen - mit symbol getstrichelt und farbe je nach symbol
for zeit, sym in zip(ann_zeiten, ann_symbole): 
    if sym == "+":
        continue
    farbe = "violet" if sym == "N" else "tomato"
    ax.axvline(x=zeit, color=farbe, linewidth=0.8, linestyle="--", alpha=0.7) #zeichnet eine vertikale linie an der zeit der annotation, farbe je nach symbol, linienbreite = 0.8, gestrichelt, transparenz = 0.7
    ax.text(zeit + 0.02, max(sig) * 0.9, sym, color=farbe, fontsize=8) #schreibt das symbol der annotation leicht rechts von der linie, auf 90% der maximalen signalhöhe, in der gleichen farbe wie die linie, schriftgröße = 8

#beschriftung
ax.set_xlabel("Zeit (Sekunden)") 
ax.set_ylabel("Amplitude (mV)")
ax.set_title(f"EKG Rohdaten – MIT-BIH Record {Path(record_pfad).name}")
ax.legend() #legende anzeigen
ax.grid(True, alpha=0.3) #gitterlinien im hintergrund, transparenz = 0.3

plt.tight_layout() #verhindert dass beschriftungen abgeschnitten werden

# Ausgabeverzeichnis im Unterordner ergebnisse
output_dir = Path(__file__).resolve().parent / "ergebnisse" #erstellt den pfad zum ordner "ergebnisse" im gleichen verzeichnis wie das python skript
output_dir.mkdir(parents=True, exist_ok=True) #eigentlich unnötig, da der ordner schon existiert, aber falls nicht, wird er hiermit erstellt
output_path = output_dir / f"{Path(record_pfad).name}_plot.png" #erstellt den pfad für die ausgabedatei, z.b. "ergebnisse/100.hea_plot.png", indem es den namen der eingabedatei nimmt und "_plot.png" anhängt

plt.savefig(output_path, dpi=150) #speichert den plot als png datei im angegebenen pfad, dpi = dots per inch = auflösung des bildes, 150 ist eine gute auflösung für web und druck
plt.show() #öffnet ein fenster mit dem plot

print(f"Plot gespeichert als {output_path}") #gibt speicherort aus 


#jetzt fourier transformation machen

N = len(sig) #welches intervall deckt das signal ab 

# DFT diskrete fourier transformation berechnen
X = np.fft.fft(sig) 

# Nur positive Frequenzen (die andere Hälfte ist gespiegelt)
X_halb = X[:N//2] #warum?

# Frequenzachse bauen
freqs = np.fft.fftfreq(N, d=1/abtastrate)[:N//2]

# Amplitudenspektrum (Betrag)
amplitude = np.abs(X_halb)

# Plotten
fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(freqs, amplitude, color="darkorange", linewidth=0.9)
ax.set_xlabel("Frequenz (Hz)")
ax.set_ylabel("|X[k]| – Amplitude")
ax.set_title("Frequenzspektrum des EKG-Signals")
ax.set_xlim(0, 50)   # EKG-relevanter Bereich
ax.grid(True, alpha=0.3)
plt.tight_layout()

# Speichern des Frequenzspektrums
spectrum_output_path = output_dir / f"{Path(record_pfad).name}_spectrum_fft.png"
plt.savefig(spectrum_output_path, dpi=150)
plt.show()

print(f"Frequenzspektrum gespeichert als {spectrum_output_path}")


# ── Kombinierter Plot: EKG + Frequenzspektrum ───────────────────────────── 

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=False)  # 2 Zeilen, 1 Spalte

# Oberer Plot: EKG Signal
ax1.plot(t, sig, color="steelblue", linewidth=0.9, label="EKG Signal (MLII)")
for zeit, sym in zip(ann_zeiten, ann_symbole): 
    if sym == "+":
        continue
    farbe = "violet" if sym == "N" else "tomato"
    ax1.axvline(x=zeit, color=farbe, linewidth=0.8, linestyle="--", alpha=0.7)
    ax1.text(zeit + 0.02, max(sig) * 0.9, sym, color=farbe, fontsize=8)
ax1.set_ylabel("Amplitude (mV)")
ax1.set_title(f"EKG Rohdaten – MIT-BIH Record {Path(record_pfad).name}")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Unterer Plot: Frequenzspektrum
ax2.plot(freqs, amplitude, color="darkorange", linewidth=0.9)
ax2.set_xlabel("Frequenz (Hz)")
ax2.set_ylabel("|X[k]| – Amplitude")
ax2.set_title("Frequenzspektrum des EKG-Signals")
ax2.set_xlim(0, 50)
ax2.grid(True, alpha=0.3)

plt.tight_layout()

# Speichern des kombinierten Plots
combined_output_path = output_dir / f"{Path(record_pfad).name}_combined_plot.png"
plt.savefig(combined_output_path, dpi=150)
plt.show()

print(f"Kombinierter Plot gespeichert als {combined_output_path}")
