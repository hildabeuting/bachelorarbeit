import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def haar_wavelet(t):
    psi = np.zeros_like(t)
    psi[(t >= 0) & (t < 0.5)] = 1.0
    psi[(t >= 0.5) & (t < 1.0)] = -1.0
    return psi

def haar_wavelet_ab(t, a, b):
    tau = (t - b) / a
    psi = np.zeros_like(tau)
    psi[(tau >= 0) & (tau < 0.5)] = 1.0 / np.sqrt(a)
    psi[(tau >= 0.5) & (tau < 1.0)] = -1.0 / np.sqrt(a)
    return psi

# Signal definieren
a = 5.0
b = 7.0
t = np.linspace(0, 2 * np.pi, 1000)
signal = np.sin(a * t) + np.cos(b * t)

# Haar-Mutter-Wavelet
tau = np.linspace(-0.2, 1.2, 400)
psi0 = haar_wavelet(tau)

# Skalierte und verschobene Wavelets
scales = [0.25, 0.5, 1.0, 2.0]
shifts = [0.0, 0.5, 1.0]
wavelet_examples = []
for scale in scales:
    for shift in shifts[:2]:
        wavelet_examples.append((scale, shift, haar_wavelet_ab(tau, scale, shift)))

# Haar-CWT manuell berechnen
def haar_cwt(signal, t, scales, translations):
    dt = t[1] - t[0]
    coeffs = np.zeros((len(scales), len(translations)))
    for i, scale in enumerate(scales):
        for j, trans in enumerate(translations):
            psi = haar_wavelet_ab(t, scale, trans)
            coeffs[i, j] = np.sum(signal * psi) * dt
    return coeffs

translations = np.linspace(0, 2 * np.pi, 200)
coeffs = haar_cwt(signal, t, scales, translations)

# Fourier-Transformation des Signals
fft_vals = np.fft.fft(signal)
fft_freq = np.fft.fftfreq(signal.size, d=t[1] - t[0])
fft_power = np.abs(fft_vals)
positive = fft_freq >= 0

# Plotten
fig, axs = plt.subplots(5, 1, figsize=(10, 18))

axs[0].plot(t, signal, color="tab:blue")
axs[0].set_title("Signal: sin(a x) + cos(b x)")
axs[0].set_xlabel("t")
axs[0].set_ylabel("Amplitude")
axs[0].grid(True)

axs[1].plot(tau, psi0, label="Haar-Mutter-Wavelet")
axs[1].axhline(0, color="k", lw=0.5)
axs[1].set_title("Haar-Wavelet (Mutter-Wavelet)")
axs[1].legend()
axs[1].grid(True)

for scale, shift, psi in wavelet_examples:
    axs[2].plot(tau, psi, label=f"scale={scale}, shift={shift}")
axs[2].set_title("Haar-Wavelet: verschiedene Skalen und Verschiebungen")
axs[2].legend()
axs[2].grid(True)

im = axs[3].imshow(
    coeffs,
    aspect="auto",
    extent=[translations[0], translations[-1], scales[-1], scales[0]],
    cmap="seismic",
    vmin=-np.max(np.abs(coeffs)),
    vmax=np.max(np.abs(coeffs)),
)
axs[3].set_title("Haar-Wavelet-Transform von sin(a x) + cos(b x)")
axs[3].set_ylabel("Skala")
axs[3].set_xlabel("Verschiebung")
fig.colorbar(im, ax=axs[3], label="Wavelet-Koeffizient")

axs[4].stem(fft_freq[positive], fft_power[positive], linefmt='tab:green', markerfmt=' ', basefmt='k-')
axs[4].set_title('Fourier-Transformierte des gleichen Signals')
axs[4].set_xlabel('Frequenz')
axs[4].set_ylabel('Amplitude')
axs[4].grid(True)

plt.tight_layout()

fig2 = plt.figure(figsize=(10, 6))
ax3d = fig2.add_subplot(111, projection='3d')
T, S = np.meshgrid(translations, scales)
surf = ax3d.plot_surface(
    T,
    S,
    coeffs,
    cmap='seismic',
    edgecolor='none',
    rstride=1,
    cstride=1,
    antialiased=True,
)
ax3d.set_title('3D Haar-Wavelet-Transform')
ax3d.set_xlabel('Verschiebung')
ax3d.set_ylabel('Skala')
ax3d.set_zlabel('Koeffizient')
ax3d.view_init(elev=30, azim=-60)
fig2.colorbar(surf, ax=ax3d, shrink=0.5, aspect=10)

plt.show()