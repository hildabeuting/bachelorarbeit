import numpy as np
import matplotlib.pyplot as plt


def mexikanerhut_wavelet(x: np.ndarray) -> np.ndarray:
    """Mexikanerhut (Ricker) wavelet function."""
    factor = 2.0 / (np.sqrt(3.0) * np.pi ** 0.25)
    return factor * (1.0 - x ** 2) * np.exp(-x ** 2 / 2.0)


def wavelet_transform(a: float, b_values: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Compute Wf(a, b) for a fixed scale a and multiple translations b."""
    x = (t[:, None] - b_values[None, :]) / a
    psi = mexikanerhut_wavelet(x)
    indicator = np.logical_and(t >= 1.0, t <= 2.0).astype(float)
    f_t = np.sin(2.0 * np.pi * t) * indicator
    integrand = f_t[:, None] * (psi / np.sqrt(np.abs(a)))
    integral = np.trapezoid(integrand, t, axis=0)
    return integral


def main() -> None:
    a_values = np.linspace(0.05, 3.0, 150)
    b_values = np.linspace(-0.5, 3.5, 200)
    t = np.linspace(-5.0, 8.0, 3001)

    W = np.empty((len(a_values), len(b_values)))
    for i, a in enumerate(a_values):
        W[i, :] = wavelet_transform(a, b_values, t)

    # --- Plot 1: Scalogram (2D) ---
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), gridspec_kw={'height_ratios': [1, 2]})

    # Signal oben
    t_plot = np.linspace(0, 3.5, 1000)
    indicator = np.logical_and(t_plot >= 1.0, t_plot <= 2.0).astype(float)
    f_plot = np.sin(2.0 * np.pi * t_plot) * indicator
    axes[0].plot(t_plot, f_plot, color='steelblue', linewidth=1.5)
    axes[0].set_xlim(-0.5, 3.5)
    axes[0].set_ylabel(r'$f(t)$')
    axes[0].set_title(r'Signal $f(t) = \sin(2\pi t)\,\mathbf{1}_{[1,2]}(t)$')
    axes[0].axvspan(1, 2, alpha=0.1, color='steelblue')
    axes[0].grid(True, alpha=0.3)

    # Scalogram unten
    im = axes[1].contourf(b_values, a_values, W, levels=60, cmap='viridis')
    fig.colorbar(im, ax=axes[1], label=r'$Wf(a,b)$')
    axes[1].set_xlabel('Translationsparameter $b$')
    axes[1].set_ylabel('Skalierungsparameter $a$')
    axes[1].set_title('Scalogram $Wf(a,b)$ mit Mexikanerhut-Wavelet')
    axes[1].set_xlim(-0.5, 3.5)

    plt.tight_layout()
    #plt.savefig('scalogram.png', dpi=150, bbox_inches='tight')
    #plt.show()

    # --- Plot 2: 3D-Plot (angepasster b-Bereich) ---
    A, B = np.meshgrid(a_values, b_values, indexing='ij')

    fig2 = plt.figure(figsize=(11, 7))
    ax3d = fig2.add_subplot(111, projection='3d')
    surf = ax3d.plot_surface(A, B, W, cmap='viridis', edgecolor='none',
                              linewidth=0, antialiased=True)

    ax3d.set_title(r'3D-Plot von $Wf(a, b)$ mit Mexikanerhut-Wavelet')
    ax3d.set_xlabel('Skalierungsparameter $a$')
    ax3d.set_ylabel('Translationsparameter $b$')
    ax3d.set_zlabel('$Wf(a, b)$')
    fig2.colorbar(surf, shrink=0.6, aspect=12, label=r'Wert von $Wf(a, b)$')

    plt.tight_layout()
    
    
 
    plt.show()

    print("Gespeichert: scalogram.png und wavelet_3d.png")


if __name__ == '__main__':
    main()