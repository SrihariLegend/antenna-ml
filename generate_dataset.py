"""Generate a synthetic microstrip patch antenna dataset using design equations.

The classical rectangular microstrip patch antenna formulas (Bahl & Trivedi,
1977; Pozar) are used to compute patch dimensions from substrate and frequency
parameters.  Sweeping over substrate height, dielectric constant, and target
frequency produces a large, physically-grounded dataset that covers common
WiFi, 5G and IoT frequency bands (1 – 10 GHz).

Generated CSV columns
---------------------
Frequency(GHz)          : target resonant frequency
Substrate_Height(mm)    : dielectric slab thickness
Dielectric_Constant     : relative permittivity of the substrate
Patch_Length(mm)        : resonant length of the patch
Patch_Width(mm)         : width of the patch
Substrate_Length(mm)    : length of the ground-plane / substrate board
Substrate_Width(mm)     : width  of the ground-plane / substrate board
"""

import numpy as np
import pandas as pd

C0 = 3e8  # speed of light (m/s)


def patch_dimensions(freq_hz: float, eps_r: float, h_m: float):
    """Return (W_m, L_m) for a rectangular microstrip patch antenna.

    Args:
        freq_hz: Desired resonant frequency in Hz.
        eps_r:   Relative dielectric constant of the substrate.
        h_m:     Substrate height in metres.

    Returns:
        (W_m, L_m) in metres, or None if the geometry is degenerate.
    """
    # Patch width (TM_010 mode)
    W = (C0 / (2.0 * freq_hz)) * np.sqrt(2.0 / (eps_r + 1.0))
    if W <= 0:
        return None

    # Effective dielectric constant (Hammerstad & Jensen approximation)
    eps_eff = (eps_r + 1.0) / 2.0 + (eps_r - 1.0) / 2.0 * (
        1.0 + 12.0 * h_m / W
    ) ** (-0.5)

    # Fringing length extension (Schneider)
    dL = (
        0.412
        * h_m
        * ((eps_eff + 0.3) * (W / h_m + 0.264))
        / ((eps_eff - 0.258) * (W / h_m + 0.8))
    )

    # Resonant patch length
    L = C0 / (2.0 * freq_hz * np.sqrt(eps_eff)) - 2.0 * dL
    if L <= 0:
        return None

    return W, L


def generate(
    freq_range: tuple = (1.0, 10.0),
    n_freqs: int = 50,
    h_range: tuple = (0.5, 5.0),
    n_heights: int = 20,
    eps_range: tuple = (2.0, 12.0),
    n_eps: int = 15,
) -> pd.DataFrame:
    """Sweep design-parameter space and return a DataFrame of valid designs.

    Args:
        freq_range: (min, max) frequency in GHz.
        n_freqs:    Number of frequency points to sample.
        h_range:    (min, max) substrate height in mm.
        n_heights:  Number of height values.
        eps_range:  (min, max) relative dielectric constant.
        n_eps:      Number of ε_r values.

    Returns:
        DataFrame with one valid antenna design per row.
    """
    freqs_ghz = np.linspace(freq_range[0], freq_range[1], n_freqs)
    heights_mm = np.linspace(h_range[0], h_range[1], n_heights)
    eps_values = np.linspace(eps_range[0], eps_range[1], n_eps)

    rows = []
    for f_ghz in freqs_ghz:
        for h_mm in heights_mm:
            for eps in eps_values:
                result = patch_dimensions(f_ghz * 1e9, eps, h_mm * 1e-3)
                if result is None:
                    continue
                W_m, L_m = result
                W_mm = W_m * 1e3
                L_mm = L_m * 1e3
                # Substrate: add 6 × h clearance on every side (standard rule of thumb)
                margin_mm = 6.0 * h_mm
                rows.append(
                    {
                        "Frequency(GHz)": round(f_ghz, 4),
                        "Substrate_Height(mm)": round(h_mm, 4),
                        "Dielectric_Constant": round(eps, 4),
                        "Patch_Length(mm)": round(L_mm, 4),
                        "Patch_Width(mm)": round(W_mm, 4),
                        "Substrate_Length(mm)": round(L_mm + margin_mm, 4),
                        "Substrate_Width(mm)": round(W_mm + margin_mm, 4),
                    }
                )

    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("Generating patch antenna dataset …")
    df = generate()
    print(f"Generated {len(df)} valid designs")
    print(f"Frequency range : {df['Frequency(GHz)'].min()} – {df['Frequency(GHz)'].max()} GHz")
    print(f"Patch length    : {df['Patch_Length(mm)'].min():.1f} – {df['Patch_Length(mm)'].max():.1f} mm")
    print(f"Patch width     : {df['Patch_Width(mm)'].min():.1f} – {df['Patch_Width(mm)'].max():.1f} mm")
    output = "dataset_patch_antenna.csv"
    df.to_csv(output, index=False)
    print(f"Saved → {output}")
