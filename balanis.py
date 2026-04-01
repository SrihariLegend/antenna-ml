"""Antenna dimension helpers matching the CST simulation geometry rules.

The CST dataset was generated with fixed rules:
  - Patch width:       38 mm (constant across all designs)
  - Substrate length:  2 × patch_length
  - Substrate width:   2 × patch_width  (= 76 mm)

The Balanis analytical functions are kept for reference / future use.
"""

import math

C = 299_792_458.0  # speed of light m/s
DEFAULT_ER = 4.4   # FR-4

# --- CST dataset geometry constants ---
CST_PATCH_WIDTH_MM = 38.0


def patch_width(freq_ghz: float, er: float = DEFAULT_ER) -> float:
    """Balanis analytical patch width (mm). Kept for reference."""
    f = freq_ghz * 1e9
    return (C / (2 * f) * math.sqrt(2 / (er + 1))) * 1e3


def effective_er(er: float, h_mm: float, w_mm: float) -> float:
    """Effective dielectric constant accounting for fringing."""
    h = h_mm * 1e-3
    w = w_mm * 1e-3
    return (er + 1) / 2 + (er - 1) / 2 * 1 / math.sqrt(1 + 12 * h / w)


def full_dimensions(freq_ghz: float, patch_length_mm: float, substrate_height_mm: float, er: float = DEFAULT_ER) -> dict:
    """Compute all antenna dimensions matching the CST simulation rules.

    Returns dict with keys: patch_length_mm, patch_width_mm,
    substrate_height_mm, substrate_width_mm, substrate_length_mm.
    """
    pw = CST_PATCH_WIDTH_MM
    sl = 2 * patch_length_mm
    sw = 2 * pw
    return {
        "patch_length_mm": patch_length_mm,
        "patch_width_mm": pw,
        "substrate_height_mm": substrate_height_mm,
        "substrate_length_mm": sl,
        "substrate_width_mm": sw,
    }
