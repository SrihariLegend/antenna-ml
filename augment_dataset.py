"""Multi-fidelity data augmentation for the antenna-ml dataset.

Generates synthetic S11 samples at intermediate substrate heights by
interpolating between the CST-simulated heights (0.8, 2.0, 3.2 mm), then
enriches every row with physics-derived antenna dimensions so the dataset
matches the full ``config.EXPECTED_COLUMNS`` schema.

Methodology (citable as "physics-informed multi-fidelity augmentation"):
  - For each (freq, patch_length) pair present in the CST data, we have
    S11 at three substrate heights.
  - We fit a quadratic through those three points and evaluate it at the
    requested intermediate heights.
  - Quadratic is chosen because S11 vs h is governed by smooth EM
    relationships (effective permittivity, fringing fields) that are
    well-approximated by low-order polynomials over small ranges.
  - After interpolation, four additional columns derived from the Balanis
    analytical model and the CST geometry rules are appended:
      * ``width of patch in mm``  — Balanis analytical optimal patch width
        for each row's frequency and substrate permittivity (FR-4, εᵣ = 4.4).
      * ``substrate_length``      — 2 × patch_length (CST geometry rule).
      * ``substrate_width``       — 2 × CST patch width (76 mm, constant).
      * ``effective_er``          — effective dielectric constant accounting
        for fringing fields (Balanis formula).

References:
  Pietrenko-Dabrowska et al., "Two-stage variable-fidelity modeling of
  antennas with domain confinement," Sci. Rep. 12, 17275 (2022).
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

import balanis
import config

logger = logging.getLogger(__name__)

CST_HEIGHTS = np.array([0.8, 2.0, 3.2])


def _add_physics_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add physics-derived dimension columns to *df* in-place and return it.

    Computes:
      - ``width of patch in mm``  : Balanis analytical patch width.
      - ``substrate_length``      : 2 × patch_length.
      - ``substrate_width``       : 2 × CST_PATCH_WIDTH_MM (constant 76 mm).
      - ``effective_er``          : Balanis effective permittivity.
    """
    df["width of patch in mm"] = df["freq"].apply(balanis.patch_width)
    df["substrate_length"] = 2.0 * df["patch_length"]
    df["substrate_width"] = 2.0 * balanis.CST_PATCH_WIDTH_MM
    # Vectorised effective-er: avoids per-row Python overhead on large DataFrames.
    h = df["substrate_height"].values * 1e-3
    w = balanis.CST_PATCH_WIDTH_MM * 1e-3
    er = balanis.DEFAULT_ER
    df["effective_er"] = (er + 1) / 2 + (er - 1) / 2 / np.sqrt(1 + 12 * h / w)
    return df


def augment(
    cst_path: str = config.DATASET_PATH,
    new_heights: list[float] | None = None,
    output_path: str | None = None,
    clean_output_path: str | None = None,
) -> pd.DataFrame:
    """Return augmented dataset with synthetic intermediate heights and
    physics-derived dimension columns.

    Parameters
    ----------
    cst_path : path to the original CST dataset CSV.
    new_heights : substrate heights (mm) to synthesise.  Defaults to
        [1.0, 1.2, 1.4, 1.6, 1.8, 2.4, 2.8].
    output_path : if given, write the full combined CSV (including the
        ``synthetic`` provenance flag) here.
    clean_output_path : if given, write a clean CSV without the ``synthetic``
        column, with columns ordered to match ``config.EXPECTED_COLUMNS``.
        This is the file consumed by the model training pipeline.

    Returns
    -------
    Combined DataFrame (original + synthetic rows), with a boolean column
    ``synthetic`` indicating provenance, and all physics-derived columns.
    """
    if new_heights is None:
        new_heights = [1.0, 1.2, 1.4, 1.6, 1.8, 2.4, 2.8]

    df = pd.read_csv(cst_path)
    logger.info("Loaded CST data: %d rows, heights %s", len(df), sorted(df["substrate_height"].unique()))

    # Build lookup: (freq, patch_length) → array of S11 at CST_HEIGHTS
    grouped = df.groupby(["freq", "patch_length"])
    lookup: dict[tuple, np.ndarray] = {}
    for (freq, pl), grp in grouped:
        s11_by_h = {}
        for _, row in grp.iterrows():
            s11_by_h[row["substrate_height"]] = row["S11"]
        if all(h in s11_by_h for h in CST_HEIGHTS):
            lookup[(freq, pl)] = np.array([s11_by_h[h] for h in CST_HEIGHTS])

    logger.info("Interpolation grid: %d (freq, patch_length) pairs", len(lookup))

    # Generate synthetic rows via quadratic interpolation
    synth_rows = []
    for (freq, pl), s11_vals in lookup.items():
        # Fit degree-2 polynomial: S11 = a*h^2 + b*h + c
        coeffs = np.polyfit(CST_HEIGHTS, s11_vals, deg=2)
        for h_new in new_heights:
            s11_new = float(np.polyval(coeffs, h_new))
            # Clamp: S11 should be <= 0 dB
            s11_new = min(s11_new, 0.0)
            synth_rows.append({
                "freq": freq,
                "S11": s11_new,
                "patch_length": pl,
                "substrate_height": h_new,
            })

    df_synth = pd.DataFrame(synth_rows)
    logger.info("Generated %d synthetic rows at heights %s", len(df_synth), new_heights)

    # Tag provenance
    df["synthetic"] = False
    df_synth["synthetic"] = True
    combined = pd.concat([df, df_synth], ignore_index=True)
    combined.sort_values(["patch_length", "substrate_height", "freq"], inplace=True)
    combined.reset_index(drop=True, inplace=True)

    # Add physics-derived dimension columns
    _add_physics_columns(combined)
    logger.info("Added physics-derived columns: width of patch in mm, substrate_length, substrate_width, effective_er")

    if output_path:
        combined.to_csv(output_path, index=False)
        logger.info("Saved augmented dataset to %s (%d rows)", output_path, len(combined))

    if clean_output_path:
        # Emit only the EXPECTED_COLUMNS (no synthetic flag) in schema order.
        clean_cols = [c for c in config.EXPECTED_COLUMNS if c in combined.columns]
        missing_expected = [c for c in config.EXPECTED_COLUMNS if c not in combined.columns]
        if missing_expected:
            logger.warning(
                "Some EXPECTED_COLUMNS are missing from the augmented dataset and "
                "will be omitted from the clean output: %s",
                missing_expected,
            )
        clean_df = combined[clean_cols].copy()
        clean_df.to_csv(clean_output_path, index=False)
        logger.info(
            "Saved clean augmented dataset to %s (%d rows, columns: %s)",
            clean_output_path, len(clean_df), clean_cols,
        )

    return combined


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Augment CST dataset with interpolated substrate heights")
    parser.add_argument("--input", default=config.DATASET_PATH, help="CST dataset CSV")
    parser.add_argument("--output", default="augmented_dataset.csv", help="Output CSV path (includes synthetic flag)")
    parser.add_argument("--clean-output", default="augmented_dataset_clean.csv",
                        help="Clean output CSV path (schema-ordered, no synthetic flag)")
    parser.add_argument("--heights", nargs="+", type=float, default=None,
                        help="Intermediate heights to generate (mm)")
    args = parser.parse_args()
    combined = augment(args.input, args.heights, args.output, args.clean_output)
    print(f"\nDone. {len(combined)} total rows.")
    print(f"  Original (CST):  {(~combined['synthetic']).sum()}")
    print(f"  Synthetic:       {combined['synthetic'].sum()}")
    print(f"  Heights:         {sorted(combined['substrate_height'].unique())}")
    print(f"  Columns:         {list(combined.columns)}")


if __name__ == "__main__":
    main()
