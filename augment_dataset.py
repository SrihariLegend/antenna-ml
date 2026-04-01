"""Multi-fidelity data augmentation for the antenna-ml dataset.

Generates synthetic S11 samples at intermediate substrate heights by
interpolating between the CST-simulated heights (0.8, 2.0, 3.2 mm).

Methodology (citable as "physics-informed multi-fidelity augmentation"):
  - For each (freq, patch_length) pair present in the CST data, we have
    S11 at three substrate heights.
  - We fit a quadratic through those three points and evaluate it at the
    requested intermediate heights.
  - Quadratic is chosen because S11 vs h is governed by smooth EM
    relationships (effective permittivity, fringing fields) that are
    well-approximated by low-order polynomials over small ranges.

References:
  Pietrenko-Dabrowska et al., "Two-stage variable-fidelity modeling of
  antennas with domain confinement," Sci. Rep. 12, 17275 (2022).
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)

CST_HEIGHTS = np.array([0.8, 2.0, 3.2])


def augment(
    cst_path: str = config.DATASET_PATH,
    new_heights: list[float] | None = None,
    output_path: str | None = None,
) -> pd.DataFrame:
    """Return augmented dataset with synthetic intermediate heights.

    Parameters
    ----------
    cst_path : path to the original CST dataset CSV.
    new_heights : substrate heights (mm) to synthesise.  Defaults to
        [1.0, 1.2, 1.4, 1.6, 1.8, 2.4, 2.8].
    output_path : if given, write the combined CSV here.

    Returns
    -------
    Combined DataFrame (original + synthetic rows), with a boolean column
    ``synthetic`` indicating provenance.
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

    if output_path:
        combined.to_csv(output_path, index=False)
        logger.info("Saved augmented dataset to %s (%d rows)", output_path, len(combined))

    return combined


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Augment CST dataset with interpolated substrate heights")
    parser.add_argument("--input", default=config.DATASET_PATH, help="CST dataset CSV")
    parser.add_argument("--output", default="augmented_dataset.csv", help="Output CSV path")
    parser.add_argument("--heights", nargs="+", type=float, default=None,
                        help="Intermediate heights to generate (mm)")
    args = parser.parse_args()
    combined = augment(args.input, args.heights, args.output)
    print(f"\nDone. {len(combined)} total rows.")
    print(f"  Original (CST):  {(~combined['synthetic']).sum()}")
    print(f"  Synthetic:       {combined['synthetic'].sum()}")
    print(f"  Heights:         {sorted(combined['substrate_height'].unique())}")


if __name__ == "__main__":
    main()
