#!/usr/bin/env python3
"""Build the per-event feature matrices the coating classifiers train on.

Both classifiers ask the same question: given one particle's scatter signature,
which coating is it wearing? The features therefore come from the no-probe
control acquisitions, where nothing but the coating differs between conditions.

Two matrices are produced, cached as Parquet under ``cache/``:

``coating_features_ss_full.parquet``
    The nine side-scatter axes only. This is the matrix that matters
    scientifically, because side scatter needs no fluorescent label and so is
    the only signature available on an unknown sample.

``coating_features_3coat_fluor.parquet``
    The same nine axes plus the two fluorescence channels. Used solely for the
    ablation that asks how much accuracy the labels would add, as an upper
    bound the label-free classifier is compared against.

Events are kept when side scatter clears the noise floor and every channel in
the matrix reads positive, since the features are log-transformed.

Run::

    python scripts/02_build_features.py [--force]

The build takes a few minutes and is skipped when the cache already exists.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from nfcm import constants as K
from nfcm.io import read_fcs, sample_path
from nfcm.paths import CACHE_DIR, require_data_root

#: The no-probe controls are drawn from the streptavidin session, which has the
#: deepest acquisitions of the two and so gives the classifiers the most events.
FEATURE_SESSION = "streptavidin"

MATRICES = {
    "coating_features_ss_full.parquet": K.SS_AXES,
    "coating_features_3coat_fluor.parquet": K.SS_AXES + K.FLUOR_AXES,
}


def build(channels: list[str]) -> pd.DataFrame:
    """Assemble a log-transformed feature matrix over the given channels."""
    frames = []
    for coating in K.COATINGS:
        for replicate in K.REPLICATES:
            path = sample_path(
                FEATURE_SESSION, "N", coating, replicate, with_probe=False
            )
            if not path.exists():
                print(f"  {coating:<10} {replicate}: missing, skipped")
                continue

            events = read_fcs(path)
            events = events[events[K.CH_SS] >= K.SS_NOISE_FLOOR]
            # Every feature is log-transformed, so a single non-positive
            # reading disqualifies the whole event.
            events = events[(events[channels] > 0).all(axis=1)]

            frame = pd.DataFrame(
                {f"log_{c}": np.log10(events[c].to_numpy()) for c in channels}
            )
            frame["Coating"] = coating
            frame["Replicate"] = replicate
            frames.append(frame)
            print(f"  {coating:<10} {replicate}: {len(frame):>8,} events")

    if not frames:
        raise FileNotFoundError("no no-probe control files found")

    matrix = pd.concat(frames, ignore_index=True)
    matrix["Coating"] = matrix["Coating"].astype("category")
    matrix["Replicate"] = matrix["Replicate"].astype("category")
    return matrix


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true", help="rebuild even if the cache exists"
    )
    args = parser.parse_args()

    require_data_root()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    for filename, channels in MATRICES.items():
        out = CACHE_DIR / filename
        if out.exists() and not args.force:
            cached = pd.read_parquet(out)
            print(f"{filename}: cached, {len(cached):,} events (--force to rebuild)")
            continue

        print(f"\n{filename}  ({len(channels)} channels)")
        matrix = build(channels)
        matrix.to_parquet(out, index=False)
        print(f"  total {len(matrix):,} events across "
              f"{matrix.Coating.nunique()} coatings -> {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
