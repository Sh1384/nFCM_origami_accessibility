"""Filesystem layout for the nFCM origami accessibility analysis.

Raw flow cytometry data is not stored in this repository. It is archived
separately (see the data link in README.md) and must be downloaded before any
script that reads FCS files will run.

Point the code at your local copy in either of two ways:

1.  Set the environment variable ``NFCM_DATA_ROOT``::

        export NFCM_DATA_ROOT=/Volumes/external/nFCM_raw

2.  Or place the acquisition folders inside ``data/`` at the repository root,
    which is the default and is git-ignored.

Every path helper below resolves against that single root, so no script in this
repository contains an absolute path.
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------- repository
#: Repository root, resolved from this file's location.
REPO_ROOT: Path = Path(__file__).resolve().parent.parent

CONFIG_DIR: Path = REPO_ROOT / "config"
RESULTS_DIR: Path = REPO_ROOT / "results"
FIGURES_DIR: Path = REPO_ROOT / "figures"

#: Metrics and statistics quoted in the main text and its tables.
RESULTS_MAIN: Path = RESULTS_DIR / "main"
#: Tables reported in the Supporting Information.
RESULTS_SI: Path = RESULTS_DIR / "si"
#: Per-sample intermediates that several figures read but the paper does not
#: tabulate directly, such as acquisition-rate and distribution-shape summaries.
RESULTS_INTERMEDIATE: Path = RESULTS_DIR / "intermediate"
FIGURES_MAIN: Path = FIGURES_DIR / "main"
FIGURES_SI: Path = FIGURES_DIR / "supplementary"

#: Cache for intermediate feature matrices and trained models. Git-ignored.
CACHE_DIR: Path = REPO_ROOT / "cache"


# ---------------------------------------------------------------- raw data
def data_root() -> Path:
    """Return the directory holding the raw acquisition folders.

    Resolution order is ``$NFCM_DATA_ROOT`` then ``<repo>/data``. The directory
    is not required to exist for imports to succeed, so that scripts which only
    read cached results still work without the raw archive.
    """
    env = os.environ.get("NFCM_DATA_ROOT")
    return Path(env).expanduser().resolve() if env else REPO_ROOT / "data"


#: Acquisition folders, keyed by the role each plays in the analysis.
#:
#: The values are the original instrument folder names as they appear in the
#: archived dataset. Keeping them verbatim means a downloaded copy works with no
#: renaming, while the keys give call sites a name that says what the data is.
DATASETS: dict[str, str] = {
    # Main experiment 1: streptavidin-AF488 probe, 3 coatings x 4 extensions x 3 reps.
    "streptavidin": "Shervin5_25",
    # Main experiment 2: anti-biotin IgG-AF488 probe, same design.
    # The archive nests this folder one level deeper than the others.
    "antibody": "Shervin4_18/Shervin4_18",
    # Coating chemistry with no origami, used as the matched debris reference.
    "coating_only": "Shervin2_13_2026",
    # Independent session used only to test cross-session generalisation.
    "cross_session": "Shervin5_03",
}


def dataset_dir(name: str) -> Path:
    """Return the directory for one acquisition session.

    Parameters
    ----------
    name
        A key of :data:`DATASETS`, for example ``"streptavidin"``.
    """
    try:
        return data_root() / DATASETS[name]
    except KeyError:
        raise KeyError(
            f"unknown dataset {name!r}; expected one of {sorted(DATASETS)}"
        ) from None


def require_data_root() -> Path:
    """Return the raw data root, raising a helpful error if it is absent.

    Call this at the top of any script that reads FCS files so the failure is a
    clear instruction rather than a missing-file traceback deep in a loop.
    """
    root = data_root()
    if not root.is_dir():
        raise FileNotFoundError(
            f"raw data directory not found: {root}\n"
            "Download the archive linked in README.md, then either set "
            "NFCM_DATA_ROOT to its location or place the acquisition folders "
            f"in {REPO_ROOT / 'data'}."
        )
    return root
