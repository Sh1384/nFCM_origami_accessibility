"""Reading FCS files and resolving the sample naming convention.

Every acquisition in the two main sessions follows one filename pattern::

    {extension}_{coating}_{probe_tag}_{replicate}.fcs

for example ``6_Silica_S_B.fcs`` is the 6 nt biotin extension on silica-coated
origami, incubated with streptavidin, replicate B. The ``probe_tag`` differs
per session (``S``/``NoS`` for streptavidin, ``AB``/``NoAB`` for antibody),
which is why call sites go through :func:`sample_path` rather than formatting
filenames by hand.
"""
from __future__ import annotations

import functools
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from fcsparser import parse

# The instrument writes a vendor block into the FCS ANALYSIS segment that
# fcsparser cannot interpret. It holds acquisition metadata only, never event
# data, so the warning is noise on every single file read.
warnings.filterwarnings(
    "ignore",
    message=r"There appears to be some information in the ANALYSIS segment",
    module="fcsparser.*",
)

from .constants import CH_SS, DEBRIS_FILES, SESSIONS
from .paths import dataset_dir


def read_fcs(path: str | Path) -> pd.DataFrame:
    """Read one FCS file and return its events as a DataFrame.

    One row is one detected particle; one column is one detector channel.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"FCS file not found: {path}")
    _, df = parse(str(path), reformat_meta=True)
    return df


@functools.lru_cache(maxsize=256)
def _read_fcs_cached(path: str) -> pd.DataFrame:
    """Memoised :func:`read_fcs`.

    Several figures read the same control files a dozen times each. Caching on
    the resolved path turns a multi-minute figure build into a few seconds. The
    returned frame is shared, so treat it as read-only.
    """
    return read_fcs(path)


def sample_path(
    session: str, extension: str, coating: str, replicate: str, *, with_probe: bool
) -> Path:
    """Build the path to one acquisition.

    Parameters
    ----------
    session
        ``"streptavidin"`` or ``"antibody"``.
    extension
        One of ``"N"``, ``"3"``, ``"6"``, ``"12"``.
    coating
        One of ``"NoCoating"``, ``"PLL"``, ``"Silica"``.
    replicate
        ``"A"``, ``"B"``, or ``"C"``.
    with_probe
        ``True`` selects the probe-incubated acquisition, ``False`` the matched
        no-probe control from the same session.
    """
    try:
        cfg = SESSIONS[session]
    except KeyError:
        raise KeyError(
            f"unknown session {session!r}; expected one of {sorted(SESSIONS)}"
        ) from None
    tag = cfg["probe"] if with_probe else cfg["no_probe"]
    return dataset_dir(cfg["dataset"]) / f"{extension}_{coating}_{tag}_{replicate}.fcs"


def debris_path(session: str, coating: str) -> Path:
    """Return the matched particle-free reference for one coating.

    Uncoated origami is referenced to the buffer blank acquired in its own
    session, because with no coating chemistry present the only background is
    the buffer itself. The two coatings each have a dedicated acquisition of the
    coating chemistry with no origami, which is session-independent.
    """
    if coating in DEBRIS_FILES:
        key, fname = DEBRIS_FILES[coating]
        return dataset_dir(key) / fname
    return dataset_dir(SESSIONS[session]["dataset"]) / "Buffer.fcs"


def load_channel(
    path: str | Path,
    channel: str,
    *,
    ss_window: tuple[float, float] | None = None,
    ss_floor: float | None = None,
    positive_only: bool = True,
) -> np.ndarray:
    """Read one channel, optionally restricted by side scatter.

    Parameters
    ----------
    path
        FCS file to read.
    channel
        Column to return, for example ``"V525-H"``.
    ss_window
        Inclusive ``(low, high)`` bounds on :data:`~nfcm.constants.CH_SS`.
        Mutually exclusive with ``ss_floor``.
    ss_floor
        Lower bound only. Used when deriving the side-scatter window itself,
        where no window exists yet.
    positive_only
        Drop non-positive values. Log-scale plots and log-ratio metrics require
        this; the instrument writes occasional zero or negative baselines.
    """
    if ss_window is not None and ss_floor is not None:
        raise ValueError("pass ss_window or ss_floor, not both")

    df = _read_fcs_cached(str(Path(path)))
    if ss_window is not None:
        low, high = ss_window
        df = df[(df[CH_SS] >= low) & (df[CH_SS] <= high)]
    elif ss_floor is not None:
        df = df[df[CH_SS] >= ss_floor]

    values = df[channel].to_numpy()
    return values[values > 0] if positive_only else values


def acquisition_summary(path: str | Path) -> dict[str, float]:
    """Return acquisition duration, event count, and mean event rate.

    The ``Time`` channel is recorded in seconds, so the duration is the span of
    that channel and the rate follows directly.
    """
    df = read_fcs(path)
    time = df["Time"].to_numpy()
    duration = float(time.max() - time.min())
    n = int(len(df))
    return {
        "n_events": n,
        "duration_s": duration,
        "events_per_s": n / duration if duration > 0 else float("nan"),
    }
