from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def toy_frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 500
    df = pd.DataFrame(
        {
            "a": rng.normal(0, 1, n),
            "b": rng.gamma(2, 2, n),
            "c": rng.integers(0, 5, n).astype(float),
        }
    )
    df["target"] = (df["a"] + rng.normal(0, 1, n) > 0).astype(int)
    df["score"] = 1 / (1 + np.exp(-(df["a"] + rng.normal(0, 0.5, n))))
    df.loc[:9, "b"] = np.nan
    return df


# --------------------------------------------------------------------------- #
# Quarantine
# --------------------------------------------------------------------------- #
# One list, read here, so that local runs, core-ci, the scheduled workflows and
# release certification all exclude exactly the same modules. Previously the
# exclusions lived as --ignore flags inside one workflow only, and the scheduled
# workflows failed nightly on tests the pull-request gate never ran.
#
# Set START_RUN_QUARANTINED=1 to run quarantined modules deliberately.
_QUARANTINE_FILE = Path(__file__).with_name("quarantine.txt")


def _read_quarantine() -> list[str]:
    if os.environ.get("START_RUN_QUARANTINED") == "1" or not _QUARANTINE_FILE.exists():
        return []
    names: list[str] = []
    for raw in _QUARANTINE_FILE.read_text(encoding="utf-8").splitlines():
        entry = raw.split("#", 1)[0].strip()
        if entry:
            # collect_ignore paths are relative to this conftest's directory.
            names.append(entry.removeprefix("tests/"))
    return names


collect_ignore = _read_quarantine()
