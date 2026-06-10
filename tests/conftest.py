from __future__ import annotations

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
