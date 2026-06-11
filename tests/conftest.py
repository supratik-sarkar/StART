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


class FakeLLM:
    name = "fake"

    @property
    def available(self):
        return True

    def __init__(self, responses=None):
        self.responses = list(responses or ["Synthetic check passed with AUC 0.91 [EV-TEST-0001]."])
        self.calls = []
        self.prompts = []

    def complete(self, system, user, *, max_tokens=1024):
        self.prompts.append((system, user, max_tokens))
        if hasattr(self, "calls"):
            self.calls.append((system, user))
        if self.responses:
            return self.responses.pop(0)
        return "Synthetic check passed with AUC 0.91 [EV-TEST-0001]."

    def generate(self, prompt, *, system=None, metadata=None):
        max_tokens = 1024
        if isinstance(metadata, dict):
            max_tokens = metadata.get("max_tokens", 1024)
        return self.complete(system or "", prompt, max_tokens=max_tokens)
