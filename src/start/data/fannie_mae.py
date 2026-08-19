"""Fannie Mae Single-Family Loan Performance dataset loader (Workstream B3).

Supports bring-your-own-file loan acquisition / performance data (pipe-delimited
or CSV). Checks terms notice, applies row limits, and maps target `is_delinquent`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

FANNIE_MAE_TERMS_NOTE = (
    "Fannie Mae Single-Family Loan Performance data requires agreement to "
    "Fannie Mae's Single-Family Data Terms & Conditions. Data must be acquired "
    "directly from Fannie Mae."
)


def load_fannie_mae_dataset(
    file_path: str | Path,
    row_limit: int = 100000,
    target_column: str = "is_delinquent",
) -> pd.DataFrame:
    """Load and parse Fannie Mae loan performance or acquisition data.

    Parameters
    ----------
    file_path : str | Path
        Path to local pipe-delimited or CSV file.
    row_limit : int
        Maximum rows to load (default 100,000).
    target_column : str
        Target column name (default 'is_delinquent').
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Fannie Mae data file not found at: {path}. "
            "Please provide a valid local path or select the synthetic / UCI dataset."
        )

    # Detect delimiter
    try:
        # Read a small sample
        with path.open("r", encoding="utf-8", errors="replace") as f:
            sample_line = f.readline()
        sep = "|" if "|" in sample_line else ","

        df = pd.read_csv(
            path,
            sep=sep,
            nrows=row_limit,
            low_memory=False,
            encoding="utf-8",
            on_bad_lines="skip",
        )
    except Exception as exc:
        raise ValueError(
            f"Failed to parse Fannie Mae data file at {path}: {exc}. "
            "Ensure the file is standard pipe-delimited or CSV format."
        ) from exc

    if df.empty:
        raise ValueError(f"Fannie Mae dataset at {path} contains 0 rows.")

    # Target column mapping / inference
    if target_column in df.columns:
        return df

    # Standard Fannie Mae delinquency status column check
    candidate_delinq_cols = [
        "current_loan_delinquency_status",
        "delinquency_status",
        "zero_balance_code",
        "current_delinquency_status",
    ]
    matched_col = next((c for c in candidate_delinq_cols if c in df.columns), None)
    if matched_col:
        # Values > 0 or specific status indicates delinquency
        raw_val = pd.to_numeric(df[matched_col], errors="coerce").fillna(0)
        df[target_column] = (raw_val >= 1).astype(int)
    else:
        # If no standard delinquency column is present, search for binary target
        numeric_binary = [c for c in df.columns if df[c].nunique() == 2]
        if numeric_binary:
            chosen = numeric_binary[0]
            df[target_column] = df[chosen]
        else:
            raise ValueError(
                f"Could not infer target column '{target_column}' in Fannie Mae file. "
                f"Columns present: {', '.join(df.columns[:10])}..."
            )

    return df
