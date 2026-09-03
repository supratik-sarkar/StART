"""Gate A slice A1 — context protocol, registry extension, target dispatch.

Each test names the property it defends or the incident it prevents.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from start.registry import TestContext
from start.registry.contexts import (
    DEFAULT_TOLERANCE,
    ContextKind,
    ContextMismatch,
    Determinism,
    ReviewContext,
    context_methods_for_test_context,
)
from start.tests._target_dispatch import (
    AMBIGUOUS_INTEGER_MAX_UNIQUE,
    TargetType,
    infer_target_type,
    require_target_type,
)


def _frame(target, n=200, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    df["y"] = target
    return df


def _ctx(target, extra=None, n=200):
    df = _frame(target, n=n)
    return TestContext(train=df, test=df.copy(), target_column="y", extra=extra or {})


# --------------------------------------------------------------- protocol --
def test_test_context_satisfies_protocol_without_field_changes():
    """The whole point of a Protocol over a base class: zero change to the model."""
    assert isinstance(_ctx(np.zeros(200, dtype=int)), ReviewContext)


def test_context_kind_is_tabular():
    assert _ctx(np.zeros(200, dtype=int)).context_kind() == ContextKind.TABULAR.value


def test_describe_carries_shapes_not_data():
    """A describe() that carried frames would leak data into logs and evidence."""
    block = _ctx(np.zeros(200, dtype=int)).describe()
    assert block["train_shape"] == (200, 3)
    for value in block.values():
        assert not isinstance(value, pd.DataFrame)


def test_validate_reports_violations_rather_than_raising():
    """A test whose context is unusable must SKIP, not abort the review."""
    assert TestContext(train=None, target_column="y").validate_context()
    df = pd.DataFrame({"a": [1, 2], "y": [0, 1]})
    violations = TestContext(train=df, target_column="nope").validate_context()
    assert any("not a column" in v for v in violations)


def test_clean_context_has_no_violations():
    assert _ctx(np.zeros(200, dtype=int)).validate_context() == []


def test_missing_test_columns_are_detected():
    train = _frame(np.zeros(200, dtype=int))
    test = train.drop(columns=["b"])
    violations = TestContext(train=train, test=test, target_column="y").validate_context()
    assert any("missing" in v for v in violations)


def test_methods_are_additive_only():
    """No field was added, renamed or removed on TestContext."""
    added = set(context_methods_for_test_context())
    assert added == {"context_kind", "describe", "validate_context"}
    ctx = _ctx(np.zeros(200, dtype=int))
    for field in ("train", "test", "target_column", "prediction_column", "score_column",
                  "timestamp_column", "entity_id_column", "model", "seed", "extra"):
        assert hasattr(ctx, field)


# ------------------------------------------------------------- determinism --
def test_determinism_classes_are_the_three_declared():
    assert {d.value for d in Determinism} == {"exact", "seeded", "numerical"}


def test_tolerances_distinguish_closed_form_from_iterative():
    """One tolerance for both would be wrong for both."""
    assert DEFAULT_TOLERANCE["closed_form"] < DEFAULT_TOLERANCE["iterative"]


def test_context_mismatch_is_a_typed_error():
    assert issubclass(ContextMismatch, TypeError)


# ---------------------------------------------------------- target dispatch --
@pytest.mark.parametrize(
    "target,expected,confidence",
    [
        (np.random.default_rng(1).integers(0, 2, 200), TargetType.BINARY, "high"),
        (np.random.default_rng(2).choice(list("ABCD"), 200), TargetType.MULTICLASS, "high"),
        (np.random.default_rng(3).normal(size=200), TargetType.CONTINUOUS, "high"),
        (np.random.default_rng(4).integers(0, 9, 200), TargetType.MULTICLASS, "ambiguous"),
        (np.random.default_rng(5).integers(0, 500, 200), TargetType.CONTINUOUS, "high"),
    ],
)
def test_target_inference(target, expected, confidence):
    inference = infer_target_type(_ctx(target))
    assert inference.target_type is expected
    assert inference.confidence == confidence


def test_non_numeric_dtype_spelling_does_not_break_inference():
    """Regression: pandas 2.x reports 'string', 3.x reports 'str'. Matching on the
    exact spelling classified a string target as continuous."""
    ctx = _ctx(np.random.default_rng(6).choice(list("ABCD"), 200))
    ctx.train["y"] = ctx.train["y"].astype("category")
    assert infer_target_type(ctx).target_type is TargetType.MULTICLASS


def test_explicit_target_type_is_never_overridden():
    """A reviewer who has stated the target type outranks every heuristic."""
    counts = np.random.default_rng(7).integers(0, 9, 200)
    inference = infer_target_type(_ctx(counts, extra={"target_type": "continuous"}))
    assert inference.target_type is TargetType.CONTINUOUS
    assert inference.source == "explicit"


def test_invalid_explicit_target_type_is_rejected():
    with pytest.raises(ValueError, match="not recognised"):
        infer_target_type(_ctx(np.zeros(200, dtype=int), extra={"target_type": "regression"}))


def test_ambiguous_integer_announces_itself():
    """A count target routed to multiclass would be handed AUC and IV, which mean
    nothing for it — and the numbers would look perfectly reasonable."""
    inference = infer_target_type(_ctx(np.random.default_rng(8).integers(0, 9, 200)))
    assert inference.is_ambiguous
    assert "count-regression" in inference.detail
    assert "target_type" in inference.detail  # tells the reviewer how to override


def test_absent_target_is_not_an_error():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    inference = infer_target_type(TestContext(train=df))
    assert inference.target_type is TargetType.NONE
    assert inference.source == "absent"


def test_single_valued_target_is_none():
    inference = infer_target_type(_ctx(np.ones(200, dtype=int)))
    assert inference.target_type is TargetType.NONE


def test_guard_skips_on_unsupported_target():
    inference, skip = require_target_type(_ctx(np.random.default_rng(9).normal(size=200)), "binary")
    assert skip is not None
    assert skip.status == "skipped"
    assert "continuous" in skip.interpretation
    assert skip.limitations


def test_guard_proceeds_on_supported_target():
    inference, skip = require_target_type(
        _ctx(np.random.default_rng(10).integers(0, 2, 200)), "binary", "multiclass"
    )
    assert skip is None
    assert inference.target_type is TargetType.BINARY


def test_guard_records_the_inference_in_params():
    """The inference must be visible in evidence, not discovered from a nonsensical IV."""
    inference, _ = require_target_type(_ctx(np.random.default_rng(11).integers(0, 2, 200)), "binary")
    params = inference.as_params()
    assert set(params) == {
        "target_type_inferred", "target_type_source",
        "target_type_confidence", "target_n_unique",
    }


def test_absent_target_skips_with_a_useful_reason():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    _, skip = require_target_type(TestContext(train=df), "binary")
    assert skip is not None
    assert "no usable target column" in skip.interpretation


def test_ambiguity_boundary_is_explicit():
    """One above the boundary is continuous; at the boundary it is ambiguous."""
    n = 400
    rng = np.random.default_rng(12)
    at = rng.integers(0, AMBIGUOUS_INTEGER_MAX_UNIQUE, n)
    above = rng.integers(0, AMBIGUOUS_INTEGER_MAX_UNIQUE + 60, n)
    assert infer_target_type(_ctx(at, n=n)).is_ambiguous
    assert infer_target_type(_ctx(above, n=n)).target_type is TargetType.CONTINUOUS
