"""Deterministic test registry.

Tests are pure functions registered with :func:`register_test`. They receive a
:class:`TestContext` plus keyword params and return a ``TestResult``. The
agentic layer may *choose* and *sequence* tests, but never computes metrics.

External test packs can extend StART by exposing a ``start.test_packs`` entry
point whose target is a module that registers tests on import.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from importlib import import_module, metadata
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from start.core.schemas import TestResult
from start.registry.contexts import context_methods_for_test_context


class TestContext(BaseModel):
    """Bag of artifacts available to a test engine."""

    __test__ = False  # not a pytest class
    model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())

    train: Any = None  # pandas DataFrame
    test: Any = None  # pandas DataFrame
    target_column: str | None = None
    prediction_column: str | None = None
    score_column: str | None = None
    timestamp_column: str | None = None
    entity_id_column: str | None = None
    model: Any = None
    seed: int = 42
    extra: dict[str, Any] = Field(default_factory=dict)


for _name, _fn in context_methods_for_test_context().items():
    setattr(TestContext, _name, _fn)


TestFn = Callable[..., TestResult]


@dataclass(frozen=True)
class TestSpec:
    __test__ = False

    test_id: str
    family: str
    name: str
    fn: TestFn
    description: str = ""
    requires: tuple[str, ...] = field(default_factory=tuple)
    default_params: dict[str, Any] = field(default_factory=dict)
    context_type: str = "tabular"
    risk_stripes: tuple[str, ...] = ()
    risk_dimensions: tuple[str, ...] = ()
    object_kinds: tuple[str, ...] = ()


_REGISTRY: dict[str, TestSpec] = {}
_BUILTIN_FAMILY_MODULES = (
    "start.tests.preprocessing",
    "start.tests.eda",
    "start.tests.feature_engineering",
    "start.tests.portfolio",
    "start.tests.attribution",
    "start.tests.traded_risk",
    "start.tests.covariance",
    "start.tests.supervised",
    "start.tests.xai",
    "start.tests.genai",
)
_loaded = False


def register_test(
    test_id: str,
    *,
    family: str,
    name: str | None = None,
    description: str = "",
    requires: tuple[str, ...] = (),
    default_params: dict[str, Any] | None = None,
    context_type: str = "tabular",
    risk_stripes: tuple[str, ...] = (),
    risk_dimensions: tuple[str, ...] = (),
    object_kinds: tuple[str, ...] = (),
) -> Callable[[TestFn], TestFn]:
    def decorator(fn: TestFn) -> TestFn:
        if test_id in _REGISTRY:
            raise ValueError(f"Duplicate test_id: {test_id}")
        _REGISTRY[test_id] = TestSpec(
            test_id=test_id,
            family=family,
            name=name or test_id,
            fn=fn,
            description=description or (fn.__doc__ or "").strip().split("\n")[0],
            requires=requires,
            default_params=default_params or {},
            context_type=context_type,
            risk_stripes=risk_stripes,
            risk_dimensions=risk_dimensions,
            object_kinds=object_kinds,
        )
        return fn

    return decorator


def load_builtin_tests() -> None:
    global _loaded
    if _loaded:
        return
    for module in _BUILTIN_FAMILY_MODULES:
        import_module(module)
    _load_entry_point_packs()
    # F1 (v4.2.0 Gate A closeout): attach risk metadata to the 19 pre-v4.2.0
    # registrations. Declared as data in legacy_metadata.py and applied here so the
    # legacy family modules are not edited at all. Metadata only — no test function,
    # parameter, threshold or result behaviour is touched.
    from start.registry.legacy_metadata import apply_legacy_metadata

    apply_legacy_metadata()
    _loaded = True


def _load_entry_point_packs() -> None:
    try:
        eps = metadata.entry_points(group="start.test_packs")
    except Exception:  # pragma: no cover - defensive
        return
    for ep in eps:
        try:
            ep.load()
        except Exception:  # pragma: no cover - plugin failures must not break core
            continue


def get_test(test_id: str) -> TestSpec:
    load_builtin_tests()
    if test_id not in _REGISTRY:
        raise KeyError(f"Unknown test_id: {test_id}")
    return _REGISTRY[test_id]


def list_tests(family: str | None = None) -> list[TestSpec]:
    load_builtin_tests()
    specs = sorted(_REGISTRY.values(), key=lambda s: s.test_id)
    return [s for s in specs if family is None or s.family == family]


def list_families() -> list[str]:
    load_builtin_tests()
    return sorted({s.family for s in _REGISTRY.values()})
