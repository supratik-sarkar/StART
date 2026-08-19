"""Risk-stripe-agnostic review core.

Three orthogonal axes, deliberately kept separate:

    stripe     what risk is being borne          start.risk.stripes
    object     what artefact bears it            start.risk.objects
    dimension  what a reviewer must establish    start.risk.dimensions

and two things computed from them:

    plan       deterministic, hash-stable scope  start.risk.plan
    controls   traceability to public frameworks start.risk.controls

Everything in this subpackage is standard-library only. It imports and runs in
an environment with nothing pip-installed, which is what makes it usable as the
skeleton of a review inside a locked-down environment where the scientific
stack may not be available.

Quick start::

    from start.risk import RiskObject, synthesise_plan

    obj = RiskObject(object_id="M-1042", kind="vendor_model", materiality="high")
    plan = synthesise_plan(stripe_id="financial_crime", obj=obj)
    print("\\n".join(plan.summary_lines()))
    print(plan.plan_hash())
"""

from start.risk.controls import (
    MAPPING_VERSION,
    ControlFramework,
    Expectation,
    coverage_report,
    framework,
    framework_ids,
)
from start.risk.dimensions import (
    DIMENSIONS,
    Applicability,
    Dimension,
    dimension,
    dimension_ids,
)
from start.risk.objects import (
    OBJECT_KINDS,
    CapabilityProfile,
    DimensionVerdict,
    RiskObject,
    RiskObjectKind,
    applicability,
    object_kind,
    object_kind_ids,
)
from start.risk.plan import (
    MATERIALITY_LEVELS,
    PlannedDimension,
    ReviewPlan,
    synthesise_plan,
)
from start.risk.stripes import (
    STRIPES,
    RiskStripe,
    load_entry_point_stripes,
    register_stripe,
    stripe,
    stripe_ids,
)

__all__ = [
    # dimensions
    "Applicability",
    "Dimension",
    "DIMENSIONS",
    "dimension",
    "dimension_ids",
    # objects
    "CapabilityProfile",
    "RiskObject",
    "RiskObjectKind",
    "OBJECT_KINDS",
    "DimensionVerdict",
    "applicability",
    "object_kind",
    "object_kind_ids",
    # stripes
    "RiskStripe",
    "STRIPES",
    "stripe",
    "stripe_ids",
    "register_stripe",
    "load_entry_point_stripes",
    # controls
    "ControlFramework",
    "Expectation",
    "framework",
    "framework_ids",
    "coverage_report",
    "MAPPING_VERSION",
    # plan
    "ReviewPlan",
    "PlannedDimension",
    "synthesise_plan",
    "MATERIALITY_LEVELS",
]
