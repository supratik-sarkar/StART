#!/usr/bin/env python3
"""StART demonstration flight — the script behind the README recording.

Runs the full arc in about two minutes:

    profile detection  ->  risk plan synthesis  ->  deterministic evidence
    ->  narrative invariance  ->  disclosure envelope  ->  seal  ->  verify

Two properties make this recordable rather than merely runnable.

**It works with no API keys.** Without credentials it runs the deterministic
path on both sides of the invariance check, which trivially passes — and says
so, rather than implying an LLM was involved. With ``--provider openai`` (or
``anthropic``, ``deepseek``, ``gateway``) it routes the narrative through a real
model and the invariance check becomes a real test. Anyone who clones the repo
sees the same flow; only the interesting part requires a key.

**Timing is deterministic.** ``--speed`` scales every pause by a fixed factor,
so a re-recording lands on the same beats. ``--speed 0`` disables pauses
entirely for CI.

Standard library only — it must run before anyone has installed the modelling
stack, because the first thing a visitor does is run it.

    python scripts/demo_flight.py
    python scripts/demo_flight.py --provider openai --model gpt-4.1-mini
    python scripts/demo_flight.py --speed 0 --no-clear      # CI / smoke test
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from start.attestation import (  # noqa: E402
    POLICIES,
    attest_narrative_invariance,
    build_envelope,
    build_seal,
    policy_for,
    verify_prompt_covered,
    verify_seal,
)
from start.risk import RiskObject, synthesise_plan  # noqa: E402
from start.runtime_profile import (  # noqa: E402
    ProfileViolation,
    assert_provider_allowed,
    egress_policy,
    profile_manifest,
)

SPEED = 1.0
WIDTH = min(shutil.get_terminal_size((88, 24)).columns, 92)

_C = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "red": "\033[31m",
    "cyan": "\033[36m",
    "magenta": "\033[35m",
}


def c(text: str, *styles: str) -> str:
    if not sys.stdout.isatty() or os.environ.get("NO_COLOR"):
        return text
    return "".join(_C[s] for s in styles) + text + _C["reset"]


def pause(seconds: float) -> None:
    if SPEED > 0:
        time.sleep(seconds * SPEED)


def type_out(text: str, delay: float = 0.012) -> None:
    """Character-by-character output. Reads as live work on a recording."""
    if SPEED <= 0:
        print(text)
        return
    for ch in text:
        sys.stdout.write(ch)
        sys.stdout.flush()
        time.sleep(delay * SPEED)
    print()


def act(number: int, title: str) -> None:
    print()
    print(c("━" * WIDTH, "cyan"))
    print(c(f"  ACT {number}  ·  {title}", "cyan", "bold"))
    print(c("━" * WIDTH, "cyan"))
    pause(0.6)


def note(text: str) -> None:
    for line in _wrap(text, WIDTH - 4):
        print(c(f"  {line}", "dim"))


def _wrap(text: str, width: int) -> list[str]:
    words, lines, current = text.split(), [], ""
    for word in words:
        if len(current) + len(word) + 1 > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines


# --------------------------------------------------------------------------- #
# The evidence the demo reasons over.
#
# Deliberately synthetic and deliberately small: the point of the demo is the
# machinery around the numbers, and a reviewer watching a recording should be
# able to check the arithmetic in their head.
# --------------------------------------------------------------------------- #
EVIDENCE = [
    {
        "evidence_id": "EV-7f3a1c8b0d21",
        "test_id": "supervised.cohort_metrics_comparison",
        "test_name": "Cohort metric comparison",
        "status": "warn",
        "metrics": {"train_auc": 0.8421, "test_auc": 0.7714, "oos_auc": 0.7602, "gap": 0.0707},
        "thresholds": [{"metric": "gap", "warn": 0.05, "fail": 0.10, "direction": "upper"}],
        "interpretation": "Holdout degradation exceeds the warn threshold.",
    },
    {
        "evidence_id": "EV-2b9e4d7a6c05",
        "test_id": "supervised.calibration",
        "test_name": "Calibration",
        "status": "fail",
        "metrics": {"ece": 0.1382, "brier": 0.1041, "slope": 0.83},
        "thresholds": [{"metric": "ece", "warn": 0.05, "fail": 0.10, "direction": "upper"}],
        "interpretation": "Expected calibration error breaches the fail threshold.",
    },
    {
        "evidence_id": "EV-c04f81a2e937",
        "test_id": "preprocessing.population_stability",
        "test_name": "Population stability",
        "status": "pass",
        "metrics": {"max_psi": 0.0912, "features_above_010": 0},
        "thresholds": [{"metric": "max_psi", "warn": 0.10, "fail": 0.25, "direction": "upper"}],
        "interpretation": "No feature exceeds the PSI warning level.",
    },
]

DETERMINISTIC_NARRATIVE = (
    "Discriminatory power degrades across cohorts: train AUC 0.8421 against test AUC 0.7714 "
    "and out-of-sample 0.7602 [EV-7f3a1c8b0d21]. The resulting gap of 0.0707 exceeds the warn "
    "threshold of 0.05 but remains below the fail threshold of 0.1. Calibration fails: expected "
    "calibration error is 0.1382 against a warn threshold of 0.05 and a fail threshold of 0.1, "
    "alongside a Brier score of 0.1041 and a calibration slope of 0.83 [EV-2b9e4d7a6c05]. "
    "Population stability passes, with a maximum PSI of 0.0912 below the warn threshold of 0.1 "
    "and fail threshold of 0.25, with 0 features above the 0.1 level [EV-c04f81a2e937]."
)

SYSTEM_PROMPT = (
    "You are a model validation reviewer. Write one paragraph summarising the evidence below "
    "for a validation report. Cite evidence using the identifiers shown in the evidence block, "
    "in square brackets, for example [rA] or [EV-7f3a1c8b0d21]. Do NOT use numbered footnote "
    "markers such as [1], [2], [3]. State every figure exactly as given. Do not introduce any "
    "number that does not appear in the evidence. Do not speculate about causes."
)


def _load_local_env() -> None:
    env_file = Path(__file__).resolve().parents[1] / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip()
                if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
                    v = v[1:-1]
                if k and v and k not in os.environ:
                    os.environ[k] = v


def _call_provider(provider_name: str, model: str, system: str, user: str) -> str:
    _load_local_env()
    from start.core.config import LLMConfig
    from start.providers.llm import get_llm_provider

    provider = get_llm_provider(LLMConfig(provider=provider_name, model=model))
    if not provider.available:
        raise RuntimeError(f"provider '{provider_name}' is not available: SDK missing or credential unset")
    return provider.complete(system, user, max_tokens=600)


def main(argv: list[str] | None = None) -> int:
    _load_local_env()
    global SPEED

    parser = argparse.ArgumentParser(description="StART demonstration flight.")
    parser.add_argument(
        "--provider", default="none", help="none | openai | anthropic | deepseek | gemini | grok | gateway"
    )
    parser.add_argument("--model", default="", help="model name for the chosen provider")
    parser.add_argument("--speed", type=float, default=1.0, help="pause multiplier; 0 disables all pauses")
    parser.add_argument("--no-clear", action="store_true", help="do not clear the screen")
    args = parser.parse_args(argv)
    SPEED = max(0.0, args.speed)

    if not args.no_clear and sys.stdout.isatty():
        os.system("clear" if os.name != "nt" else "cls")  # noqa: S605

    print()
    print(c("  StART", "bold", "magenta"), c("— Standardized Agentic Reusable Tests", "bold"))
    print(c("  deterministic engines compute · agents reason · evidence proves · seals attest", "dim"))
    pause(1.2)

    # ---------------------------------------------------------------- ACT 1 --
    act(1, "Where am I running, and what am I allowed to touch?")
    note(
        "Before anything else, StART establishes its containment regime. This is the "
        "difference between a repository that supports two environments by convention and "
        "one that enforces the distinction."
    )
    pause(0.8)
    policy = egress_policy()
    print()
    type_out(c("  $ ", "dim") + "start doctor --egress")
    print(f"    runtime profile     {c(policy.profile.value, 'bold', 'green')}")
    print(f"    permitted providers {', '.join(sorted(policy.allowed))}")
    print(f"    refused providers   {c(', '.join(sorted(policy.denied)) or '(none)', 'yellow')}")
    print(f"    manifest hash       {profile_manifest()['manifest_hash'][:24]}…")
    pause(1.0)
    print()
    note(
        "Under an enterprise profile the public SaaS providers are not merely unconfigured — "
        "they are refused at the routing boundary, whatever SDKs are installed and whatever "
        "keys happen to be exported. Watch:"
    )
    pause(0.6)
    print()
    saved = os.environ.get("START_PROFILE")
    os.environ["START_PROFILE"] = "enterprise"
    try:
        assert_provider_allowed("openai")
        print(c("    (unexpectedly permitted)", "red"))
    except ProfileViolation as exc:
        print(c("    START_PROFILE=enterprise  →  start review --provider openai", "dim"))
        print(c(f"    ✗ ProfileViolation: {str(exc).splitlines()[0]}", "red"))
    finally:
        if saved is None:
            os.environ.pop("START_PROFILE", None)
        else:
            os.environ["START_PROFILE"] = saved
    pause(1.4)

    # ---------------------------------------------------------------- ACT 2 --
    act(2, "What does this review owe? (and why it is not an ML question)")
    note(
        "StART reviews a risk object, not a fitted estimator. Here is the same machinery "
        "applied to a vendor black box in the financial-crime stripe — no training data, no "
        "labels, no internals."
    )
    pause(0.8)
    obj = RiskObject(
        object_id="M-1042",
        kind="vendor_model",
        name="third-party transaction monitoring engine",
        materiality="high",
        capability_overrides={"affects_individuals": True},
    )
    plan = synthesise_plan(stripe_id="financial_crime", obj=obj)
    print()
    for line in plan.summary_lines()[:8]:
        print(f"  {line}")
    print(c("    …", "dim"))
    pause(1.0)
    print()
    note(
        "The interesting column is the substitutions. A vendor model cannot support "
        "discriminatory power or calibration — but the obligation does not evaporate, it "
        "transfers:"
    )
    print()
    for sub in plan.substitutions:
        print(
            f"    {c(sub['dimension'], 'yellow'):<40} → {c(', '.join(sub['burden_transferred_to']), 'green')}"
        )
    print()
    print(f"    plan hash  {c(plan.plan_hash(), 'bold')}")
    note("Recompute this at sign-off. If it differs, the scope moved after it was agreed.")
    pause(1.6)

    # ---------------------------------------------------------------- ACT 3 --
    act(3, "Deterministic evidence")
    note("Engines compute. Agents never do. Three records, hash-chained into the ledger:")
    print()
    for record in EVIDENCE:
        colour = {"pass": "green", "warn": "yellow", "fail": "red"}[record["status"]]
        metrics = ", ".join(f"{k}={v}" for k, v in list(record["metrics"].items())[:3])
        badge = c(f"{record['status'].upper():<4}", colour)
        print(f"    [{badge}] {record['test_id']:<42}")
        print(c(f"           {metrics}", "dim"))
    pause(1.2)

    # ---------------------------------------------------------------- ACT 4 --
    act(4, "What may leave this process?")
    note(
        "Prompts are not assembled from evidence directly. They are assembled from a "
        "policy-derived projection of it, and the projection is hashed."
    )
    pause(0.6)
    print()
    for policy_id in ("public_demo", "restricted", "minimal"):
        env = build_envelope(EVIDENCE, policy=POLICIES[policy_id])
        print(
            f"    {policy_id:<14} projected {len(env.projected):>2} fields, "
            f"withheld {len(env.withheld_paths):>2}   {c(env.envelope_hash()[:16], 'dim')}"
        )
    pause(1.0)

    envelope = build_envelope(EVIDENCE, policy=policy_for())
    print()
    note(
        f"Active policy for this profile: {envelope.policy_id}. Now the egress check — every "
        "number in the outbound prompt must exist in the envelope:"
    )
    print()
    try:
        verify_prompt_covered("Summarise: gap 0.0707 against threshold 0.05.", envelope)
        print(c("    ✓ prompt covered by envelope — send permitted", "green"))
    except Exception as exc:
        print(c(f"    ✗ {exc}", "red"))
    try:
        verify_prompt_covered("Account 4471982 shows a balance of 128455.30.", envelope)
        print(c("    (unexpectedly permitted)", "red"))
    except Exception as exc:
        print(c(f"    ✗ refused: {str(exc)[: WIDTH - 14]}", "red"))
    pause(1.6)

    # ---------------------------------------------------------------- ACT 5 --
    act(5, "Is the narrative load-bearing?")
    note(
        "The question a model risk function will actually ask: if the language model had said "
        "something different, would the conclusion have changed? StART tests it rather than "
        "asserting it."
    )
    pause(0.8)

    provider_used = "deterministic (no provider selected)"
    model_narrative = DETERMINISTIC_NARRATIVE
    narration_path = "deterministic_only"
    req_provider = None
    req_model = None
    fb_reason = None
    fb_detail = None
    fb_at = None

    if args.provider != "none":
        req_provider = args.provider
        req_model = args.model or ""
        print()
        type_out(
            c("  $ ", "dim")
            + f"start review --provider {args.provider} "
            f"{'--model ' + args.model if args.model else ''}".rstrip()
        )
        try:
            assert_provider_allowed(args.provider)
            prompt = envelope.render()
            verify_prompt_covered(prompt, envelope)
            model_narrative = _call_provider(args.provider, args.model, SYSTEM_PROMPT, prompt)
            provider_used = f"{args.provider}" + (f" / {args.model}" if args.model else "")
            narration_path = "model_narrated"
            print(c(f"    ✓ narrative generated via {provider_used}", "green"))
        except Exception as exc:
            import datetime

            narration_path = "deterministic_fallback"
            fb_reason = "provider_error"
            fb_detail = f"{type(exc).__name__}: {exc}"
            fb_at = datetime.datetime.now(datetime.UTC).isoformat()
            print(c(f"    ! {exc}", "yellow"))
            note(
                "Falling back to the deterministic narrative. The invariance check below is "
                "therefore trivial — it is comparing the deterministic path with itself, and "
                "says so rather than implying a model was involved."
            )
    else:
        print()
        note(
            "No provider selected, so both sides of the check are the deterministic path. "
            "Re-run with --provider openai (or anthropic, deepseek, gateway) to make this a "
            "real test."
        )

    attestation = attest_narrative_invariance(
        section="model_performance",
        deterministic_narrative=DETERMINISTIC_NARRATIVE,
        model_narrative=model_narrative,
        evidence=EVIDENCE,
        provider_name=provider_used,
        narration_path=narration_path,
        requested_provider=req_provider,
        requested_model=req_model,
        fallback_reason=fb_reason,
        fallback_detail=fb_detail,
        fallback_at=fb_at,
    )
    print()
    verdict = "INVARIANT" if attestation.invariant else "DIVERGENT"
    colour = "green" if attestation.invariant else "red"
    print(f"    {c(verdict, colour, 'bold')}  ({provider_used})")
    print(
        f"    claims bound      {attestation.model_binding['bound_claims']}"
        f"/{attestation.model_binding['total_claims']}"
    )
    print(f"    unbound rate      {attestation.unbound_claim_rate}")
    print(f"    blocking issues   {len(attestation.blocking_divergences())}")
    for divergence in attestation.divergences[:4]:
        marker = {"high": "✗", "medium": "!", "low": "·"}[divergence.severity]
        tint = {"high": "red", "medium": "yellow", "low": "dim"}[divergence.severity]
        print(c(f"      {marker} {divergence.kind}: {divergence.detail[: WIDTH - 14]}", tint))
    pause(1.6)

    # ---------------------------------------------------------------- ACT 6 --
    act(6, "Seal it")
    note(
        "One string committing to the scope, the policy, the evidence head, every "
        "attestation, and the containment regime. Paste it into the memo."
    )
    examined = {
        "discriminatory_power",
        "accuracy_calibration",
        "stability",
        "monitoring",
        "third_party_diligence",
        "benchmarking",
        "outcomes_analysis",
    }
    coverage = plan.coverage(examined)
    seal = build_seal(
        review_id="R-2026-0042",
        plan=plan.as_dict(),
        policy={"disclosure_policy": envelope.policy_id, "envelope": envelope.envelope_hash()},
        evidence_head=EVIDENCE[-1]["evidence_id"],
        attestations=[attestation.as_dict()],
        profile=profile_manifest(),
        environment={"python": sys.version.split()[0], "platform": sys.platform},
        controls=coverage,
    )
    print()
    print(
        f"    control coverage  {coverage['expectations_covered']}"
        f"/{coverage['expectations_total']} expectations "
        f"({coverage['overall_coverage_ratio']:.0%})"
    )
    if coverage["unmapped_frameworks"]:
        print(c(f"    unmapped          {', '.join(coverage['unmapped_frameworks'])}", "dim"))
    print()
    print(f"    {c(seal.seal_string(), 'bold', 'magenta')}")
    pause(1.2)

    print()
    note("And it verifies — including localising a change to the exact leaf:")
    print()
    manifest = seal.manifest()
    result = verify_seal(manifest, seal.seal_string())
    print(c(f"    ✓ {result['reason']}", "green"))
    manifest["payloads"]["plan"]["materiality"] = "low"
    tampered = verify_seal(manifest, seal.seal_string())
    print(c(f"    ✗ after altering the plan: {tampered['reason']}", "red"))
    pause(1.0)

    print()
    print(c("━" * WIDTH, "cyan"))
    print(c("  Every quantitative claim traced to evidence. Every disclosure bounded by policy.", "bold"))
    print(c("  Every review reducible to one verifiable string.", "bold"))
    print(c("━" * WIDTH, "cyan"))
    print()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\ninterrupted")
        raise SystemExit(130) from None
