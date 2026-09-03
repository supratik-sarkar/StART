from __future__ import annotations

from start.interactive_checkpoints import (
    render_checkpoint,
    resolve_checkpoint,
)


def test_agent_agrees_non_interactive_keep():
    d = resolve_checkpoint("architecture", "mlp", "mlp", "good fit", evidence_id="ARCH-01")
    # v3.1.1: non-interactive mode keeps user value (no auto-accept for agreement)
    assert d.choice == "non_interactive_keep"
    assert d.effective_value == "mlp"


def test_auto_accept_takes_recommendation():
    d = resolve_checkpoint("architecture", "wide_deep", "mlp", "small data", auto_accept=True)
    assert d.choice == "auto_accept"
    assert d.effective_value == "mlp"


def test_non_interactive_keeps_user_choice_not_silent():
    d = resolve_checkpoint("architecture", "wide_deep", "mlp", "small data")
    assert d.choice == "non_interactive_keep"  # explicit, recorded
    assert d.effective_value == "wide_deep"


def test_interactive_accept():
    answers = iter(["A"])
    d = resolve_checkpoint(
        "metric",
        "auc_roc",
        "pr_auc",
        "false negatives costly",
        interactive=True,
        ask=lambda p: next(answers),
    )
    assert d.choice == "accept"
    assert d.effective_value == "pr_auc"


def test_interactive_keep():
    answers = iter(["K"])
    d = resolve_checkpoint(
        "metric",
        "auc_roc",
        "pr_auc",
        "reason",
        interactive=True,
        ask=lambda p: next(answers),
    )
    assert d.choice == "keep"
    assert d.effective_value == "auc_roc"


def test_interactive_empty_defaults_to_keep():
    answers = iter([""])
    d = resolve_checkpoint(
        "split",
        "random",
        "stratified",
        "reason",
        interactive=True,
        ask=lambda p: next(answers),
    )
    assert d.choice == "keep"


def test_interactive_explain_then_choose():
    answers = iter(["E", "A"])
    msgs = []
    d = resolve_checkpoint(
        "architecture",
        "wide_deep",
        "mlp",
        "short reason",
        explanation="MLP lowers overfitting risk on small tabular data",
        interactive=True,
        ask=lambda p: next(answers),
        emit=msgs.append,
    )
    assert d.choice == "accept"
    assert any("overfitting" in m for m in msgs)


def test_interactive_invalid_then_valid():
    answers = iter(["x", "z", "K"])
    msgs = []
    d = resolve_checkpoint(
        "metric",
        "auc_roc",
        "pr_auc",
        "reason",
        interactive=True,
        ask=lambda p: next(answers),
        emit=msgs.append,
    )
    assert d.choice == "keep"
    assert any("A, O, C" in m for m in msgs)


def test_decision_to_dict():
    d = resolve_checkpoint(
        "architecture", "wide_deep", "mlp", "reason", evidence_id="ARCH-01", auto_accept=True
    )
    out = d.to_dict()
    for key in (
        "checkpoint",
        "user_value",
        "recommended_value",
        "reason",
        "evidence_id",
        "choice",
        "effective_value",
    ):
        assert key in out


def test_render_checkpoint_shows_options_when_disagreeing():
    out = render_checkpoint("architecture", "wide_deep", "mlp", "small data", "ARCH-01")
    assert "[A] Accept" in out and "[O] Override" in out and "[C] Challenge" in out


def test_render_checkpoint_notes_agreement():
    out = render_checkpoint("architecture", "mlp", "mlp", "good fit")
    assert "agrees" in out
