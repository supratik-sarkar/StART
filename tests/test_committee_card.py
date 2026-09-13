from __future__ import annotations

from start.committee_card import CommitteeCard, render_card_markdown, render_card_rich


def test_evidence_backed_flag():
    assert CommitteeCard(agent="A", purpose="p", evidence=["x"]).evidence_backed is True
    assert CommitteeCard(agent="A", purpose="p").evidence_backed is False


def test_non_evidence_backed_states_explicitly():
    md = render_card_markdown(CommitteeCard(agent="A", purpose="p", recommendation="do x"))
    assert "not evidence-backed" in md


def test_card_has_evidence_first_structure():
    md = render_card_markdown(CommitteeCard(
        agent="ArchitectureReviewAgent", purpose="Model selection",
        evidence=["569 rows"], recommendation="MLP",
        alternatives=["MLP", "WideDeep"], risks=["overfitting"],
        artifacts_used=["data_statistics"], open_questions=["interpretability?"],
        decision="pending",
    ))
    # evidence appears before recommendation
    assert md.index("Evidence") < md.index("Recommendation")
    assert "Alternatives" in md and "Risks" in md


def test_rich_render_returns_renderable():
    card = CommitteeCard(agent="A", purpose="p", evidence=["e"], recommendation="r")
    panel = render_card_rich(card)
    assert panel is not None


def test_to_dict_complete():
    d = CommitteeCard(agent="A", purpose="p", evidence=["e"]).to_dict()
    for key in ("agent", "purpose", "evidence", "evidence_backed",
                "recommendation", "alternatives", "risks", "open_questions",
                "artifacts_used", "decision"):
        assert key in d
