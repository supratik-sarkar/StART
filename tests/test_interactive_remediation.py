from __future__ import annotations

from start.interactive_review import (
    ReviewConfig,
    _ask_split_proportions,
    _infer_cost_priority,
    prompt_review_config,
)


# -- Section M: cost priority inference --------------------------------------- #
def test_false_negative_text_maps_to_false_negatives():
    assert _infer_cost_priority("false negatives are significantly costlier") == "false_negatives"
    assert _infer_cost_priority("missed churn is costly") == "false_negatives"
    assert _infer_cost_priority("prioritize recall") == "false_negatives"


def test_false_positive_text_maps_to_false_positives():
    assert _infer_cost_priority("false positives costly") == "false_positives"
    assert _infer_cost_priority("avoid unnecessary intervention") == "false_positives"
    assert _infer_cost_priority("prioritize precision") == "false_positives"


def test_neutral_text_returns_none():
    assert _infer_cost_priority("just build a good model") is None


# -- Section D: split proportion validation ----------------------------------- #
def test_valid_split_proportions():
    answers = iter(["0.70", "0.15", "0.15"])
    assert _ask_split_proportions(lambda p: next(answers)) == (0.70, 0.15, 0.15)


def test_invalid_split_falls_back_to_default():
    answers = iter(["0.50", "0.30", "0.30"])  # sums to 1.10
    assert _ask_split_proportions(lambda p: next(answers)) == (0.60, 0.20, 0.20)


def test_blank_split_uses_defaults():
    answers = iter(["", "", ""])
    assert _ask_split_proportions(lambda p: next(answers)) == (0.60, 0.20, 0.20)


# -- Section C: interactive run_dl defaults true ------------------------------ #
def test_interactive_run_dl_defaults_true():
    answers = iter([
        "n",  # committee workflow? -> legacy
        "", "attrition", "stratified", "mlp", "relu",
        "standard",  # robustness
        "1",  # Select backend -> 1: deterministic
        "",  # run_dl blank -> default Y
        "0.60", "0.20", "0.20",  # split proportions
        "bounded_random_search",  # tuning strategy
        "5",  # trials
        "integrated_gradients",  # explainability (explain_method)
    ])
    cfg = prompt_review_config(ReviewConfig(), ask=lambda *_: next(answers))
    assert cfg.run_dl is True


def test_interactive_run_dl_explicit_no():
    answers = iter([
        "n",  # committee workflow? -> legacy
        "", "attrition", "stratified", "mlp", "relu",
        "standard",  # robustness
        "1",  # Select backend -> 1: deterministic
        "n",  # run_dl no -> diagnostics only
    ])
    cfg = prompt_review_config(ReviewConfig(), ask=lambda *_: next(answers))
    assert cfg.run_dl is False


# -- Section Q: enterprise gateway does not require a public key --------------- #
def test_enterprise_gateway_no_public_key_required():
    from start.providers.keys import key_required

    assert key_required("enterprise_llm_gateway") is False
    assert key_required("openai") is True


def test_missing_key_non_interactive_does_not_prompt():
    from start.providers.keys import ensure_provider_key

    # non-interactive + no key => status missing, never prompts/raises
    status = ensure_provider_key("openai", prompt_for_key=False, interactive=False)
    assert status.source in ("missing", "env")
