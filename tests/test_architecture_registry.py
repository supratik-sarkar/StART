from __future__ import annotations

import warnings

import pytest

from start.modeling.architecture_registry import (
    ACTIVATIONS,
    family_available,
    list_families,
    resolve_architecture,
    torchvision_available,
)


def test_activation_set_complete():
    assert set(ACTIVATIONS) == {"relu", "leaky_relu", "gelu", "tanh", "selu", "elu", "swish", "mish", "sigmoid", "softplus"}



def test_new_family_activation_form():
    r = resolve_architecture(family="mlp", activation="gelu")
    assert r.family == "mlp" and r.activation == "gelu" and r.modality == "tabular"
    assert r.from_alias is None and not r.warnings_emitted


def test_all_activations_resolve():
    for act in ACTIVATIONS:
        r = resolve_architecture(family="residual_mlp", activation=act)
        assert r.activation == act


def test_default_activation_per_family():
    assert resolve_architecture(family="mlp").activation == "relu"
    assert resolve_architecture(family="lstm").activation == "tanh"
    assert resolve_architecture(family="transformer").activation == "gelu"


def test_deprecated_alias_resolves_with_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        r = resolve_architecture("leaky_relu_mlp")
    assert r.family == "mlp" and r.activation == "leaky_relu"
    assert r.from_alias == "leaky_relu_mlp"
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
    assert r.warnings_emitted  # surfaced for evidence/logging too


def test_alias_does_not_override_explicit_activation():
    # If the user explicitly passes activation, the alias default yields to it.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        r = resolve_architecture("leaky_relu_mlp", activation="gelu")
    assert r.family == "mlp" and r.activation == "gelu"


def test_unknown_family_and_activation_raise():
    with pytest.raises(ValueError, match="[Uu]nknown architecture family"):
        resolve_architecture(family="quantum_net")
    with pytest.raises(ValueError, match="[Uu]nknown activation"):
        resolve_architecture(family="mlp", activation="sigmoid_squared")


def test_list_families_by_modality():
    tabular = list_families("tabular")
    assert {"mlp", "residual_mlp", "wide_deep"} <= set(tabular)
    sequence = list_families("sequence")
    assert {"rnn", "gru", "lstm", "bi_lstm"} <= set(sequence)
    vision = list_families("vision")
    assert {"simple_cnn", "resnet18"} <= set(vision)


def test_family_availability_reporting():
    ok, reason = family_available("mlp")
    assert ok and reason == ""

    ok, reason = family_available("tcn")  # roadmap
    assert not ok and "roadmap" in reason

    ok, reason = family_available("simple_cnn")  # pure-torch, always available with torch
    assert ok

    # resnet18 is gated on torchvision: explicit availability, not a silent stub
    ok, reason = family_available("resnet18")
    if torchvision_available():
        assert ok
    else:
        assert not ok and "torchvision" in reason


def test_unknown_family_availability():
    ok, reason = family_available("does_not_exist")
    assert not ok and "Unknown family" in reason
