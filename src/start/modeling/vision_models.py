"""CNN architectures for the vision track.

``simple_cnn`` is fully implemented in pure PyTorch (no torchvision needed),
configurable by conv blocks, base channels, kernel size, pooling, dropout, and
dense size. Three named presets resolve to concrete configs:

    simple_cnn_small  - 2 conv blocks, 16 base channels
    simple_cnn_medium - 3 conv blocks, 32 base channels
    simple_cnn_deep   - 4 conv blocks, 32 base channels

Every built network reports a resolved, evidence-stampable architecture name
and config. ``resnet18`` is an OPTIONAL preset, available only when torchvision
is installed; availability is reported explicitly (never a silent stub).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

CNN_PRESETS = ("simple_cnn_small", "simple_cnn_medium", "simple_cnn_deep")

_PRESET_CONFIG = {
    "simple_cnn_small": {"n_blocks": 2, "base_channels": 16, "dense": 64},
    "simple_cnn_medium": {"n_blocks": 3, "base_channels": 32, "dense": 128},
    "simple_cnn_deep": {"n_blocks": 4, "base_channels": 32, "dense": 128},
}


@dataclass
class CNNConfig:
    n_blocks: int = 2
    base_channels: int = 16
    kernel_size: int = 3
    pooling: str = "max"  # max | avg
    dropout: float = 0.1
    dense: int = 64
    name: str = "simple_cnn_custom"
    extra: dict[str, Any] = field(default_factory=dict)

    def stamp(self) -> dict[str, Any]:
        """Evidence-stampable architecture descriptor."""
        return {
            "architecture": self.name,
            "n_blocks": self.n_blocks,
            "base_channels": self.base_channels,
            "kernel_size": self.kernel_size,
            "pooling": self.pooling,
            "dropout": self.dropout,
            "dense": self.dense,
        }


def config_from_preset(preset: str, **overrides: Any) -> CNNConfig:
    if preset not in _PRESET_CONFIG:
        raise ValueError(f"Unknown CNN preset '{preset}'. Known: {CNN_PRESETS}")
    cfg = dict(_PRESET_CONFIG[preset])
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    return CNNConfig(name=preset, **cfg)


def torchvision_available() -> bool:
    try:
        import torchvision  # noqa: F401

        return True
    except ImportError:
        return False


def build_simple_cnn(config: CNNConfig, channels: int, image_size: int, n_classes: int) -> Any:
    """Pure-PyTorch configurable CNN."""
    import torch
    import torch.nn as nn

    pool_cls = nn.MaxPool2d if config.pooling == "max" else nn.AvgPool2d
    pad = config.kernel_size // 2

    class SimpleCNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            blocks: list[Any] = []
            in_ch = channels
            out_ch = config.base_channels
            size = image_size
            for _ in range(config.n_blocks):
                blocks += [
                    nn.Conv2d(in_ch, out_ch, config.kernel_size, padding=pad),
                    nn.ReLU(),
                    pool_cls(2),
                ]
                in_ch = out_ch
                out_ch = out_ch * 2
                size = size // 2
            self.features = nn.Sequential(*blocks)
            flat = in_ch * max(size, 1) * max(size, 1)
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(flat, config.dense),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.dense, n_classes),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.classifier(self.features(x))

    return SimpleCNN()


def build_resnet18(channels: int, n_classes: int, pretrained: bool = False) -> Any:
    """Optional ResNet-18 preset. Requires torchvision; raises a clear,
    availability-style error otherwise (explicit, not a silent stub)."""
    if not torchvision_available():
        raise ImportError(
            "resnet18 unavailable: torchvision not installed (pip install torchvision). "
            "Use a simple_cnn preset instead."
        )
    import torch.nn as nn
    import torchvision

    model = torchvision.models.resnet18(weights=None)
    if channels != 3:
        model.conv1 = nn.Conv2d(channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    return model


def build_vision_network(
    architecture: str,
    channels: int,
    image_size: int,
    n_classes: int,
    *,
    config: CNNConfig | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Build a vision network and return (module, architecture_stamp).

    architecture: a simple_cnn preset name, "simple_cnn" (with explicit config),
    or "resnet18" (gated on torchvision).
    """
    if architecture == "resnet18":
        net = build_resnet18(channels, n_classes)
        return net, {"architecture": "resnet18", "backend": "torchvision"}
    if architecture in CNN_PRESETS:
        cfg = config or config_from_preset(architecture)
        return build_simple_cnn(cfg, channels, image_size, n_classes), cfg.stamp()
    if architecture == "simple_cnn":
        cfg = config or CNNConfig()
        return build_simple_cnn(cfg, channels, image_size, n_classes), cfg.stamp()
    raise ValueError(
        f"Unknown vision architecture '{architecture}'. "
        f"Known: {CNN_PRESETS} + 'simple_cnn' (configurable) + 'resnet18' (optional)."
    )


def count_parameters(net: Any) -> int:
    """Total trainable parameter count of a built network."""
    return int(sum(p.numel() for p in net.parameters() if p.requires_grad))


def describe_cnn(
    architecture: str,
    channels: int,
    image_size: int,
    n_classes: int,
    *,
    config: CNNConfig | None = None,
) -> dict[str, Any]:
    """Build the CNN and return a full evidence-stamped descriptor: preset,
    conv blocks, channels, kernel, pooling, dense, dropout, image size, and the
    real trainable parameter count. Used by the notebook/dashboard CNN UX so all
    architecture choices become evidence-backed metadata."""
    net, stamp = build_vision_network(
        architecture, channels, image_size, n_classes, config=config
    )
    descriptor = dict(stamp)
    descriptor.update(
        {
            "channels": channels,
            "image_size": image_size,
            "n_classes": n_classes,
            "param_count": count_parameters(net),
        }
    )
    # Per-block channel progression for the configurable CNN (transparency).
    if architecture in CNN_PRESETS or architecture == "simple_cnn":
        cfg = config or (
            config_from_preset(architecture) if architecture in CNN_PRESETS else CNNConfig()
        )
        chans, c = [], cfg.base_channels
        for _ in range(cfg.n_blocks):
            chans.append(c)
            c *= 2
        descriptor["conv_block_channels"] = chans
    return descriptor
