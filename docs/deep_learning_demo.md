# StART v0.5 Deep Learning Demo

This patch adds a laptop-safe deep-learning model review workflow.

## Local validation

```bash
cd ~/Desktop/StART
source .venv-start/bin/activate
python -m pip install torch matplotlib tabulate
python -m pytest
ruff check src tests
python examples/deep_learning_sequence_demo.py
python notebooks/03_deep_learning_model_review.py
```

## Databricks demo

Import `notebooks/03_deep_learning_model_review.py` as a Databricks notebook.
Attach a CPU cluster first. The default MLP workflow is intentionally small and
runs without GPU. Use deterministic mode first. LLM-assisted mode can be enabled
only after configuring a Databricks secret scope for the provider key.

## What to show

1. Cohort metrics across train/test/OOS.
2. Learning curve figure.
3. Calibration curve.
4. Integrated Gradients attribution if Captum is installed, otherwise permutation attribution.
5. Top-feature shock sensitivity.
6. Noise and masking robustness.
7. Evidence-backed markdown report.
