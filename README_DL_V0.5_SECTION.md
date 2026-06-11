## Deep Learning Model Review Demo

StART v0.5 adds a laptop-safe deep-learning workflow for binary classification.
The default model is an MLP trained for a small number of epochs on a public
synthetic dataset, split into train/test/OOS cohorts.

Run locally:

```bash
python -m pip install torch matplotlib tabulate
python examples/deep_learning_sequence_demo.py
```

Run the Databricks-compatible notebook locally or in Databricks:

```bash
python notebooks/03_deep_learning_model_review.py
```

Generated outputs:

```text
start_output/
├── evidence_store/RUN-DL-*/
├── figures/deep_learning/RUN-DL-*/
├── ledger.jsonl
└── reports/RUN-DL-*.md
```

The DL suite currently implements MLP, Residual MLP, and Wide & Deep. RNN, LSTM,
GRU, TCN, and Transformer-based workflows remain roadmap items for genuinely
sequential data.
