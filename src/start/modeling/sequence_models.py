"""Sequence network modules for StART's sequence DL track.

Real trainable PyTorch recurrent classifiers over inputs shaped
(batch, timesteps, features):

    rnn      - vanilla Elman RNN
    gru      - gated recurrent unit
    lstm     - long short-term memory
    bi_lstm  - bidirectional LSTM

Each takes the last (or pooled) hidden state into a linear head producing
``n_outputs`` logits. Imports torch lazily so module import never fails when
torch is absent. These are NEVER applied to tabular data — the sequence track
builds genuinely sequential tensors.
"""

from __future__ import annotations

from typing import Any

SEQUENCE_FAMILIES = ("rnn", "gru", "lstm", "bi_lstm")


def build_sequence_network(
    family: str,
    n_features: int,
    hidden_size: int = 32,
    num_layers: int = 1,
    dropout: float = 0.1,
    n_outputs: int = 1,
) -> Any:
    """Build a recurrent sequence classifier module."""
    import torch
    import torch.nn as nn

    if family not in SEQUENCE_FAMILIES:
        raise ValueError(f"Unknown sequence family '{family}'. Known: {SEQUENCE_FAMILIES}")

    bidirectional = family == "bi_lstm"
    rnn_dropout = dropout if num_layers > 1 else 0.0

    class SequenceClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            if family == "rnn":
                self.rnn = nn.RNN(
                    n_features,
                    hidden_size,
                    num_layers,
                    batch_first=True,
                    nonlinearity="tanh",
                    dropout=rnn_dropout,
                )
            elif family == "gru":
                self.rnn = nn.GRU(
                    n_features,
                    hidden_size,
                    num_layers,
                    batch_first=True,
                    dropout=rnn_dropout,
                )
            else:  # lstm or bi_lstm
                self.rnn = nn.LSTM(
                    n_features,
                    hidden_size,
                    num_layers,
                    batch_first=True,
                    dropout=rnn_dropout,
                    bidirectional=bidirectional,
                )
            self.drop = nn.Dropout(dropout)
            head_in = hidden_size * (2 if bidirectional else 1)
            self.head = nn.Linear(head_in, n_outputs)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out, _ = self.rnn(x)
            last = out[:, -1, :]  # last timestep representation
            return self.head(self.drop(last))

    return SequenceClassifier()
