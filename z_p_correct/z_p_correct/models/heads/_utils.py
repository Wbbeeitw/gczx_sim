"""Shared helpers for phase/progress heads."""

import torch.nn as nn


def build_mlp(input_dim, hidden_dim, output_dim, depth, dropout=0.0, activation=nn.GELU):
    """Build a simple MLP trunk.

    Args:
        input_dim: Input feature dimension.
        hidden_dim: Hidden layer dimension.
        output_dim: Output dimension.
        depth: Number of Linear layers.
        dropout: Dropout probability applied between hidden layers.
        activation: Activation class to use between hidden layers.

    Returns:
        nn.Sequential MLP.
    """
    layers = []
    for i in range(depth):
        in_d = input_dim if i == 0 else hidden_dim
        out_d = output_dim if i == depth - 1 else hidden_dim
        layers.append(nn.Linear(in_d, out_d))
        if i < depth - 1:
            layers.append(activation())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)
