"""Small MLP: delayed obs history -> current single-step obs."""
from __future__ import annotations

import torch
import torch.nn as nn


def make_mlp(in_dim, out_dim, hidden_dims, activation):
    layers = []
    width = in_dim
    for h in hidden_dims:
        layers += [nn.Linear(width, h), activation]
        width = h
    layers.append(nn.Linear(width, out_dim))
    return nn.Sequential(*layers)


class ObsPredictor(nn.Module):
    """Stage 1 target: one frame of actor obs. Stage 2: same net inside ActorCritic."""

    def __init__(
        self,
        history_dim,
        frame_dim,
        hidden_dims=None,
        activation="elu",
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256]
        act = nn.ELU() if activation == "elu" else nn.ReLU()
        self.history_dim = history_dim
        self.frame_dim = frame_dim
        self.net = make_mlp(history_dim, frame_dim, hidden_dims, act)

    def forward(self, history):
        return self.net(history)
