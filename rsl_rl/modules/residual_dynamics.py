# Tiny residual dynamics model for v2b smoke tests.
# Predicts x_{t+1} = x_t + f(x_t, u_t). Copy baseline is just x_{t+1} ≈ x_t.

from __future__ import annotations

import torch
import torch.nn as nn


class ResidualDynamics(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden=(256, 256)):
        super().__init__()
        layers = []
        in_dim = state_dim + action_dim
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ELU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, state_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x, u):
        # residual: next = x + delta(x, u)
        return x + self.net(torch.cat([x, u], dim=-1))


def mse(a, b):
    return ((a - b) ** 2).mean().item()


def copy_baseline_mse(x_t, x_tp1):
    return mse(x_t, x_tp1)


def train_residual(model, x_t, u_t, x_tp1, epochs=40, batch_size=4096, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = x_t.shape[0]
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n, device=x_t.device)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            pred = model(x_t[idx], u_t[idx])
            loss = ((pred - x_tp1[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        return mse(model(x_t, u_t), x_tp1)
