import torch
import torch.nn as nn


class PolicyNetwork(nn.Module):
    def __init__(
        self,
        state_dim: int = 4,
        action_dim: int = 2,
        hidden_dim: int = 64,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)
