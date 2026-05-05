from __future__ import annotations

from ..utils.common import *

# Focused module: adversaries.

class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: float):
        ctx.weight = float(weight)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return (-ctx.weight) * grad_output, None


def grad_reverse(x: torch.Tensor, weight: float = 1.0) -> torch.Tensor:
    return GradientReversalFunction.apply(x, weight)


class FactorVAEDiscriminator(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int = 512) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class VectorEmotionAdversary(nn.Module):
    def __init__(self, vector_dim: int, num_labels: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(vector_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_labels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualEmotionAdversary(nn.Module):
    def __init__(self, residual_dim: int, num_labels: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(residual_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_labels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def permute_latent_dims(z: torch.Tensor) -> torch.Tensor:
    columns = []
    for dim_idx in range(z.size(1)):
        permutation = torch.randperm(z.size(0), device=z.device)
        columns.append(z[permutation, dim_idx])
    return torch.stack(columns, dim=1)
