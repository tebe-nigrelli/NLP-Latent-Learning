from __future__ import annotations

from ..utils.common import *
from ..schemas import FactorVAEOutput

# Focused module: vae.

class FactorVAEEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_scalar_factors: int,
        vector_latent_dim: int,
        hidden_dim: int = 1024,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_scalar_factors = num_scalar_factors
        self.vector_latent_dim = vector_latent_dim
        self.total_latent_dim = num_scalar_factors + vector_latent_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.mu = nn.Linear(hidden_dim, self.total_latent_dim)
        self.logvar = nn.Linear(hidden_dim, self.total_latent_dim)

    def split(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scalar = tensor[..., : self.num_scalar_factors]
        vector = tensor[..., self.num_scalar_factors :]
        return scalar, vector

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor, sample: bool) -> torch.Tensor:
        if not sample:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor, sample: bool = True) -> FactorVAEOutput:
        hidden = self.net(x)
        mu = self.mu(hidden)
        logvar = self.logvar(hidden)
        z = self.reparameterize(mu=mu, logvar=logvar, sample=sample)

        scalar_z, vector_z = self.split(z)
        scalar_mu, vector_mu = self.split(mu)
        scalar_logvar, vector_logvar = self.split(logvar)
        return FactorVAEOutput(
            z=z,
            mu=mu,
            logvar=logvar,
            scalar_z=scalar_z,
            vector_z=vector_z,
            scalar_mu=scalar_mu,
            vector_mu=vector_mu,
            scalar_logvar=scalar_logvar,
            vector_logvar=vector_logvar,
        )


class FactorVAEDecoder(nn.Module):
    def __init__(
        self,
        output_dim: int,
        num_scalar_factors: int,
        vector_latent_dim: int,
        hidden_dim: int = 1024,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        total_latent_dim = num_scalar_factors + vector_latent_dim
        self.net = nn.Sequential(
            nn.Linear(total_latent_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)
