from __future__ import annotations

from ..utils.common import *
from ..modeling.adversaries import grad_reverse, permute_latent_dims
from ..config import LossConfig, ModelConfig
from ..schemas import FactorVAEOutput
from ..modeling.t5_utils import masked_mean

# Focused module: losses.

def tc_subspace_dim(loss_config: LossConfig, model_config: ModelConfig) -> int:
    if loss_config.tc_subspace == "none":
        return 0
    if loss_config.tc_subspace == "scalar":
        return int(model_config.num_scalar_factors)
    if loss_config.tc_subspace == "vector":
        return int(model_config.vector_latent_dim)
    if loss_config.tc_subspace == "full":
        return int(model_config.num_scalar_factors + model_config.vector_latent_dim)
    raise ValueError(f"Unsupported tc_subspace={loss_config.tc_subspace}")


def tc_latents_for_tc(
    vae_out: FactorVAEOutput,
    attention_mask: Optional[torch.Tensor],
    tc_subspace: str,
    tc_mode: str,
) -> Optional[torch.Tensor]:
    if tc_subspace == "none":
        return None
    if tc_subspace == "scalar":
        z = vae_out.scalar_z
    elif tc_subspace == "vector":
        z = vae_out.vector_z
    elif tc_subspace == "full":
        z = vae_out.z
    else:
        raise ValueError(f"Unsupported tc_subspace={tc_subspace}")

    if tc_mode == "example":
        return masked_mean(z, attention_mask)
    if tc_mode == "per_token":
        if attention_mask is None:
            return z.reshape(-1, z.size(-1))
        keep = attention_mask.bool()
        if keep.sum() == 0:
            return z.new_zeros((0, z.size(-1)))
        return z[keep]
    raise ValueError(f"Unsupported tc_mode={tc_mode}")


def pooled_latents_for_tc(
    vae_out: FactorVAEOutput,
    attention_mask: Optional[torch.Tensor],
    tc_subspace: str,
) -> Optional[torch.Tensor]:
    return tc_latents_for_tc(
        vae_out=vae_out,
        attention_mask=attention_mask,
        tc_subspace=tc_subspace,
        tc_mode="example",
    )


def factorvae_tc_loss(discriminator: nn.Module, z_batch: torch.Tensor) -> torch.Tensor:
    if z_batch is None or z_batch.size(0) < 2:
        if z_batch is None:
            return torch.zeros((), device=device)
        return z_batch.new_zeros(())
    logits = discriminator(z_batch)
    return (logits[:, 0] - logits[:, 1]).mean()


def discriminator_loss(discriminator: nn.Module, z_batch: torch.Tensor) -> torch.Tensor:
    if z_batch is None or z_batch.size(0) < 2:
        if z_batch is None:
            return torch.zeros((), device=device)
        return z_batch.new_zeros(())
    z_detached = z_batch.detach()
    z_perm = permute_latent_dims(z_detached)
    real_logits = discriminator(z_detached)
    perm_logits = discriminator(z_perm)
    real_target = torch.zeros(z_detached.size(0), dtype=torch.long, device=z_detached.device)
    perm_target = torch.ones(z_detached.size(0), dtype=torch.long, device=z_detached.device)
    ce = nn.CrossEntropyLoss()
    return 0.5 * (ce(real_logits, real_target) + ce(perm_logits, perm_target))


def multilabel_bce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    pos_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if pos_weight is not None:
        pos_weight = pos_weight.to(device=logits.device, dtype=logits.dtype)
    return F.binary_cross_entropy_with_logits(logits, labels.float(), pos_weight=pos_weight)


def multilabel_mse_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """MSE loss for continuous regression targets in [0, 1]."""
    probs = torch.sigmoid(logits)
    return F.mse_loss(probs, labels.float(), reduction="mean")


def adversarial_multilabel_loss(
    adversary: nn.Module,
    summary: torch.Tensor,
    labels: torch.Tensor,
    pos_weight: Optional[torch.Tensor],
    grl_weight: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    logits = adversary(grad_reverse(summary, grl_weight))
    loss = multilabel_bce_loss(logits, labels, pos_weight=pos_weight)
    return logits, loss


def probe_multilabel_loss(
    adversary: nn.Module,
    summary: torch.Tensor,
    labels: torch.Tensor,
    pos_weight: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    logits = adversary(summary)
    loss = multilabel_bce_loss(logits, labels, pos_weight=pos_weight)
    return logits, loss


def vector_adversarial_loss(
    vector_adversary: nn.Module,
    vector_summary: torch.Tensor,
    labels: torch.Tensor,
    pos_weight: Optional[torch.Tensor],
    grl_weight: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return adversarial_multilabel_loss(
        adversary=vector_adversary,
        summary=vector_summary,
        labels=labels,
        pos_weight=pos_weight,
        grl_weight=grl_weight,
    )


def vector_probe_loss(
    vector_adversary: nn.Module,
    vector_summary: torch.Tensor,
    labels: torch.Tensor,
    pos_weight: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    return probe_multilabel_loss(
        adversary=vector_adversary,
        summary=vector_summary,
        labels=labels,
        pos_weight=pos_weight,
    )


def residual_adversarial_loss(
    residual_adversary: nn.Module,
    residual_summary: torch.Tensor,
    labels: torch.Tensor,
    pos_weight: Optional[torch.Tensor],
    grl_weight: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return adversarial_multilabel_loss(
        adversary=residual_adversary,
        summary=residual_summary,
        labels=labels,
        pos_weight=pos_weight,
        grl_weight=grl_weight,
    )


def residual_probe_loss(
    residual_adversary: nn.Module,
    residual_summary: torch.Tensor,
    labels: torch.Tensor,
    pos_weight: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    return probe_multilabel_loss(
        adversary=residual_adversary,
        summary=residual_summary,
        labels=labels,
        pos_weight=pos_weight,
    )


def transfer_strength_margin_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    labels = labels.float()
    positive_penalty = F.relu(margin - logits) * labels
    negative_penalty = F.relu(margin + logits) * (1.0 - labels)
    return (positive_penalty + negative_penalty).mean()


def cross_covariance_penalty(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.size(0) <= 1 or b.size(0) <= 1:
        return a.new_zeros(())
    a_centered = a - a.mean(dim=0, keepdim=True)
    b_centered = b - b.mean(dim=0, keepdim=True)
    cov = a_centered.transpose(0, 1) @ b_centered
    cov = cov / float(max(a.size(0) - 1, 1))
    return cov.pow(2).mean()
