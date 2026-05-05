from __future__ import annotations

from ..utils.common import *

# Focused module: pooling.

class ScalarLatentAttentionPooling(nn.Module):
    def __init__(
        self,
        num_scalar_factors: int,
        source_dim: int,
        num_heads: int = 4,
        pooling_mode: str = "per_scalar_dim",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        valid_modes = {"per_scalar_dim", "joint_scalar_vector"}
        if pooling_mode not in valid_modes:
            raise ValueError(f"Unsupported pooling_mode={pooling_mode}. Expected one of {sorted(valid_modes)}")

        self.num_scalar_factors = num_scalar_factors
        self.source_dim = source_dim
        self.num_heads = num_heads
        self.pooling_mode = pooling_mode
        self.dropout = nn.Dropout(dropout)

        if pooling_mode == "per_scalar_dim":
            self.score_weight = nn.Parameter(torch.empty(num_scalar_factors, num_heads, source_dim))
            self.score_bias = nn.Parameter(torch.zeros(num_scalar_factors, num_heads))
        else:
            self.score_weight = nn.Parameter(torch.empty(num_heads, source_dim))
            self.score_bias = nn.Parameter(torch.zeros(num_heads))
        nn.init.xavier_uniform_(self.score_weight)

    def forward(
        self,
        source_sequence: torch.Tensor,
        scalar_sequence: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        source_sequence = self.dropout(source_sequence)

        if self.pooling_mode == "per_scalar_dim":
            logits = torch.einsum("bsd,nhd->bnhs", source_sequence, self.score_weight)
            logits = logits + self.score_bias.unsqueeze(0).unsqueeze(-1)
        else:
            shared_logits = torch.einsum("bsd,hd->bhs", source_sequence, self.score_weight)
            shared_logits = shared_logits + self.score_bias.unsqueeze(0).unsqueeze(-1)
            logits = shared_logits.unsqueeze(1).expand(-1, self.num_scalar_factors, -1, -1)

        if attention_mask is not None:
            keep_mask = attention_mask.bool()
            logits = logits.masked_fill(
                ~keep_mask.unsqueeze(1).unsqueeze(1),
                torch.finfo(source_sequence.dtype).min,
            )

        weights = torch.softmax(logits, dim=-1)
        values = scalar_sequence.transpose(1, 2).unsqueeze(2)
        pooled = torch.sum(weights * values, dim=-1)
        return pooled, weights
