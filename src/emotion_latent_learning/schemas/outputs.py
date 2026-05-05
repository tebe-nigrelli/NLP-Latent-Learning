from __future__ import annotations

from ..utils.common import *
from ..config import DataConfig, ExperimentConfig, LossConfig, ModelConfig, PromptConfig, ScheduleConfig

# Focused module: schemas.

@dataclass
class FactorVAEOutput:
    z: torch.Tensor
    mu: torch.Tensor
    logvar: torch.Tensor
    scalar_z: torch.Tensor
    vector_z: torch.Tensor
    scalar_mu: torch.Tensor
    vector_mu: torch.Tensor
    scalar_logvar: torch.Tensor
    vector_logvar: torch.Tensor


@dataclass
class T5FactorVAEModelOutput:
    t5_encoder_sequence: torch.Tensor
    attention_weights: torch.Tensor
    pooled_scalar_tensor: torch.Tensor
    vae: FactorVAEOutput
    vae_decoded_sequence: torch.Tensor
    residual_memory: torch.Tensor
    decoder_memory: torch.Tensor
    classification_logits: torch.Tensor
    base_loss: Optional[torch.Tensor] = None
    loss_terms: Optional[Dict[str, torch.Tensor]] = None


@dataclass
class TrainingState:
    model: nn.Module
    discriminator: Optional[nn.Module]
    vector_adversary: Optional[nn.Module]
    residual_adversary: Optional[nn.Module]
    optimizer: torch.optim.Optimizer
    disc_optimizer: Optional[torch.optim.Optimizer]
    vector_adv_optimizer: Optional[torch.optim.Optimizer]
    residual_adv_optimizer: Optional[torch.optim.Optimizer]
    lr_scheduler: torch.optim.lr_scheduler.LambdaLR
    threshold: Any
    best_val_micro_f1: float
    step: int
    history: List[Dict[str, Any]]


@dataclass
class ExperimentContext:
    tokenizer: AutoTokenizer
    emotion_names: Sequence[str]
    device: torch.device
    data_config: DataConfig
    prompt_config: PromptConfig
    model_config: ModelConfig
    loss_config: LossConfig
    schedule_config: ScheduleConfig
    experiment_config: ExperimentConfig
    pos_weight: torch.Tensor
