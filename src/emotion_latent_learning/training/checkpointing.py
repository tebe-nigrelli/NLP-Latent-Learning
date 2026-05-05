from __future__ import annotations

from ..utils.common import *
from ..schemas import ExperimentContext, TrainingState
from ..utils import to_serializable

# Focused module: checkpointing.

def state_dict_to_cpu(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cpu_state: Dict[str, torch.Tensor] = {}
    for name, tensor in state_dict.items():
        if isinstance(tensor, torch.Tensor):
            cpu_state[name] = tensor.detach().cpu()
        else:
            cpu_state[name] = tensor
    return cpu_state


def compact_model_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    compact_state: Dict[str, torch.Tensor] = {}
    skip_prefixes = (
        "t5_encoder.",
        "t5_decoder.",
    )
    for name, tensor in model.state_dict().items():
        if name.startswith(skip_prefixes):
            continue
        if name.startswith("shared_t5"):
            if ".lora_" not in name:
                continue
        compact_state[name] = tensor.detach().cpu()
    return compact_state


def load_compact_model_state_dict(model: nn.Module, state_dict: Dict[str, torch.Tensor]) -> None:
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    unexpected_keys = list(unexpected_keys)
    ignored_missing_prefixes = (
        "shared_t5",
        "t5_encoder.",
        "t5_decoder.",
    )
    missing_keys = [
        key for key in missing_keys
        if not key.startswith(ignored_missing_prefixes)
    ]
    if unexpected_keys:
        print(f"warning: unexpected keys while loading compact checkpoint: {unexpected_keys[:10]}")
    if missing_keys:
        print(f"warning: non-frozen keys missing while loading compact checkpoint: {missing_keys[:10]}")


def metric_summary_for_checkpoint(metrics: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if metrics is None:
        return None
    drop_keys = {"logits", "labels", "probs", "preds", "classification_report_text"}
    keep = {
        "threshold",
        "threshold_mode",
        "threshold_mean",
        "threshold_min",
        "threshold_max",
        "avg_total_loss",
        "avg_base_loss",
        "avg_recon",
        "avg_kl",
        "avg_cls",
        "avg_tc",
        "avg_copy",
        "avg_vec_adv",
        "avg_residual_adv",
        "avg_sep",
        "avg_orth",
        "avg_transfer",
        "weighted_recon",
        "weighted_kl",
        "weighted_cls",
        "weighted_tc",
        "weighted_copy",
        "weighted_vec_adv",
        "weighted_residual_adv",
        "weighted_sep",
        "weighted_orth",
        "weighted_transfer",
        "micro_f1",
        "macro_f1",
        "weighted_f1",
        "subset_accuracy",
        "hamming_acc",
        "jaccard_micro",
        "average_precision_micro",
        "label_ranking_average_precision",
    }
    serial = to_serializable(metrics)
    return {k: v for k, v in serial.items() if k in keep and k not in drop_keys}


def save_checkpoint(
    path: Path,
    state: TrainingState,
    ctx: ExperimentContext,
    epoch: int,
    val_metrics: Optional[Dict[str, Any]] = None,
    test_metrics: Optional[Dict[str, Any]] = None,
    include_training_state: bool = False,
    include_history: bool = False,
) -> None:
    compact_model_state = compact_model_state_dict(state.model)
    payload = {
        "epoch": epoch,
        "step": state.step,
        "threshold": state.threshold,
        "best_val_micro_f1": state.best_val_micro_f1,
        "model_state_dict": compact_model_state,
        "checkpoint_format": "compact_no_frozen_t5",
        "discriminator_state_dict": None if state.discriminator is None else state_dict_to_cpu(state.discriminator.state_dict()),
        "vector_adversary_state_dict": None if state.vector_adversary is None else state_dict_to_cpu(state.vector_adversary.state_dict()),
        "residual_adversary_state_dict": None if state.residual_adversary is None else state_dict_to_cpu(state.residual_adversary.state_dict()),
        "configs": {
            "data_config": asdict(ctx.data_config),
            "prompt_config": asdict(ctx.prompt_config),
            "model_config": asdict(ctx.model_config),
            "loss_config": asdict(ctx.loss_config),
            "schedule_config": asdict(ctx.schedule_config),
            "experiment_config": asdict(ctx.experiment_config),
        },
        "emotion_names": list(ctx.emotion_names),
        "val_metrics": metric_summary_for_checkpoint(val_metrics),
        "test_metrics": metric_summary_for_checkpoint(test_metrics),
    }
    if include_training_state:
        payload.update({
            "optimizer_state_dict": state.optimizer.state_dict(),
            "disc_optimizer_state_dict": None if state.disc_optimizer is None else state.disc_optimizer.state_dict(),
            "vector_adv_optimizer_state_dict": None if state.vector_adv_optimizer is None else state.vector_adv_optimizer.state_dict(),
            "residual_adv_optimizer_state_dict": None if state.residual_adv_optimizer is None else state.residual_adv_optimizer.state_dict(),
            "lr_scheduler_state_dict": state.lr_scheduler.state_dict(),
        })
    if include_history:
        payload["history"] = state.history
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
