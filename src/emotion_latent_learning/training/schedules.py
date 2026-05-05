from __future__ import annotations

from ..utils.common import *
from ..schemas import ExperimentContext, TrainingState
from ..modeling.t5_utils import set_lora_trainable

# Focused module: schedules.

def linear_value_schedule(
    epoch: int,
    start_epoch: int,
    end_epoch: int,
    start_value: float,
    end_value: float,
) -> float:
    if epoch <= start_epoch:
        return float(start_value)
    if epoch >= end_epoch:
        return float(end_value)
    span = float(max(1, end_epoch - start_epoch))
    progress = float(epoch - start_epoch) / span
    return float(start_value + progress * (end_value - start_value))


def current_loss_weights(epoch: int, ctx: ExperimentContext) -> Dict[str, float]:
    vector_adv_scale = min(1.0, float(epoch) / float(max(1, ctx.schedule_config.vector_adv_warmup_epochs)))
    residual_adv_scale = min(1.0, float(epoch) / float(max(1, ctx.schedule_config.residual_adv_warmup_epochs)))
    residual_scale = 0.0
    if ctx.model_config.use_skip_connection:
        residual_scale = linear_value_schedule(
            epoch=epoch,
            start_epoch=ctx.schedule_config.residual_decay_start_epoch,
            end_epoch=ctx.schedule_config.residual_decay_end_epoch,
            start_value=ctx.schedule_config.residual_scale_start,
            end_value=ctx.schedule_config.residual_scale_end,
        )
    # KL warmup schedule: autoencoder mode (B=0) for kl_zero_epochs, then gradually turn on KL
    kl_weight = linear_value_schedule(
        epoch=epoch,
        start_epoch=ctx.schedule_config.kl_zero_epochs,
        end_epoch=ctx.schedule_config.kl_warmup_end_epoch,
        start_value=0.0,
        end_value=ctx.loss_config.kl_weight,
    )
    tc_weight = ctx.loss_config.tc_weight if kl_weight > 0.0 else 0.0
    return {
        "classification_weight": ctx.loss_config.classification_weight,
        "recon_weight": ctx.loss_config.recon_weight,
        "kl_weight": kl_weight,
        "tc_weight": tc_weight,
        "copy_weight": ctx.loss_config.copy_weight if epoch >= ctx.schedule_config.copy_loss_start_epoch else 0.0,
        "vector_adv_weight": ctx.loss_config.vector_adv_weight * vector_adv_scale,
        "residual_adv_weight": (ctx.loss_config.residual_adv_weight * residual_adv_scale) if ctx.model_config.use_skip_connection else 0.0,
        "vector_sep_weight": ctx.loss_config.vector_sep_weight,
        "orthogonality_weight": ctx.loss_config.orthogonality_weight,
        "transfer_strength_weight": ctx.loss_config.transfer_strength_weight,
        "residual_scale": residual_scale,
    }


def copy_loss_should_freeze_vae(epoch: int, ctx: ExperimentContext) -> bool:
    freeze_epochs = max(0, int(ctx.schedule_config.copy_loss_vae_freeze_epochs))
    if freeze_epochs <= 0:
        return False
    freeze_end_epoch = ctx.schedule_config.copy_loss_start_epoch + freeze_epochs
    return ctx.schedule_config.copy_loss_start_epoch <= epoch < freeze_end_epoch


def ensure_lora_schedule(epoch: int, state: TrainingState, ctx: ExperimentContext) -> None:
    enable_lora = epoch >= ctx.schedule_config.lora_start_epoch
    set_lora_trainable(state.model.shared_t5, enabled=enable_lora)


def slim_history_row(epoch: int, split_name: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
    keep_keys = [
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
        "intensity_mae",
        "intensity_rmse",
        "intensity_pearson_macro",
        "semeval_ei_reg_official_score",
        "semeval_ei_reg_pearson_macro",
        "semeval_ei_reg_pearson_high_gold_macro",
        "factor_dci_disentanglement",
        "factor_dci_completeness",
        "factor_effective_num_factors",
        "factor_active_scalar_factors",
        "emotion_meaning_separation_r2",
        "emotion_meaning_separation_pearson",
        "emotion_leakage_vector_r2",
        "emotion_in_scalar_r2",
        "scalar_vector_mean_abs_correlation",
    ]
    row = {"epoch": epoch, "split": split_name}
    for key in keep_keys:
        if key in metrics:
            value = metrics[key]
            row[key] = float(value) if isinstance(value, (int, float, np.floating, np.integer)) else value
    return row
