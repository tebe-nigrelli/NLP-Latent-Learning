#!/bin/bash
#SBATCH --job-name=nlp-factorvae-summary
#SBATCH --partition=debug_gpuh200
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=output/%x_%j.out
#SBATCH --error=output/%x_%j.err

set -euo pipefail

# Evaluate the base model variant:
#   T5 hidden-state VAE + scalar attention pooling
#   no skip connection, no LoRA, no vector/residual adversarial losses, no auxiliary edit/copy losses.
#
# Submit from the repository root:
#   sbatch evaluate_base_vae_attention_disentanglement.slurm
#
# Required by default:
#   BASE_CHECKPOINT_PATH=/path/to/base/best_checkpoint.pt
# or place a checkpoint at:
#   ${BASE_OUTPUT_DIR}/best_checkpoint.pt
#
# Useful overrides:
#   sbatch --export=ALL,BASE_CHECKPOINT_PATH=base_vae_attention_artifacts/best_checkpoint.pt evaluate_base_vae_attention_disentanglement.slurm
#   sbatch --export=ALL,BASE_TEST_SAMPLES=1024,BASE_LATENT_DIAG_MAX_SAMPLES=1024 evaluate_base_vae_attention_disentanglement.slurm

PROJECT_DIR="${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
cd "$PROJECT_DIR"

mkdir -p output

export HF_HOME="${HF_HOME:-$PROJECT_DIR/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PROJECT_DIR/.cache/uv}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

export BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-base_vae_attention_from_scratch_artifacts}"
export BASE_EVAL_OUTPUT_DIR="${BASE_EVAL_OUTPUT_DIR:-base_vae_attention_from_scratch_artifacts}"
export BASE_CHECKPOINT_PATH="${BASE_CHECKPOINT_PATH:-$BASE_OUTPUT_DIR/best_checkpoint.pt}"
export BASE_MODEL_NAME="${BASE_MODEL_NAME:-}"
export BASE_MAX_LENGTH="${BASE_MAX_LENGTH:-128}"
export BASE_EVAL_BATCH_SIZE="${BASE_EVAL_BATCH_SIZE:-16}"
export BASE_NUM_WORKERS="${BASE_NUM_WORKERS:-0}"
export BASE_SEED="${BASE_SEED:-42}"

# Set a sample count to 0 to use the full split.
export BASE_THRESHOLD_SAMPLES="${BASE_THRESHOLD_SAMPLES:-0}"
export BASE_VAL_SAMPLES="${BASE_VAL_SAMPLES:-0}"
export BASE_TEST_SAMPLES="${BASE_TEST_SAMPLES:-0}"

# Retune thresholds on the calibration split before validation/test evaluation.
export BASE_RETUNE_THRESHOLDS="${BASE_RETUNE_THRESHOLDS:-1}"
export BASE_THRESHOLD_MODE="${BASE_THRESHOLD_MODE:-per_label}"
export BASE_THRESHOLD_GRID_POINTS="${BASE_THRESHOLD_GRID_POINTS:-101}"

# Latent/disentanglement diagnostics.
export BASE_LATENT_DIAG_MAX_SAMPLES="${BASE_LATENT_DIAG_MAX_SAMPLES:-4096}"
export BASE_LATENT_DIAG_RIDGE_ALPHA="${BASE_LATENT_DIAG_RIDGE_ALPHA:-1.0}"
export BASE_LATENT_DIAG_PROBE_EVAL_FRACTION="${BASE_LATENT_DIAG_PROBE_EVAL_FRACTION:-0.30}"
export BASE_LATENT_DIAG_MIN_PROBE_TRAIN="${BASE_LATENT_DIAG_MIN_PROBE_TRAIN:-64}"
export BASE_LATENT_ACTIVE_VARIANCE_THRESHOLD="${BASE_LATENT_ACTIVE_VARIANCE_THRESHOLD:-0.0001}"

# Base-model evaluation weights. Auxiliary losses are intentionally zeroed.
export BASE_CLASSIFICATION_WEIGHT="${BASE_CLASSIFICATION_WEIGHT:-4.0}"
export BASE_RECON_WEIGHT="${BASE_RECON_WEIGHT:-2.0}"
export BASE_KL_WEIGHT="${BASE_KL_WEIGHT:-0.05}"
export BASE_TC_WEIGHT="${BASE_TC_WEIGHT:-0.1}"

# If set to 1, evaluate a freshly initialized base model when the checkpoint is missing.
# Default is 0 because that usually indicates a bad path.
export BASE_ALLOW_UNTRAINED="${BASE_ALLOW_UNTRAINED:-0}"

printf 'Project dir: %s\n' "$PROJECT_DIR"
printf 'Checkpoint: %s\n' "$BASE_CHECKPOINT_PATH"
printf 'Eval output dir: %s\n' "$BASE_EVAL_OUTPUT_DIR"
printf 'Split sample caps threshold/val/test: %s/%s/%s\n' "$BASE_THRESHOLD_SAMPLES" "$BASE_VAL_SAMPLES" "$BASE_TEST_SAMPLES"
printf 'Latent diagnostics max samples: %s\n' "$BASE_LATENT_DIAG_MAX_SAMPLES"

uv run python - <<'PY'
from __future__ import annotations

import dataclasses
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path.cwd()
SRC = PROJECT_ROOT / "src"
if str(SRC.resolve()) not in sys.path:
    sys.path.insert(0, str(SRC.resolve()))

from emotion_latent_learning.data import EmotionTextCollator
from emotion_latent_learning.evaluation.metrics import evaluate_loader, summarize_eval_metrics
from emotion_latent_learning.evaluation.thresholds import restore_threshold
from emotion_latent_learning.training.checkpointing import load_compact_model_state_dict
from emotion_latent_learning.training.runtime import (
    build_data_bundle,
    build_runtime,
    default_configs,
    get_device,
    runtime_summary,
)
from emotion_latent_learning.training.schedules import current_loss_weights
from emotion_latent_learning.utils import save_json, set_seed, to_serializable


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value is None or value == "" else int(value)


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return default if value is None or value == "" else float(value)


def env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def apply_dataclass_payload(obj: Any, payload: Optional[Dict[str, Any]]) -> None:
    if not payload:
        return
    field_names = {field.name for field in dataclasses.fields(obj)}
    for key, value in payload.items():
        if key in field_names:
            setattr(obj, key, value)


def subset_or_full(dataset: Any, n: int) -> Any:
    if n is None or int(n) <= 0:
        return dataset
    n = max(1, min(int(n), len(dataset)))
    return Subset(dataset, list(range(n)))


def make_loader(dataset: Any, batch_size: int, collator: Any, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        collate_fn=collator,
    )


def metric_get(payload: Optional[Dict[str, Any]], key: str, default: float = 0.0) -> float:
    if not isinstance(payload, dict):
        return default
    value = payload.get(key, default)
    try:
        return float(value)
    except Exception:
        return default


def print_probe(name: str, probe: Optional[Dict[str, Any]]) -> None:
    if not isinstance(probe, dict):
        print(f"  {name}: unavailable")
        return
    print(
        f"  {name}: "
        f"r2_macro={metric_get(probe, 'ridge_r2_macro'):.4f} "
        f"pearson_macro={metric_get(probe, 'ridge_pearson_macro'):.4f} "
        f"mae_macro={metric_get(probe, 'ridge_mae_macro'):.4f} "
        f"heldout={bool(probe.get('probe_is_heldout', False))} "
        f"train={int(probe.get('probe_train_examples', 0) or 0)} "
        f"eval={int(probe.get('probe_eval_examples', 0) or 0)}"
    )


def print_disentanglement(prefix: str, metrics: Dict[str, Any], max_labels: int = 28) -> None:
    diag = metrics.get("latent_diagnostics") or {}
    factor = diag.get("factor_power") or metrics.get("factor_power") or {}
    split = diag.get("emotion_meaning_split") or metrics.get("emotion_meaning_split") or {}
    dci = factor.get("dci") or {}
    align = factor.get("diagonal_alignment") or {}
    clf_power = factor.get("classifier_weight_power") or {}
    branch_dci = split.get("branch_group_dci") or {}
    branch_mig = split.get("branch_group_mig") or {}

    print(f"\n=== {prefix} disentanglement diagnostics ===")
    print(
        "diagnostic_sample: "
        f"examples={int(diag.get('num_examples_used', 0) or 0)} "
        f"probe_train={int(diag.get('probe_train_examples', split.get('probe_train_examples', 0)) or 0)} "
        f"probe_eval={int(diag.get('probe_eval_examples', split.get('probe_eval_examples', 0)) or 0)} "
        f"heldout={bool(diag.get('probe_is_heldout', split.get('probe_is_heldout', False)))}"
    )
    print(
        "factor_power: "
        f"dci_disentanglement={metric_get(dci, 'dci_disentanglement'):.4f} "
        f"dci_completeness={metric_get(dci, 'dci_completeness'):.4f} "
        f"effective_num_factors={metric_get(factor, 'effective_num_factors'):.4f} "
        f"active_scalar_factors={int(factor.get('active_scalar_factors', 0) or 0)} "
        f"mean_abs_factor_label_corr={metric_get(factor, 'mean_abs_factor_label_correlation'):.4f} "
        f"max_abs_factor_label_corr={metric_get(factor, 'max_abs_factor_label_correlation'):.4f}"
    )
    print(
        "diagonal_alignment: "
        f"mean_abs_diag_corr={metric_get(align, 'mean_abs_diagonal_correlation'):.4f} "
        f"mean_abs_offdiag_corr={metric_get(align, 'mean_abs_offdiagonal_correlation'):.4f} "
        f"diagonal_dominance={metric_get(align, 'diagonal_dominance'):.4f}"
    )
    if clf_power.get("available"):
        print(
            "classifier_weight_power: "
            f"diag_power_mean={metric_get(clf_power, 'diagonal_power_mean'):.4f} "
            f"offdiag_power_mean={metric_get(clf_power, 'offdiagonal_power_mean'):.4f} "
            f"diagonal_dominance={metric_get(clf_power, 'diagonal_dominance'):.4f}"
        )
    else:
        print(f"classifier_weight_power: unavailable ({clf_power.get('reason', 'not exposed')})")

    print(
        "branch_split: "
        f"branch_dci_disentanglement={metric_get(branch_dci, 'dci_disentanglement'):.4f} "
        f"branch_dci_completeness={metric_get(branch_dci, 'dci_completeness'):.4f} "
        f"signed_group_mig={metric_get(branch_mig, 'emotion_vs_meaning_mig_signed_macro'):.4f} "
        f"positive_group_mig={metric_get(branch_mig, 'emotion_vs_meaning_mig_positive_macro'):.4f} "
        f"standard_group_mig={metric_get(branch_mig, 'standard_group_mig_macro'):.4f} "
        f"scalar_win_rate={metric_get(branch_mig, 'emotion_scalar_win_rate'):.4f}"
    )
    print(
        "branch_leakage: "
        f"emotion_branch_importance_share={metric_get(split, 'emotion_branch_importance_share'):.4f} "
        f"meaning_branch_emotion_leakage_share={metric_get(split, 'meaning_branch_emotion_leakage_share'):.4f} "
        f"residual_emotion_leakage_share={metric_get(split, 'residual_emotion_leakage_share'):.4f} "
        f"emotion_scalar_share_r2={metric_get(split, 'emotion_scalar_share_r2'):.4f} "
        f"emotion_leakage_ratio_r2={metric_get(split, 'emotion_leakage_ratio_r2'):.4f}"
    )
    print(
        "branch_predictivity: "
        f"emotion_in_scalar_r2={metric_get(split, 'emotion_in_scalar_r2'):.4f} "
        f"emotion_leakage_vector_r2={metric_get(split, 'emotion_leakage_vector_r2'):.4f} "
        f"emotion_leakage_residual_r2={metric_get(split, 'emotion_leakage_residual_r2'):.4f} "
        f"separation_r2={metric_get(split, 'emotion_meaning_separation_r2'):.4f} "
        f"separation_pearson={metric_get(split, 'emotion_meaning_separation_pearson'):.4f}"
    )
    print(
        "scalar_vector_correlation: "
        f"mean_abs={metric_get(split, 'scalar_vector_mean_abs_correlation'):.4f} "
        f"max_abs={metric_get(split, 'scalar_vector_max_abs_correlation'):.4f}"
    )

    print("probe_scores:")
    print_probe("scalar_emotion_probe", split.get("scalar_emotion_probe"))
    print_probe("vector_emotion_probe", split.get("vector_emotion_probe"))
    print_probe("residual_emotion_probe", split.get("residual_emotion_probe"))
    print_probe("full_emotion_probe", split.get("full_emotion_probe"))

    per_label = branch_mig.get("per_label") or {}
    if per_label:
        print("per_label_branch_winners:")
        for idx, (label, row) in enumerate(per_label.items()):
            if idx >= max_labels:
                remaining = len(per_label) - max_labels
                if remaining > 0:
                    print(f"  ... {remaining} more labels omitted")
                break
            print(
                f"  {label}: winner={row.get('winning_group')} "
                f"signed_mig={metric_get(row, 'emotion_vs_meaning_mig_signed'):.4f} "
                f"scalar_mi={metric_get(row, 'emotion_scalar_max_mi'):.4f} "
                f"vector_mi={metric_get(row, 'meaning_vector_max_mi'):.4f} "
                f"residual_mi={metric_get(row, 'residual_max_mi'):.4f}"
            )
    print(f"=== end {prefix} disentanglement diagnostics ===\n")


def load_checkpoint_if_available(runtime: Any, checkpoint_path: Path, allow_untrained: bool) -> Dict[str, Any]:
    state = runtime.state
    ctx = runtime.ctx
    if not checkpoint_path.exists():
        if allow_untrained:
            print(f"warning: checkpoint not found at {checkpoint_path}; evaluating randomly initialized model")
            return {"missing_checkpoint": str(checkpoint_path)}
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Set BASE_CHECKPOINT_PATH or BASE_ALLOW_UNTRAINED=1."
        )
    checkpoint = torch.load(checkpoint_path, map_location=ctx.device)
    load_compact_model_state_dict(state.model, checkpoint["model_state_dict"])
    if state.discriminator is not None and checkpoint.get("discriminator_state_dict") is not None:
        state.discriminator.load_state_dict(checkpoint["discriminator_state_dict"], strict=False)
    if state.vector_adversary is not None and checkpoint.get("vector_adversary_state_dict") is not None:
        state.vector_adversary.load_state_dict(checkpoint["vector_adversary_state_dict"], strict=False)
    if state.residual_adversary is not None and checkpoint.get("residual_adversary_state_dict") is not None:
        state.residual_adversary.load_state_dict(checkpoint["residual_adversary_state_dict"], strict=False)
    if "threshold" in checkpoint:
        state.threshold = restore_threshold(
            checkpoint["threshold"],
            num_labels=len(ctx.emotion_names),
            mode=ctx.experiment_config.threshold_mode,
        )
    print(f"loaded checkpoint: {checkpoint_path}")
    print(f"checkpoint epoch: {checkpoint.get('epoch')} step: {checkpoint.get('step')}")
    return checkpoint


checkpoint_path = Path(os.environ["BASE_CHECKPOINT_PATH"]).expanduser()
checkpoint_payload: Dict[str, Any] = {}
if checkpoint_path.exists():
    checkpoint_payload = torch.load(checkpoint_path, map_location="cpu")

DATA_CONFIG, PROMPT_CONFIG, MODEL_CONFIG, LOSS_CONFIG, SCHEDULE_CONFIG, EXPERIMENT_CONFIG = default_configs()
ckpt_configs = checkpoint_payload.get("configs", {}) if isinstance(checkpoint_payload, dict) else {}
apply_dataclass_payload(DATA_CONFIG, ckpt_configs.get("data_config"))
apply_dataclass_payload(PROMPT_CONFIG, ckpt_configs.get("prompt_config"))
apply_dataclass_payload(MODEL_CONFIG, ckpt_configs.get("model_config") or checkpoint_payload.get("model_config"))
apply_dataclass_payload(LOSS_CONFIG, ckpt_configs.get("loss_config"))
apply_dataclass_payload(SCHEDULE_CONFIG, ckpt_configs.get("schedule_config"))
apply_dataclass_payload(EXPERIMENT_CONFIG, ckpt_configs.get("experiment_config"))

# Enforce the intended base model variant. These settings define the ablation.
EXPERIMENT_CONFIG.seed = env_int("BASE_SEED", int(getattr(EXPERIMENT_CONFIG, "seed", 42)))
EXPERIMENT_CONFIG.output_dir = os.environ.get("BASE_EVAL_OUTPUT_DIR", "base_vae_attention_eval_artifacts")
EXPERIMENT_CONFIG.threshold_mode = os.environ.get("BASE_THRESHOLD_MODE", "per_label")
EXPERIMENT_CONFIG.threshold_grid_points = env_int("BASE_THRESHOLD_GRID_POINTS", 101)
EXPERIMENT_CONFIG.sample_posterior_eval = False
EXPERIMENT_CONFIG.show_num_examples = 0
EXPERIMENT_CONFIG.monitor_num_examples = 0
EXPERIMENT_CONFIG.monitor_edit_steps = 0
EXPERIMENT_CONFIG.monitor_attention_enabled = False
EXPERIMENT_CONFIG.prompt_eval_num_examples = 0
EXPERIMENT_CONFIG.eval_num_copy_examples = 0
EXPERIMENT_CONFIG.latent_diagnostics_enabled = True
EXPERIMENT_CONFIG.latent_diagnostics_max_samples = env_int("BASE_LATENT_DIAG_MAX_SAMPLES", 4096)
EXPERIMENT_CONFIG.latent_diagnostics_ridge_alpha = env_float("BASE_LATENT_DIAG_RIDGE_ALPHA", 1.0)
EXPERIMENT_CONFIG.latent_diagnostics_probe_eval_fraction = env_float("BASE_LATENT_DIAG_PROBE_EVAL_FRACTION", 0.30)
EXPERIMENT_CONFIG.latent_diagnostics_min_probe_train_examples = env_int("BASE_LATENT_DIAG_MIN_PROBE_TRAIN", 64)
EXPERIMENT_CONFIG.latent_active_variance_threshold = env_float("BASE_LATENT_ACTIVE_VARIANCE_THRESHOLD", 1e-4)

DATA_CONFIG.max_length = env_int("BASE_MAX_LENGTH", int(getattr(DATA_CONFIG, "max_length", 128)))
DATA_CONFIG.eval_batch_size = env_int("BASE_EVAL_BATCH_SIZE", int(getattr(DATA_CONFIG, "eval_batch_size", 16)))
DATA_CONFIG.num_workers = env_int("BASE_NUM_WORKERS", 0)
DATA_CONFIG.text_denoising_enabled = False

model_name_override = os.environ.get("BASE_MODEL_NAME", "").strip()
if model_name_override:
    MODEL_CONFIG.model_name = model_name_override
MODEL_CONFIG.use_lora = False
MODEL_CONFIG.use_skip_connection = False
MODEL_CONFIG.attention_source = getattr(MODEL_CONFIG, "attention_source", "encoder_sequence") or "encoder_sequence"
MODEL_CONFIG.pooling_mode = getattr(MODEL_CONFIG, "pooling_mode", "per_scalar_dim") or "per_scalar_dim"
MODEL_CONFIG.classifier_mode = getattr(MODEL_CONFIG, "classifier_mode", "per_emotion_mlp") or "per_emotion_mlp"
MODEL_CONFIG.decoder_conditioning_mode = getattr(MODEL_CONFIG, "decoder_conditioning_mode", "split_film") or "split_film"

LOSS_CONFIG.classification_weight = env_float("BASE_CLASSIFICATION_WEIGHT", 4.0)
LOSS_CONFIG.recon_weight = env_float("BASE_RECON_WEIGHT", 2.0)
LOSS_CONFIG.kl_weight = env_float("BASE_KL_WEIGHT", 0.05)
LOSS_CONFIG.tc_weight = env_float("BASE_TC_WEIGHT", 0.1)
LOSS_CONFIG.vector_adv_weight = 0.0
LOSS_CONFIG.residual_adv_weight = 0.0
LOSS_CONFIG.vector_sep_weight = 0.0
LOSS_CONFIG.scalar_alignment_weight = 0.0
LOSS_CONFIG.vector_label_leakage_weight = 0.0
LOSS_CONFIG.orthogonality_weight = 0.0
LOSS_CONFIG.transfer_strength_weight = 0.0
LOSS_CONFIG.edit_sensitivity_weight = 0.0
LOSS_CONFIG.copy_weight = 0.0

SCHEDULE_CONFIG.kl_zero_epochs = 0
SCHEDULE_CONFIG.kl_warmup_end_epoch = 1
SCHEDULE_CONFIG.copy_loss_start_epoch = 10**9
SCHEDULE_CONFIG.lora_start_epoch = 10**9
SCHEDULE_CONFIG.vector_adv_warmup_epochs = 1
SCHEDULE_CONFIG.residual_adv_warmup_epochs = 1

set_seed(EXPERIMENT_CONFIG.seed)
device = get_device()
print("device:", device)
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))

print("base evaluation config:")
print(json.dumps({
    "model_name": MODEL_CONFIG.model_name,
    "num_scalar_factors": MODEL_CONFIG.num_scalar_factors,
    "vector_latent_dim": MODEL_CONFIG.vector_latent_dim,
    "latent_pool_heads": MODEL_CONFIG.latent_pool_heads,
    "attention_source": MODEL_CONFIG.attention_source,
    "pooling_mode": MODEL_CONFIG.pooling_mode,
    "classifier_mode": MODEL_CONFIG.classifier_mode,
    "use_skip_connection": MODEL_CONFIG.use_skip_connection,
    "use_lora": MODEL_CONFIG.use_lora,
    "tc_weight": LOSS_CONFIG.tc_weight,
    "vector_adv_weight": LOSS_CONFIG.vector_adv_weight,
    "residual_adv_weight": LOSS_CONFIG.residual_adv_weight,
    "latent_diagnostics_max_samples": EXPERIMENT_CONFIG.latent_diagnostics_max_samples,
}, indent=2))

print("loading data...")
data = build_data_bundle(
    data_config=DATA_CONFIG,
    model_config=MODEL_CONFIG,
    experiment_config=EXPERIMENT_CONFIG,
)
collator = EmotionTextCollator(tokenizer=data.tokenizer, max_length=DATA_CONFIG.max_length)

data.threshold_tune_dataset = subset_or_full(data.threshold_tune_dataset, env_int("BASE_THRESHOLD_SAMPLES", 0))
data.val_dataset = subset_or_full(data.val_dataset, env_int("BASE_VAL_SAMPLES", 0))
data.test_dataset = subset_or_full(data.test_dataset, env_int("BASE_TEST_SAMPLES", 0))
data.threshold_loader = make_loader(data.threshold_tune_dataset, DATA_CONFIG.eval_batch_size, collator, DATA_CONFIG.num_workers)
data.val_loader = make_loader(data.val_dataset, DATA_CONFIG.eval_batch_size, collator, DATA_CONFIG.num_workers)
data.test_loader = make_loader(data.test_dataset, DATA_CONFIG.eval_batch_size, collator, DATA_CONFIG.num_workers)
data.text_denoising_dataset = None
data.text_denoising_loader = None

runtime = build_runtime(
    data=data,
    data_config=DATA_CONFIG,
    prompt_config=PROMPT_CONFIG,
    model_config=MODEL_CONFIG,
    loss_config=LOSS_CONFIG,
    schedule_config=SCHEDULE_CONFIG,
    experiment_config=EXPERIMENT_CONFIG,
    device=device,
)
print("runtime summary:")
print(json.dumps(runtime_summary(runtime), indent=2, default=str))

checkpoint = load_checkpoint_if_available(
    runtime=runtime,
    checkpoint_path=checkpoint_path,
    allow_untrained=env_bool("BASE_ALLOW_UNTRAINED", False),
)

# Use a fully warmed base loss profile for evaluation.
epoch_for_weights = max(1, int(checkpoint.get("epoch", 1) or 1)) if isinstance(checkpoint, dict) else 1
weights = current_loss_weights(epoch=epoch_for_weights, ctx=runtime.ctx)
print("evaluation weights:")
print(json.dumps({k: float(v) for k, v in weights.items()}, indent=2))

metrics_payload: Dict[str, Any] = {
    "checkpoint_path": str(checkpoint_path),
    "checkpoint_epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
    "config": {
        "data_config": dataclasses.asdict(DATA_CONFIG),
        "prompt_config": dataclasses.asdict(PROMPT_CONFIG),
        "model_config": dataclasses.asdict(MODEL_CONFIG),
        "loss_config": dataclasses.asdict(LOSS_CONFIG),
        "schedule_config": dataclasses.asdict(SCHEDULE_CONFIG),
        "experiment_config": dataclasses.asdict(EXPERIMENT_CONFIG),
    },
}

if env_bool("BASE_RETUNE_THRESHOLDS", True):
    threshold_metrics = evaluate_loader(
        state=runtime.state,
        ctx=runtime.ctx,
        data_loader=runtime.data.threshold_loader,
        desc="base threshold calibration",
        tune_threshold=True,
        weights=weights,
    )
    runtime.state.threshold = threshold_metrics["threshold"]
    summarize_eval_metrics(prefix="[base] calib ", metrics=threshold_metrics, num_labels=len(runtime.ctx.emotion_names))
    print_disentanglement("calibration", threshold_metrics)
    metrics_payload["threshold_tune"] = threshold_metrics
else:
    print("threshold retuning disabled; using checkpoint/default threshold")

val_metrics = evaluate_loader(
    state=runtime.state,
    ctx=runtime.ctx,
    data_loader=runtime.data.val_loader,
    desc="base validation",
    tune_threshold=False,
    weights=weights,
)
summarize_eval_metrics(prefix="[base] val   ", metrics=val_metrics, num_labels=len(runtime.ctx.emotion_names))
print_disentanglement("validation", val_metrics)
metrics_payload["validation"] = val_metrics

test_metrics = evaluate_loader(
    state=runtime.state,
    ctx=runtime.ctx,
    data_loader=runtime.data.test_loader,
    desc="base test",
    tune_threshold=False,
    weights=weights,
)
summarize_eval_metrics(prefix="[base] test  ", metrics=test_metrics, num_labels=len(runtime.ctx.emotion_names))
print_disentanglement("test", test_metrics)
metrics_payload["test"] = test_metrics

output_dir = Path(EXPERIMENT_CONFIG.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)
save_json(output_dir / "base_eval_metrics_with_disentanglement.json", metrics_payload)

compact = {
    "checkpoint_path": str(checkpoint_path),
    "validation": {
        "micro_f1": val_metrics.get("micro_f1"),
        "macro_f1": val_metrics.get("macro_f1"),
        "factor_dci_disentanglement": val_metrics.get("factor_dci_disentanglement"),
        "branch_mig": val_metrics.get("branch_mig"),
        "emotion_branch_importance_share": val_metrics.get("emotion_branch_importance_share"),
        "meaning_branch_emotion_leakage_share": val_metrics.get("meaning_branch_emotion_leakage_share"),
        "emotion_meaning_separation_r2": val_metrics.get("emotion_meaning_separation_r2"),
        "scalar_vector_mean_abs_correlation": val_metrics.get("scalar_vector_mean_abs_correlation"),
    },
    "test": {
        "micro_f1": test_metrics.get("micro_f1"),
        "macro_f1": test_metrics.get("macro_f1"),
        "factor_dci_disentanglement": test_metrics.get("factor_dci_disentanglement"),
        "branch_mig": test_metrics.get("branch_mig"),
        "emotion_branch_importance_share": test_metrics.get("emotion_branch_importance_share"),
        "meaning_branch_emotion_leakage_share": test_metrics.get("meaning_branch_emotion_leakage_share"),
        "emotion_meaning_separation_r2": test_metrics.get("emotion_meaning_separation_r2"),
        "scalar_vector_mean_abs_correlation": test_metrics.get("scalar_vector_mean_abs_correlation"),
    },
}
save_json(output_dir / "base_eval_summary.json", compact)
print("saved full metrics:", (output_dir / "base_eval_metrics_with_disentanglement.json").resolve())
print("saved compact summary:", (output_dir / "base_eval_summary.json").resolve())
print("base VAE attention evaluation completed")
PY
