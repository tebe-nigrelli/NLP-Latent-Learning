#!/bin/bash
#SBATCH --job-name=nlp-skip-adv-dlora
#SBATCH --partition=medium_gpuh200
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --array=0-1
#SBATCH --output=output/%x_%A_%a.out
#SBATCH --error=output/%x_%A_%a.err

set -euo pipefail

export BASE_COPY_WEIGHT="${ABLATION_COPY_WEIGHT:-${BASE_COPY_WEIGHT:-1.0}}"
export BASE_COPY_LOSS_START_EPOCH="${ABLATION_COPY_LOSS_START_EPOCH:-${BASE_COPY_LOSS_START_EPOCH:-0}}"
export BASE_COPY_LOSS_DECAY_START_EPOCH="${ABLATION_COPY_LOSS_DECAY_START_EPOCH:-${BASE_COPY_LOSS_DECAY_START_EPOCH:-10**9}}"
export BASE_COPY_LOSS_DECAY_END_EPOCH="${ABLATION_COPY_LOSS_DECAY_END_EPOCH:-${BASE_COPY_LOSS_DECAY_END_EPOCH:-10**9}}"

# Skip-connection + decoder-only LoRA + adversarial-loss pooling ablation.
#
# Focused grid: 2 pooling modes only.
#   pooling modes: per_scalar_dim, joint_scalar_vector
#   vector dim:    64 fixed
#   scalar/classifier: 56 scalar factors with per_emotion_pair_mlp by default
#   skip:          enabled
#   decoder conditioning: standard (split_film disabled)
#   adversaries:   vector/meaning adversary and residual adversary enabled
#   LoRA:          decoder-only, fixed rank/targets by default
#
# Submit from the repository root:
#   sbatch train_eval_skip_adv_decoder_lora_pooling_ablation.slurm
#
# Fast subset test:
#   sbatch --export=ALL,ABLATION_TRAIN_SAMPLES=4096,ABLATION_VAL_SAMPLES=1024,ABLATION_TEST_SAMPLES=1024 train_eval_skip_adv_decoder_lora_pooling_ablation.slurm
#
# Optional overrides:
#   ABLATION_VECTOR_ADV_WEIGHT=0.5
#   ABLATION_RESIDUAL_ADV_WEIGHT=1.0
#   ABLATION_DECODER_CONDITIONING_MODE=standard
#   ABLATION_LORA_R=16
#   ABLATION_LORA_TARGET_MODULES="q v"
#   ABLATION_CLASSIFIER_SETUP="per_emotion_pair_mlp:56:standard:32:128"
#
PROJECT_DIR="${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
cd "$PROJECT_DIR"

mkdir -p output

export HF_HOME="${HF_HOME:-$PROJECT_DIR/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PROJECT_DIR/.cache/uv}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# Ablation grid. The current Slurm task id indexes the Cartesian product.
export ABLATION_OUTPUT_ROOT="${ABLATION_OUTPUT_ROOT:-skip_adv_decoder_lora_pooling_ablation_artifacts}"
export ABLATION_POOLING_MODES="${ABLATION_POOLING_MODES:-per_scalar_dim joint_scalar_vector}"
export ABLATION_VECTOR_DIMS="${ABLATION_VECTOR_DIMS:-64}"
# Format per entry: classifier_mode:num_scalar_factors:classifier_parameterization:per_emotion_hidden_dim:joint_mlp_hidden_dim
export ABLATION_CLASSIFIER_SETUPS="${ABLATION_CLASSIFIER_SETUPS:-${ABLATION_CLASSIFIER_SETUP:-per_emotion_pair_mlp:56:standard:32:128}}"
export ABLATION_USE_SKIP_CONNECTION="${ABLATION_USE_SKIP_CONNECTION:-true}"
export ABLATION_RESIDUAL_SCALE_START="${ABLATION_RESIDUAL_SCALE_START:-1.0}"
export ABLATION_RESIDUAL_SCALE_END="${ABLATION_RESIDUAL_SCALE_END:-1.0}"
export ABLATION_RESIDUAL_ADV_WEIGHT="${ABLATION_RESIDUAL_ADV_WEIGHT:-1.0}"
export ABLATION_VECTOR_ADV_WEIGHT="${ABLATION_VECTOR_ADV_WEIGHT:-0.5}"
export ABLATION_USE_LORA="${ABLATION_USE_LORA:-true}"
export ABLATION_LORA_START_EPOCH="${ABLATION_LORA_START_EPOCH:-1}"
export ABLATION_LORA_R="${ABLATION_LORA_R:-16}"
export ABLATION_LORA_ALPHA="${ABLATION_LORA_ALPHA:-32}"
export ABLATION_LORA_DROPOUT="${ABLATION_LORA_DROPOUT:-0.1}"
export ABLATION_LORA_TARGET_MODULES="${ABLATION_LORA_TARGET_MODULES:-q v}"

read -r -a _POOLING_MODES <<< "$ABLATION_POOLING_MODES"
read -r -a _VECTOR_DIMS <<< "$ABLATION_VECTOR_DIMS"
read -r -a _CLASSIFIER_SETUPS <<< "$ABLATION_CLASSIFIER_SETUPS"

_TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
_COMBO_COUNT=$((${#_POOLING_MODES[@]} * ${#_VECTOR_DIMS[@]} * ${#_CLASSIFIER_SETUPS[@]}))
if (( _TASK_ID >= _COMBO_COUNT )); then
  echo "Task ${_TASK_ID} is outside combination count ${_COMBO_COUNT}; exiting cleanly."
  exit 0
fi

_SETUP_INDEX=$((_TASK_ID % ${#_CLASSIFIER_SETUPS[@]}))
_TMP=$((_TASK_ID / ${#_CLASSIFIER_SETUPS[@]}))
_VECTOR_INDEX=$((_TMP % ${#_VECTOR_DIMS[@]}))
_POOLING_INDEX=$((_TMP / ${#_VECTOR_DIMS[@]}))

export BASE_POOLING_MODE="${_POOLING_MODES[$_POOLING_INDEX]}"
export BASE_VECTOR_LATENT_DIM="${_VECTOR_DIMS[$_VECTOR_INDEX]}"

IFS=':' read -r BASE_CLASSIFIER_MODE BASE_NUM_SCALAR_FACTORS BASE_CLASSIFIER_PARAMETERIZATION BASE_PER_EMOTION_HIDDEN_DIM BASE_JOINT_MLP_HIDDEN_DIM <<< "${_CLASSIFIER_SETUPS[$_SETUP_INDEX]}"
export BASE_CLASSIFIER_MODE
export BASE_NUM_SCALAR_FACTORS
export BASE_CLASSIFIER_PARAMETERIZATION
export BASE_PER_EMOTION_HIDDEN_DIM
export BASE_JOINT_MLP_HIDDEN_DIM
export BASE_TOTAL_LATENT_DIM="$((BASE_NUM_SCALAR_FACTORS + BASE_VECTOR_LATENT_DIM))"
export BASE_USE_SKIP_CONNECTION="$ABLATION_USE_SKIP_CONNECTION"
export BASE_RESIDUAL_SCALE_START="$ABLATION_RESIDUAL_SCALE_START"
export BASE_RESIDUAL_SCALE_END="$ABLATION_RESIDUAL_SCALE_END"
export BASE_RESIDUAL_ADV_WEIGHT="$ABLATION_RESIDUAL_ADV_WEIGHT"
export BASE_VECTOR_ADV_WEIGHT="$ABLATION_VECTOR_ADV_WEIGHT"

_POOLING_TAG="${BASE_POOLING_MODE}"
_POOLING_TAG="${_POOLING_TAG//joint_scalar_vector/joint}"
_POOLING_TAG="${_POOLING_TAG//per_scalar_dim/peremotion}"
_CLASSIFIER_TAG="${BASE_CLASSIFIER_MODE}-${BASE_CLASSIFIER_PARAMETERIZATION}"
_CLASSIFIER_TAG="${_CLASSIFIER_TAG//_/-}"
_SKIP_TAG="skip-${BASE_USE_SKIP_CONNECTION}_rscale-${BASE_RESIDUAL_SCALE_START}-${BASE_RESIDUAL_SCALE_END}_vadv-${BASE_VECTOR_ADV_WEIGHT}_radv-${BASE_RESIDUAL_ADV_WEIGHT}_dec-${ABLATION_DECODER_CONDITIONING_MODE:-standard}"
_SKIP_TAG="${_SKIP_TAG//./p}"
_LORA_TAG="decoder-lora-r${ABLATION_LORA_R}-a${ABLATION_LORA_ALPHA}-start${ABLATION_LORA_START_EPOCH}"
export ABLATION_RUN_ID="pool-${_POOLING_TAG}_scalar-${BASE_NUM_SCALAR_FACTORS}_vector-${BASE_VECTOR_LATENT_DIM}_total-${BASE_TOTAL_LATENT_DIM}_classifier-${_CLASSIFIER_TAG}_${_SKIP_TAG}_${_LORA_TAG}_seed-${ABLATION_SEED:-42}"
export BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-${ABLATION_OUTPUT_ROOT}/${ABLATION_RUN_ID}}"

# Main run configuration.
export BASE_MODEL_NAME="${ABLATION_MODEL_NAME:-${BASE_MODEL_NAME:-google/flan-t5-base}}"
export BASE_USE_LORA="${ABLATION_USE_LORA:-${BASE_USE_LORA:-true}}"
export BASE_LORA_START_EPOCH="${ABLATION_LORA_START_EPOCH:-${BASE_LORA_START_EPOCH:-1}}"
export BASE_LORA_R="${ABLATION_LORA_R:-${BASE_LORA_R:-16}}"
export BASE_LORA_ALPHA="${ABLATION_LORA_ALPHA:-${BASE_LORA_ALPHA:-32}}"
export BASE_LORA_DROPOUT="${ABLATION_LORA_DROPOUT:-${BASE_LORA_DROPOUT:-0.1}}"
export BASE_LORA_TARGET_MODULES="${ABLATION_LORA_TARGET_MODULES:-${BASE_LORA_TARGET_MODULES:-q v}}"
export BASE_DECODER_CONDITIONING_MODE="${ABLATION_DECODER_CONDITIONING_MODE:-${BASE_DECODER_CONDITIONING_MODE:-standard}}"
export BASE_NUM_EPOCHS="${ABLATION_NUM_EPOCHS:-${BASE_NUM_EPOCHS:-10}}"
export BASE_SEED="${ABLATION_SEED:-${BASE_SEED:-42}}"
export BASE_MAX_LENGTH="${ABLATION_MAX_LENGTH:-${BASE_MAX_LENGTH:-128}}"
export BASE_TRAIN_BATCH_SIZE="${ABLATION_TRAIN_BATCH_SIZE:-${BASE_TRAIN_BATCH_SIZE:-8}}"
export BASE_EVAL_BATCH_SIZE="${ABLATION_EVAL_BATCH_SIZE:-${BASE_EVAL_BATCH_SIZE:-16}}"
export BASE_NUM_WORKERS="${ABLATION_NUM_WORKERS:-${BASE_NUM_WORKERS:-0}}"

# Set to 0 for full split. Defaults are full train/eval, but you can cap them for fast checks.
export BASE_TRAIN_SAMPLES="${ABLATION_TRAIN_SAMPLES:-${BASE_TRAIN_SAMPLES:-0}}"
export BASE_THRESHOLD_SAMPLES="${ABLATION_THRESHOLD_SAMPLES:-${BASE_THRESHOLD_SAMPLES:-0}}"
export BASE_VAL_SAMPLES="${ABLATION_VAL_SAMPLES:-${BASE_VAL_SAMPLES:-0}}"
export BASE_TEST_SAMPLES="${ABLATION_TEST_SAMPLES:-${BASE_TEST_SAMPLES:-0}}"

# Base architecture knobs.
export BASE_VAE_HIDDEN_DIM="${ABLATION_VAE_HIDDEN_DIM:-${BASE_VAE_HIDDEN_DIM:-1024}}"
export BASE_LATENT_POOL_HEADS="${ABLATION_LATENT_POOL_HEADS:-${BASE_LATENT_POOL_HEADS:-8}}"
export BASE_ATTENTION_SOURCE="${ABLATION_ATTENTION_SOURCE:-${BASE_ATTENTION_SOURCE:-encoder_sequence}}"
export BASE_CLASSIFIER_MODE="${BASE_CLASSIFIER_MODE:-per_emotion_mlp}"
export BASE_CLASSIFIER_PARAMETERIZATION="${BASE_CLASSIFIER_PARAMETERIZATION:-standard}"
export BASE_PER_EMOTION_HIDDEN_DIM="${BASE_PER_EMOTION_HIDDEN_DIM:-32}"
export BASE_JOINT_MLP_HIDDEN_DIM="${BASE_JOINT_MLP_HIDDEN_DIM:-128}"

# Thresholding and diagnostics.
export BASE_THRESHOLD_MODE="${ABLATION_THRESHOLD_MODE:-${BASE_THRESHOLD_MODE:-per_label}}"
export BASE_THRESHOLD_GRID_POINTS="${ABLATION_THRESHOLD_GRID_POINTS:-${BASE_THRESHOLD_GRID_POINTS:-101}}"
export BASE_LATENT_DIAG_MAX_SAMPLES="${ABLATION_LATENT_DIAG_MAX_SAMPLES:-${BASE_LATENT_DIAG_MAX_SAMPLES:-4096}}"
export BASE_LATENT_DIAG_RIDGE_ALPHA="${ABLATION_LATENT_DIAG_RIDGE_ALPHA:-${BASE_LATENT_DIAG_RIDGE_ALPHA:-1.0}}"
export BASE_LATENT_DIAG_PROBE_EVAL_FRACTION="${ABLATION_LATENT_DIAG_PROBE_EVAL_FRACTION:-${BASE_LATENT_DIAG_PROBE_EVAL_FRACTION:-0.30}}"
export BASE_LATENT_DIAG_MIN_PROBE_TRAIN="${ABLATION_LATENT_DIAG_MIN_PROBE_TRAIN:-${BASE_LATENT_DIAG_MIN_PROBE_TRAIN:-64}}"
export BASE_LATENT_ACTIVE_VARIANCE_THRESHOLD="${ABLATION_LATENT_ACTIVE_VARIANCE_THRESHOLD:-${BASE_LATENT_ACTIVE_VARIANCE_THRESHOLD:-0.0001}}"

# Base-model loss weights. Keeps VAE reconstruction/KL/classification, disables add-ons.
export BASE_CLASSIFICATION_WEIGHT="${ABLATION_CLASSIFICATION_WEIGHT:-${BASE_CLASSIFICATION_WEIGHT:-4.0}}"
export BASE_RECON_WEIGHT="${ABLATION_RECON_WEIGHT:-${BASE_RECON_WEIGHT:-2.0}}"
export BASE_KL_WEIGHT="${ABLATION_KL_WEIGHT:-${BASE_KL_WEIGHT:-0.05}}"
export BASE_TC_WEIGHT="${ABLATION_TC_WEIGHT:-${BASE_TC_WEIGHT:-0.1}}"
export BASE_ORTHOGONALITY_WEIGHT="${ABLATION_ORTHOGONALITY_WEIGHT:-${BASE_ORTHOGONALITY_WEIGHT:-0.0}}"
export BASE_VECTOR_SEP_WEIGHT="${ABLATION_VECTOR_SEP_WEIGHT:-${BASE_VECTOR_SEP_WEIGHT:-0.0}}"
export BASE_SCALAR_ALIGNMENT_WEIGHT="${ABLATION_SCALAR_ALIGNMENT_WEIGHT:-${BASE_SCALAR_ALIGNMENT_WEIGHT:-0.0}}"

# Optimizer/schedule.
export BASE_LR="${ABLATION_LR:-${BASE_LR:-2e-4}}"
export BASE_WEIGHT_DECAY="${ABLATION_WEIGHT_DECAY:-${BASE_WEIGHT_DECAY:-1e-2}}"
export BASE_WARMUP_RATIO="${ABLATION_WARMUP_RATIO:-${BASE_WARMUP_RATIO:-0.1}}"
export BASE_MIN_LR_SCALE="${ABLATION_MIN_LR_SCALE:-${BASE_MIN_LR_SCALE:-0.1}}"

printf 'Ablation task: %s / %s\n' "$_TASK_ID" "$((_COMBO_COUNT - 1))"
printf 'Ablation run id: %s\n' "$ABLATION_RUN_ID"
printf 'Pooling mode: %s\n' "$BASE_POOLING_MODE"
printf 'Scalar factors: %s\n' "$BASE_NUM_SCALAR_FACTORS"
printf 'Vector latent dim: %s\n' "$BASE_VECTOR_LATENT_DIM"
printf 'Total latent dim: %s\n' "$BASE_TOTAL_LATENT_DIM"

printf 'Project dir: %s\n' "$PROJECT_DIR"
printf 'Output dir: %s\n' "$BASE_OUTPUT_DIR"
printf 'Model: %s\n' "$BASE_MODEL_NAME"
printf 'Epochs: %s\n' "$BASE_NUM_EPOCHS"
printf 'Train/threshold/val/test sample caps: %s/%s/%s/%s\n' \
  "$BASE_TRAIN_SAMPLES" "$BASE_THRESHOLD_SAMPLES" "$BASE_VAL_SAMPLES" "$BASE_TEST_SAMPLES"
printf 'Classifier mode: %s; parameterization: %s; per-emotion hidden dim: %s\n' "$BASE_CLASSIFIER_MODE" "$BASE_CLASSIFIER_PARAMETERIZATION" "$BASE_PER_EMOTION_HIDDEN_DIM"
printf 'Skip + adversarial + decoder-only LoRA settings: skip=%s residual_scale=%s->%s vector_adv_weight=%s residual_adv_weight=%s decoder_mode=%s lora=%s lora_start=%s r=%s alpha=%s dropout=%s targets=%s; diagnostics on final eval.\n' \
  "$BASE_USE_SKIP_CONNECTION" "$BASE_RESIDUAL_SCALE_START" "$BASE_RESIDUAL_SCALE_END" "$ABLATION_VECTOR_ADV_WEIGHT" "$BASE_RESIDUAL_ADV_WEIGHT" "${ABLATION_DECODER_CONDITIONING_MODE:-standard}" \
  "$ABLATION_USE_LORA" "$ABLATION_LORA_START_EPOCH" "$ABLATION_LORA_R" "$ABLATION_LORA_ALPHA" "$ABLATION_LORA_DROPOUT" "$ABLATION_LORA_TARGET_MODULES"

uv run python - <<'PY'
from __future__ import annotations

import dataclasses
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path.cwd()
SRC = PROJECT_ROOT / "src"
if str(SRC.resolve()) not in sys.path:
    sys.path.insert(0, str(SRC.resolve()))

from emotion_latent_learning.data import EmotionTextCollator
from emotion_latent_learning.evaluation.metrics import evaluate_loader, summarize_eval_metrics
from emotion_latent_learning.evaluation.thresholds import restore_threshold
from emotion_latent_learning.training.checkpointing import save_checkpoint
from emotion_latent_learning.training.runtime import (
    build_data_bundle,
    build_runtime,
    default_configs,
    get_device,
    runtime_summary,
)
from emotion_latent_learning.training.runners import run_training
from emotion_latent_learning.training.schedules import current_loss_weights
from emotion_latent_learning.utils import save_json, set_seed, to_serializable


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value is None or value == "" else int(value)


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return default if value is None or value == "" else float(value)


def env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None or value == "" else value


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def subset_or_full(dataset: Any, n: int) -> Any:
    if n is None or int(n) <= 0:
        return dataset
    n = max(1, min(int(n), len(dataset)))
    return Subset(dataset, list(range(n)))


def make_train_loader(dataset: Any, batch_size: int, collator: Any, seed: int, num_workers: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        collate_fn=collator,
        generator=generator,
    )


def make_eval_loader(dataset: Any, batch_size: int, collator: Any, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        collate_fn=collator,
    )


def metric_float(payload: Optional[Dict[str, Any]], key: str, default: float = 0.0) -> float:
    if not isinstance(payload, dict):
        return default
    try:
        return float(payload.get(key, default))
    except Exception:
        return default


def print_probe(name: str, probe: Optional[Dict[str, Any]]) -> None:
    if not isinstance(probe, dict):
        print(f"  {name}: unavailable")
        return
    print(
        f"  {name}: "
        f"r2_macro={metric_float(probe, 'ridge_r2_macro'):.4f} "
        f"pearson_macro={metric_float(probe, 'ridge_pearson_macro'):.4f} "
        f"mae_macro={metric_float(probe, 'ridge_mae_macro'):.4f} "
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

    print(f"\n=== {prefix} latent/disentanglement diagnostics ===")
    print(
        "diagnostic sample: "
        f"examples={int(diag.get('num_examples_used', 0) or 0)} "
        f"probe_train={int(diag.get('probe_train_examples', split.get('probe_train_examples', 0)) or 0)} "
        f"probe_eval={int(diag.get('probe_eval_examples', split.get('probe_eval_examples', 0)) or 0)} "
        f"heldout={bool(diag.get('probe_is_heldout', split.get('probe_is_heldout', False)))}"
    )
    print(
        "factor DCI: "
        f"disentanglement={metric_float(dci, 'dci_disentanglement'):.4f} "
        f"completeness={metric_float(dci, 'dci_completeness'):.4f} "
        f"effective_num_factors={metric_float(factor, 'effective_num_factors'):.4f} "
        f"active_scalar_factors={int(factor.get('active_scalar_factors', 0) or 0)} "
        f"mean_abs_factor_label_corr={metric_float(factor, 'mean_abs_factor_label_correlation'):.4f} "
        f"max_abs_factor_label_corr={metric_float(factor, 'max_abs_factor_label_correlation'):.4f}"
    )
    print(
        "scalar-label diagonal alignment: "
        f"mean_abs_diag_corr={metric_float(align, 'mean_abs_diagonal_correlation'):.4f} "
        f"mean_abs_offdiag_corr={metric_float(align, 'mean_abs_offdiagonal_correlation'):.4f} "
        f"diagonal_dominance={metric_float(align, 'diagonal_dominance'):.4f}"
    )
    block_align = factor.get("block_alignment") or {}
    if block_align:
        print(
            "scalar-label block alignment: "
            f"factors_per_label={int(block_align.get('scalar_factors_per_label', 0) or 0)} "
            f"assigned_corr_dominance={metric_float(block_align, 'assigned_block_correlation_dominance'):.4f} "
            f"assigned_importance_dominance={metric_float(block_align, 'assigned_block_importance_dominance'):.4f} "
            f"top_corr_block_match_rate={metric_float(block_align, 'top_correlation_block_match_rate'):.4f}"
        )
    if clf_power.get("available"):
        print(
            "classifier weight power: "
            f"diag_mean={metric_float(clf_power, 'diagonal_power_mean'):.4f} "
            f"offdiag_mean={metric_float(clf_power, 'offdiagonal_power_mean'):.4f} "
            f"diagonal_dominance={metric_float(clf_power, 'diagonal_dominance'):.4f}"
        )
    else:
        print(f"classifier weight power: unavailable ({clf_power.get('reason', 'not exposed')})")

    print(
        "branch DCI/MIG: "
        f"branch_dci_disentanglement={metric_float(branch_dci, 'dci_disentanglement'):.4f} "
        f"branch_dci_completeness={metric_float(branch_dci, 'dci_completeness'):.4f} "
        f"branch_mig_signed_macro={metric_float(branch_mig, 'emotion_vs_meaning_mig_signed_macro'):.4f} "
        f"branch_mig_positive_macro={metric_float(branch_mig, 'emotion_vs_meaning_mig_positive_macro'):.4f} "
        f"emotion_scalar_win_rate={metric_float(branch_mig, 'emotion_scalar_win_rate'):.4f}"
    )
    print(
        "branch leakage/separation: "
        f"emotion_branch_importance_share={metric_float(split, 'emotion_branch_importance_share'):.4f} "
        f"meaning_branch_emotion_leakage_share={metric_float(split, 'meaning_branch_emotion_leakage_share'):.4f} "
        f"residual_emotion_leakage_share={metric_float(split, 'residual_emotion_leakage_share'):.4f} "
        f"emotion_meaning_separation_r2={metric_float(split, 'emotion_meaning_separation_r2'):.4f} "
        f"emotion_meaning_separation_pearson={metric_float(split, 'emotion_meaning_separation_pearson'):.4f}"
    )
    print(
        "scalar/vector dependence: "
        f"mean_abs_corr={metric_float(split, 'scalar_vector_mean_abs_correlation'):.4f} "
        f"max_abs_corr={metric_float(split, 'scalar_vector_max_abs_correlation'):.4f}"
    )

    print("probe scores by branch:")
    print_probe("scalar -> emotion", split.get("scalar_emotion_probe"))
    print_probe("vector -> emotion", split.get("vector_emotion_probe"))
    print_probe("residual -> emotion", split.get("residual_emotion_probe"))
    print_probe("full -> emotion", split.get("full_emotion_probe"))
    print_probe("scalar factors -> emotion", factor.get("ridge_probe"))

    print("per-label branch winners:")
    per_label = branch_mig.get("per_label") or {}
    for idx, (label, row) in enumerate(per_label.items()):
        if idx >= max_labels:
            remaining = len(per_label) - max_labels
            if remaining > 0:
                print(f"  ... {remaining} additional labels omitted")
            break
        print(
            f"  {label}: winner={row.get('winning_group')} "
            f"signed_mig={metric_float(row, 'emotion_vs_meaning_mig_signed'):.4f} "
            f"scalar_mi={metric_float(row, 'emotion_scalar_max_mi'):.4f} "
            f"vector_mi={metric_float(row, 'meaning_vector_max_mi'):.4f} "
            f"residual_mi={metric_float(row, 'residual_max_mi'):.4f}"
        )

    print("per-label top scalar factors:")
    top_factors = factor.get("per_label_top_factor") or {}
    for idx, (label, row) in enumerate(top_factors.items()):
        if idx >= max_labels:
            remaining = len(top_factors) - max_labels
            if remaining > 0:
                print(f"  ... {remaining} additional labels omitted")
            break
        print(
            f"  {label}: factor={row.get('factor_name')} "
            f"factor_index={row.get('factor_index')} "
            f"abs_corr={metric_float(row, 'abs_correlation'):.4f} "
            f"corr={metric_float(row, 'correlation'):.4f}"
        )


def serializable_metric_subset(metrics: Dict[str, Any]) -> Dict[str, Any]:
    drop = {"logits", "labels", "probs", "preds", "classification_report_text"}
    return {key: to_serializable(value) for key, value in metrics.items() if key not in drop}


DATA_CONFIG, PROMPT_CONFIG, MODEL_CONFIG, LOSS_CONFIG, SCHEDULE_CONFIG, EXPERIMENT_CONFIG = default_configs()

EXPERIMENT_CONFIG.seed = env_int("BASE_SEED", 42)
EXPERIMENT_CONFIG.output_dir = env_str("BASE_OUTPUT_DIR", "base_vae_attention_from_scratch_artifacts")
EXPERIMENT_CONFIG.threshold_mode = env_str("BASE_THRESHOLD_MODE", "per_label")
EXPERIMENT_CONFIG.threshold_grid_points = env_int("BASE_THRESHOLD_GRID_POINTS", 101)
EXPERIMENT_CONFIG.show_num_examples = 0
EXPERIMENT_CONFIG.monitor_num_examples = 0
EXPERIMENT_CONFIG.monitor_edit_steps = 0
EXPERIMENT_CONFIG.monitor_attention_enabled = False
EXPERIMENT_CONFIG.eval_num_copy_examples = 0
EXPERIMENT_CONFIG.prompt_eval_num_examples = 0
# Disable during training validation; turn on for final explicit validation/test diagnostics.
EXPERIMENT_CONFIG.latent_diagnostics_enabled = False
EXPERIMENT_CONFIG.latent_diagnostics_max_samples = env_int("BASE_LATENT_DIAG_MAX_SAMPLES", 4096)
EXPERIMENT_CONFIG.latent_diagnostics_ridge_alpha = env_float("BASE_LATENT_DIAG_RIDGE_ALPHA", 1.0)
EXPERIMENT_CONFIG.latent_diagnostics_probe_eval_fraction = env_float("BASE_LATENT_DIAG_PROBE_EVAL_FRACTION", 0.30)
EXPERIMENT_CONFIG.latent_diagnostics_min_probe_train_examples = env_int("BASE_LATENT_DIAG_MIN_PROBE_TRAIN", 64)
EXPERIMENT_CONFIG.latent_active_variance_threshold = env_float("BASE_LATENT_ACTIVE_VARIANCE_THRESHOLD", 0.0001)

DATA_CONFIG.max_length = env_int("BASE_MAX_LENGTH", 128)
DATA_CONFIG.train_batch_size = env_int("BASE_TRAIN_BATCH_SIZE", 8)
DATA_CONFIG.eval_batch_size = env_int("BASE_EVAL_BATCH_SIZE", 16)
DATA_CONFIG.num_workers = env_int("BASE_NUM_WORKERS", 0)
DATA_CONFIG.text_denoising_enabled = False

MODEL_CONFIG.model_name = env_str("BASE_MODEL_NAME", "google/flan-t5-base")
MODEL_CONFIG.num_scalar_factors = env_int("BASE_NUM_SCALAR_FACTORS", 28)
MODEL_CONFIG.vector_latent_dim = env_int("BASE_VECTOR_LATENT_DIM", 64)
MODEL_CONFIG.vae_hidden_dim = env_int("BASE_VAE_HIDDEN_DIM", 1024)
MODEL_CONFIG.latent_pool_heads = env_int("BASE_LATENT_POOL_HEADS", 8)
MODEL_CONFIG.attention_source = env_str("BASE_ATTENTION_SOURCE", "encoder_sequence")
MODEL_CONFIG.pooling_mode = env_str("BASE_POOLING_MODE", "per_scalar_dim")
MODEL_CONFIG.classifier_mode = env_str("BASE_CLASSIFIER_MODE", "per_emotion_pair_mlp")
MODEL_CONFIG.classifier_parameterization = env_str("BASE_CLASSIFIER_PARAMETERIZATION", "standard")
MODEL_CONFIG.per_emotion_hidden_dim = env_int("BASE_PER_EMOTION_HIDDEN_DIM", 32)
MODEL_CONFIG.joint_mlp_hidden_dim = env_int("BASE_JOINT_MLP_HIDDEN_DIM", 128)
MODEL_CONFIG.use_skip_connection = env_bool("BASE_USE_SKIP_CONNECTION", True)
MODEL_CONFIG.use_lora = env_bool("BASE_USE_LORA", True)
MODEL_CONFIG.lora_r = env_int("BASE_LORA_R", 16)
MODEL_CONFIG.lora_alpha = env_int("BASE_LORA_ALPHA", 32)
MODEL_CONFIG.lora_dropout = env_float("BASE_LORA_DROPOUT", 0.1)
_lora_targets = env_str("BASE_LORA_TARGET_MODULES", "q v").strip()
MODEL_CONFIG.lora_target_modules = tuple(_lora_targets.split()) if _lora_targets else None
MODEL_CONFIG.decoder_conditioning_mode = env_str("BASE_DECODER_CONDITIONING_MODE", "standard")

LOSS_CONFIG.classification_weight = env_float("BASE_CLASSIFICATION_WEIGHT", 4.0)
LOSS_CONFIG.recon_weight = env_float("BASE_RECON_WEIGHT", 2.0)
LOSS_CONFIG.kl_weight = env_float("BASE_KL_WEIGHT", 0.05)
LOSS_CONFIG.tc_weight = env_float("BASE_TC_WEIGHT", 0.1)
LOSS_CONFIG.copy_weight = 1.0
LOSS_CONFIG.vector_adv_weight = env_float("BASE_VECTOR_ADV_WEIGHT", 0.5)
LOSS_CONFIG.residual_adv_weight = env_float("BASE_RESIDUAL_ADV_WEIGHT", 0.0)
LOSS_CONFIG.vector_sep_weight = env_float("BASE_VECTOR_SEP_WEIGHT", 0.0)
LOSS_CONFIG.scalar_alignment_weight = env_float("BASE_SCALAR_ALIGNMENT_WEIGHT", 0.0)
LOSS_CONFIG.vector_label_leakage_weight = 0.0
LOSS_CONFIG.orthogonality_weight = env_float("BASE_ORTHOGONALITY_WEIGHT", 0.0)
LOSS_CONFIG.transfer_strength_weight = 0.0
LOSS_CONFIG.edit_sensitivity_weight = 0.0

SCHEDULE_CONFIG.num_epochs = env_int("BASE_NUM_EPOCHS", 1)
SCHEDULE_CONFIG.lr = env_float("BASE_LR", 2e-4)
SCHEDULE_CONFIG.weight_decay = env_float("BASE_WEIGHT_DECAY", 1e-2)
SCHEDULE_CONFIG.warmup_ratio = env_float("BASE_WARMUP_RATIO", 0.1)
SCHEDULE_CONFIG.min_lr_scale = env_float("BASE_MIN_LR_SCALE", 0.1)
# LoRA is disabled; these only keep schedule logs sane.
SCHEDULE_CONFIG.lora_start_epoch = env_int("BASE_LORA_START_EPOCH", 1)
SCHEDULE_CONFIG.copy_loss_start_epoch = 0
SCHEDULE_CONFIG.copy_loss_decay_start_epoch = 10**9
SCHEDULE_CONFIG.copy_loss_decay_end_epoch = 10**9
SCHEDULE_CONFIG.vector_adv_warmup_epochs = 0 if LOSS_CONFIG.vector_adv_weight > 0 else 10**9
SCHEDULE_CONFIG.residual_adv_warmup_epochs = 0 if LOSS_CONFIG.residual_adv_weight > 0 else 10**9
SCHEDULE_CONFIG.kl_zero_epochs = 0
SCHEDULE_CONFIG.kl_warmup_end_epoch = max(1, SCHEDULE_CONFIG.num_epochs)
SCHEDULE_CONFIG.residual_scale_start = env_float("BASE_RESIDUAL_SCALE_START", 1.0)
SCHEDULE_CONFIG.residual_scale_end = env_float("BASE_RESIDUAL_SCALE_END", 1.0)

set_seed(EXPERIMENT_CONFIG.seed)
device = get_device()
print("device:", device)
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))

print("loading data...")
data = build_data_bundle(
    data_config=DATA_CONFIG,
    model_config=MODEL_CONFIG,
    experiment_config=EXPERIMENT_CONFIG,
)

collator = EmotionTextCollator(tokenizer=data.tokenizer, max_length=DATA_CONFIG.max_length)
train_subset = subset_or_full(data.train_core_dataset, env_int("BASE_TRAIN_SAMPLES", 0))
threshold_subset = subset_or_full(data.threshold_tune_dataset, env_int("BASE_THRESHOLD_SAMPLES", 0))
val_subset = subset_or_full(data.val_dataset, env_int("BASE_VAL_SAMPLES", 0))
test_subset = subset_or_full(data.test_dataset, env_int("BASE_TEST_SAMPLES", 0))

data.train_core_dataset = train_subset
data.threshold_tune_dataset = threshold_subset
data.val_dataset = val_subset
data.test_dataset = test_subset
data.train_loader = make_train_loader(
    train_subset,
    DATA_CONFIG.train_batch_size,
    collator,
    EXPERIMENT_CONFIG.seed,
    DATA_CONFIG.num_workers,
)
data.threshold_loader = make_eval_loader(threshold_subset, DATA_CONFIG.eval_batch_size, collator, DATA_CONFIG.num_workers)
data.val_loader = make_eval_loader(val_subset, DATA_CONFIG.eval_batch_size, collator, DATA_CONFIG.num_workers)
data.test_loader = make_eval_loader(test_subset, DATA_CONFIG.eval_batch_size, collator, DATA_CONFIG.num_workers)
data.text_denoising_dataset = None
data.text_denoising_loader = None

audit = {
    "dataset_name": data.dataset_name,
    "target_kind": data.target_kind,
    "num_labels": data.num_labels,
    "emotion_names": list(data.emotion_names),
    "train_samples": len(data.train_core_dataset),
    "threshold_samples": len(data.threshold_tune_dataset),
    "val_samples": len(data.val_dataset),
    "test_samples": len(data.test_dataset),
    "train_batches": len(data.train_loader),
    "threshold_batches": len(data.threshold_loader),
    "val_batches": len(data.val_loader),
    "test_batches": len(data.test_loader),
    "model_name": MODEL_CONFIG.model_name,
    "base_architecture": {
        "use_lora": MODEL_CONFIG.use_lora,
        "decoder_only_lora": True,
        "lora_r": MODEL_CONFIG.lora_r,
        "lora_alpha": MODEL_CONFIG.lora_alpha,
        "lora_dropout": MODEL_CONFIG.lora_dropout,
        "lora_target_modules": None if MODEL_CONFIG.lora_target_modules is None else list(MODEL_CONFIG.lora_target_modules),
        "lora_start_epoch": SCHEDULE_CONFIG.lora_start_epoch,
        "use_skip_connection": MODEL_CONFIG.use_skip_connection,
        "decoder_conditioning_mode": MODEL_CONFIG.decoder_conditioning_mode,
        "attention_source": MODEL_CONFIG.attention_source,
        "pooling_mode": MODEL_CONFIG.pooling_mode,
        "classifier_mode": MODEL_CONFIG.classifier_mode,
        "classifier_parameterization": MODEL_CONFIG.classifier_parameterization,
        "joint_mlp_hidden_dim": MODEL_CONFIG.joint_mlp_hidden_dim,
        "per_emotion_hidden_dim": MODEL_CONFIG.per_emotion_hidden_dim,
        "num_scalar_factors": MODEL_CONFIG.num_scalar_factors,
        "vector_latent_dim": MODEL_CONFIG.vector_latent_dim,
        "total_latent_dim": MODEL_CONFIG.num_scalar_factors + MODEL_CONFIG.vector_latent_dim,
        "latent_pool_heads": MODEL_CONFIG.latent_pool_heads,
    },
    "losses": {
        "classification_weight": LOSS_CONFIG.classification_weight,
        "recon_weight": LOSS_CONFIG.recon_weight,
        "kl_weight": LOSS_CONFIG.kl_weight,
        "tc_weight": LOSS_CONFIG.tc_weight,
        "copy_weight": LOSS_CONFIG.copy_weight,
        "vector_adv_weight": LOSS_CONFIG.vector_adv_weight,
        "residual_adv_weight": LOSS_CONFIG.residual_adv_weight,
        "vector_sep_weight": LOSS_CONFIG.vector_sep_weight,
        "scalar_alignment_weight": LOSS_CONFIG.scalar_alignment_weight,
        "vector_label_leakage_weight": LOSS_CONFIG.vector_label_leakage_weight,
        "transfer_strength_weight": LOSS_CONFIG.transfer_strength_weight,
        "edit_sensitivity_weight": LOSS_CONFIG.edit_sensitivity_weight,
    },
}
print("run audit:")
print(json.dumps(audit, indent=2))

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

# Decoder-only LoRA audit. The repo freeze helper only enables decoder LoRA
# parameters; encoder LoRA parameters should remain frozen.
lora_params = []
for name, param in runtime.state.model.shared_t5.named_parameters():
    if "lora_" in name:
        lower = name.lower()
        branch = "decoder" if (("decoder.block" in lower) or ("decoder.layers" in lower)) else "non_decoder"
        lora_params.append({"name": name, "branch": branch, "requires_grad_initial": bool(param.requires_grad), "numel": int(param.numel())})
print("decoder-only LoRA audit:")
print(json.dumps({
    "use_lora": MODEL_CONFIG.use_lora,
    "lora_target_modules": None if MODEL_CONFIG.lora_target_modules is None else list(MODEL_CONFIG.lora_target_modules),
    "lora_start_epoch": SCHEDULE_CONFIG.lora_start_epoch,
    "decoder_lora_numel": sum(p["numel"] for p in lora_params if p["branch"] == "decoder"),
    "non_decoder_lora_numel": sum(p["numel"] for p in lora_params if p["branch"] != "decoder"),
    "decoder_lora_param_count": sum(1 for p in lora_params if p["branch"] == "decoder"),
    "non_decoder_lora_param_count": sum(1 for p in lora_params if p["branch"] != "decoder"),
    "sample_params": lora_params[:12],
}, indent=2))

output_dir = Path(EXPERIMENT_CONFIG.output_dir)
save_json(output_dir / "base_train_eval_audit.json", audit)

print("\n=== training base model from scratch ===")
state = run_training(runtime, monitor=False, save_each_epoch=True)

weights = current_loss_weights(epoch=SCHEDULE_CONFIG.num_epochs, ctx=runtime.ctx)
# Explicitly keep the final eval on the base objective, without auxiliary terms.
for key in [
    "tc_weight",
    "copy_weight",
    "vector_adv_weight",
    "residual_adv_weight",
    "vector_sep_weight",
    "scalar_alignment_weight",
    "vector_label_leakage_weight",
    "transfer_strength_weight",
    "edit_sensitivity_weight",
]:
    weights[key] = 0.0
weights["classification_weight"] = LOSS_CONFIG.classification_weight
weights["recon_weight"] = LOSS_CONFIG.recon_weight
weights["kl_weight"] = LOSS_CONFIG.kl_weight
weights["orthogonality_weight"] = LOSS_CONFIG.orthogonality_weight
weights["residual_scale"] = env_float("BASE_RESIDUAL_SCALE_END", 1.0)

print("\n=== retuning thresholds on calibration split ===")
# Enable diagnostics only on validation/test. Threshold tuning does not need them.
runtime.ctx.experiment_config.latent_diagnostics_enabled = False
threshold_metrics = evaluate_loader(
    state=state,
    ctx=runtime.ctx,
    data_loader=runtime.data.threshold_loader,
    desc="final [threshold tune]",
    tune_threshold=True,
    weights=weights,
)
state.threshold = restore_threshold(
    threshold_metrics["threshold"],
    num_labels=len(runtime.ctx.emotion_names),
    mode=runtime.ctx.experiment_config.threshold_mode,
)
summarize_eval_metrics(prefix="[final] calib ", metrics=threshold_metrics, num_labels=len(runtime.ctx.emotion_names))

print("\n=== final validation with disentanglement diagnostics ===")
runtime.ctx.experiment_config.latent_diagnostics_enabled = True
val_metrics = evaluate_loader(
    state=state,
    ctx=runtime.ctx,
    data_loader=runtime.data.val_loader,
    desc="final [val diagnostics]",
    tune_threshold=False,
    weights=weights,
)
summarize_eval_metrics(prefix="[final] val   ", metrics=val_metrics, num_labels=len(runtime.ctx.emotion_names))
print_disentanglement("validation", val_metrics)

print("\n=== final test with disentanglement diagnostics ===")
test_metrics = evaluate_loader(
    state=state,
    ctx=runtime.ctx,
    data_loader=runtime.data.test_loader,
    desc="final [test diagnostics]",
    tune_threshold=False,
    weights=weights,
)
summarize_eval_metrics(prefix="[final] test  ", metrics=test_metrics, num_labels=len(runtime.ctx.emotion_names))
print_disentanglement("test", test_metrics)

save_json(output_dir / "base_threshold_metrics.json", serializable_metric_subset(threshold_metrics))
save_json(output_dir / "base_val_metrics_with_disentanglement.json", serializable_metric_subset(val_metrics))
save_json(output_dir / "base_test_metrics_with_disentanglement.json", serializable_metric_subset(test_metrics))

# Save a final checkpoint with final validation/test summaries attached, even if best checkpoint already exists.
save_checkpoint(
    path=output_dir / "final_checkpoint.pt",
    state=state,
    ctx=runtime.ctx,
    epoch=SCHEDULE_CONFIG.num_epochs,
    val_metrics=val_metrics,
    test_metrics=test_metrics,
)


def nested_float(mapping: Dict[str, Any], path: str, default: float = 0.0) -> float:
    node: Any = mapping
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    try:
        return float(node)
    except Exception:
        return default

_diag_test = test_metrics.get("latent_diagnostics") or {}
_factor_test = _diag_test.get("factor_power") or {}
_split_test = _diag_test.get("emotion_meaning_split") or {}
_align_test = _factor_test.get("diagonal_alignment") or {}
_block_align_test = _factor_test.get("block_alignment") or {}
_dci_test = _factor_test.get("dci") or {}
_branch_mig_test = _split_test.get("branch_group_mig") or {}
_branch_dci_test = _split_test.get("branch_group_dci") or {}
_scalar_probe_test = _split_test.get("scalar_emotion_probe") or {}
_vector_probe_test = _split_test.get("vector_emotion_probe") or {}
_residual_probe_test = _split_test.get("residual_emotion_probe") or {}

ten_metric_summary = {
    "emotion_in_scalar_r2": float(_split_test.get("emotion_in_scalar_r2", _scalar_probe_test.get("ridge_r2_macro", 0.0)) or 0.0),
    "emotion_leakage_vector_r2": float(_split_test.get("emotion_leakage_vector_r2", _vector_probe_test.get("ridge_r2_macro", 0.0)) or 0.0),
    "emotion_leakage_residual_r2": float(_split_test.get("emotion_leakage_residual_r2", _residual_probe_test.get("ridge_r2_macro", 0.0)) or 0.0),
    "emotion_meaning_separation_r2": float(_split_test.get("emotion_meaning_separation_r2", 0.0) or 0.0),
    "emotion_meaning_separation_pearson": float(_split_test.get("emotion_meaning_separation_pearson", 0.0) or 0.0),
    "branch_mig": float(_branch_mig_test.get("emotion_vs_meaning_mig_signed_macro", test_metrics.get("branch_mig", 0.0)) or 0.0),
    "emotion_branch_importance_share": float(_split_test.get("emotion_branch_importance_share", 0.0) or 0.0),
    "meaning_branch_emotion_leakage_share": float(_split_test.get("meaning_branch_emotion_leakage_share", 0.0) or 0.0),
    "residual_emotion_leakage_share": float(_split_test.get("residual_emotion_leakage_share", 0.0) or 0.0),
    "diagonal_dominance": None if "diagonal_dominance" not in _align_test else float(_align_test.get("diagonal_dominance", 0.0) or 0.0),
    "block_assigned_correlation_dominance": None if "assigned_block_correlation_dominance" not in _block_align_test else float(_block_align_test.get("assigned_block_correlation_dominance", 0.0) or 0.0),
    "block_assigned_importance_dominance": None if "assigned_block_importance_dominance" not in _block_align_test else float(_block_align_test.get("assigned_block_importance_dominance", 0.0) or 0.0),
    "block_top_correlation_match_rate": None if "top_correlation_block_match_rate" not in _block_align_test else float(_block_align_test.get("top_correlation_block_match_rate", 0.0) or 0.0),
    "scalar_factors_per_label": None if "scalar_factors_per_label" not in _block_align_test else int(_block_align_test.get("scalar_factors_per_label", 0) or 0),
    "factor_dci_completeness": float(_dci_test.get("dci_completeness", test_metrics.get("factor_dci_completeness", 0.0)) or 0.0),
}

summary = {
    "run_id": os.environ.get("ABLATION_RUN_ID", ""),
    "pooling_mode": MODEL_CONFIG.pooling_mode,
    "classifier_mode": MODEL_CONFIG.classifier_mode,
    "classifier_parameterization": MODEL_CONFIG.classifier_parameterization,
    "joint_mlp_hidden_dim": MODEL_CONFIG.joint_mlp_hidden_dim,
        "per_emotion_hidden_dim": MODEL_CONFIG.per_emotion_hidden_dim,
    "num_scalar_factors": MODEL_CONFIG.num_scalar_factors,
    "vector_latent_dim": MODEL_CONFIG.vector_latent_dim,
    "total_latent_dim": MODEL_CONFIG.num_scalar_factors + MODEL_CONFIG.vector_latent_dim,
    "use_skip_connection": MODEL_CONFIG.use_skip_connection,
    "use_lora": MODEL_CONFIG.use_lora,
    "decoder_only_lora": True,
    "lora_start_epoch": SCHEDULE_CONFIG.lora_start_epoch,
    "lora_r": MODEL_CONFIG.lora_r,
    "lora_alpha": MODEL_CONFIG.lora_alpha,
    "lora_dropout": MODEL_CONFIG.lora_dropout,
    "lora_target_modules": None if MODEL_CONFIG.lora_target_modules is None else list(MODEL_CONFIG.lora_target_modules),
    "residual_scale_start": SCHEDULE_CONFIG.residual_scale_start,
    "residual_scale_end": SCHEDULE_CONFIG.residual_scale_end,
    "vector_adv_weight": LOSS_CONFIG.vector_adv_weight,
    "residual_adv_weight": LOSS_CONFIG.residual_adv_weight,
    "decoder_conditioning_mode": MODEL_CONFIG.decoder_conditioning_mode,
    "seed": EXPERIMENT_CONFIG.seed,
    "num_epochs": SCHEDULE_CONFIG.num_epochs,
    "model_name": MODEL_CONFIG.model_name,
    "ten_metric_summary": ten_metric_summary,
    "best_val_micro_f1_during_training": state.best_val_micro_f1,
    "final_val_micro_f1": float(val_metrics.get("micro_f1", 0.0)),
    "final_val_macro_f1": float(val_metrics.get("macro_f1", 0.0)),
    "final_test_micro_f1": float(test_metrics.get("micro_f1", 0.0)),
    "final_test_macro_f1": float(test_metrics.get("macro_f1", 0.0)),
    "final_test_factor_dci_disentanglement": float(_dci_test.get("dci_disentanglement", 0.0) or 0.0),
    "final_test_factor_dci_completeness": float(_dci_test.get("dci_completeness", 0.0) or 0.0),
    "final_test_branch_dci_completeness": float(_branch_dci_test.get("dci_completeness", 0.0) or 0.0),
    "final_test_branch_mig": float(_branch_mig_test.get("emotion_vs_meaning_mig_signed_macro", 0.0) or 0.0),
    "final_test_emotion_branch_importance_share": float(_split_test.get("emotion_branch_importance_share", 0.0) or 0.0),
    "final_test_meaning_branch_emotion_leakage_share": float(_split_test.get("meaning_branch_emotion_leakage_share", 0.0) or 0.0),
    "final_test_residual_emotion_leakage_share": float(_split_test.get("residual_emotion_leakage_share", 0.0) or 0.0),
    "final_test_scalar_vector_mean_abs_correlation": float(_split_test.get("scalar_vector_mean_abs_correlation", 0.0) or 0.0),
    "final_test_block_assigned_correlation_dominance": None if "assigned_block_correlation_dominance" not in _block_align_test else float(_block_align_test.get("assigned_block_correlation_dominance", 0.0) or 0.0),
    "final_test_block_assigned_importance_dominance": None if "assigned_block_importance_dominance" not in _block_align_test else float(_block_align_test.get("assigned_block_importance_dominance", 0.0) or 0.0),
    "final_test_block_top_correlation_match_rate": None if "top_correlation_block_match_rate" not in _block_align_test else float(_block_align_test.get("top_correlation_block_match_rate", 0.0) or 0.0),
    "threshold_mode": runtime.ctx.experiment_config.threshold_mode,
    "threshold_mean": float(test_metrics.get("threshold_mean", 0.0)),
    "best_checkpoint": str((output_dir / "best_checkpoint.pt").resolve()),
    "final_checkpoint": str((output_dir / "final_checkpoint.pt").resolve()),
    "val_metrics_path": str((output_dir / "base_val_metrics_with_disentanglement.json").resolve()),
    "test_metrics_path": str((output_dir / "base_test_metrics_with_disentanglement.json").resolve()),
}
save_json(output_dir / "base_train_eval_summary.json", summary)
save_json(output_dir / "ablation_result_summary.json", summary)

print("\n=== saved outputs ===")
print(json.dumps(summary, indent=2))
PY
