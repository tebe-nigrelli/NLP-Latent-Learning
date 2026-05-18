#!/bin/bash
#SBATCH --job-name=beta-class-samples
#SBATCH --partition=debug_gpuh200
#SBATCH --qos=debug
#SBATCH --output=output/beta-class-samples_%j.out
#SBATCH --error=output/beta-class-samples_%j.err
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p "$SUBMIT_DIR/output"

if [[ -d "$SUBMIT_DIR/src/emotion_latent_learning" ]]; then
  PROJECT_DIR="$SUBMIT_DIR"
elif [[ -d "$SUBMIT_DIR/../src/emotion_latent_learning" ]]; then
  PROJECT_DIR="$(cd "$SUBMIT_DIR/.." && pwd)"
else
  PROJECT_DIR="$SUBMIT_DIR"
fi

DEFAULT_CHECKPOINT_REL="config_matched_beta_schedule_ablation_artifacts/pool-joint_scalar-56_vector-64_total-120_classifier-per-emotion-pair-mlp-standard_skip-false_rscale-1p0-1p0_vadv-0p5_radv-0p0_dec-split_film_decoder-lora-r16-a32-start1_klstart-4-klend-8-klw-0p05_seed-42/best_checkpoint.pt"

if [[ -n "${BETA_CLASSIFICATION_CHECKPOINT_PATH:-}" ]]; then
  CHECKPOINT_PATH="$BETA_CLASSIFICATION_CHECKPOINT_PATH"
elif [[ -f "$SUBMIT_DIR/$DEFAULT_CHECKPOINT_REL" ]]; then
  CHECKPOINT_PATH="$SUBMIT_DIR/$DEFAULT_CHECKPOINT_REL"
elif [[ -f "$PROJECT_DIR/$DEFAULT_CHECKPOINT_REL" ]]; then
  CHECKPOINT_PATH="$PROJECT_DIR/$DEFAULT_CHECKPOINT_REL"
else
  CHECKPOINT_PATH="$SUBMIT_DIR/$DEFAULT_CHECKPOINT_REL"
fi

if [[ "$CHECKPOINT_PATH" != /* ]]; then
  CHECKPOINT_PATH="$SUBMIT_DIR/$CHECKPOINT_PATH"
fi

OUTPUT_DIR="${BETA_CLASSIFICATION_OUTPUT_DIR:-$SUBMIT_DIR/beta_classification_samples_artifacts}"
mkdir -p "$OUTPUT_DIR"

export BETA_CLASSIFICATION_CHECKPOINT_PATH="$CHECKPOINT_PATH"
export BETA_CLASSIFICATION_OUTPUT_DIR="$OUTPUT_DIR"
export BETA_CLASSIFICATION_SPLIT="${BETA_CLASSIFICATION_SPLIT:-test}"
export BETA_CLASSIFICATION_MAX_SAMPLES="${BETA_CLASSIFICATION_MAX_SAMPLES:-128}"
export BETA_CLASSIFICATION_BATCH_SIZE="${BETA_CLASSIFICATION_BATCH_SIZE:-16}"
export BETA_CLASSIFICATION_NUM_WORKERS="${BETA_CLASSIFICATION_NUM_WORKERS:-0}"
export BETA_CLASSIFICATION_TOP_K="${BETA_CLASSIFICATION_TOP_K:-5}"

printf 'Submit dir: %s\n' "$SUBMIT_DIR"
printf 'Project dir: %s\n' "$PROJECT_DIR"
printf 'Checkpoint: %s\n' "$BETA_CLASSIFICATION_CHECKPOINT_PATH"
printf 'Output dir: %s\n' "$BETA_CLASSIFICATION_OUTPUT_DIR"
printf 'Split/max samples/batch: %s/%s/%s\n' "$BETA_CLASSIFICATION_SPLIT" "$BETA_CLASSIFICATION_MAX_SAMPLES" "$BETA_CLASSIFICATION_BATCH_SIZE"

cd "$PROJECT_DIR"

uv run python - <<'PY'
from __future__ import annotations

import csv
import dataclasses
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from sklearn.metrics import f1_score

PROJECT_ROOT = Path.cwd()
SRC = PROJECT_ROOT / "src"
if str(SRC.resolve()) not in sys.path:
    sys.path.insert(0, str(SRC.resolve()))

from emotion_latent_learning.config import (
    DataConfig,
    ExperimentConfig,
    LossConfig,
    ModelConfig,
    PromptConfig,
    ScheduleConfig,
)
from emotion_latent_learning.evaluation.thresholds import restore_threshold
from emotion_latent_learning.training.checkpointing import load_compact_model_state_dict
from emotion_latent_learning.training.runtime import build_data_bundle, build_runtime, get_device
from emotion_latent_learning.training.schedules import current_loss_weights


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value is None or value == "" else int(value)


def dataclass_from_dict(cls: Any, payload: Optional[Dict[str, Any]]) -> Any:
    obj = cls()
    if not payload:
        return obj
    valid = {f.name for f in dataclasses.fields(cls)}
    for key, value in payload.items():
        if key not in valid:
            continue
        default_value = getattr(obj, key, None)
        if isinstance(default_value, tuple) and isinstance(value, list):
            value = tuple(value)
        setattr(obj, key, value)
    return obj


def configs_from_checkpoint(checkpoint: Dict[str, Any]) -> Tuple[DataConfig, PromptConfig, ModelConfig, LossConfig, ScheduleConfig, ExperimentConfig]:
    configs = checkpoint.get("configs") or {}
    return (
        dataclass_from_dict(DataConfig, configs.get("data_config")),
        dataclass_from_dict(PromptConfig, configs.get("prompt_config")),
        dataclass_from_dict(ModelConfig, configs.get("model_config")),
        dataclass_from_dict(LossConfig, configs.get("loss_config")),
        dataclass_from_dict(ScheduleConfig, configs.get("schedule_config")),
        dataclass_from_dict(ExperimentConfig, configs.get("experiment_config")),
    )


def to_jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return str(value)


def threshold_vector_from_payload(payload: Any, num_labels: int) -> List[float]:
    """Best-effort parser for serialized threshold payloads; fallback is 0.5."""
    if payload is None:
        return [0.5] * num_labels
    if isinstance(payload, (int, float)):
        return [float(payload)] * num_labels
    if isinstance(payload, (list, tuple)):
        vals = [float(x) for x in payload]
        if len(vals) == num_labels:
            return vals
        if len(vals) == 1:
            return vals * num_labels
    if isinstance(payload, dict):
        for key in ["thresholds", "values", "per_label_thresholds", "label_thresholds", "per_label"]:
            if key in payload:
                return threshold_vector_from_payload(payload[key], num_labels)
        for key in ["threshold", "global_threshold", "value"]:
            if key in payload:
                return threshold_vector_from_payload(payload[key], num_labels)
    return [0.5] * num_labels


def logits_from_output(output: Any) -> torch.Tensor:
    for attr in ["classification_logits", "logits", "label_logits"]:
        value = getattr(output, attr, None)
        if value is not None:
            return value
    if isinstance(output, dict):
        for key in ["classification_logits", "logits", "label_logits"]:
            if key in output and output[key] is not None:
                return output[key]
    raise RuntimeError("Could not find classification logits in model output.")


def labels_to_names(values: Sequence[int], names: Sequence[str]) -> List[str]:
    return [names[i] for i, v in enumerate(values) if int(v) == 1]


checkpoint_path = Path(os.environ["BETA_CLASSIFICATION_CHECKPOINT_PATH"]).expanduser().resolve()
output_dir = Path(os.environ["BETA_CLASSIFICATION_OUTPUT_DIR"]).expanduser().resolve()
split_name = os.environ.get("BETA_CLASSIFICATION_SPLIT", "test")
max_samples = env_int("BETA_CLASSIFICATION_MAX_SAMPLES", 128)
batch_size = env_int("BETA_CLASSIFICATION_BATCH_SIZE", 16)
num_workers = env_int("BETA_CLASSIFICATION_NUM_WORKERS", 0)
top_k = env_int("BETA_CLASSIFICATION_TOP_K", 5)

if not checkpoint_path.exists():
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

output_dir.mkdir(parents=True, exist_ok=True)
print(f"Loading checkpoint: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location="cpu")
data_config, prompt_config, model_config, loss_config, schedule_config, experiment_config = configs_from_checkpoint(checkpoint)

data_config.eval_batch_size = batch_size
data_config.num_workers = num_workers
experiment_config.output_dir = str(checkpoint_path.parent)
experiment_config.sample_posterior_eval = False
experiment_config.latent_diagnostics_enabled = False

device = get_device()
print(f"Device: {device}")

data = build_data_bundle(
    data_config=data_config,
    model_config=model_config,
    experiment_config=experiment_config,
)
runtime = build_runtime(
    data=data,
    data_config=data_config,
    prompt_config=prompt_config,
    model_config=model_config,
    loss_config=loss_config,
    schedule_config=schedule_config,
    experiment_config=experiment_config,
    device=device,
    save_config=False,
)
load_compact_model_state_dict(runtime.state.model, checkpoint["model_state_dict"])

if checkpoint.get("threshold") is not None:
    runtime.state.threshold = restore_threshold(
        checkpoint["threshold"],
        num_labels=len(runtime.ctx.emotion_names),
        mode=runtime.ctx.experiment_config.threshold_mode,
    )

model = runtime.state.model
model.eval()
labels_names = list(runtime.ctx.emotion_names)
num_labels = len(labels_names)
thresholds = threshold_vector_from_payload(checkpoint.get("threshold"), num_labels)
threshold_tensor = torch.tensor(thresholds, dtype=torch.float32, device=device).view(1, -1)

loader = getattr(data, f"{split_name}_loader")
weights = current_loss_weights(epoch=int(getattr(schedule_config, "num_epochs", 0)), ctx=runtime.ctx)
residual_scale = float(weights.get("residual_scale", 0.0))

rows: List[Dict[str, Any]] = []
all_probs: List[torch.Tensor] = []
all_true: List[torch.Tensor] = []
seen = 0

with torch.no_grad():
    for batch in loader:
        if seen >= max_samples:
            break
        remaining = max_samples - seen
        texts = list(batch["texts"][:remaining])
        input_ids = batch["input_ids"][:remaining].to(device)
        attention_mask = batch["attention_mask"][:remaining].to(device)
        labels = batch["labels"][:remaining].to(device)

        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=None,
            sample_posterior=False,
            classification_weight=0.0,
            recon_weight=0.0,
            kl_weight=0.0,
            residual_scale=residual_scale,
        )
        logits = logits_from_output(output)
        probs = torch.sigmoid(logits)
        preds = (probs >= threshold_tensor).long()

        all_probs.append(probs.detach().cpu())
        all_true.append(labels.detach().cpu().long())

        top_vals, top_idx = torch.topk(probs, k=min(top_k, num_labels), dim=-1)
        for j, text in enumerate(texts):
            true_vec = labels[j].detach().cpu().long().tolist()
            pred_vec = preds[j].detach().cpu().long().tolist()
            prob_vec = probs[j].detach().cpu().tolist()
            top_pairs = [f"{labels_names[int(idx)]}:{float(val):.4f}" for idx, val in zip(top_idx[j].detach().cpu(), top_vals[j].detach().cpu())]
            row: Dict[str, Any] = {
                "example_index": seen + j,
                "split": split_name,
                "checkpoint": str(checkpoint_path),
                "run_dir": str(checkpoint_path.parent),
                "text": text,
                "true_labels": ";".join(labels_to_names(true_vec, labels_names)),
                "predicted_labels": ";".join(labels_to_names(pred_vec, labels_names)),
                "top_predictions": ";".join(top_pairs),
                "num_true_labels": int(sum(true_vec)),
                "num_predicted_labels": int(sum(pred_vec)),
            }
            for idx, name in enumerate(labels_names):
                safe = name.replace(" ", "_").replace("/", "_")
                row[f"prob_{safe}"] = float(prob_vec[idx])
                row[f"pred_{safe}"] = int(pred_vec[idx])
                row[f"true_{safe}"] = int(true_vec[idx])
                row[f"threshold_{safe}"] = float(thresholds[idx])
            rows.append(row)
        seen += len(texts)

if not rows:
    raise RuntimeError("No classification samples were collected.")

csv_path = output_dir / "beta_classification_samples.csv"
fieldnames = list(rows[0].keys())
with csv_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

probs_cpu = torch.cat(all_probs, dim=0).numpy()
true_cpu = torch.cat(all_true, dim=0).numpy()
pred_cpu = (probs_cpu >= torch.tensor(thresholds).numpy().reshape(1, -1)).astype(int)
summary = {
    "checkpoint": str(checkpoint_path),
    "run_dir": str(checkpoint_path.parent),
    "split": split_name,
    "num_examples": int(len(rows)),
    "threshold_mode": str(getattr(runtime.ctx.experiment_config, "threshold_mode", "unknown")),
    "micro_f1": float(f1_score(true_cpu, pred_cpu, average="micro", zero_division=0)),
    "macro_f1": float(f1_score(true_cpu, pred_cpu, average="macro", zero_division=0)),
    "weighted_f1": float(f1_score(true_cpu, pred_cpu, average="weighted", zero_division=0)),
    "mean_predicted_labels_per_example": float(pred_cpu.sum(axis=1).mean()),
    "mean_true_labels_per_example": float(true_cpu.sum(axis=1).mean()),
    "model_config": to_jsonable(dataclasses.asdict(model_config)),
    "loss_config": to_jsonable(dataclasses.asdict(loss_config)),
    "schedule_config": to_jsonable(dataclasses.asdict(schedule_config)),
}
json_path = output_dir / "beta_classification_samples_summary.json"
json_path.write_text(json.dumps(summary, indent=2, sort_keys=True))

print(f"Wrote {len(rows)} sample rows to {csv_path}")
print(f"Wrote summary to {json_path}")
print(json.dumps({k: summary[k] for k in ["num_examples", "micro_f1", "macro_f1", "weighted_f1"]}, indent=2))
PY
