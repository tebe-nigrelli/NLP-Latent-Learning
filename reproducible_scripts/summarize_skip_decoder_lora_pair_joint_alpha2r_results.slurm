#!/bin/bash
#SBATCH --job-name=nlp-alpha2r-sum
#SBATCH --partition=debug_gpuh200
#SBATCH --mem=8G
#SBATCH --qos=debug
#SBATCH --time=00:15:00
#SBATCH --output=output/%x_%j.out
#SBATCH --error=output/%x_%j.err


set -euo pipefail

# Summarize the focused skip + decoder-LoRA alpha=2*r ablation.
# Expected layout:
#   skip_decoder_lora_pair_joint_alpha2r_artifacts/
#     pool-joint_scalar-56_vector-64_..._decoder-lora-r8-a16-.../
#     pool-joint_scalar-56_vector-64_..._decoder-lora-r16-a32-.../
#     pool-joint_scalar-56_vector-64_..._decoder-lora-r32-a64-.../
# Each run directory should contain ablation_result_summary.json.
# Submit from the repo root or from factor_jobs:
#   sbatch summarize_skip_decoder_lora_pair_joint_alpha2r_results.slurm

PROJECT_DIR="${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
cd "$PROJECT_DIR"
mkdir -p output

export UV_CACHE_DIR="${UV_CACHE_DIR:-$PROJECT_DIR/.cache/uv}"
export PYTHONUNBUFFERED=1

export ABLATION_OUTPUT_ROOT="${ABLATION_OUTPUT_ROOT:-skip_decoder_lora_pair_joint_alpha2r_artifacts}"
export ABLATION_SUMMARY_CSV="${ABLATION_SUMMARY_CSV:-${ABLATION_OUTPUT_ROOT}/ablation_10_metric_summary.csv}"
export ABLATION_SUMMARY_JSON="${ABLATION_SUMMARY_JSON:-${ABLATION_OUTPUT_ROOT}/ablation_10_metric_summary.json}"

uv run python - <<'PY'
from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

root = Path(os.environ.get("ABLATION_OUTPUT_ROOT", "skip_decoder_lora_pair_joint_alpha2r_artifacts"))
csv_path = Path(os.environ.get("ABLATION_SUMMARY_CSV", root / "ablation_10_metric_summary.csv"))
json_path = Path(os.environ.get("ABLATION_SUMMARY_JSON", root / "ablation_10_metric_summary.json"))

metric_names = [
    "emotion_in_scalar_r2",
    "emotion_leakage_vector_r2",
    "emotion_leakage_residual_r2",
    "emotion_meaning_separation_r2",
    "emotion_meaning_separation_pearson",
    "branch_mig",
    "emotion_branch_importance_share",
    "meaning_branch_emotion_leakage_share",
    "residual_emotion_leakage_share",
    "diagonal_dominance",
    "block_assigned_correlation_dominance",
    "block_assigned_importance_dominance",
    "block_top_correlation_match_rate",
    "scalar_factors_per_label",
    "factor_dci_completeness",
]

# Some older summaries did not expose every config field at top level.
# Infer stable fields from the run directory name as a fallback.
def parse_run_id(run_id: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    if "pool-joint" in run_id:
        out["pooling_mode"] = "joint_scalar_vector"
    elif "pool-peremotion" in run_id or "pool-per-scalar" in run_id or "pool-per_scalar" in run_id:
        out["pooling_mode"] = "per_scalar_dim"

    m = re.search(r"scalar-(\d+)", run_id)
    if m:
        out["num_scalar_factors"] = int(m.group(1))
    m = re.search(r"vector-(\d+)", run_id)
    if m:
        out["vector_latent_dim"] = int(m.group(1))
    m = re.search(r"total-(\d+)", run_id)
    if m:
        out["total_latent_dim"] = int(m.group(1))

    if "classifier-per-emotion-pair-mlp" in run_id:
        out["classifier_mode"] = "per_emotion_pair_mlp"
    elif "classifier-per-emotion-mlp" in run_id:
        out["classifier_mode"] = "per_emotion_mlp"
    elif "classifier-joint-mlp" in run_id:
        out["classifier_mode"] = "joint_mlp"

    if "classifier-per-emotion-pair-mlp-standard" in run_id or "classifier-per-emotion-mlp-standard" in run_id or "classifier-joint-mlp-standard" in run_id:
        out["classifier_parameterization"] = "standard"
    elif "orthogonal" in run_id:
        out["classifier_parameterization"] = "orthogonal"

    if "skip-true" in run_id:
        out["use_skip_connection"] = True
    elif "skip-false" in run_id:
        out["use_skip_connection"] = False

    m = re.search(r"rscale-([^-_]+)-([^-_]+)", run_id)
    if m:
        out["residual_scale_start"] = m.group(1).replace("p", ".")
        out["residual_scale_end"] = m.group(2).replace("p", ".")

    m = re.search(r"radv-([^-_]+)", run_id)
    if m:
        out["residual_adv_weight"] = m.group(1).replace("p", ".")

    m = re.search(r"decoder-lora-r(\d+)-a(\d+)", run_id)
    if m:
        out["lora_r"] = int(m.group(1))
        out["lora_alpha"] = int(m.group(2))
        out["use_lora"] = True
        out["decoder_only_lora"] = True

    m = re.search(r"targets-([a-z-]+)-start", run_id)
    if m:
        out["lora_target_modules"] = m.group(1).replace("-", " ")

    m = re.search(r"start(\d+)", run_id)
    if m:
        out["lora_start_epoch"] = int(m.group(1))

    m = re.search(r"seed-(\d+)", run_id)
    if m:
        out["seed"] = int(m.group(1))

    return out


def first_not_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def as_target_modules(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, Iterable):
        return " ".join(str(x) for x in value)
    return str(value)

rows = []
for path in sorted(root.glob("*/ablation_result_summary.json")):
    with path.open("r", encoding="utf-8") as fh:
        payload: Dict[str, Any] = json.load(fh)

    run_id = str(payload.get("run_id") or path.parent.name)
    inferred = parse_run_id(run_id)
    metrics = payload.get("ten_metric_summary") or {}

    lora_r = first_not_none(payload.get("lora_r"), inferred.get("lora_r"))
    lora_alpha = first_not_none(payload.get("lora_alpha"), inferred.get("lora_alpha"))
    lora_alpha_over_r: Optional[float] = None
    try:
        if lora_r is not None and float(lora_r) != 0 and lora_alpha is not None:
            lora_alpha_over_r = float(lora_alpha) / float(lora_r)
    except Exception:
        lora_alpha_over_r = None

    row = {
        "run_id": run_id,
        "pooling_mode": first_not_none(payload.get("pooling_mode"), inferred.get("pooling_mode")),
        "classifier_mode": first_not_none(payload.get("classifier_mode"), inferred.get("classifier_mode")),
        "classifier_parameterization": first_not_none(payload.get("classifier_parameterization"), inferred.get("classifier_parameterization")),
        "joint_mlp_hidden_dim": payload.get("joint_mlp_hidden_dim"),
        "per_emotion_hidden_dim": payload.get("per_emotion_hidden_dim"),
        "num_scalar_factors": first_not_none(payload.get("num_scalar_factors"), inferred.get("num_scalar_factors")),
        "vector_latent_dim": first_not_none(payload.get("vector_latent_dim"), inferred.get("vector_latent_dim")),
        "total_latent_dim": first_not_none(payload.get("total_latent_dim"), inferred.get("total_latent_dim")),
        "use_skip_connection": first_not_none(payload.get("use_skip_connection"), inferred.get("use_skip_connection")),
        "use_lora": first_not_none(payload.get("use_lora"), inferred.get("use_lora")),
        "decoder_only_lora": first_not_none(payload.get("decoder_only_lora"), inferred.get("decoder_only_lora")),
        "lora_start_epoch": first_not_none(payload.get("lora_start_epoch"), inferred.get("lora_start_epoch")),
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_alpha_over_r": lora_alpha_over_r,
        "lora_dropout": payload.get("lora_dropout"),
        "lora_target_modules": as_target_modules(first_not_none(payload.get("lora_target_modules"), inferred.get("lora_target_modules"))),
        "residual_scale_start": first_not_none(payload.get("residual_scale_start"), inferred.get("residual_scale_start")),
        "residual_scale_end": first_not_none(payload.get("residual_scale_end"), inferred.get("residual_scale_end")),
        "residual_adv_weight": first_not_none(payload.get("residual_adv_weight"), inferred.get("residual_adv_weight")),
        "num_epochs": payload.get("num_epochs"),
        "seed": first_not_none(payload.get("seed"), inferred.get("seed")),
        "model_name": payload.get("model_name"),
        "final_test_micro_f1": payload.get("final_test_micro_f1"),
        "final_test_macro_f1": payload.get("final_test_macro_f1"),
    }
    for name in metric_names:
        row[name] = metrics.get(name)
    rows.append(row)

if not rows:
    raise SystemExit(f"No completed ablation_result_summary.json files found under {root.resolve()}")

rows.sort(
    key=lambda r: (
        int(r.get("lora_r") or 0),
        int(r.get("lora_alpha") or 0),
        str(r.get("pooling_mode") or ""),
        str(r.get("classifier_mode") or ""),
        int(r.get("seed") or 0),
    )
)

csv_path.parent.mkdir(parents=True, exist_ok=True)
json_path.parent.mkdir(parents=True, exist_ok=True)
fieldnames = list(rows[0].keys())
with csv_path.open("w", newline="", encoding="utf-8") as fh:
    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
with json_path.open("w", encoding="utf-8") as fh:
    json.dump(rows, fh, indent=2)

print(f"Found {len(rows)} completed ablation runs under {root.resolve()}")
print(f"CSV:  {csv_path.resolve()}")
print(f"JSON: {json_path.resolve()}")
print("\nRows by LoRA rank:")
for row in rows:
    print(
        f"  r={row.get('lora_r')} alpha={row.get('lora_alpha')} "
        f"alpha/r={row.get('lora_alpha_over_r')} "
        f"micro={row.get('final_test_micro_f1')} "
        f"macro={row.get('final_test_macro_f1')} "
        f"sep_r2={row.get('emotion_meaning_separation_r2')} "
        f"vec_leak={row.get('emotion_leakage_vector_r2')} "
        f"branch_mig={row.get('branch_mig')} "
        f"emo_share={row.get('emotion_branch_importance_share')} "
        f"leak_share={row.get('meaning_branch_emotion_leakage_share')}"
    )
PY
