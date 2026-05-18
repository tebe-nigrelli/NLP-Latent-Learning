#!/bin/bash
#SBATCH --job-name=nlp-skip-summary
#SBATCH --partition=debug_gpuh200
#SBATCH --qos=debug
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=output/%x_%j.out
#SBATCH --error=output/%x_%j.err

set -euo pipefail

# Summarize completed skip-connection ablation runs.
# Submit after the array finishes, from the repository root:
#   sbatch summarize_skip_connection_ablation_results.slurm
# Or as a dependency:
#   jid=$(sbatch --parsable train_eval_skip_connection_ablation_array.slurm)
#   sbatch --dependency=afterok:$jid summarize_skip_connection_ablation_results.slurm

PROJECT_DIR="${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
cd "$PROJECT_DIR"
mkdir -p output

export UV_CACHE_DIR="${UV_CACHE_DIR:-$PROJECT_DIR/.cache/uv}"
export PYTHONUNBUFFERED=1
export ABLATION_OUTPUT_ROOT="${ABLATION_OUTPUT_ROOT:-skip_connection_ablation_artifacts}"
export ABLATION_SUMMARY_CSV="${ABLATION_SUMMARY_CSV:-${ABLATION_OUTPUT_ROOT}/ablation_10_metric_summary.csv}"
export ABLATION_SUMMARY_JSON="${ABLATION_SUMMARY_JSON:-${ABLATION_OUTPUT_ROOT}/ablation_10_metric_summary.json}"

uv run python - <<'PY'
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Dict

root = Path(os.environ.get("ABLATION_OUTPUT_ROOT", "latent_dim_ablation_artifacts"))
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

rows = []
for path in sorted(root.glob("*/ablation_result_summary.json")):
    with path.open("r", encoding="utf-8") as fh:
        payload: Dict[str, Any] = json.load(fh)
    metrics = payload.get("ten_metric_summary") or {}
    row = {
        "run_id": payload.get("run_id", path.parent.name),
        "pooling_mode": payload.get("pooling_mode"),
        "classifier_mode": payload.get("classifier_mode"),
        "classifier_parameterization": payload.get("classifier_parameterization"),
        "joint_mlp_hidden_dim": payload.get("joint_mlp_hidden_dim"),
        "per_emotion_hidden_dim": payload.get("per_emotion_hidden_dim"),
        "num_scalar_factors": payload.get("num_scalar_factors"),
        "vector_latent_dim": payload.get("vector_latent_dim"),
        "total_latent_dim": payload.get("total_latent_dim"),
        "use_skip_connection": payload.get("use_skip_connection"),
        "residual_scale_start": payload.get("residual_scale_start"),
        "residual_scale_end": payload.get("residual_scale_end"),
        "residual_adv_weight": payload.get("residual_adv_weight"),
        "num_epochs": payload.get("num_epochs"),
        "seed": payload.get("seed"),
        "model_name": payload.get("model_name"),
        "final_test_micro_f1": payload.get("final_test_micro_f1"),
        "final_test_macro_f1": payload.get("final_test_macro_f1"),
    }
    for name in metric_names:
        row[name] = metrics.get(name)
    rows.append(row)

if not rows:
    raise SystemExit(f"No completed ablation_result_summary.json files found under {root}")

rows.sort(key=lambda r: (str(r.get("pooling_mode")), int(r.get("num_scalar_factors") or 0), int(r.get("vector_latent_dim") or 0), int(r.get("seed") or 0)))

csv_path.parent.mkdir(parents=True, exist_ok=True)
json_path.parent.mkdir(parents=True, exist_ok=True)
with csv_path.open("w", newline="", encoding="utf-8") as fh:
    writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
with json_path.open("w", encoding="utf-8") as fh:
    json.dump(rows, fh, indent=2)

print(f"Found {len(rows)} completed ablation runs")
print(f"CSV:  {csv_path.resolve()}")
print(f"JSON: {json_path.resolve()}")
print("\nTop rows by emotion_meaning_separation_r2:")
for row in sorted(rows, key=lambda r: float(r.get("emotion_meaning_separation_r2") or -999), reverse=True)[:10]:
    print(
        f"  {row['run_id']}: "
        f"micro={row.get('final_test_micro_f1')} "
        f"macro={row.get('final_test_macro_f1')} "
        f"sep_r2={row.get('emotion_meaning_separation_r2')} "
        f"vec_leak={row.get('emotion_leakage_vector_r2')} "
        f"resid_leak={row.get('emotion_leakage_residual_r2')} "
        f"resid_share={row.get('residual_emotion_leakage_share')} "
        f"branch_mig={row.get('branch_mig')} "
        f"block_imp={row.get('block_assigned_importance_dominance')} "
        f"block_match={row.get('block_top_correlation_match_rate')}"
    )
PY
