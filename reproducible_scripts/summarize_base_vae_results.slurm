#!/usr/bin/env bash
#SBATCH --job-name=summarize-base-vae
#SBATCH --partition=debug_gpuh200
#SBATCH --qos=debug
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=output/%x_%j.out
#SBATCH --error=output/%x_%j.err
set -euo pipefail

mkdir -p logs

# Space-separated list of base VAE artifact directories to summarize.
# Override example:
# sbatch --export=ALL,BASE_VAE_RESULT_DIRS="dir_a dir_b" summarize_base_vae_results.slurm
export BASE_VAE_RESULT_DIRS="${BASE_VAE_RESULT_DIRS:-base_vae_joint_attention_from_scratch_artifacts base_vae_attention_from_scratch_artifacts}"
export BASE_VAE_SUMMARY_DIR="${BASE_VAE_SUMMARY_DIR:-base_vae_summary_artifacts}"

mkdir -p "${BASE_VAE_SUMMARY_DIR}"

uv run python - <<'PY'
import csv
import json
import os
from pathlib import Path
from statistics import mean

result_dirs = [Path(p) for p in os.environ.get("BASE_VAE_RESULT_DIRS", "").split() if p.strip()]
summary_dir = Path(os.environ.get("BASE_VAE_SUMMARY_DIR", "base_vae_summary_artifacts"))
summary_dir.mkdir(parents=True, exist_ok=True)

preferred_names = [
    "base_test_metrics_with_disentanglement.json",
    "test_metrics_with_disentanglement.json",
    "base_eval_metrics_with_disentanglement.json",
    "base_val_metrics_with_disentanglement.json",
    "val_metrics_with_disentanglement.json",
    "ablation_result_summary.json",
    "base_train_eval_summary.json",
]

classification_fields = [
    "micro_f1",
    "macro_f1",
    "weighted_f1",
    "subset_accuracy",
    "hamming_acc",
    "jaccard_micro",
    "average_precision_micro",
    "label_ranking_average_precision",
]

disentanglement_fields = [
    "emotion_in_scalar_r2",
    "emotion_leakage_vector_r2",
    "emotion_leakage_residual_r2",
    "emotion_meaning_separation_r2",
    "emotion_meaning_separation_pearson",
    "branch_mig",
    "emotion_branch_importance_share",
    "meaning_branch_emotion_leakage_share",
    "diagonal_dominance",
    "factor_dci_completeness",
]

extra_fields = [
    "factor_dci_disentanglement",
    "factor_effective_num_factors",
    "factor_active_scalar_factors",
    "scalar_vector_mean_abs_correlation",
    "scalar_vector_max_abs_correlation",
    "residual_emotion_leakage_share",
    "block_assigned_correlation_dominance",
    "block_assigned_importance_dominance",
    "block_top_correlation_match_rate",
    "scalar_factors_per_label",
]

all_metric_fields = classification_fields + disentanglement_fields + extra_fields


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def infer_pooling_mode(path: Path, metrics: dict) -> str:
    text = str(path).lower()
    if metrics.get("pooling_mode"):
        return str(metrics["pooling_mode"])
    if "joint" in text:
        return "joint_scalar_vector"
    return "per_scalar_dim"


def find_metric_file(root: Path):
    if not root.exists():
        return None
    for name in preferred_names:
        p = root / name
        if p.exists():
            return p
    candidates = sorted(root.rglob("*metrics*disentanglement*.json"))
    if candidates:
        test_candidates = [p for p in candidates if "test" in p.name.lower()]
        return test_candidates[0] if test_candidates else candidates[0]
    candidates = sorted(root.rglob("*.json"))
    if candidates:
        # Prefer files that look like test/eval outputs and avoid huge checkpoints metadata if present.
        scored = []
        for p in candidates:
            name = p.name.lower()
            score = 0
            if "test" in name:
                score -= 10
            if "metric" in name:
                score -= 5
            if "summary" in name:
                score -= 3
            scored.append((score, str(p), p))
        return sorted(scored)[0][2]
    return None


def nested_get(d, *keys):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def get_metric(metrics: dict, key: str):
    if key in metrics:
        return metrics[key]

    ld = metrics.get("latent_diagnostics") or {}
    fp = ld.get("factor_power") or {}
    ems = ld.get("emotion_meaning_split") or {}

    mapping = {
        "emotion_in_scalar_r2": (ems, "emotion_in_scalar_r2"),
        "emotion_leakage_vector_r2": (ems, "emotion_leakage_vector_r2"),
        "emotion_leakage_residual_r2": (ems, "emotion_leakage_residual_r2"),
        "emotion_meaning_separation_r2": (ems, "emotion_meaning_separation_r2"),
        "emotion_meaning_separation_pearson": (ems, "emotion_meaning_separation_pearson"),
        "emotion_branch_importance_share": (ems, "emotion_branch_importance_share"),
        "meaning_branch_emotion_leakage_share": (ems, "meaning_branch_emotion_leakage_share"),
        "residual_emotion_leakage_share": (ems, "residual_emotion_leakage_share"),
        "scalar_vector_mean_abs_correlation": (ems, "scalar_vector_mean_abs_correlation"),
        "scalar_vector_max_abs_correlation": (ems, "scalar_vector_max_abs_correlation"),
        "factor_effective_num_factors": (fp, "effective_num_factors"),
        "factor_active_scalar_factors": (fp, "active_scalar_factors"),
    }
    if key in mapping:
        src, subkey = mapping[key]
        return src.get(subkey)

    if key == "branch_mig":
        return nested_get(ems, "branch_group_mig", "emotion_vs_meaning_mig_signed_macro")
    if key == "factor_dci_disentanglement":
        return nested_get(fp, "dci", "dci_disentanglement")
    if key == "factor_dci_completeness":
        return nested_get(fp, "dci", "dci_completeness")
    if key == "diagonal_dominance":
        return nested_get(fp, "diagonal_alignment", "diagonal_dominance")

    block = fp.get("block_alignment") or {}
    block_mapping = {
        "block_assigned_correlation_dominance": "assigned_block_correlation_dominance",
        "block_assigned_importance_dominance": "assigned_block_importance_dominance",
        "block_top_correlation_match_rate": "top_correlation_block_match_rate",
        "scalar_factors_per_label": "scalar_factors_per_label",
    }
    if key in block_mapping:
        return block.get(block_mapping[key])

    return None


def flatten_summary_json(metrics: dict):
    # Some jobs save a wrapper with keys like test/val/final. Prefer test.
    if isinstance(metrics.get("test"), dict):
        return metrics["test"]
    if isinstance(metrics.get("final_test"), dict):
        return metrics["final_test"]
    return metrics

rows = []
missing = []

for root in result_dirs:
    metric_file = find_metric_file(root)
    if metric_file is None:
        missing.append(str(root))
        continue

    raw = load_json(metric_file)
    metrics = flatten_summary_json(raw)

    row = {
        "run_dir": str(root),
        "metrics_file": str(metric_file),
        "model_family": "base_vae",
        "pooling_mode": infer_pooling_mode(root, metrics),
        "classifier_mode": metrics.get("classifier_mode", ""),
        "num_scalar_factors": metrics.get("num_scalar_factors", ""),
        "vector_latent_dim": metrics.get("vector_latent_dim", ""),
        "total_latent_dim": metrics.get("total_latent_dim", ""),
    }

    for field in all_metric_fields:
        value = get_metric(metrics, field)
        row[field] = value if value is not None else ""

    rows.append(row)

if not rows:
    raise SystemExit(
        "No metric JSON files found. Checked: " + ", ".join(str(p) for p in result_dirs)
    )

sort_order = {"joint_scalar_vector": 0, "per_scalar_dim": 1}
rows.sort(key=lambda r: (sort_order.get(r.get("pooling_mode"), 99), r.get("run_dir", "")))

fieldnames = [
    "run_dir",
    "metrics_file",
    "model_family",
    "pooling_mode",
    "classifier_mode",
    "num_scalar_factors",
    "vector_latent_dim",
    "total_latent_dim",
] + all_metric_fields

csv_path = summary_dir / "base_vae_10_metric_summary.csv"
json_path = summary_dir / "base_vae_10_metric_summary.json"

with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

with json_path.open("w", encoding="utf-8") as f:
    json.dump({"rows": rows, "missing_dirs": missing}, f, indent=2)

print(f"Wrote {csv_path}")
print(f"Wrote {json_path}")
print(f"Summarized {len(rows)} base VAE runs")
if missing:
    print("Missing or empty dirs:")
    for item in missing:
        print(f"  - {item}")

print("\nCompact summary:")
for r in rows:
    print(
        f"{r['pooling_mode']:20s} "
        f"micro_f1={r.get('micro_f1','')} "
        f"macro_f1={r.get('macro_f1','')} "
        f"sep_r2={r.get('emotion_meaning_separation_r2','')} "
        f"branch_mig={r.get('branch_mig','')} "
        f"vec_leak_r2={r.get('emotion_leakage_vector_r2','')}"
    )
PY
