#!/bin/bash
#SBATCH --job-name=translation-beta-split-ppl
#SBATCH --partition=debug_gpuh200
#SBATCH --output=logs/translation_beta_split_ppl_%j.out
#SBATCH --error=logs/translation_beta_split_ppl_%j.err
#SBATCH --time=00:15:00
#SBATCH --qos=debug
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

set -euo pipefail
mkdir -p logs

SCRIPT_PATH="${TRANSLATION_PPL_SCRIPT:-../src/emotion_latent_learning/evaluation/check_translation_gpt2_ppl.py}"
OUTPUT_DIR="${TRANSLATION_PPL_OUTPUT_DIR:-translation_config_matched_beta_split_film_artifacts}"
MAX_EXAMPLES="${TRANSLATION_PPL_MAX_EXAMPLES:-32}"
CHECKPOINT_NAMES="${TRANSLATION_PPL_CHECKPOINT_NAMES:-best_checkpoint.pt}"
TARGET_LABELS="${TRANSLATION_PPL_TARGET_LABELS:-joy sadness anger gratitude neutral}"
EDIT_STEPS="${TRANSLATION_PPL_EDIT_STEPS:-24}"
EDIT_METHOD="${TRANSLATION_PPL_EDIT_METHOD:-backprop}"
EDIT_TARGET_MODE="${TRANSLATION_PPL_EDIT_TARGET_MODE:-add_only}"
GPT2_MODEL="${TRANSLATION_PPL_GPT2_MODEL:-gpt2}"
GPT2_DEVICE="${TRANSLATION_PPL_GPT2_DEVICE:-auto}"
DEVICE="${TRANSLATION_PPL_DEVICE:-auto}"
SPLIT="${TRANSLATION_PPL_SPLIT:-test}"

if [[ ! -f "$SCRIPT_PATH" ]]; then
  echo "Missing script: $SCRIPT_PATH" >&2
  echo "Copy check_translation_gpt2_ppl.py to src/emotion_latent_learning/evaluation/ or set TRANSLATION_PPL_SCRIPT." >&2
  exit 2
fi

ARTIFACT_ROOT="config_matched_beta_schedule_ablation_artifacts"
RUN_KL0="pool-joint_scalar-56_vector-64_total-120_classifier-per-emotion-pair-mlp-standard_skip-false_rscale-1p0-1p0_vadv-0p5_radv-0p0_dec-split_film_decoder-lora-r16-a32-start1_klstart-0-klend-4-klw-0p05_seed-42"
RUN_KL2="pool-joint_scalar-56_vector-64_total-120_classifier-per-emotion-pair-mlp-standard_skip-false_rscale-1p0-1p0_vadv-0p5_radv-0p0_dec-split_film_decoder-lora-r16-a32-start1_klstart-2-klend-6-klw-0p05_seed-42"
RUN_KL4="pool-joint_scalar-56_vector-64_total-120_classifier-per-emotion-pair-mlp-standard_skip-false_rscale-1p0-1p0_vadv-0p5_radv-0p0_dec-split_film_decoder-lora-r16-a32-start1_klstart-4-klend-8-klw-0p05_seed-42"
RUN_KL6="pool-joint_scalar-56_vector-64_total-120_classifier-per-emotion-pair-mlp-standard_skip-false_rscale-1p0-1p0_vadv-0p5_radv-0p0_dec-split_film_decoder-lora-r16-a32-start1_klstart-6-klend-10-klw-0p05_seed-42"

ARTIFACT_ROOTS=(
  "$ARTIFACT_ROOT/$RUN_KL0"
  "$ARTIFACT_ROOT/$RUN_KL2"
  "$ARTIFACT_ROOT/$RUN_KL4"
  "$ARTIFACT_ROOT/$RUN_KL6"
)

for root in "${ARTIFACT_ROOTS[@]}"; do
  if [[ ! -d "$root" ]]; then
    echo "Missing artifact directory: $root" >&2
    exit 2
  fi
done

read -r -a CHECKPOINT_NAME_ARGS <<< "$CHECKPOINT_NAMES"
read -r -a TARGET_LABEL_ARGS <<< "$TARGET_LABELS"

uv run python "$SCRIPT_PATH" \
  --artifact-roots "${ARTIFACT_ROOTS[@]}" \
  --output-dir "$OUTPUT_DIR" \
  --split "$SPLIT" \
  --max-examples "$MAX_EXAMPLES" \
  --checkpoint-names "${CHECKPOINT_NAME_ARGS[@]}" \
  --target-labels "${TARGET_LABEL_ARGS[@]}" \
  --edit-method "$EDIT_METHOD" \
  --edit-target-mode "$EDIT_TARGET_MODE" \
  --edit-steps "$EDIT_STEPS" \
  --gpt2-model "$GPT2_MODEL" \
  --gpt2-device "$GPT2_DEVICE" \
  --device "$DEVICE" \
  --save-examples
