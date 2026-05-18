#!/bin/bash
#SBATCH --job-name=translation-gpt2-ppl
#SBATCH --partition=medium_gpuh200
#SBATCH --output=recon_exact_f1_%j.out
#SBATCH --error=recon_exact_f1_%j.err
#SBATCH --time=03:00:00
#SBATCH --mem=24G
#SBATCH --gres=gpu:1

set -euo pipefail
mkdir -p logs

SCRIPT_PATH="${TRANSLATION_PPL_SCRIPT:-../src/emotion_latent_learning/evaluation/check_translation_gpt2_ppl.py}"
OUTPUT_DIR="${TRANSLATION_PPL_OUTPUT_DIR:-translation_gpt2_ppl_artifacts}"
MAX_EXAMPLES="${TRANSLATION_PPL_MAX_EXAMPLES:-32}"
CHECKPOINT_NAMES="${TRANSLATION_PPL_CHECKPOINT_NAMES:-best_checkpoint.pt}"
TARGET_LABELS="${TRANSLATION_PPL_TARGET_LABELS:-joy sadness anger gratitude neutral}"
EDIT_STEPS="${TRANSLATION_PPL_EDIT_STEPS:-24}"
EDIT_METHOD="${TRANSLATION_PPL_EDIT_METHOD:-backprop}"
GPT2_MODEL="${TRANSLATION_PPL_GPT2_MODEL:-gpt2}"
GPT2_DEVICE="${TRANSLATION_PPL_GPT2_DEVICE:-auto}"
SPLIT="${TRANSLATION_PPL_SPLIT:-test}"

if [[ ! -f "$SCRIPT_PATH" ]]; then
  echo "Missing script: $SCRIPT_PATH" >&2
  echo "Copy check_translation_gpt2_ppl.py to ../src/emotion_latent_learning/evaluation/ or set TRANSLATION_PPL_SCRIPT." >&2
  exit 2
fi

read -r -a CHECKPOINT_NAME_ARGS <<< "$CHECKPOINT_NAMES"
read -r -a TARGET_LABEL_ARGS <<< "$TARGET_LABELS"

uv run python "$SCRIPT_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --split "$SPLIT" \
  --max-examples "$MAX_EXAMPLES" \
  --checkpoint-names "${CHECKPOINT_NAME_ARGS[@]}" \
  --target-labels "${TARGET_LABEL_ARGS[@]}" \
  --edit-method "$EDIT_METHOD" \
  --edit-steps "$EDIT_STEPS" \
  --gpt2-model "$GPT2_MODEL" \
  --gpt2-device "$GPT2_DEVICE" \
  --save-examples
