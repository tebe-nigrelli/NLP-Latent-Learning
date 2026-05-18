#!/bin/bash
#SBATCH --job-name=recon_exact_f1
#SBATCH --partition=medium_gpuh200
#SBATCH --output=recon_exact_f1_%j.out
#SBATCH --error=recon_exact_f1_%j.err
#SBATCH --time=03:00:00
#SBATCH --mem=24G
#SBATCH --gres=gpu:1

set -euo pipefail

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

SCRIPT_PATH="${SCRIPT_PATH:-../src/emotion_latent_learning/evaluation/check_reconstruction_exact_tokenf1.py}"
OUTPUT_DIR="${OUTPUT_DIR:-reconstruction_exact_tokenf1_artifacts}"
MAX_EXAMPLES="${MAX_EXAMPLES:-128}"
BATCH_SIZE="${BATCH_SIZE:-8}"
SPLIT="${SPLIT:-test}"
MODES="${MODES:-full latent_only}"
CHECKPOINT_NAMES="${CHECKPOINT_NAMES:-best_checkpoint.pt final_checkpoint.pt}"
ARTIFACT_ROOTS="${ARTIFACT_ROOTS:-}"
SAVE_EXAMPLES="${SAVE_EXAMPLES:-1}"

if [[ ! -f "${SCRIPT_PATH}" ]]; then
  echo "Could not find ${SCRIPT_PATH}. Copy check_reconstruction_exact_tokenf1.py into ../src/emotion_latent_learning/evaluation/ first." >&2
  exit 2
fi

cmd=(uv run python "${SCRIPT_PATH}"
  --output-dir "${OUTPUT_DIR}"
  --max-examples "${MAX_EXAMPLES}"
  --batch-size "${BATCH_SIZE}"
  --split "${SPLIT}"
  --modes ${MODES}
  --checkpoint-names ${CHECKPOINT_NAMES}
)

if [[ -n "${ARTIFACT_ROOTS}" ]]; then
  cmd+=(--artifact-roots ${ARTIFACT_ROOTS})
fi

if [[ "${SAVE_EXAMPLES}" == "1" ]]; then
  cmd+=(--save-examples)
fi

printf 'Running:'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"
