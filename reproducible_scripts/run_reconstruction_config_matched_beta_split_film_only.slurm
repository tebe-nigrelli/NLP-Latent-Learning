#!/bin/bash
#SBATCH --job-name=recon_beta_split_film
#SBATCH --output=logs/recon_beta_split_film_%j.out
#SBATCH --error=logs/recon_beta_split_film_%j.err
#SBATCH --partition=debug_gpuh200
#SBATCH --qos=debug 
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

set -euo pipefail

mkdir -p logs

# Run from the repository root.
cd "${SLURM_SUBMIT_DIR:-$PWD}"

SCRIPT_PATH="${RECON_SCRIPT_PATH:-../src/emotion_latent_learning/evaluation/check_reconstruction_exact_tokenf1.py}"
OUTPUT_DIR="${RECON_OUTPUT_DIR:-reconstruction_config_matched_beta_split_film_artifacts}"
MAX_EXAMPLES="${RECON_MAX_EXAMPLES:-128}"
BATCH_SIZE="${RECON_BATCH_SIZE:-8}"
SPLIT="${RECON_SPLIT:-test}"
CHECKPOINT_NAMES="${RECON_CHECKPOINT_NAMES:-best_checkpoint.pt final_checkpoint.pt}"

# These runs have use_skip_connection=false, so full and latent_only are equivalent.
# Use full by default to avoid duplicate work. Override with RECON_MODES="full latent_only" if needed.
MODES="${RECON_MODES:-full}"

ROOT="config_matched_beta_schedule_ablation_artifacts"
RUN_0="$ROOT/pool-joint_scalar-56_vector-64_total-120_classifier-per-emotion-pair-mlp-standard_skip-false_rscale-1p0-1p0_vadv-0p5_radv-0p0_dec-split_film_decoder-lora-r16-a32-start1_klstart-0-klend-4-klw-0p05_seed-42"
RUN_2="$ROOT/pool-joint_scalar-56_vector-64_total-120_classifier-per-emotion-pair-mlp-standard_skip-false_rscale-1p0-1p0_vadv-0p5_radv-0p0_dec-split_film_decoder-lora-r16-a32-start1_klstart-2-klend-6-klw-0p05_seed-42"
RUN_4="$ROOT/pool-joint_scalar-56_vector-64_total-120_classifier-per-emotion-pair-mlp-standard_skip-false_rscale-1p0-1p0_vadv-0p5_radv-0p0_dec-split_film_decoder-lora-r16-a32-start1_klstart-4-klend-8-klw-0p05_seed-42"
RUN_6="$ROOT/pool-joint_scalar-56_vector-64_total-120_classifier-per-emotion-pair-mlp-standard_skip-false_rscale-1p0-1p0_vadv-0p5_radv-0p0_dec-split_film_decoder-lora-r16-a32-start1_klstart-6-klend-10-klw-0p05_seed-42"

for d in "$RUN_0" "$RUN_2" "$RUN_4" "$RUN_6"; do
  if [[ ! -d "$d" ]]; then
    echo "Missing run directory: $d" >&2
    exit 2
  fi
done

if [[ ! -f "$SCRIPT_PATH" ]]; then
  echo "Missing reconstruction script: $SCRIPT_PATH" >&2
  echo "Expected location: src/emotion_latent_learning/evaluation/check_reconstruction_exact_tokenf1.py" >&2
  exit 2
fi

echo "Running reconstruction check on beta split-FiLM runs"
echo "Output dir: $OUTPUT_DIR"
echo "Max examples per checkpoint/mode: $MAX_EXAMPLES"
echo "Modes: $MODES"
echo "Checkpoints: $CHECKPOINT_NAMES"

uv run python "$SCRIPT_PATH" \
  --artifact-roots "$RUN_0" "$RUN_2" "$RUN_4" "$RUN_6" \
  --checkpoint-names $CHECKPOINT_NAMES \
  --output-dir "$OUTPUT_DIR" \
  --split "$SPLIT" \
  --max-examples "$MAX_EXAMPLES" \
  --batch-size "$BATCH_SIZE" \
  --modes $MODES \
  --save-examples

echo "Done. Summary: $OUTPUT_DIR/reconstruction_summary.csv"
