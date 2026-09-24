# Reproducing the Emo-DiVA experiments

This guide contains the full environment, cluster, training, evaluation, and output details intentionally kept out of the project README.

## Contents

- [Environment setup](#environment-setup)
- [Data](#data)
- [Cluster setup](#cluster-setup)
- [Recommended smoke run](#recommended-smoke-run)
- [Training and ablations](#training-and-ablations)
- [Reconstruction and translation](#reconstruction-and-translation)
- [Attention analysis](#attention-analysis)
- [Outputs and metrics](#outputs-and-metrics)

## Environment setup

The project requires Python 3.14 or newer and uses `uv` for dependency locking.

```bash
git clone https://github.com/tebe-nigrelli/NLP-Latent-Learning.git
cd NLP-Latent-Learning
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.14.2
uv sync
```

Activation is optional because commands can be run through `uv run`:

```bash
source .venv/bin/activate
```

Confirm the main runtime packages are importable:

```bash
uv run python -c "import torch, datasets, transformers; print(torch.__version__)"
```

The locked dependencies include PyTorch, Transformers, Datasets, PEFT, scikit-learn, and the analysis/notebook stack. CUDA compatibility depends on the GPU driver and PyTorch build available on the target machine.

## Data

The main experiments use the simplified configuration of `google-research-datasets/go_emotions`: 27 emotion labels plus `neutral`. Dataset construction downloads and caches the data automatically through the Hugging Face Datasets interface; no manual raw-data download is required.

By default, the Slurm scripts keep project-local caches under `.cache/`:

```text
.cache/huggingface/datasets
.cache/huggingface/transformers
.cache/uv
```

To inspect the data and plots independently of training:

```bash
uv run jupyter lab eda/EDA.ipynb
```

## Cluster setup

Run all commands from the repository root. Create the shared log directory before submitting jobs:

```bash
mkdir -p output
```

The checked-in scripts target the original Slurm cluster and request a single GPU. Review their `#SBATCH` partition, GPU, memory, time, array, and notification directives before using another cluster.

Common experiment defaults are:

| Setting | Default |
| --- | --- |
| Backbone | `google/flan-t5-base` |
| Epochs | 10 |
| Learning rate | `2e-4` |
| Train / evaluation batch | 8 / 16 |
| Weight decay | `1e-2` |
| Scheduler warmup ratio | 0.1 |
| Seed | 42 |

Python defaults live in `src/emotion_latent_learning/config/defaults.py`. Slurm scripts override them with environment variables, so a run is defined jointly by the Python defaults and its submitted script.

## Recommended smoke run

Before launching full arrays, verify the environment, data access, GPU, and artifact writing with a small base-model run:

```bash
sbatch --export=ALL,BASE_MODEL_NAME=google/flan-t5-small,BASE_NUM_EPOCHS=1,BASE_TRAIN_SAMPLES=4096,BASE_THRESHOLD_SAMPLES=1024,BASE_VAL_SAMPLES=1024,BASE_TEST_SAMPLES=1024 \
  reproducible_scripts/train_eval_base_vae_attention_disentanglement.slurm
```

Values of `0` for the sample-cap variables mean “use the full split.”

## Training and ablations

### Architecture 1: base T5 + FactorVAE

```bash
sbatch reproducible_scripts/train_eval_base_vae_attention_disentanglement.slurm
sbatch reproducible_scripts/train_eval_base_vae_joint_attention_disentanglement.slurm
sbatch reproducible_scripts/evaluate_base_vae_attention_disentanglement.slurm
sbatch reproducible_scripts/summarize_base_vae_results.slurm
```

Latent-size and classifier ablations:

```bash
sbatch reproducible_scripts/train_eval_latent_dim_ablation_array.slurm
sbatch reproducible_scripts/summarize_latent_dim_ablation_results.slurm

sbatch reproducible_scripts/train_eval_56scalar_adapted_mlp_ablation_array.slurm
sbatch reproducible_scripts/summarize_56scalar_adapted_mlp_ablation_results.slurm

sbatch reproducible_scripts/train_eval_56scalar_pair_mlp_ablation_array.slurm
sbatch reproducible_scripts/summarize_56scalar_pair_mlp_ablation_results.slurm
```

### Architecture 2: residual bottleneck

```bash
sbatch reproducible_scripts/train_eval_skip_connection_ablation_array.slurm
sbatch reproducible_scripts/summarize_skip_connection_ablation_results.slurm
```

### Architecture 3: decoder LoRA + copy loss

The decoder-LoRA grid is split across six job arrays:

```bash
for set in 1 2 3 4 5 6; do
  sbatch "reproducible_scripts/train_eval_skip_decoder_lora_ablation_set${set}.slurm"
done

sbatch reproducible_scripts/summarize_skip_decoder_lora_ablation_results.slurm
```

### Architecture 4: adversarial regularization

```bash
sbatch reproducible_scripts/train_eval_skip_adv_decoder_lora_pooling_ablation.slurm
sbatch reproducible_scripts/summarize_skip_adv_decoder_lora_pooling_ablation_results.slurm
```

### Architecture 5: split-FiLM and KL/TC schedule

```bash
sbatch reproducible_scripts/train_eval_skip_adv_beta_schedule_config_matched.slurm
sbatch reproducible_scripts/summarize_skip_adv_beta_schedule_ablation_results.slurm
```

This array compares 0, 2, 4, and 6 zero-weight epochs before a four-epoch KL/total-correlation warmup. The report's best trade-off is the four-zero-epoch configuration.

## Reconstruction and translation

Generated reconstruction is scored with exact match and token-F1:

```bash
sbatch reproducible_scripts/run_reconstruction_exact_tokenf1.slurm
sbatch reproducible_scripts/run_reconstruction_config_matched_beta_split_film_only.slurm
```

Emotion translation is inspected with generated samples, copy behavior, and GPT-2 perplexity:

```bash
sbatch reproducible_scripts/run_translation_gpt2_ppl.slurm
sbatch reproducible_scripts/run_translation_config_matched_beta_split_film_only.slurm
```

Perplexity measures fluency, not whether an output expresses the requested emotion while preserving content. Always inspect the generated examples and copy rate alongside it.

## Attention analysis

Run the analysis scripts against a compatible saved checkpoint:

```bash
uv run python analyze_attention_maps.py \
  --checkpoint path/to/best_checkpoint.pt \
  --split test \
  --output-dir attention_map_analysis \
  --max-examples 0

uv run python visualize_attention_examples.py \
  --checkpoint path/to/best_checkpoint.pt \
  --split test \
  --output-dir attention_examples

uv run python aggregate_attention_words_by_emotion.py \
  --checkpoint path/to/best_checkpoint.pt \
  --split test \
  --output-dir attention_word_aggregation
```

Collect classification examples from the beta-schedule family with:

```bash
sbatch reproducible_scripts/collect_beta_classification_samples.slurm
```

## Outputs and metrics

A typical run artifact directory contains:

```text
config.json
history.json
best_checkpoint.pt
checkpoint_epoch_001.pt
checkpoint_epoch_002.pt
final_checkpoint.pt
final_metrics.json
ablation_result_summary.json
```

Summary jobs produce CSV and JSON aggregations. Small curated examples are checked into `ablations_results/`; complete checkpoints, caches, logs, generated samples, and full output directories are not tracked.

The main metric groups are:

- **classification:** micro-F1, macro-F1, weighted-F1, subset accuracy, hamming accuracy, Jaccard, average precision, and LRAP;
- **reconstruction:** training reconstruction losses, generated exact match, and token-F1;
- **disentanglement:** emotion and leakage R², separation R², branch MIG, and DCI-style diagnostics;
- **translation diagnostics:** generated samples, copy rate, GPT-2 perplexity, and qualitative review.

Because GoEmotions is imbalanced, report macro-F1 and per-label behavior in addition to micro-F1. For translation, generated examples are essential: a fluent or low-perplexity copy is not a successful emotion edit.

