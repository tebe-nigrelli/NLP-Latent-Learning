# Emo-DiVA: Emotion Disentanglement via Variational Autoencoders

This repository contains the code used for the **Emo-DiVA** NLP project: **Emotion Disentanglement via Variational Autoencoders**.

The project studies whether the hidden states of an encoder-decoder language model can be pushed into a more interpretable latent space. In particular, we place a FactorVAE between the encoder and decoder of FLAN-T5 and try to separate the latent representation into an emotion branch and a remaining context branch. The goal is not only emotion classification, but also controllable emotion translation: changing the emotional tone of a sentence while preserving its content.

The main result is mixed. The best models learn a useful emotion subspace for multi-label emotion classification and sentence reconstruction, but they do not reliably perform emotion translation.

## Paper summary

The paper evaluates a sequence of architectures, each designed to fix a specific failure mode found in the previous experiments.

1. **Base T5 + FactorVAE**: insert a FactorVAE between the FLAN-T5 encoder and decoder. The latent space is split into emotion factors and a context vector branch. This gives usable emotion classification, but reconstruction and translation are poor.
2. **Residual bottleneck**: add a skip/residual path around the VAE to make training more stable. This helps information flow, but reconstruction is still weak and the shortcut can carry information around the intended latent split.
3. **Decoder LoRA + Copy Loss**: train the T5 decoder through LoRA and add a token-level copy loss. Reconstruction becomes very strong, but the model mostly learns to copy rather than edit emotion.
4. **Adversarial regularization**: add adversaries on the context and residual branches to push emotion information out of the non-emotion paths. This improves separation metrics, but translation still mostly behaves like reconstruction.
5. **Split-FiLM decoder**: remove the residual shortcut, split the VAE decoder into emotion and context decoders, and inject a pooled sentence-level emotion summary through FiLM-style modulation. This gives the best overall classification and disentanglement trade-off, but emotion translation remains unreliable.

The final split-FiLM family reaches roughly **0.59 micro-F1** and **0.52 macro-F1** on GoEmotions classification, with improved separation metrics. Translation samples show small lexical edits or unchanged copies rather than stable emotion transfer, so the conclusion is that the model organizes emotion well enough for recognition but not well enough for independent manipulation.

## Repository layout

```text
.
├── src/emotion_latent_learning/      # Main package: config, modeling, training, evaluation
├── reproducible_scripts/             # Slurm scripts for training, ablations, summaries, and evaluation
├── eda/                              # EDA notebook used to inspect GoEmotions and auxiliary data
├── ablations_results/                # Checked-in summary CSVs from earlier ablations
├── architecture_plots/               # Architecture figures used while writing the report
├── data/                             # Local/generated dataset cache placeholder; ignored by git
├── models/                           # Local/generated model/checkpoint placeholder; ignored by git
├── output/                           # Local/generated Slurm logs and artifacts; ignored by git
├── img/                              # Project image assets
├── pyproject.toml                    # Python package metadata and dependencies
├── uv.lock                           # Locked dependency versions for uv
└── README.md                         # This file
```

Large files are intentionally not tracked. Dataset caches, model checkpoints, Slurm logs, generated JSON files, generated CSV summaries, and full output folders should stay outside git unless they are small and deliberately curated.

## What is implemented

The codebase implements a FLAN-T5 based sequence-to-sequence model with an inserted VAE-style latent bottleneck. The default experimental direction is:

```text
text
  -> FLAN-T5 encoder
  -> FactorVAE encoder
  -> emotion latent factors + context latent vector
  -> VAE / split-FiLM decoder back to T5 decoder memory
  -> FLAN-T5 decoder
  -> reconstructed or edited text
```

The model supports the main components used in the paper:

- FactorVAE KL and total-correlation regularization;
- a split latent space with scalar emotion factors and vector context latents;
- attention pooling over latent sequences for multi-label emotion classification;
- per-emotion, per-emotion-pair, and joint MLP classifier heads;
- optional skip/residual connections;
- decoder LoRA for efficient decoder adaptation;
- copy loss for token-level reconstruction;
- adversarial losses on non-emotion branches;
- split-FiLM decoder conditioning;
- KL/TC warmup schedules;
- reconstruction, translation, latent diagnostic, and attention analysis scripts.

Most configuration defaults live in:

```text
src/emotion_latent_learning/config/defaults.py
```

The Slurm scripts override these defaults through environment variables, so the exact configuration of a run is usually defined by both the Python defaults and the submitted script.

## Dataset

The main paper experiments use **GoEmotions**, a multi-label Reddit emotion dataset with 27 emotion labels plus `neutral`. The code uses the Hugging Face datasets interface:

```python
dataset_name = "goemotions"
dataset_repo = "google-research-datasets/go_emotions"
dataset_config = "simplified"
```

The data is downloaded automatically the first time a training or evaluation script builds the dataset bundle. You do **not** need to manually download GoEmotions or place raw files in `data/`.

The repository also contains some configuration and EDA support for SemEval-2018 EI-reg. That path is useful as auxiliary exploration, but the paper draft reports the main results on GoEmotions.

## What the EDA shows

The EDA notebook is:

```text
eda/EDA.ipynb
```

The useful takeaways for the paper are:

- GoEmotions is strongly imbalanced. `neutral` is much more frequent than most emotion labels.
- Several labels, such as `grief`, `pride`, `relief`, and `nervousness`, are rare.
- Most samples have one label, but the task is still multi-label, so the model uses independent label probabilities and per-label thresholding.
- Texts are short Reddit comments, usually around one sentence, but they include artifacts such as usernames, punctuation fragments, repeated characters, tables, and informal spelling.
- The imbalance makes macro-F1 important. Micro-F1 alone can hide weak performance on rare emotions.
- The short and lexically marked nature of some emotions helps classification, but it does not provide parallel examples of the same content written with different emotions. This is one reason emotion translation is much harder than emotion recognition.

The EDA is mainly used to justify the evaluation choices: multi-label classification, per-label thresholds, macro-F1 reporting, and qualitative inspection of generated translation outputs.

## Setup with uv

### 1. Clone the repository

Replace the URL with the actual repository URL.

```bash
git clone https://github.com/tebe-nigrelli/NLP-Latent-Learning
cd NLP-Latent-Learning
```

### 2. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Restart your shell if `uv` is not available immediately.

### 3. Install Python and dependencies

The project declares Python `>=3.14` and includes a `.python-version` file.

```bash
uv python install 3.14.2
uv sync
```

This creates a local `.venv` and installs the locked dependencies from `uv.lock`.

### 4. Optional: activate the environment

You can run everything through `uv run`, so activation is optional.

```bash
source .venv/bin/activate
```

## Running the EDA notebook

```bash
uv run jupyter lab eda/EDA.ipynb
```

The notebook is for inspection and reporting. The training scripts do not require manually prepared GoEmotions files because the dataset is downloaded automatically at runtime.

## Reproducing the paper experiments

The main reproduction path is through the Slurm scripts in:

```text
reproducible_scripts/
```

Submit jobs from the repository root. Before running jobs, create the common output folders:

```bash
mkdir -p output logs
```

The scripts were written for the our GPU cluster, so you may need to edit the `#SBATCH` partition, QoS, memory, and time settings before running them elsewhere.

### Architecture 1: base T5 + FactorVAE

Train the base attention model:

```bash
sbatch reproducible_scripts/train_eval_base_vae_attention_disentanglement.slurm
```

Train the joint-attention base variant:

```bash
sbatch reproducible_scripts/train_eval_base_vae_joint_attention_disentanglement.slurm
```

Evaluate an existing base checkpoint:

```bash
sbatch reproducible_scripts/evaluate_base_vae_attention_disentanglement.slurm
```

Summarize base results:

```bash
sbatch reproducible_scripts/summarize_base_vae_results.slurm
```

### Architecture 1 ablations: latent dimensions and classifiers

Run the latent-dimension ablation. This varies pooling mode, number of emotion factors, and context-vector size.

```bash
sbatch reproducible_scripts/train_eval_latent_dim_ablation_array.slurm
```

Summarize it:

```bash
sbatch reproducible_scripts/summarize_latent_dim_ablation_results.slurm
```

For the 56-scalar classifier variants used after the first ablation:

```bash
sbatch reproducible_scripts/train_eval_56scalar_adapted_mlp_ablation_array.slurm
sbatch reproducible_scripts/summarize_56scalar_adapted_mlp_ablation_results.slurm
```

```bash
sbatch reproducible_scripts/train_eval_56scalar_pair_mlp_ablation_array.slurm
sbatch reproducible_scripts/summarize_56scalar_pair_mlp_ablation_results.slurm
```

The paper uses these experiments to motivate keeping 56 emotion factors and using classifier variants that map pairs of emotion factors to labels.

### Architecture 2: residual / skip connection

Run the skip-connection ablation:

```bash
sbatch reproducible_scripts/train_eval_skip_connection_ablation_array.slurm
sbatch reproducible_scripts/summarize_skip_connection_ablation_results.slurm
```

This corresponds to the residual bottleneck experiments in the draft. It tests whether routing encoder information around the VAE stabilizes training and preserves information.

### Architecture 3: decoder LoRA + Copy Loss

The decoder-LoRA grid is split into six Slurm array shards:

```bash
sbatch reproducible_scripts/train_eval_skip_decoder_lora_ablation_set1.slurm
sbatch reproducible_scripts/train_eval_skip_decoder_lora_ablation_set2.slurm
sbatch reproducible_scripts/train_eval_skip_decoder_lora_ablation_set3.slurm
sbatch reproducible_scripts/train_eval_skip_decoder_lora_ablation_set4.slurm
sbatch reproducible_scripts/train_eval_skip_decoder_lora_ablation_set5.slurm
sbatch reproducible_scripts/train_eval_skip_decoder_lora_ablation_set6.slurm
```

Then summarize:

```bash
sbatch reproducible_scripts/summarize_skip_decoder_lora_ablation_results.slurm
```

The paper reports that this family greatly improves reconstruction but encourages copying, which is bad for emotion translation.

### Architecture 4: adversarial regularization

Run the skip/decoder-LoRA/adversarial pooling ablation:

```bash
sbatch reproducible_scripts/train_eval_skip_adv_decoder_lora_pooling_ablation.slurm
sbatch reproducible_scripts/summarize_skip_adv_decoder_lora_pooling_ablation_results.slurm
```

This tests whether emotion can be pushed out of the context and residual branches using adversarial classifiers.

### Architecture 5: split-FiLM decoder and beta schedule

Run the config-matched split-FiLM beta-schedule ablation:

```bash
sbatch reproducible_scripts/train_eval_skip_adv_beta_schedule_config_matched.slurm
sbatch reproducible_scripts/summarize_skip_adv_beta_schedule_ablation_results.slurm
```

This is the closest script family to the final architecture in the paper. The important idea is that KL and total-correlation weights start at zero for a few epochs, allowing the model to learn reconstruction before stronger VAE regularization is introduced.

The paper compares warmup start epochs such as 0, 2, 4, and 6. The best trade-off in the draft is around the 4-zero-epoch configuration: classification stays near the best runs, while separation metrics improve.

## Reconstruction and translation evaluation

Reconstruction is evaluated with exact match and token-F1:

```bash
sbatch reproducible_scripts/run_reconstruction_exact_tokenf1.slurm
```

For the config-matched split-FiLM beta-schedule checkpoints:

```bash
sbatch reproducible_scripts/run_reconstruction_config_matched_beta_split_film_only.slurm
```

Emotion translation is evaluated with generated samples and GPT-2 perplexity:

```bash
sbatch reproducible_scripts/run_translation_gpt2_ppl.slurm
```

For the config-matched split-FiLM checkpoints:

```bash
sbatch reproducible_scripts/run_translation_config_matched_beta_split_film_only.slurm
```

The translation scripts are diagnostic. Low perplexity does not prove successful emotion transfer, and we explicitly treats translation as a failure because many outputs are unchanged copies or semantically weak edits.

## Attention analysis

The paper includes attention examples and top attended words by emotion. These scripts reproduce that type of analysis from a checkpoint.

Analyze attention maps:

```bash
uv run python analyze_attention_maps.py \
  --checkpoint path/to/best_checkpoint.pt \
  --split test \
  --output-dir attention_map_analysis \
  --max-examples 0
```

Render selected examples:

```bash
uv run python visualize_attention_examples.py \
  --checkpoint path/to/best_checkpoint.pt \
  --split test \
  --output-dir attention_examples
```

Aggregate high-attention words by emotion:

```bash
uv run python aggregate_attention_words_by_emotion.py \
  --checkpoint path/to/best_checkpoint.pt \
  --split test \
  --output-dir attention_word_aggregation
```

Collect classification examples from the beta-schedule model family:

```bash
sbatch reproducible_scripts/collect_beta_classification_samples.slurm
```

## Expected outputs

Training runs normally create an artifact folder containing files like:

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

Summary scripts collect metrics into CSV/JSON files. The checked-in `ablations_results/` folder contains example summaries from earlier experiments.

Important metric groups are:

- **classification**: micro-F1, macro-F1, weighted-F1, subset accuracy, hamming accuracy, Jaccard, average precision, LRAP;
- **reconstruction**: hidden-state reconstruction losses during training, plus generated exact match and token-F1 in evaluation;
- **disentanglement**: emotion R2, context/vector leakage R2, residual leakage R2, separation R2, branch MIG, DCI-style diagnostics;
- **translation diagnostics**: generated samples, copy rate, GPT-2 perplexity, and qualitative inspection.

For reporting, macro-F1 and per-label thresholds matter because GoEmotions is imbalanced. For translation, qualitative samples matter because perplexity alone can look acceptable even when the output does not perform the requested emotion change.
