from __future__ import annotations

from ..utils.common import *

# Focused module: config.

@dataclass
class DataConfig:
    """Dataset and dataloader settings.

    `dataset_name` selects the backend used by `build_data_bundle`.

    Supported values:
    - `"goemotions"`: Hugging Face Google GoEmotions simplified split.
    - `"semval2018_ei_reg"`: SemEval-2018 Task 1 EI-reg local TSV files.

    For SemEval EI-reg, set `semeval_data_dir` to the folder where the official
    files should live. If the directory is missing, the runtime can try to
    download/extract the data automatically. You can also set
    `semeval_download_url` to a local or remote official archive. Expected files
    look like `2018-EI-reg-En-anger-train.txt`,
    `2018-EI-reg-En-anger-dev.txt`, and `2018-EI-reg-En-anger-test-gold.txt`.
    """

    dataset_name: str = "goemotions"
    dataset_repo: str = "google-research-datasets/go_emotions"
    dataset_config: str = "simplified"
    max_length: int = 128
    train_batch_size: int = 8
    eval_batch_size: int = 16
    num_workers: int = 0
    calibration_fraction: float = 0.05
    pos_weight_max_ratio: float = 20.0

    # SemEval-2018 Task 1, EI-reg options.
    semeval_data_dir: Optional[str] = None
    semeval_auto_download: bool = True
    semeval_download_url: Optional[str] = None
    semeval_language: str = "En"
    semeval_emotions: Tuple[str, ...] = ("anger", "fear", "joy", "sadness")
    semeval_merge_by_text: bool = True
    semeval_binary_threshold: float = 0.5
    # Set to True when targets are continuous regression values instead of binary labels
    is_regression: bool = False
    # Loss function for classification: 'bce' for multi-label binary classification, 'mse' for regression
    classification_loss_type: str = "bce"


@dataclass
class PromptConfig:
    use_prompt: bool = False
    prompt_text: str = "I am exactly repeating: "
    mask_prompt_loss: bool = True
    max_decoder_length: int = 128


@dataclass
class ModelConfig:
    model_name: str = "google/flan-t5-small"
    num_scalar_factors: int = 28
    vector_latent_dim: int = 512
    vae_hidden_dim: int = 1024
    residual_bottleneck_dim: int = 64
    residual_bottleneck_dropout: float = 0.5
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: Optional[Tuple[str, ...]] = None
    latent_pool_heads: int = 8
    attention_source: str = "encoder_sequence"
    pooling_mode: str = "per_scalar_dim"
    classifier_mode: str = "per_emotion_mlp"
    classifier_parameterization: str = "standard"
    classifier_dropout: float = 0.1
    per_emotion_hidden_dim: int = 32
    joint_mlp_hidden_dim: int = 128
    classifier_uses_mu: bool = False
    use_skip_connection: bool = True
    vae_dropout: float = 0.1
    vector_adversary_hidden_dim: int = 128
    residual_adversary_hidden_dim: int = 128
    decoder_conditioning_mode: str = "split_film"
    vector_branch_dropout: float = 0.10
    scalar_decoder_hidden_dim: int = 512
    film_hidden_dim: int = 512


@dataclass
class LossConfig:
    classification_weight: float = 4.0
    recon_weight: float = 2.0
    kl_weight: float = 1.0
    tc_weight: float = 1.0
    tc_subspace: str = "full"
    tc_mode: str = "per_token"
    copy_weight: float = 2.0
    vector_adv_weight: float = 0.5
    vector_sep_weight: float = 0.01
    orthogonality_weight: float = 0.01
    transfer_strength_weight: float = 0.25
    transfer_strength_margin: float = 2.0
    residual_adv_weight: float = 1.0


@dataclass
class ScheduleConfig:
    num_epochs: int = 30
    lr: float = 2e-4
    disc_lr: float = 5e-5
    vector_adv_lr: float = 2e-4
    residual_adv_lr: float = 2e-4
    weight_decay: float = 1e-2
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.1
    min_lr_scale: float = 0.1
    lora_start_epoch: int = 4
    copy_loss_start_epoch: int = 7
    copy_loss_vae_freeze_epochs: int = 3
    vector_adv_warmup_epochs: int = 5
    residual_adv_warmup_epochs: int = 5
    residual_decay_start_epoch: int = 4
    residual_decay_end_epoch: int = 30
    residual_scale_start: float = 1.0
    residual_scale_end: float = 0.1
    eval_every_epoch: bool = True
    kl_zero_epochs: int = 4
    kl_warmup_end_epoch: int = 10


@dataclass
class ExperimentConfig:
    seed: int = 42
    output_dir: str = "factorvae_tokenlevel_clean_artifacts"
    sample_posterior_train: bool = True
    sample_posterior_eval: bool = False
    prompt_eval_num_examples: int = 32
    prompt_candidates: Tuple[str, ...] = (
        "",
        "I am exactly repeating: ",
        "Repeat exactly: ",
        "Copy exactly: ",
    )
    eval_num_copy_examples: int = 32
    show_num_examples: int = 2
    monitor_num_examples: int = 2
    monitor_edit_num_labels: int = 2
    monitor_edit_steps: int = 24
    threshold_grid_points: int = 101
    threshold_mode: str = "per_label"
    monitor_edit_method: str = "backprop"
    edit_force_scale: float = 1.5
    edit_ridge: float = 1e-3
    edit_backsolve_ridge: float = 1e-3
    edit_target_margin: float = 6.0
    edit_target_mode: str = "rewrite"

    # Evaluation diagnostics. These are intentionally capped because latent
    # probe metrics can become large on full validation/test splits.
    latent_diagnostics_enabled: bool = True
    latent_diagnostics_max_samples: int = 4096
    latent_diagnostics_ridge_alpha: float = 1.0
    latent_active_variance_threshold: float = 1e-4
