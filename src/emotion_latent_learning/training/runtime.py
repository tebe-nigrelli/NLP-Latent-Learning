from __future__ import annotations

from ..utils.common import *
from ..modeling.adversaries import FactorVAEDiscriminator, ResidualEmotionAdversary, VectorEmotionAdversary
from ..config import DataConfig, ExperimentConfig, LossConfig, ModelConfig, PromptConfig, ScheduleConfig
from ..data import (
    EmotionTextCollator,
    EmotionTextDataset,
    GoEmotionsExampleDataset,
    SemEvalEIRegDataset,
    compute_pos_weight,
    labels_are_continuous,
    load_semval2018_ei_reg,
    download_semval2018_ei_reg,
    semval_download_instructions,
)
from .losses import tc_subspace_dim
from ..modeling import T5FactorVAEModel
from ..schemas import ExperimentContext, TrainingState
from ..modeling.t5_utils import set_lora_trainable
from ..evaluation.thresholds import default_threshold
from ..utils import optimizable_model_parameters, save_json

# Focused module: runtime.

@dataclass
class DataBundle:
    """Dataset, tokenizer, loaders, labels, and monitor examples for the experiment."""

    tokenizer: Any
    raw_dataset: Any
    train_split: Any
    val_split: Any
    test_split: Any
    emotion_names: Sequence[str]
    num_labels: int
    dataset_name: str
    target_kind: str
    train_dataset: EmotionTextDataset
    val_dataset: EmotionTextDataset
    test_dataset: EmotionTextDataset
    train_core_dataset: Subset
    threshold_tune_dataset: Subset
    train_loader: DataLoader
    threshold_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    pos_weight: torch.Tensor
    monitor_example_indices: List[int]
    monitor_target_labels: List[List[str]]


@dataclass
class ExperimentRuntime:
    """All mutable objects needed to train and evaluate one run."""

    ctx: ExperimentContext
    state: TrainingState
    output_dir: Path
    data: DataBundle


def default_configs() -> Tuple[DataConfig, PromptConfig, ModelConfig, LossConfig, ScheduleConfig, ExperimentConfig]:
    """Create fresh default configuration objects."""

    return (
        DataConfig(),
        PromptConfig(),
        ModelConfig(),
        LossConfig(),
        ScheduleConfig(),
        ExperimentConfig(),
    )


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Return cuda when available unless disabled."""

    return torch.device("cuda" if prefer_cuda and torch.cuda.is_available() else "cpu")


def _normalized_dataset_name(name: str) -> str:
    return name.lower().strip().replace("-", "_")


def _load_dataset_splits(
    data_config: DataConfig,
) -> Tuple[Any, Any, Any, List[str], str, str, Any]:
    """Return train/val/test splits plus metadata for the selected dataset."""

    dataset_name = _normalized_dataset_name(data_config.dataset_name)
    if dataset_name in {"goemotions", "go_emotions"}:
        raw_dataset = load_dataset(data_config.dataset_repo, data_config.dataset_config)
        train_split = raw_dataset["train"]
        val_split = raw_dataset["validation"] if "validation" in raw_dataset else raw_dataset["test"]
        test_split = raw_dataset["test"] if "test" in raw_dataset else val_split
        emotion_names = list(train_split.features["labels"].feature.names)
        return train_split, val_split, test_split, emotion_names, "goemotions", "multilabel", raw_dataset

    if dataset_name in {"semval2018_ei_reg", "semval_2018_ei_reg", "semval_ei_reg", "ei_reg"}:
        if not data_config.semeval_data_dir:
            raise ValueError(
                "DataConfig.semeval_data_dir must point to the SemEval-2018 EI-reg data directory "
                "when dataset_name='semval2018_ei_reg'."
            )

        data_dir = Path(data_config.semeval_data_dir).expanduser()
        if getattr(data_config, "semeval_auto_download", True):
            download_semval2018_ei_reg(
                data_dir=data_dir,
                url=getattr(data_config, "semeval_download_url", None),
                language=data_config.semeval_language,
                emotions=data_config.semeval_emotions,
            )
        elif not data_dir.exists():
            raise FileNotFoundError(
                semval_download_instructions(
                    data_dir=data_dir,
                    language=data_config.semeval_language,
                    emotions=data_config.semeval_emotions,
                )
            )

        semval = load_semval2018_ei_reg(
            data_dir=data_dir,
            language=data_config.semeval_language,
            emotions=data_config.semeval_emotions,
            merge_by_text=data_config.semeval_merge_by_text,
        )
        return (
            semval.train,
            semval.validation,
            semval.test,
            list(semval.emotion_names),
            "semval2018_ei_reg",
            "intensity_regression",
            semval,
        )

    raise ValueError(
        f"Unsupported dataset_name={data_config.dataset_name!r}. "
        "Use 'goemotions' or 'semval2018_ei_reg'."
    )


def _make_text_dataset(split: Any, num_labels: int, dataset_name: str) -> EmotionTextDataset:
    if dataset_name == "goemotions":
        return GoEmotionsExampleDataset(split, num_labels=num_labels)
    if dataset_name == "semval2018_ei_reg":
        return SemEvalEIRegDataset(split)
    raise ValueError(f"Unsupported dataset_name={dataset_name!r}")


def build_data_bundle(
    data_config: DataConfig,
    model_config: ModelConfig,
    experiment_config: ExperimentConfig,
    tokenizer: Optional[Any] = None,
) -> DataBundle:
    """Load the selected dataset, build datasets/loaders, and choose fixed monitor examples."""

    tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_config.model_name)
    train_split, val_split, test_split, emotion_names, dataset_name, target_kind, raw_dataset = _load_dataset_splits(
        data_config
    )
    num_labels = len(emotion_names)

    train_dataset = _make_text_dataset(train_split, num_labels=num_labels, dataset_name=dataset_name)
    val_dataset = _make_text_dataset(val_split, num_labels=num_labels, dataset_name=dataset_name)
    test_dataset = _make_text_dataset(test_split, num_labels=num_labels, dataset_name=dataset_name)

    all_train_indices = np.arange(len(train_dataset))
    rng = np.random.default_rng(experiment_config.seed)
    rng.shuffle(all_train_indices)
    num_calibration = max(1, int(len(all_train_indices) * data_config.calibration_fraction))
    calibration_indices = all_train_indices[:num_calibration].tolist()
    train_core_indices = all_train_indices[num_calibration:].tolist()

    train_core_dataset = Subset(train_dataset, train_core_indices)
    threshold_tune_dataset = Subset(train_dataset, calibration_indices)
    collator = EmotionTextCollator(tokenizer=tokenizer, max_length=data_config.max_length)

    generator = torch.Generator()
    generator.manual_seed(experiment_config.seed)

    train_loader = DataLoader(
        train_core_dataset,
        batch_size=data_config.train_batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=data_config.num_workers,
        collate_fn=collator,
        generator=generator,
    )
    threshold_loader = DataLoader(
        threshold_tune_dataset,
        batch_size=data_config.eval_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=data_config.num_workers,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=data_config.eval_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=data_config.num_workers,
        collate_fn=collator,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=data_config.eval_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=data_config.num_workers,
        collate_fn=collator,
    )

    pos_weight = compute_pos_weight(train_dataset.labels, max_ratio=data_config.pos_weight_max_ratio)
    if labels_are_continuous(train_dataset.labels) and data_config.pos_weight_max_ratio > 1.0:
        # Keep regression-like intensity training stable. Heavy class imbalance weights
        # are useful for sparse binary labels but can over-amplify continuous scores.
        pos_weight = pos_weight.clamp(max=min(data_config.pos_weight_max_ratio, 5.0))

    monitor_rng = random.Random(experiment_config.seed + 7)
    monitor_example_indices = monitor_rng.sample(
        list(range(len(val_dataset))),
        k=min(experiment_config.monitor_num_examples, len(val_dataset)),
    )
    monitor_target_labels = [
        monitor_rng.sample(list(emotion_names), k=min(experiment_config.monitor_edit_num_labels, len(emotion_names)))
        for _ in monitor_example_indices
    ]

    return DataBundle(
        tokenizer=tokenizer,
        raw_dataset=raw_dataset,
        train_split=train_split,
        val_split=val_split,
        test_split=test_split,
        emotion_names=emotion_names,
        num_labels=num_labels,
        dataset_name=dataset_name,
        target_kind=target_kind,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        train_core_dataset=train_core_dataset,
        threshold_tune_dataset=threshold_tune_dataset,
        train_loader=train_loader,
        threshold_loader=threshold_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        pos_weight=pos_weight,
        monitor_example_indices=monitor_example_indices,
        monitor_target_labels=monitor_target_labels,
    )


def _make_lr_lambda(schedule_config: ScheduleConfig, total_steps: int):
    warmup_steps = max(1, int(schedule_config.warmup_ratio * total_steps))

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step + 1) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return schedule_config.min_lr_scale + (1.0 - schedule_config.min_lr_scale) * cosine

    return lr_lambda


def build_runtime(
    data: DataBundle,
    data_config: DataConfig,
    prompt_config: PromptConfig,
    model_config: ModelConfig,
    loss_config: LossConfig,
    schedule_config: ScheduleConfig,
    experiment_config: ExperimentConfig,
    device: Optional[torch.device] = None,
    save_config: bool = True,
) -> ExperimentRuntime:
    """Instantiate model, discriminators, optimizers, scheduler, context, and training state."""

    device = device or get_device()
    
    # Auto-configure loss type based on is_regression flag
    if data_config.is_regression:
        data_config.classification_loss_type = "mse"
    else:
        data_config.classification_loss_type = "bce"
    
    model = T5FactorVAEModel(
        model_config=model_config,
        num_labels=data.num_labels,
        pos_weight=data.pos_weight,
        classification_loss_type=data_config.classification_loss_type,
    ).to(device)

    discriminator = None
    if loss_config.tc_weight > 0.0 and loss_config.tc_subspace != "none":
        discriminator = FactorVAEDiscriminator(
            latent_dim=tc_subspace_dim(loss_config, model_config),
            hidden_dim=512,
        ).to(device)

    vector_adversary = None
    if loss_config.vector_adv_weight > 0.0:
        vector_adversary = VectorEmotionAdversary(
            vector_dim=model_config.vector_latent_dim,
            num_labels=data.num_labels,
            hidden_dim=model_config.vector_adversary_hidden_dim,
        ).to(device)

    residual_adversary = None
    if model_config.use_skip_connection and loss_config.residual_adv_weight > 0.0:
        residual_adversary = ResidualEmotionAdversary(
            residual_dim=model.hidden_size,
            num_labels=data.num_labels,
            hidden_dim=model_config.residual_adversary_hidden_dim,
        ).to(device)

    set_lora_trainable(model.shared_t5, enabled=False)

    optimizer = torch.optim.AdamW(
        optimizable_model_parameters(model),
        lr=schedule_config.lr,
        weight_decay=schedule_config.weight_decay,
    )
    disc_optimizer = None if discriminator is None else torch.optim.AdamW(
        discriminator.parameters(),
        lr=schedule_config.disc_lr,
        weight_decay=schedule_config.weight_decay,
    )
    vector_adv_optimizer = None if vector_adversary is None else torch.optim.AdamW(
        vector_adversary.parameters(),
        lr=schedule_config.vector_adv_lr,
        weight_decay=schedule_config.weight_decay,
    )
    residual_adv_optimizer = None if residual_adversary is None else torch.optim.AdamW(
        residual_adversary.parameters(),
        lr=schedule_config.residual_adv_lr,
        weight_decay=schedule_config.weight_decay,
    )

    scheduler_total_steps = max(1, schedule_config.num_epochs * len(data.train_loader))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=_make_lr_lambda(schedule_config, scheduler_total_steps),
    )

    ctx = ExperimentContext(
        tokenizer=data.tokenizer,
        emotion_names=data.emotion_names,
        device=device,
        data_config=data_config,
        prompt_config=prompt_config,
        model_config=model_config,
        loss_config=loss_config,
        schedule_config=schedule_config,
        experiment_config=experiment_config,
        pos_weight=data.pos_weight.to(device),
    )

    state = TrainingState(
        model=model,
        discriminator=discriminator,
        vector_adversary=vector_adversary,
        residual_adversary=residual_adversary,
        optimizer=optimizer,
        disc_optimizer=disc_optimizer,
        vector_adv_optimizer=vector_adv_optimizer,
        residual_adv_optimizer=residual_adv_optimizer,
        lr_scheduler=lr_scheduler,
        threshold=default_threshold(num_labels=data.num_labels, mode=experiment_config.threshold_mode),
        best_val_micro_f1=float("-inf"),
        step=0,
        history=[],
    )

    output_dir = Path(experiment_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if save_config:
        save_json(
            output_dir / "config.json",
            {
                "data_config": asdict(data_config),
                "prompt_config": asdict(prompt_config),
                "model_config": asdict(model_config),
                "loss_config": asdict(loss_config),
                "schedule_config": asdict(schedule_config),
                "experiment_config": asdict(experiment_config),
                "emotion_names": list(data.emotion_names),
                "dataset_name": data.dataset_name,
                "target_kind": data.target_kind,
            },
        )

    return ExperimentRuntime(ctx=ctx, state=state, output_dir=output_dir, data=data)


def runtime_summary(runtime: ExperimentRuntime) -> Dict[str, Any]:
    """Return a compact summary of a runtime after model construction."""

    model = runtime.state.model
    trainable_names = [name for name, param in model.named_parameters() if param.requires_grad]
    summary = {
        "device": str(runtime.ctx.device),
        "hidden_size": model.hidden_size,
        "latent_dim": runtime.ctx.model_config.num_scalar_factors + runtime.ctx.model_config.vector_latent_dim,
        "num_labels": len(runtime.ctx.emotion_names),
        "dataset_name": runtime.data.dataset_name,
        "target_kind": runtime.data.target_kind,
        "train_core_samples": len(runtime.data.train_core_dataset),
        "threshold_tune_samples": len(runtime.data.threshold_tune_dataset),
        "train_batches": len(runtime.data.train_loader),
        "threshold_batches": len(runtime.data.threshold_loader),
        "val_batches": len(runtime.data.val_loader),
        "test_batches": len(runtime.data.test_loader),
        "trainable_parameter_groups": len(trainable_names),
        "first_trainable_names": trainable_names[:10],
        "output_dir": str(runtime.output_dir.resolve()),
    }
    if getattr(model, "lora_target_modules", tuple()):
        summary["lora_target_modules"] = list(model.lora_target_modules)
    if torch.cuda.is_available():
        summary["gpu_total_gb"] = torch.cuda.get_device_properties(runtime.ctx.device).total_memory / (1024 ** 3)
    return summary
