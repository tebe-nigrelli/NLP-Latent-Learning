import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from datasets import Dataset, load_dataset, load_from_disk
import torch
import platform
import sys
from importlib.metadata import version, PackageNotFoundError
import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    TrainerCallback
)
from sklearn.model_selection import train_test_split
from pathlib import Path


GO_EMOTIONS_REPO = "google-research-datasets/go_emotions"

DATA_DIR = Path("../data")
GOEMOTIONS_DIR = DATA_DIR / "goemotions_simplified"
HF_CACHE_DIR = DATA_DIR / "huggingface_cache"

def parse_label_ids(label_value) -> list[int]:
    if label_value is None:
        return []
    if isinstance(label_value, str):
        text = label_value.strip()
        if not text:
            return []
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
        return [int(x.strip()) for x in text.split(",") if x.strip()]
    if isinstance(label_value, np.ndarray):
        return [int(x) for x in label_value.tolist()]
    if isinstance(label_value, (list, tuple, set)):
        return [int(x) for x in label_value]
    return [int(label_value)]

def load_goemotions_dataset():
    if GOEMOTIONS_DIR.exists():
        return load_from_disk(str(GOEMOTIONS_DIR))

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(
        GO_EMOTIONS_REPO,
        "simplified",
        cache_dir=str(HF_CACHE_DIR),
    )
    ds.save_to_disk(str(GOEMOTIONS_DIR))
    return ds

def load_and_split_goemotions():
    ds = load_goemotions_dataset()
    label_cols = list(ds["train"].features["labels"].feature.names)

    def split_to_df(split_name):
        out = ds[split_name].to_pandas().copy()
        out["labels"] = out["labels"].apply(parse_label_ids)

        for i, label in enumerate(label_cols):
            out[label] = out["labels"].apply(lambda xs, i=i: int(i in xs))

        out["split"] = split_name
        return out

    train_df = split_to_df("train")
    val_df = split_to_df("validation")
    test_df = split_to_df("test")
    df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    return df, label_cols, train_df, val_df, test_df


def _pkg_version(pkg_name: str):
    try:
        return version(pkg_name)
    except PackageNotFoundError:
        return None


def _get_model_config(model):
    if hasattr(model, "config"):
        return model.config
    if hasattr(model, "base_model") and hasattr(model.base_model, "config"):
        return model.base_model.config
    if hasattr(model, "base_model") and hasattr(model.base_model, "model") and hasattr(model.base_model.model, "config"):
        return model.base_model.model.config
    return None


def _extract_peft_info(model):
    if not hasattr(model, "peft_config") or not model.peft_config:
        return None

    cfg = next(iter(model.peft_config.values()))
    return {
        "task_type": str(getattr(cfg, "task_type", None)),
        "r": getattr(cfg, "r", None),
        "lora_alpha": getattr(cfg, "lora_alpha", None),
        "lora_dropout": getattr(cfg, "lora_dropout", None),
        "bias": getattr(cfg, "bias", None),
        "target_modules": getattr(cfg, "target_modules", None),
        "modules_to_save": getattr(cfg, "modules_to_save", None),
    }

def _save_run_config(
    trainer,
    output_dir: str | Path,
    *,
    model_name: str,
    text_col: str,
    label_cols: list[str],
    max_length: int,
    epochs: int,
    batch_size: int,
    lr: float,
    use_4bit: bool,
    seed: int,
    threshold: float = 0.5,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    args = trainer.args
    model_cfg = _get_model_config(trainer.model)

    run_config = {
        "model_name": model_name,
        "text_col": text_col,
        "label_cols": label_cols,
        "n_labels": len(label_cols),
        "max_length": max_length,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "lr_scheduler_type": str(getattr(args, "lr_scheduler_type", None)),
        "warmup_ratio": getattr(args, "warmup_ratio", None),
        "weight_decay": getattr(args, "weight_decay", None),
        "eval_steps": getattr(args, "eval_steps", None),
        "save_steps": getattr(args, "save_steps", None),
        "logging_steps": getattr(args, "logging_steps", None),
        "gradient_checkpointing": getattr(args, "gradient_checkpointing", None),
        "fp16": getattr(args, "fp16", None),
        "bf16": getattr(args, "bf16", None),
        "optim": getattr(args, "optim", None),
        "seed": getattr(args, "seed", seed),
        "data_seed": getattr(args, "data_seed", seed),
        "metric_for_best_model": getattr(args, "metric_for_best_model", None),
        "greater_is_better": getattr(args, "greater_is_better", None),
        "threshold": threshold,
        "use_4bit": use_4bit,
        "problem_type": getattr(model_cfg, "problem_type", None) if model_cfg else None,
        "base_model_name_or_path": getattr(model_cfg, "_name_or_path", None) if model_cfg else None,
        "id2label": getattr(model_cfg, "id2label", None) if model_cfg else None,
        "label2id": getattr(model_cfg, "label2id", None) if model_cfg else None,
        "train_size": len(trainer.train_dataset) if trainer.train_dataset is not None else None,
        "eval_size": len(trainer.eval_dataset) if trainer.eval_dataset is not None else None,
        "best_metric": getattr(trainer.state, "best_metric", None),
        "best_model_checkpoint": getattr(trainer.state, "best_model_checkpoint", None),
        "global_step": getattr(trainer.state, "global_step", None),
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "transformers_version": _pkg_version("transformers"),
        "datasets_version": _pkg_version("datasets"),
        "peft_version": _pkg_version("peft"),
        "bitsandbytes_version": _pkg_version("bitsandbytes"),
        "sklearn_version": _pkg_version("scikit-learn"),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "peft": _extract_peft_info(trainer.model),
    }

    run_config_path = output_dir / "run_config.json"
    with open(run_config_path, "w") as f:
        json.dump(run_config, f, indent=2, default=str)

    print(f"Saved run config to: {run_config_path}")
    return run_config

class TrainEvalPrintCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            print(
                f"[train] step={state.global_step} "
                f"epoch={state.epoch:.4f} "
                f"loss={logs['loss']:.4f}"
            )

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            print(
                f"[valid] step={state.global_step} "
                f"epoch={state.epoch:.4f} "
                f"loss={metrics.get('eval_loss', float('nan')):.4f} "
                f"macro_f1={metrics.get('eval_macro_f1', float('nan')):.4f} "
                f"micro_f1={metrics.get('eval_micro_f1', float('nan')):.4f} "
                f"weighted_f1={metrics.get('eval_weighted_f1', float('nan')):.4f} "
                f"macro_precision={metrics.get('eval_macro_precision', float('nan')):.4f} "
                f"micro_precision={metrics.get('eval_micro_precision', float('nan')):.4f} "
                f"weighted_precision={metrics.get('eval_weighted_precision', float('nan')):.4f} "
                f"macro_recall={metrics.get('eval_macro_recall', float('nan')):.4f} "
                f"micro_recall={metrics.get('eval_micro_recall', float('nan')):.4f} "
                f"weighted_recall={metrics.get('eval_weighted_recall', float('nan')):.4f} "
                f"accuracy={metrics.get('eval_accuracy', float('nan')):.4f}"
            )

class WeightedMultilabelTrainer(Trainer):
    def __init__(self, *args, pos_weight: torch.Tensor | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weight = pos_weight
        self.model_accepts_loss_kwargs = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels").float()
        outputs = model(**inputs)
        logits = outputs.logits

        if self.pos_weight is None:
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
        else:
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits,
                labels,
                pos_weight=self.pos_weight.to(logits.device),
            )

        return (loss, outputs) if return_outputs else loss

def finetune_emotion_classifier(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    model_name: str,
    text_col: str,
    label_cols: list[str],
    output_dir: str = "./emotion_lora",
    max_length: int = 128,
    epochs: int = 3,
    batch_size: int = 4,
    lr: float = 1e-4,
    use_4bit: bool = True,
    seed: int = 42,
    early_stopping_patience: int = 5,
    pos_weight_clip: float = 20.0,
):
    def to_ds(df: pd.DataFrame):
        x = df[[text_col] + label_cols].copy()
        x["labels"] = x[label_cols].astype("float32").values.tolist()
        return Dataset.from_pandas(x[[text_col, "labels"]], preserve_index=False)

    train_ds = to_ds(train_df)
    val_ds = to_ds(val_df)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
    use_4bit = bool(use_4bit and torch.cuda.is_available())

    id2label = {i: c for i, c in enumerate(label_cols)}
    label2id = {c: i for i, c in enumerate(label_cols)}

    quant_config = None
    if use_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label_cols),
        id2label=id2label,
        label2id=label2id,
        problem_type="multi_label_classification",
        quantization_config=quant_config,
        dtype=None if use_4bit else compute_dtype,
        device_map="auto" if use_4bit else None,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    modules_to_save = []
    for name, _ in model.named_modules():
        if name.endswith("classifier"):
            modules_to_save.append("classifier")
        if name.endswith("score"):
            modules_to_save.append("score")
    modules_to_save = sorted(set(modules_to_save)) or None

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules="all-linear",
        modules_to_save=modules_to_save,
    )
    model = get_peft_model(model, peft_config)

    # Compute multilabel positive-class weights from the training set:
    # pos_weight[i] = (# negatives for label i) / (# positives for label i)
    pos_counts = train_df[label_cols].sum(axis=0).to_numpy(dtype=np.float32)
    neg_counts = (len(train_df) - pos_counts).astype(np.float32)

    pos_weight = neg_counts / np.maximum(pos_counts, 1.0)
    if pos_weight_clip is not None:
        pos_weight = np.clip(pos_weight, 1.0, pos_weight_clip)

    pos_weight = torch.tensor(pos_weight, dtype=torch.float32)

    def tokenize(batch):
        return tokenizer(batch[text_col], truncation=True, max_length=max_length)

    train_ds = train_ds.map(tokenize, batched=True, remove_columns=[text_col])
    val_ds = val_ds.map(tokenize, batched=True, remove_columns=[text_col])

    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = 1.0 / (1.0 + np.exp(-logits))
        preds = (probs >= 0.5).astype(int)
        labels = labels.astype(int)

        return {
            "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
            "micro_f1": f1_score(labels, preds, average="micro", zero_division=0),
            "weighted_f1": f1_score(labels, preds, average="weighted", zero_division=0),

            "macro_precision": precision_score(labels, preds, average="macro", zero_division=0),
            "micro_precision": precision_score(labels, preds, average="micro", zero_division=0),
            "weighted_precision": precision_score(labels, preds, average="weighted", zero_division=0),

            "macro_recall": recall_score(labels, preds, average="macro", zero_division=0),
            "micro_recall": recall_score(labels, preds, average="micro", zero_division=0),
            "weighted_recall": recall_score(labels, preds, average="weighted", zero_division=0),

            "accuracy": accuracy_score(labels, preds),  # subset accuracy in multilabel
        }

    args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=lr,
        lr_scheduler_type="linear",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        warmup_ratio=0.05,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        load_best_model_at_end=True,
        metric_for_best_model="eval_macro_f1",
        greater_is_better=True,
        logging_strategy="steps",
        logging_steps=20,
        save_total_limit=1,
        report_to="none",
        gradient_checkpointing=True,
        fp16=torch.cuda.is_available() and not use_bf16,
        bf16=use_bf16,
        optim="paged_adamw_32bit" if use_4bit else "adamw_torch",
        seed=seed,
        data_seed=seed,
    )

    trainer = WeightedMultilabelTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
        pos_weight=pos_weight,
        callbacks=[
            TrainEvalPrintCallback(),
            EarlyStoppingCallback(
                early_stopping_patience=early_stopping_patience,
                early_stopping_threshold=0.0,
            ),
        ],
    )

    trainer.train()

    # inference mode after training
    if hasattr(trainer.model, "gradient_checkpointing_disable"):
        trainer.model.gradient_checkpointing_disable()
    trainer.model.config.use_cache = True
    trainer.model.eval()

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    _save_run_config(
        trainer,
        output_dir=output_dir,
        model_name=model_name,
        text_col=text_col,
        label_cols=label_cols,
        max_length=max_length,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        use_4bit=use_4bit,
        seed=seed,
        threshold=0.5,
    )

    return trainer, tokenizer

def evaluate_emotion_classifier(
    trainer,
    test_df: pd.DataFrame,
    text_col: str,
    label_cols: list[str],
    max_length: int = 128,
    threshold: float = 0.5,
):
    tokenizer = trainer.data_collator.tokenizer

    x = test_df[[text_col] + label_cols].copy()
    x["labels"] = x[label_cols].astype("float32").values.tolist()
    test_ds = Dataset.from_pandas(x[[text_col, "labels"]], preserve_index=False)

    def tokenize(batch):
        return tokenizer(batch[text_col], truncation=True, max_length=max_length)

    test_ds = test_ds.map(tokenize, batched=True, remove_columns=[text_col])

    out = trainer.predict(test_ds)
    probs = 1.0 / (1.0 + np.exp(-out.predictions))
    preds = (probs >= threshold).astype(int)
    labels = np.array(test_ds["labels"]).astype(int)

    return {
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "micro_f1": f1_score(labels, preds, average="micro", zero_division=0),
        "weighted_f1": f1_score(labels, preds, average="weighted", zero_division=0),

        "macro_precision": precision_score(labels, preds, average="macro", zero_division=0),
        "micro_precision": precision_score(labels, preds, average="micro", zero_division=0),
        "weighted_precision": precision_score(labels, preds, average="weighted", zero_division=0),

        "macro_recall": recall_score(labels, preds, average="macro", zero_division=0),
        "micro_recall": recall_score(labels, preds, average="micro", zero_division=0),
        "weighted_recall": recall_score(labels, preds, average="weighted", zero_division=0),

        "accuracy": accuracy_score(labels, preds),
        "probs": probs,
        "preds": preds,
    }


def infer_emotions(
    texts: list[str],
    trainer,
    label_cols: list[str],
    max_length: int = 128,
    threshold: float = 0.5,
    batch_size: int = 8,
):
    tokenizer = trainer.data_collator.tokenizer
    model = trainer.model
    model.eval()

    device = next(model.parameters()).device

    old_use_cache = getattr(model.config, "use_cache", False)
    model.config.use_cache = False

    all_probs = []

    try:
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                batch_texts = texts[start:start + batch_size]

                inputs = tokenizer(
                    batch_texts,
                    truncation=True,
                    padding=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}

                logits = model(**inputs).logits
                probs = torch.sigmoid(logits).cpu().float().numpy()
                all_probs.append(probs)

                del inputs, logits
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        probs = np.vstack(all_probs)
        preds = (probs >= threshold).astype(int)

        return [
            {
                "text": text,
                "scores": {label: float(prob) for label, prob in zip(label_cols, prob_row)},
                "labels": [label for label, bit in zip(label_cols, pred_row) if bit == 1],
            }
            for text, prob_row, pred_row in zip(texts, probs, preds)
        ]
    finally:
        model.config.use_cache = old_use_cache  
    
def save_loss_plot(
    trainer,
    plot_path: str = "../output/finetune_loss.png",
    csv_path: str = "../output/finetune_loss_history.csv",
):
    plot_path = Path(plot_path)
    csv_path = Path(csv_path)
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    log_history = trainer.state.log_history

    train_steps, train_losses = [], []
    val_steps, val_losses = [], []
    rows = []

    for entry in log_history:
        step = entry.get("step")

        if "loss" in entry and "eval_loss" not in entry:
            train_steps.append(step)
            train_losses.append(entry["loss"])
            rows.append(
                {
                    "step": step,
                    "train_loss": entry["loss"],
                    "val_loss": None,
                    "epoch": entry.get("epoch"),
                }
            )

        if "eval_loss" in entry:
            val_steps.append(step)
            val_losses.append(entry["eval_loss"])
            rows.append(
                {
                    "step": step,
                    "train_loss": None,
                    "val_loss": entry["eval_loss"],
                    "epoch": entry.get("epoch"),
                }
            )

    history_df = pd.DataFrame(rows).sort_values(
        by=["step"], na_position="last"
    ).reset_index(drop=True)
    history_df.to_csv(csv_path, index=False)

    plt.figure(figsize=(8, 5))

    if train_losses:
        plt.plot(train_steps, train_losses, label="Train loss")
    if val_losses:
        plt.plot(val_steps, val_losses, label="Validation loss")

    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()

    print(f"Saved loss history CSV to: {csv_path}")
    print(f"Saved loss plot to: {plot_path}")
