import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datasets import Dataset, load_dataset, load_from_disk
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
        
def finetune_emotion_classifier(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    model_name: str,
    text_col: str,
    label_cols: list[str],
    output_dir: str = "./emotion_lora",
    max_length: int = 128,
    epochs: int = 3,
    batch_size: int = 4,   # small batch
    lr: float = 2e-4,
    use_4bit: bool = True,
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
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_strategy="steps",
        logging_steps=20,
        save_total_limit=1,
        report_to="none",
        gradient_checkpointing=True,
        fp16=torch.cuda.is_available() and not use_bf16,
        bf16=use_bf16,
        optim="paged_adamw_32bit" if use_4bit else "adamw_torch",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[TrainEvalPrintCallback()],
    )

    trainer.train()

    # inference mode after training: faster than leaving training settings on
    if hasattr(trainer.model, "gradient_checkpointing_disable"):
        trainer.model.gradient_checkpointing_disable()
    trainer.model.config.use_cache = True
    trainer.model.eval()

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
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
):
    tokenizer = trainer.data_collator.tokenizer
    model = trainer.model
    model.eval()
    model.config.use_cache = True

    device = next(model.parameters()).device

    inputs = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        logits = model(**inputs).logits
        probs = torch.sigmoid(logits).cpu().numpy()

    preds = (probs >= threshold).astype(int)

    return [
        {
            "text": text,
            "scores": {label: float(prob) for label, prob in zip(label_cols, prob_row)},
            "labels": [label for label, bit in zip(label_cols, pred_row) if bit == 1],
        }
        for text, prob_row, pred_row in zip(texts, probs, preds)
    ]
    
    
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
