from pathlib import Path
import pandas as pd

from finetuning import *

_, label_cols, train_df, val_df, test_df = load_and_split_goemotions()

trainer, tokenizer = finetune_emotion_classifier(
    train_df=train_df,
    val_df=val_df,
    model_name="../models/Qwen2.5-3B",
    text_col="text",
    label_cols=label_cols,
    output_dir="../models/finetuned/qwen_emotion_lora",
    max_length=128,
    epochs=3,
    batch_size=4,
    lr=2e-4,
    use_4bit=True,
)

save_loss_plot(trainer)

val_metrics = evaluate_emotion_classifier(
    trainer=trainer,
    test_df=val_df,
    text_col="text",
    label_cols=label_cols,
    max_length=128,
    threshold=0.5,
)

print("\nValidation statistics:")
for k, v in val_metrics.items():
    print(f"{k}: {v}")

output_dir = Path("../output")
output_dir.mkdir(parents=True, exist_ok=True)

metrics_path = output_dir / "finetune.csv"
pd.DataFrame([val_metrics]).to_csv(metrics_path, index=False)
print(f"\nSaved validation metrics to: {metrics_path}")

# Inference on the full validation set
val_preds = infer_emotions(
    val_df["text"].tolist(),
    trainer=trainer,
    label_cols=label_cols,
    max_length=128,
    threshold=0.5,
)

val_preds_df = pd.DataFrame(val_preds)
val_inference_df = pd.concat(
    [val_df.reset_index(drop=True), val_preds_df.reset_index(drop=True)],
    axis=1,
)

val_inference_path = output_dir / "validation_inference.csv"
val_inference_df.to_csv(val_inference_path, index=False)
print(f"Saved validation inference to: {val_inference_path}")

test_phrases = [
    "I am exhausted and angry",
    "I missed you so much",
    "I feel calm and content today",
    "I am terrified of what might happen",
    "This news made me so happy",
    "I feel lonely even in a crowd",
    "I am frustrated that nothing works",
    "I am deeply grateful for your help",
    "I feel guilty about what I said",
    "I am shocked by what I just heard",
    "I feel hopeful about the future",
    "I am jealous of their success",
    "I feel embarrassed about my mistake",
    "I am proud of what we achieved",
    "I feel empty and numb inside",
    "I am excited for tomorrow",
    "I feel overwhelmed by everything",
    "I am disappointed in myself",
    "I feel safe when I am with you",
    "I am confused and unsure what to do",
]

test_preds = infer_emotions(
    test_phrases,
    trainer=trainer,
    label_cols=label_cols,
    max_length=128,
    threshold=0.5,
)

print("\nTest phrase predictions:")
for text, pred in zip(test_phrases, test_preds):
    print(f"\nText: {text}")
    print(pred)