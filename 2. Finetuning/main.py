from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

from finetuning import *

df, label_cols = load_goemotions()

train_df, temp_df = train_test_split(
    df, test_size=0.2, random_state=42, shuffle=True
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, random_state=42, shuffle=True
)

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

output_path = Path("../output/finetune.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)

pd.DataFrame([val_metrics]).to_csv(output_path, index=False)
print(f"\nSaved validation metrics to: {output_path}")

preds = infer_emotions(
    [
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
    ],
    trainer=trainer,
    label_cols=label_cols,
    max_length=128,
    threshold=0.5,
)