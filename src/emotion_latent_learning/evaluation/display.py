from __future__ import annotations

from ..utils.common import *
from .generation import summarize_copy_metrics
from ..schemas import ExperimentContext, TrainingState
from .thresholds import threshold_to_tensor

# Focused module: display.

def format_label_names(names: Sequence[str]) -> str:
    if not names:
        return "(none)"
    return ", ".join(names)


def label_names_from_tensor(label_tensor: torch.Tensor, label_names: Sequence[str], threshold: Optional[float] = None) -> List[str]:
    """Extract emotion names from continuous target tensor using threshold.
    
    If threshold is None, uses 0.5 as default (standard for binary targets).
    """
    if threshold is None:
        threshold = 0.5
    active = torch.where(label_tensor.detach().cpu() >= threshold)[0].tolist()
    return [label_names[idx] for idx in active]


def predicted_names_from_logits(logits: torch.Tensor, label_names: Sequence[str], threshold: Any) -> List[str]:
    probs = torch.sigmoid(logits.detach().cpu())
    threshold_tensor = threshold_to_tensor(threshold, probs)
    active = torch.where(probs >= threshold_tensor)[0].tolist()
    return [label_names[idx] for idx in active]


def predict_emotions_for_text(state: TrainingState, ctx: ExperimentContext, text: str) -> List[str]:
    encoded = ctx.tokenizer(
        text,
        max_length=ctx.data_config.max_length,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(ctx.device)
    attention_mask = encoded["attention_mask"].to(ctx.device)
    with torch.no_grad():
        out = state.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=None,
            sample_posterior=ctx.experiment_config.sample_posterior_eval,
            classification_weight=ctx.loss_config.classification_weight,
            recon_weight=ctx.loss_config.recon_weight,
            kl_weight=ctx.loss_config.kl_weight,
        )
    return predicted_names_from_logits(out.classification_logits[0], ctx.emotion_names, state.threshold)


def pretty_print_text_block(title: str, text: str, width: int = 100) -> None:
    import textwrap

    print(title)
    wrapped = textwrap.fill(text, width=width, subsequent_indent="    ")
    print(f"    {wrapped}")


def print_generation_summary(tag: str, source_text: str, generated_text: str, predicted_emotions: Sequence[str]) -> None:
    copy_stats = summarize_copy_metrics([source_text], [generated_text])
    print(tag)
    pretty_print_text_block("  generated text:", generated_text)
    print(f"  predicted emotions on generated text: {format_label_names(predicted_emotions)}")
    print(
        "  copy metrics: "
        f"exact_match={copy_stats['exact_match']:.3f}, "
        f"token_f1={copy_stats['token_f1']:.3f}, "
        f"edit_similarity={copy_stats['edit_similarity']:.3f}"
    )
