from __future__ import annotations

from ..utils.common import *
from ..training.checkpointing import load_compact_model_state_dict, save_checkpoint
from .display import format_label_names, label_names_from_tensor, print_generation_summary
from .editing import make_partial_desired_emotion_target, optimize_emotion_latents_for_target
from .metrics import evaluate_loader, evaluate_prompt_candidates, summarize_eval_metrics
from .monitoring import run_epoch_monitor
from ..training.runtime import ExperimentRuntime
from ..training.schedules import current_loss_weights, slim_history_row
from .thresholds import restore_threshold
from ..utils import release_memory, save_json, to_serializable

# Focused module: evaluation.

def load_best_checkpoint(runtime: ExperimentRuntime, filename: str = "best_checkpoint.pt") -> Dict[str, Any]:
    """Load the best compact checkpoint into the runtime state."""

    state = runtime.state
    ctx = runtime.ctx
    checkpoint = torch.load(runtime.output_dir / filename, map_location=ctx.device)
    load_compact_model_state_dict(state.model, checkpoint["model_state_dict"])
    if state.discriminator is not None and checkpoint.get("discriminator_state_dict") is not None:
        state.discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
    if state.vector_adversary is not None and checkpoint.get("vector_adversary_state_dict") is not None:
        state.vector_adversary.load_state_dict(checkpoint["vector_adversary_state_dict"])
    if state.residual_adversary is not None and checkpoint.get("residual_adversary_state_dict") is not None:
        state.residual_adversary.load_state_dict(checkpoint["residual_adversary_state_dict"])
    state.threshold = restore_threshold(
        checkpoint["threshold"],
        num_labels=len(ctx.emotion_names),
        mode=ctx.experiment_config.threshold_mode,
    )
    return checkpoint


def final_evaluation(runtime: ExperimentRuntime, load_best: bool = True) -> Dict[str, Any]:
    """Evaluate calibration, validation, and test splits and save final metrics/checkpoint."""

    state = runtime.state
    ctx = runtime.ctx
    data = runtime.data
    output_dir = runtime.output_dir

    if load_best:
        load_best_checkpoint(runtime)

    final_weights = current_loss_weights(epoch=ctx.schedule_config.num_epochs, ctx=ctx)
    final_threshold_metrics = evaluate_loader(
        state=state,
        ctx=ctx,
        data_loader=data.threshold_loader,
        desc="final threshold set",
        tune_threshold=False,
        weights=final_weights,
    )
    final_val_metrics = evaluate_loader(
        state=state,
        ctx=ctx,
        data_loader=data.val_loader,
        desc="final validation",
        tune_threshold=False,
        weights=final_weights,
    )
    final_test_metrics = evaluate_loader(
        state=state,
        ctx=ctx,
        data_loader=data.test_loader,
        desc="final test",
        tune_threshold=False,
        weights=final_weights,
    )

    summarize_eval_metrics(prefix="[final] calib ", metrics=final_threshold_metrics, num_labels=len(ctx.emotion_names))
    summarize_eval_metrics(prefix="[final] val   ", metrics=final_val_metrics, num_labels=len(ctx.emotion_names))
    summarize_eval_metrics(prefix="[final] test  ", metrics=final_test_metrics, num_labels=len(ctx.emotion_names))

    state.history.append(slim_history_row(epoch=ctx.schedule_config.num_epochs, split_name="test", metrics=final_test_metrics))
    save_json(output_dir / "history.json", state.history)

    drop = {"logits", "labels", "probs", "preds", "classification_report_text"}
    final_payload = {
        "threshold_tune": {k: v for k, v in to_serializable(final_threshold_metrics).items() if k not in drop},
        "validation": {k: v for k, v in to_serializable(final_val_metrics).items() if k not in drop},
        "test": {k: v for k, v in to_serializable(final_test_metrics).items() if k not in drop},
    }
    save_json(output_dir / "final_metrics.json", final_payload)
    save_checkpoint(
        path=output_dir / "final_checkpoint.pt",
        state=state,
        ctx=ctx,
        epoch=ctx.schedule_config.num_epochs,
        val_metrics=final_val_metrics,
        test_metrics=final_test_metrics,
    )
    release_memory()
    return {
        "threshold_tune": final_threshold_metrics,
        "validation": final_val_metrics,
        "test": final_test_metrics,
    }


def run_prompt_experiment(runtime: ExperimentRuntime) -> List[Dict[str, Any]]:
    """Evaluate decoder prompting candidates on a small fixed validation subset."""

    ctx = runtime.ctx
    texts = [runtime.data.val_dataset[i]["text"] for i in range(min(ctx.experiment_config.prompt_eval_num_examples, len(runtime.data.val_dataset)))]
    results = evaluate_prompt_candidates(
        state=runtime.state,
        ctx=ctx,
        texts=texts,
        prompt_candidates=ctx.experiment_config.prompt_candidates,
        use_vae_memory=True,
    )
    for row in results:
        print(row)
    return results


def run_final_walkthrough(runtime: ExperimentRuntime, epoch: Optional[int] = None) -> None:
    """Print qualitative copy/edit examples on the fixed monitor set."""

    ctx = runtime.ctx
    run_epoch_monitor(
        state=runtime.state,
        ctx=ctx,
        dataset=runtime.data.val_dataset,
        example_indices=runtime.data.monitor_example_indices,
        edit_targets=runtime.data.monitor_target_labels,
        epoch=ctx.schedule_config.num_epochs if epoch is None else epoch,
    )


def single_example_latent_edit_demo(
    runtime: ExperimentRuntime,
    example_idx: Optional[int] = None,
    desired_labels: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Run one latent edit and return both source/target metadata and generated text."""

    ctx = runtime.ctx
    state = runtime.state
    model = state.model
    data = runtime.data
    example_idx = data.monitor_example_indices[0] if example_idx is None else example_idx
    example = data.val_dataset[example_idx]
    desired_labels = list(desired_labels or data.monitor_target_labels[0])

    desired_target, desired_mask = make_partial_desired_emotion_target(
        label_names=desired_labels,
        emotion_names=ctx.emotion_names,
        device=ctx.device,
    )
    encoded = ctx.tokenizer(
        [example["text"]],
        max_length=ctx.data_config.max_length,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(ctx.device)
    attention_mask = encoded["attention_mask"].to(ctx.device)

    with torch.no_grad():
        base_out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=example["labels"].unsqueeze(0).to(ctx.device),
            sample_posterior=False,
            residual_scale=current_loss_weights(ctx.schedule_config.num_epochs, ctx)["residual_scale"],
        )

    edit_result = optimize_emotion_latents_for_target(
        model=model,
        base_out=base_out,
        attention_mask=attention_mask,
        desired_target=desired_target,
        desired_mask=desired_mask,
        method=ctx.experiment_config.monitor_edit_method,
        num_steps=ctx.experiment_config.monitor_edit_steps,
        lr=0.15,
        force_scale=ctx.experiment_config.edit_force_scale,
        ridge=ctx.experiment_config.edit_ridge,
        backsolve_ridge=ctx.experiment_config.edit_backsolve_ridge,
        target_margin=ctx.experiment_config.edit_target_margin,
        target_mode=ctx.experiment_config.edit_target_mode,
    )
    summary = print_generation_summary(
        state=state,
        ctx=ctx,
        source_text=example["text"],
        gold_names=label_names_from_tensor(example["labels"], ctx.emotion_names),
        target_names=desired_labels,
        encoder_attention_mask=attention_mask,
        base_out=base_out,
        edit_result=edit_result,
    )
    return {
        "example_idx": example_idx,
        "source_text": example["text"],
        "gold_emotions": format_label_names(label_names_from_tensor(example["labels"], ctx.emotion_names)),
        "target_emotions": format_label_names(desired_labels),
        "summary": summary,
    }
