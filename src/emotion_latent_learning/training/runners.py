from __future__ import annotations

from ..utils.common import *
from .checkpointing import save_checkpoint
from ..evaluation.metrics import evaluate_loader, summarize_eval_metrics
from ..evaluation.monitoring import run_epoch_monitor
from .runtime import ExperimentRuntime
from .schedules import current_loss_weights, ensure_lora_schedule, slim_history_row
from ..schemas import TrainingState
from ..evaluation.thresholds import restore_threshold
from .step import train_one_epoch
from ..utils import release_memory, save_json

# Focused module: runners.

def run_training(runtime: ExperimentRuntime, monitor: bool = True, save_each_epoch: bool = True) -> TrainingState:
    """Run the complete epoch loop with threshold tuning, validation, monitoring, and checkpointing."""

    state = runtime.state
    ctx = runtime.ctx
    data = runtime.data
    output_dir = runtime.output_dir

    for epoch in range(1, ctx.schedule_config.num_epochs + 1):
        ensure_lora_schedule(epoch=epoch, state=state, ctx=ctx)
        weights = current_loss_weights(epoch=epoch, ctx=ctx)

        print(
            f"\n[epoch {epoch}] weights="
            f"{{'cls': {weights['classification_weight']:.2f}, 'recon': {weights['recon_weight']:.2f}, "
            f"'kl': {weights['kl_weight']:.4f}, 'tc': {weights['tc_weight']:.4f}, 'copy': {weights['copy_weight']:.4f}, "
            f"'vec_adv': {weights['vector_adv_weight']:.4f}, 'res_adv': {weights['residual_adv_weight']:.4f}, "
            f"'sep': {weights['vector_sep_weight']:.4f}, 'orth': {weights['orthogonality_weight']:.4f}, "
            f"'transfer': {weights['transfer_strength_weight']:.4f}, 'res_scale': {weights['residual_scale']:.4f}}} "
            f"lora_enabled={epoch >= ctx.schedule_config.lora_start_epoch} "
            f"tc_mode={ctx.loss_config.tc_mode} threshold_mode={ctx.experiment_config.threshold_mode}"
        )

        train_metrics = train_one_epoch(epoch=epoch, state=state, ctx=ctx, data_loader=data.train_loader, weights=weights)
        state.history.append(slim_history_row(epoch=epoch, split_name="train", metrics=train_metrics))
        summarize_eval_metrics(prefix=f"[epoch {epoch}] train ", metrics=train_metrics, num_labels=len(ctx.emotion_names))

        threshold_metrics = evaluate_loader(
            state=state,
            ctx=ctx,
            data_loader=data.threshold_loader,
            desc=f"epoch {epoch}/{ctx.schedule_config.num_epochs} [threshold]",
            tune_threshold=True,
            weights=weights,
        )
        state.threshold = restore_threshold(
            threshold_metrics["threshold"],
            num_labels=len(ctx.emotion_names),
            mode=ctx.experiment_config.threshold_mode,
        )
        summarize_eval_metrics(prefix=f"[epoch {epoch}] calib ", metrics=threshold_metrics, num_labels=len(ctx.emotion_names))
        state.history.append(slim_history_row(epoch=epoch, split_name="threshold_tune", metrics=threshold_metrics))

        val_metrics = evaluate_loader(
            state=state,
            ctx=ctx,
            data_loader=data.val_loader,
            desc=f"epoch {epoch}/{ctx.schedule_config.num_epochs} [val]",
            tune_threshold=False,
            weights=weights,
        )
        summarize_eval_metrics(prefix=f"[epoch {epoch}] val   ", metrics=val_metrics, num_labels=len(ctx.emotion_names))
        state.history.append(slim_history_row(epoch=epoch, split_name="validation", metrics=val_metrics))
        save_json(output_dir / "history.json", state.history)

        release_memory()
        if monitor:
            run_epoch_monitor(
                state=state,
                ctx=ctx,
                dataset=data.val_dataset,
                example_indices=data.monitor_example_indices,
                edit_targets=data.monitor_target_labels,
                epoch=epoch,
            )

        if float(val_metrics["micro_f1"]) > state.best_val_micro_f1:
            state.best_val_micro_f1 = float(val_metrics["micro_f1"])
            save_checkpoint(
                path=output_dir / "best_checkpoint.pt",
                state=state,
                ctx=ctx,
                epoch=epoch,
                val_metrics=val_metrics,
            )
            print(f"saved new best checkpoint at epoch {epoch} with val micro-F1={state.best_val_micro_f1:.4f}")

        if save_each_epoch:
            save_checkpoint(
                path=output_dir / f"checkpoint_epoch_{epoch:03d}.pt",
                state=state,
                ctx=ctx,
                epoch=epoch,
                val_metrics=val_metrics,
            )

    release_memory()
    return state
