from pathlib import Path
import json
import textwrap
from importlib.metadata import version, PackageNotFoundError

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _get_log_history(source):
    """
    source can be:
      - a Trainer instance
      - a checkpoint directory containing trainer_state.json
    """
    if hasattr(source, "state") and hasattr(source.state, "log_history"):
        return source.state.log_history

    source = Path(source)
    trainer_state_path = source / "trainer_state.json"
    if not trainer_state_path.exists():
        raise FileNotFoundError(f"Could not find: {trainer_state_path}")

    with open(trainer_state_path, "r") as f:
        state = json.load(f)

    return state["log_history"]


def _find_run_config_path(source):
    candidates = []

    if hasattr(source, "args"):
        candidates.append(Path(source.args.output_dir) / "run_config.json")

        best_ckpt = getattr(source.state, "best_model_checkpoint", None)
        if best_ckpt:
            best_ckpt = Path(best_ckpt)
            candidates.append(best_ckpt / "run_config.json")
            candidates.append(best_ckpt.parent / "run_config.json")
    else:
        source = Path(source)
        candidates.extend(
            [
                source / "run_config.json",
                source.parent / "run_config.json",
                source.parent.parent / "run_config.json",
            ]
        )

    for p in candidates:
        if p.exists():
            return p
    return None


def _load_run_config(source):
    run_config_path = _find_run_config_path(source)
    if run_config_path is None:
        return {}
    with open(run_config_path, "r") as f:
        return json.load(f)


def _make_history_df(log_history):
    rows = []
    for entry in log_history:
        row = {"step": entry.get("step"), "epoch": entry.get("epoch")}
        row.update(entry)
        rows.append(row)

    history_df = pd.DataFrame(rows)
    if history_df.empty:
        raise ValueError("No log history found.")

    if "step" not in history_df.columns:
        raise ValueError("No 'step' column found in log history.")

    history_df = history_df.sort_values(by=["step"], na_position="last").reset_index(drop=True)
    return history_df


def _series_or_empty(df, col):
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype=float)


def _ema(series, alpha=0.2):
    return pd.to_numeric(series, errors="coerce").ewm(alpha=alpha, adjust=False).mean()


def _epoch_markers(train_df):
    if train_df.empty or "epoch" not in train_df.columns:
        return []

    tmp = train_df[["step", "epoch"]].dropna().sort_values("epoch")
    if tmp.empty:
        return []

    max_epoch = int(np.floor(tmp["epoch"].max()))
    if max_epoch < 1:
        return []

    epoch_values = tmp["epoch"].to_numpy()
    step_values = tmp["step"].to_numpy()

    # requires monotonic increasing epoch
    if len(epoch_values) < 2 or not np.all(np.diff(epoch_values) >= 0):
        return []

    markers = []
    for e in range(1, max_epoch + 1):
        step_at_epoch = np.interp(e, epoch_values, step_values)
        markers.append((e, step_at_epoch))
    return markers


def _best_eval_row(eval_df, metric_for_best_model=None, greater_is_better=True):
    if eval_df.empty:
        return None

    metric = metric_for_best_model
    if metric and metric in eval_df.columns and eval_df[metric].notna().any():
        s = pd.to_numeric(eval_df[metric], errors="coerce")
        idx = s.idxmax() if greater_is_better else s.idxmin()
        return eval_df.loc[idx]

    if "eval_loss" in eval_df.columns and eval_df["eval_loss"].notna().any():
        s = pd.to_numeric(eval_df["eval_loss"], errors="coerce")
        idx = s.idxmin()
        return eval_df.loc[idx]

    return None


def _replication_text(run_config: dict):
    lines = []

    left = [
        f"model={run_config.get('model_name') or run_config.get('base_model_name_or_path')}",
        f"labels={run_config.get('n_labels')}",
        f"train={run_config.get('train_size')}  val={run_config.get('eval_size')}",
        f"max_len={run_config.get('max_length')}  batch={run_config.get('batch_size')}",
        f"epochs={run_config.get('epochs')}  lr={run_config.get('learning_rate')}",
        f"scheduler={run_config.get('lr_scheduler_type')}  warmup={run_config.get('warmup_steps')}",
        f"wd={run_config.get('weight_decay')}  seed={run_config.get('seed')}",
        f"4bit={run_config.get('use_4bit')}  fp16={run_config.get('fp16')}  bf16={run_config.get('bf16')}",
        f"optim={run_config.get('optim')}  grad_ckpt={run_config.get('gradient_checkpointing')}",
        f"eval_steps={run_config.get('eval_steps')}  save_steps={run_config.get('save_steps')}  log_steps={run_config.get('logging_steps')}",
    ]
    lines.extend([x for x in left if x and "None" not in x])

    peft = run_config.get("peft") or {}
    if peft:
        lines.append(
            f"LoRA: r={peft.get('r')} alpha={peft.get('lora_alpha')} dropout={peft.get('lora_dropout')} target={peft.get('target_modules')}"
        )

    env = [
        f"torch={run_config.get('torch_version')} transformers={run_config.get('transformers_version')} peft={run_config.get('peft_version')}",
        f"device={run_config.get('cuda_device_name')}",
    ]
    lines.extend([x for x in env if x and "None" not in x])

    return "\n".join(lines)


def _plot_footer(fig, run_text: str, best_text: str | None = None):
    fig.subplots_adjust(bottom=0.31)

    fig.text(
        0.01,
        0.015,
        run_text,
        ha="left",
        va="bottom",
        fontsize=8,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.9, edgecolor="0.8"),
    )

    if best_text:
        fig.text(
            0.99,
            0.015,
            best_text,
            ha="right",
            va="bottom",
            fontsize=8,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.9, edgecolor="0.8"),
        )


def _add_epoch_lines(ax, markers):
    for epoch_num, step_at_epoch in markers:
        ax.axvline(step_at_epoch, linestyle="--", alpha=0.18)
        ax.text(step_at_epoch, 1.01, f"e{epoch_num}", transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=10, alpha=0.8)


def _save_plot(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def save_training_metric_plots(
    source,
    output_dir: str = "../output/training_plots",
    history_csv_path: str = "../output/training_history_full.csv",
    run_summary_path: str = "../output/run_summary.json",
    ema_alpha: float = 0.2,
):
    """
    Saves:
      - full training history CSV
      - run summary JSON
      - loss_raw_and_smoothed.png
      - loss_gap.png
      - f1.png
      - precision.png
      - recall.png
      - accuracy.png
      - learning_rate.png (if logged)
      - grad_norm.png (if logged)
      - macro_f1_by_epoch.png (if epoch available)
    """
    output_dir = Path(output_dir)
    history_csv_path = Path(history_csv_path)
    run_summary_path = Path(run_summary_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    history_csv_path.parent.mkdir(parents=True, exist_ok=True)
    run_summary_path.parent.mkdir(parents=True, exist_ok=True)

    log_history = _get_log_history(source)
    history_df = _make_history_df(log_history)
    history_df.to_csv(history_csv_path, index=False)
    print(f"Saved full training history CSV to: {history_csv_path}")

    run_config = _load_run_config(source)

    train_df = history_df[
        history_df.get("loss", pd.Series(np.nan, index=history_df.index)).notna()
        & history_df.get("eval_loss", pd.Series(np.nan, index=history_df.index)).isna()
    ].copy()

    eval_df = history_df[
        history_df.get("eval_loss", pd.Series(np.nan, index=history_df.index)).notna()
    ].copy()

    if not train_df.empty and "loss" in train_df.columns:
        train_df["loss_ema"] = _ema(train_df["loss"], alpha=ema_alpha)

    if not eval_df.empty and "eval_loss" in eval_df.columns:
        eval_df["eval_loss_ema"] = _ema(eval_df["eval_loss"], alpha=ema_alpha)

    epoch_lines = _epoch_markers(train_df)

    metric_for_best_model = run_config.get("metric_for_best_model", "eval_macro_f1")
    greater_is_better = bool(run_config.get("greater_is_better", True))
    best_row = _best_eval_row(eval_df, metric_for_best_model, greater_is_better)

    best_text = None
    if best_row is not None:
        best_parts = [
            f"best_step={int(best_row['step'])}" if pd.notna(best_row.get("step")) else None,
            f"best_epoch={best_row['epoch']:.4f}" if pd.notna(best_row.get("epoch")) else None,
        ]
        if metric_for_best_model in best_row and pd.notna(best_row.get(metric_for_best_model)):
            best_parts.append(f"{metric_for_best_model}={best_row[metric_for_best_model]:.4f}")
        if "eval_loss" in best_row and pd.notna(best_row.get("eval_loss")):
            best_parts.append(f"eval_loss={best_row['eval_loss']:.4f}")
        if "eval_accuracy" in best_row and pd.notna(best_row.get("eval_accuracy")):
            best_parts.append(f"eval_accuracy={best_row['eval_accuracy']:.4f}")
        best_text = "\n".join([x for x in best_parts if x])

    run_text = _replication_text(run_config)

    # 1) Raw + smoothed loss
    fig, ax = plt.subplots(figsize=(11, 6.5))
    if not train_df.empty:
        ax.plot(train_df["step"], train_df["loss"], alpha=0.35, label="train_loss_raw")
        ax.plot(train_df["step"], train_df["loss_ema"], linewidth=2, label="train_loss_ema")
    if not eval_df.empty:
        ax.plot(eval_df["step"], eval_df["eval_loss"], marker="o", label="eval_loss_raw")
        ax.plot(eval_df["step"], eval_df["eval_loss_ema"], linewidth=2, label="eval_loss_ema")

    if best_row is not None and pd.notna(best_row.get("step")):
        ax.axvline(best_row["step"], linestyle="--", alpha=0.5, label="best_step")

    _add_epoch_lines(ax, epoch_lines)
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Training and Validation Loss", fontsize=16, pad=30)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=13)
    _plot_footer(fig, run_text, best_text)
    _save_plot(fig, output_dir / "loss_raw_and_smoothed.png")

    # 2) Generalization gap: eval_loss - interpolated_train_loss
    if not train_df.empty and not eval_df.empty:
        x_train = pd.to_numeric(train_df["step"], errors="coerce").to_numpy()
        y_train = pd.to_numeric(train_df["loss_ema"], errors="coerce").to_numpy()
        x_eval = pd.to_numeric(eval_df["step"], errors="coerce").to_numpy()
        y_eval = pd.to_numeric(eval_df["eval_loss"], errors="coerce").to_numpy()

        valid_train = np.isfinite(x_train) & np.isfinite(y_train)
        valid_eval = np.isfinite(x_eval) & np.isfinite(y_eval)

        if valid_train.sum() >= 2 and valid_eval.sum() >= 1:
            interp_train = np.interp(x_eval[valid_eval], x_train[valid_train], y_train[valid_train])
            gap = y_eval[valid_eval] - interp_train

            fig, ax = plt.subplots(figsize=(11, 6.5))
            ax.plot(x_eval[valid_eval], gap, marker="o", label="eval_loss - train_loss_ema")
            ax.axhline(0.0, linestyle="--", alpha=0.5)
            if best_row is not None and pd.notna(best_row.get("step")):
                ax.axvline(best_row["step"], linestyle="--", alpha=0.5, label="best_step")
            _add_epoch_lines(ax, epoch_lines)
            ax.set_xlabel("Step", fontsize=12)
            ax.set_ylabel("Gap", fontsize=12)
            ax.set_title("Generalization Gap", fontsize=16, pad=30)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=13)
            _plot_footer(fig, run_text, best_text)
            _save_plot(fig, output_dir / "loss_gap.png")

    # 3) F1
    f1_cols = ["eval_macro_f1", "eval_micro_f1", "eval_weighted_f1"]
    available_f1 = [c for c in f1_cols if c in eval_df.columns and eval_df[c].notna().any()]
    if available_f1 and not eval_df.empty:
        fig, ax = plt.subplots(figsize=(11, 6.5))
        for col in available_f1:
            ax.plot(eval_df["step"], eval_df[col], marker="o", label=col.replace("eval_", ""))
        if best_row is not None and pd.notna(best_row.get("step")):
            ax.axvline(best_row["step"], linestyle="--", alpha=0.5, label="best_step")
        _add_epoch_lines(ax, epoch_lines)
        ax.set_xlabel("Step", fontsize=12)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_title("Validation F1", fontsize=16, pad=30)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=13)
        _plot_footer(fig, run_text, best_text)
        _save_plot(fig, output_dir / "f1.png")

    # 4) Precision
    precision_cols = ["eval_macro_precision", "eval_micro_precision", "eval_weighted_precision"]
    available_precision = [c for c in precision_cols if c in eval_df.columns and eval_df[c].notna().any()]
    if available_precision and not eval_df.empty:
        fig, ax = plt.subplots(figsize=(11, 6.5))
        for col in available_precision:
            ax.plot(eval_df["step"], eval_df[col], marker="o", label=col.replace("eval_", ""))
        if best_row is not None and pd.notna(best_row.get("step")):
            ax.axvline(best_row["step"], linestyle="--", alpha=0.5, label="best_step")
        _add_epoch_lines(ax, epoch_lines)
        ax.set_xlabel("Step", fontsize=12)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_title("Validation Precision", fontsize=16, pad=30)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=13)
        _plot_footer(fig, run_text, best_text)
        _save_plot(fig, output_dir / "precision.png")

    # 5) Recall
    recall_cols = ["eval_macro_recall", "eval_micro_recall", "eval_weighted_recall"]
    available_recall = [c for c in recall_cols if c in eval_df.columns and eval_df[c].notna().any()]
    if available_recall and not eval_df.empty:
        fig, ax = plt.subplots(figsize=(11, 6.5))
        for col in available_recall:
            ax.plot(eval_df["step"], eval_df[col], marker="o", label=col.replace("eval_", ""))
        if best_row is not None and pd.notna(best_row.get("step")):
            ax.axvline(best_row["step"], linestyle="--", alpha=0.5, label="best_step")
        _add_epoch_lines(ax, epoch_lines)
        ax.set_xlabel("Step", fontsize=12)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_title("Validation Recall", fontsize=16, pad=30)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=13)
        _plot_footer(fig, run_text, best_text)
        _save_plot(fig, output_dir / "recall.png")

    # 6) Accuracy
    if "eval_accuracy" in eval_df.columns and eval_df["eval_accuracy"].notna().any():
        fig, ax = plt.subplots(figsize=(11, 6.5))
        ax.plot(eval_df["step"], eval_df["eval_accuracy"], marker="o", label="accuracy")
        if best_row is not None and pd.notna(best_row.get("step")):
            ax.axvline(best_row["step"], linestyle="--", alpha=0.5, label="best_step")
        _add_epoch_lines(ax, epoch_lines)
        ax.set_xlabel("Step", fontsize=12)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_title("Validation Accuracy", fontsize=16, pad=30)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=13)
        _plot_footer(fig, run_text, best_text)
        _save_plot(fig, output_dir / "accuracy.png")

    # 7) Learning rate
    if "learning_rate" in history_df.columns and history_df["learning_rate"].notna().any():
        lr_df = history_df[history_df["learning_rate"].notna()].copy()
        fig, ax = plt.subplots(figsize=(11, 6.5))
        ax.plot(lr_df["step"], lr_df["learning_rate"], label="learning_rate")
        if best_row is not None and pd.notna(best_row.get("step")):
            ax.axvline(best_row["step"], linestyle="--", alpha=0.5, label="best_step")
        _add_epoch_lines(ax, epoch_lines)
        ax.set_xlabel("Step", fontsize=12)
        ax.set_ylabel("LR", fontsize=12)
        ax.set_title("Learning Rate Schedule", fontsize=16, pad=30)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=13)
        _plot_footer(fig, run_text, best_text)
        _save_plot(fig, output_dir / "learning_rate.png")

    # 8) Grad norm
    if "grad_norm" in history_df.columns and history_df["grad_norm"].notna().any():
        gn_df = history_df[history_df["grad_norm"].notna()].copy()
        fig, ax = plt.subplots(figsize=(11, 6.5))
        ax.plot(gn_df["step"], gn_df["grad_norm"], label="grad_norm")
        if best_row is not None and pd.notna(best_row.get("step")):
            ax.axvline(best_row["step"], linestyle="--", alpha=0.5, label="best_step")
        _add_epoch_lines(ax, epoch_lines)
        ax.set_xlabel("Step", fontsize=12)
        ax.set_ylabel("Grad norm", fontsize=12)
        ax.set_title("Gradient Norm", fontsize=16, pad=30)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=13)
        _plot_footer(fig, run_text, best_text)
        _save_plot(fig, output_dir / "grad_norm.png")

    # 9) Macro F1 by epoch
    if "eval_macro_f1" in eval_df.columns and "epoch" in eval_df.columns and eval_df["epoch"].notna().any():
        tmp = eval_df[eval_df["eval_macro_f1"].notna() & eval_df["epoch"].notna()].copy()
        if not tmp.empty:
            fig, ax = plt.subplots(figsize=(11, 6.5))
            ax.plot(tmp["epoch"], tmp["eval_macro_f1"], marker="o", label="eval_macro_f1")
            ax.set_xlabel("Epoch", fontsize=12)
            ax.set_ylabel("Macro F1", fontsize=12)
            ax.set_title("Validation Macro F1 by Epoch", fontsize=16, pad=30)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=13)
            _plot_footer(fig, run_text, best_text)
            _save_plot(fig, output_dir / "macro_f1_by_epoch.png")

    final_row = history_df.iloc[-1].to_dict()
    summary = {
        "run_config": run_config,
        "best_eval_row": best_row.to_dict() if best_row is not None else None,
        "final_history_row": final_row,
        "history_csv_path": str(history_csv_path),
        "plots_dir": str(output_dir),
    }

    with open(run_summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"Saved run summary JSON to: {run_summary_path}")
    return history_df


def plot_emotion_heatmap(
    results: list[dict],
    n: int = 50,
    out_path: str | Path = "emotion_scores_heatmap.png",
    wrap_width: int = 24,
    cmap: str = "viridis_r",
    dpi: int = 220,
    show: bool = True,
):
    if not results:
        raise ValueError("`results` is empty.")

    subset = results[:n]
    if not subset:
        raise ValueError("No rows available after applying `n`.")

    # Preserve emotion order from the first result
    score_cols = list(subset[0]["scores"].keys())

    rows = []
    for item in subset:
        if "text" not in item or "scores" not in item:
            raise ValueError("Each item must contain 'text' and 'scores' keys.")
        rows.append(
            {
                "text": str(item["text"]),
                **{col: float(item["scores"][col]) for col in score_cols},
            }
        )

    df = pd.DataFrame(rows)
    heat = df[score_cols].astype(float)

    row_labels = [textwrap.fill(t, width=wrap_width) for t in df["text"]]

    n_rows, n_cols = heat.shape
    fig_w = max(18, n_cols * 0.8)
    fig_h = max(14, n_rows * 0.6)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Low values lighter, high values darker
    im = ax.imshow(heat.values, aspect="auto", cmap=cmap)

    # Column labels on top
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(score_cols, rotation=90, fontsize=16)
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    # Wrapped full phrases on rows
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=16)

    ax.set_xlabel("Emotion scores", fontsize=18)
    ax.set_ylabel("Input text", fontsize=18)
    ax.set_title(f"Emotion Score Heatmap (n={len(subset)})", fontsize=22, pad=28)

    # Narrow colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.ax.tick_params(labelsize=14)
    plt.tight_layout()

    out_path = str(out_path)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return out_path