from __future__ import annotations

from ..utils.common import *

# Focused module: continuous-target and SemEval EI-reg metrics.


def safe_pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson r with constant-vector and empty-input guards."""

    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2:
        return 0.0
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
    if denom <= 1e-12:
        return 0.0
    return float(np.sum(x * y) / denom)


def _rankdata_average_ties(x: np.ndarray) -> np.ndarray:
    """Small dependency-free equivalent of scipy.stats.rankdata(method='average')."""

    x = np.asarray(x, dtype=np.float64).reshape(-1)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    sorted_x = x[order]
    start = 0
    n = x.size
    while start < n:
        end = start + 1
        while end < n and sorted_x[end] == sorted_x[start]:
            end += 1
        avg_rank = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def safe_spearmanr(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rho using average ranks and safe Pearson on the ranks."""

    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 2:
        return 0.0
    return safe_pearsonr(_rankdata_average_ties(x[mask]), _rankdata_average_ties(y[mask]))


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination with constant-target guard."""

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if y_true.size == 0:
        return 0.0
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot <= 1e-12:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def continuous_regression_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    label_names: Sequence[str],
    high_threshold: Optional[float] = 0.5,
    prefix: str = "intensity",
) -> Dict[str, Any]:
    """Regression metrics for multi-output scores in [0, 1].

    For SemEval-2018 Task 1 EI-reg the headline score is the macro-average
    Pearson correlation across the four emotions. This function also reports
    per-emotion Pearson/Spearman, MAE/RMSE, R2, and Pearson on the high-intensity
    gold subset, commonly useful for inspecting strong-emotion behavior.
    
    If high_threshold=None, compute the median of targets for each label.
    """

    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if predictions.ndim != 2 or targets.ndim != 2:
        raise ValueError("predictions and targets must be rank-2 arrays shaped [n_examples, n_labels].")
    if predictions.shape != targets.shape:
        raise ValueError(f"predictions shape {predictions.shape} does not match targets shape {targets.shape}.")
    if predictions.shape[1] != len(label_names):
        raise ValueError("label_names length must match the second dimension of predictions/targets.")

    err = predictions - targets
    per_label: Dict[str, Dict[str, float]] = {}
    pearson_values: List[float] = []
    spearman_values: List[float] = []
    high_pearson_values: List[float] = []
    high_counts: Dict[str, int] = {}

    for idx, name in enumerate(label_names):
        pred_j = predictions[:, idx]
        gold_j = targets[:, idx]
        pearson_j = safe_pearsonr(pred_j, gold_j)
        spearman_j = safe_spearmanr(pred_j, gold_j)
        mae_j = float(np.mean(np.abs(pred_j - gold_j)))
        mse_j = float(np.mean((pred_j - gold_j) ** 2))
        rmse_j = float(math.sqrt(mse_j))
        r2_j = safe_r2(gold_j, pred_j)
        
        # Use data-driven threshold if not provided
        threshold_j = high_threshold if high_threshold is not None else float(np.median(gold_j))
        high_mask = gold_j >= threshold_j
        high_counts[str(name)] = int(high_mask.sum())
        high_pearson_j = safe_pearsonr(pred_j[high_mask], gold_j[high_mask]) if int(high_mask.sum()) >= 2 else 0.0
        pearson_values.append(pearson_j)
        spearman_values.append(spearman_j)
        high_pearson_values.append(high_pearson_j)
        per_label[str(name)] = {
            "pearson": pearson_j,
            "spearman": spearman_j,
            "mae": mae_j,
            "mse": mse_j,
            "rmse": rmse_j,
            "r2": r2_j,
            "pearson_high_gold": high_pearson_j,
            "high_gold_count": int(high_mask.sum()),
            "high_gold_threshold": threshold_j,
        }

    return {
        f"{prefix}_mae": float(np.mean(np.abs(err))),
        f"{prefix}_mse": float(np.mean(err ** 2)),
        f"{prefix}_rmse": float(math.sqrt(float(np.mean(err ** 2)))),
        f"{prefix}_r2_micro": safe_r2(targets.reshape(-1), predictions.reshape(-1)),
        f"{prefix}_pearson_micro": safe_pearsonr(predictions.reshape(-1), targets.reshape(-1)),
        f"{prefix}_pearson_macro": float(np.mean(pearson_values)) if pearson_values else 0.0,
        f"{prefix}_spearman_macro": float(np.mean(spearman_values)) if spearman_values else 0.0,
        f"{prefix}_pearson_high_gold_macro": float(np.mean(high_pearson_values)) if high_pearson_values else 0.0,
        f"{prefix}_high_gold_threshold": float(high_threshold) if high_threshold is not None else "data_median",
        f"{prefix}_high_gold_counts": high_counts,
        f"per_label_{prefix}": per_label,
    }


def semeval_ei_reg_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    label_names: Sequence[str],
    high_threshold: Optional[float] = 0.5,
) -> Dict[str, Any]:
    """SemEval-2018 Task 1 EI-reg friendly naming.

    `semeval_ei_reg_pearson_macro` is the official-style bottom-line score:
    average Pearson correlation across emotion-specific EI-reg files.
    
    If high_threshold=None, compute the median of targets for each label.
    """

    base = continuous_regression_metrics(
        predictions=predictions,
        targets=targets,
        label_names=label_names,
        high_threshold=high_threshold,
        prefix="semeval_ei_reg",
    )
    base["semeval_ei_reg_official_score"] = base["semeval_ei_reg_pearson_macro"]
    return base
