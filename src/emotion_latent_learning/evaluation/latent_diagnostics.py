from __future__ import annotations

from ..utils.common import *
from .regression import safe_pearsonr, safe_spearmanr

try:
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
except Exception:  # pragma: no cover - optional dependency fallback
    mutual_info_classif = None
    mutual_info_regression = None

# Focused module: FactorVAE factor-power and emotion/meaning split diagnostics.
#
# Important distinction:
# - factor_power: legacy scalar-dimension diagnostics. Useful for seeing whether
#   individual scalar dimensions align with labels, but this is NOT the primary
#   disentanglement objective for this model.
# - emotion_meaning_split: branch-level diagnostics. These ask whether emotion
#   labels live in the scalar/emotion branch and are absent from the vector/meaning
#   branch. Probes are fit on one held-out slice and reported on another slice so
#   the scores are not training-on-the-test-split artifacts.


def _as_2d_float(x: Any) -> np.ndarray:
    if x is None:
        return np.zeros((0, 0), dtype=np.float64)
    if isinstance(x, torch.Tensor):
        arr = x.detach().cpu().float().numpy()
    else:
        arr = np.asarray(x)
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        arr = arr.reshape(arr.shape[0], -1)
    return arr


def _standardize(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    mean = np.nanmean(x, axis=0, keepdims=True)
    std = np.nanstd(x, axis=0, keepdims=True)
    std = np.where(std <= 1e-8, 1.0, std)
    z = (x - mean) / std
    z = np.where(np.isfinite(z), z, 0.0)
    return z, mean.reshape(-1), std.reshape(-1)


def _safe_mean(values: Sequence[float]) -> float:
    vals = np.asarray(list(values), dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    return float(vals.mean()) if vals.size else 0.0


def _normalized_entropy(p: np.ndarray, axis: int) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    p = np.maximum(p, 0.0)
    denom = p.sum(axis=axis, keepdims=True)
    q = np.divide(p, np.maximum(denom, 1e-12))
    log_base = math.log(max(p.shape[axis], 2))
    ent = -np.sum(np.where(q > 0, q * np.log(np.maximum(q, 1e-12)), 0.0), axis=axis) / log_base
    ent = np.where(np.squeeze(denom, axis=axis) > 0, ent, 1.0)
    return ent


def _probe_split_indices(
    n: int,
    eval_fraction: float,
    min_train: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    idx = np.arange(int(n))
    if n <= 2:
        return idx, idx, False
    eval_fraction = float(np.clip(eval_fraction, 0.05, 0.50))
    eval_n = max(1, int(round(n * eval_fraction)))
    train_n = n - eval_n
    if train_n < int(min_train) and n >= int(min_train) + 1:
        train_n = int(min_train)
        eval_n = n - train_n
    if eval_n < 1 or train_n < 2:
        return idx, idx, False
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(idx)
    train_idx = np.sort(perm[:train_n])
    eval_idx = np.sort(perm[train_n:])
    return train_idx, eval_idx, True


def _ridge_fit(
    x_train: np.ndarray,
    y_train: np.ndarray,
    alpha: float = 1.0,
) -> Dict[str, np.ndarray]:
    x_train = _as_2d_float(x_train)
    y_train = _as_2d_float(y_train)
    n = min(x_train.shape[0], y_train.shape[0])
    x_train = x_train[:n]
    y_train = y_train[:n]
    if n == 0 or x_train.shape[1] == 0 or y_train.shape[1] == 0:
        return {
            "coef": np.zeros((x_train.shape[1], y_train.shape[1]), dtype=np.float64),
            "x_mean": np.zeros((x_train.shape[1],), dtype=np.float64),
            "x_std": np.ones((x_train.shape[1],), dtype=np.float64),
            "y_mean": np.zeros((y_train.shape[1],), dtype=np.float64),
            "y_std": np.ones((y_train.shape[1],), dtype=np.float64),
        }
    xz, x_mean, x_std = _standardize(x_train)
    yz, y_mean, y_std = _standardize(y_train)
    reg = float(alpha) * np.eye(xz.shape[1], dtype=np.float64)
    xtx = xz.T @ xz
    xty = xz.T @ yz
    try:
        coef = np.linalg.solve(xtx + reg, xty)
    except np.linalg.LinAlgError:
        coef = np.linalg.pinv(xtx + reg) @ xty
    return {"coef": coef, "x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std}


def _ridge_predict(x: np.ndarray, fit: Dict[str, np.ndarray]) -> np.ndarray:
    x = _as_2d_float(x)
    if x.shape[1] == 0 or fit["coef"].size == 0:
        return np.zeros((x.shape[0], fit["coef"].shape[1] if fit["coef"].ndim == 2 else 0), dtype=np.float64)
    xz = (x - fit["x_mean"].reshape(1, -1)) / np.maximum(fit["x_std"].reshape(1, -1), 1e-8)
    xz = np.where(np.isfinite(xz), xz, 0.0)
    pred_z = xz @ fit["coef"]
    return pred_z * fit["y_std"].reshape(1, -1) + fit["y_mean"].reshape(1, -1)


def _ridge_fit_predict(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Closed-form ridge regression on standardized X/Y.

    Returns predictions in the original target scale, coefficients in standardized
    space, and non-negative absolute-coefficient importances shaped [n_features, n_targets].
    This helper intentionally remains in-sample for legacy callers; branch-level
    diagnostics use `_predictive_metrics_from_features` with explicit train/eval indices.
    """

    x = _as_2d_float(x)
    y = _as_2d_float(y)
    n = min(x.shape[0], y.shape[0])
    x = x[:n]
    y = y[:n]
    fit = _ridge_fit(x, y, alpha=alpha)
    pred = _ridge_predict(x, fit)
    coef = fit["coef"]
    importance = np.abs(coef)
    return pred, coef, importance


def _predictive_metrics_from_features(
    x: np.ndarray,
    y: np.ndarray,
    label_names: Sequence[str],
    alpha: float,
    train_indices: Optional[np.ndarray] = None,
    eval_indices: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    x = _as_2d_float(x)
    y = _as_2d_float(y)
    n = min(x.shape[0], y.shape[0])
    x = x[:n]
    y = y[:n]
    if train_indices is None or eval_indices is None:
        train_indices = np.arange(n)
        eval_indices = np.arange(n)
        heldout = False
    else:
        train_indices = np.asarray(train_indices, dtype=np.int64)
        eval_indices = np.asarray(eval_indices, dtype=np.int64)
        train_indices = train_indices[(train_indices >= 0) & (train_indices < n)]
        eval_indices = eval_indices[(eval_indices >= 0) & (eval_indices < n)]
        heldout = not np.array_equal(train_indices, eval_indices)
        if train_indices.size == 0 or eval_indices.size == 0:
            train_indices = np.arange(n)
            eval_indices = np.arange(n)
            heldout = False

    fit = _ridge_fit(x[train_indices], y[train_indices], alpha=alpha)
    pred = _ridge_predict(x[eval_indices], fit)
    y_eval = y[eval_indices]
    importance = np.abs(fit["coef"])
    coef = fit["coef"]

    per_label: Dict[str, Dict[str, float]] = {}
    r2_values: List[float] = []
    pearson_values: List[float] = []
    mae_values: List[float] = []
    for idx, name in enumerate(label_names):
        if idx >= y_eval.shape[1] or idx >= pred.shape[1]:
            continue
        gold = y_eval[:, idx]
        pred_j = pred[:, idx]
        ss_res = float(np.sum((gold - pred_j) ** 2))
        ss_tot = float(np.sum((gold - gold.mean()) ** 2))
        r2 = 0.0 if ss_tot <= 1e-12 else float(1.0 - ss_res / ss_tot)
        pearson = safe_pearsonr(pred_j, gold)
        spearman = safe_spearmanr(pred_j, gold)
        mae = float(np.mean(np.abs(pred_j - gold))) if gold.size else 0.0
        r2_values.append(r2)
        pearson_values.append(pearson)
        mae_values.append(mae)
        per_label[str(name)] = {
            "r2": r2,
            "pearson": pearson,
            "spearman": spearman,
            "mae": mae,
        }
    return {
        "ridge_r2_macro": _safe_mean(r2_values),
        "ridge_pearson_macro": _safe_mean(pearson_values),
        "ridge_mae_macro": _safe_mean(mae_values),
        "ridge_importance_total": float(np.sum(importance)),
        "ridge_importance_matrix": importance.tolist(),
        "ridge_coefficients": coef.tolist(),
        "per_label": per_label,
        "probe_train_examples": int(train_indices.size),
        "probe_eval_examples": int(eval_indices.size),
        "probe_is_heldout": bool(heldout),
    }


def _correlation_matrix(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = _as_2d_float(x)
    y = _as_2d_float(y)
    n = min(x.shape[0], y.shape[0])
    x = x[:n]
    y = y[:n]
    if x.size == 0 or y.size == 0:
        return np.zeros((x.shape[1], y.shape[1]), dtype=np.float64)
    xz, _, _ = _standardize(x)
    yz, _, _ = _standardize(y)
    return (xz.T @ yz) / float(max(n - 1, 1))


def _dci_from_importance(importance: np.ndarray) -> Dict[str, Any]:
    """DCI-style scores using an importance matrix [feature_or_group, target]."""

    importance = np.asarray(importance, dtype=np.float64)
    if importance.ndim != 2 or importance.size == 0:
        return {
            "dci_disentanglement": 0.0,
            "dci_completeness": 0.0,
            "factor_importance": [],
            "label_importance": [],
        }
    importance = np.maximum(importance, 0.0)
    total = float(importance.sum())
    factor_total = importance.sum(axis=1)
    label_total = importance.sum(axis=0)
    disent_per_factor = 1.0 - _normalized_entropy(importance, axis=1)
    comp_per_label = 1.0 - _normalized_entropy(importance, axis=0)
    if total <= 1e-12:
        disent = 0.0
        completeness = 0.0
    else:
        disent = float(np.sum((factor_total / total) * disent_per_factor))
        completeness = float(np.sum((label_total / total) * comp_per_label))
    return {
        "dci_disentanglement": disent,
        "dci_completeness": completeness,
        "factor_importance": factor_total.astype(float).tolist(),
        "label_importance": label_total.astype(float).tolist(),
        "per_factor_disentanglement": disent_per_factor.astype(float).tolist(),
        "per_label_completeness": comp_per_label.astype(float).tolist(),
    }


def _is_discrete_target(y: np.ndarray) -> bool:
    y = np.asarray(y)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return True
    unique = np.unique(y)
    return bool(unique.size <= 20 and np.allclose(unique, np.round(unique), atol=1e-6))


def _target_entropy_nats(y: np.ndarray, bins: int = 10) -> float:
    y = np.asarray(y, dtype=np.float64)
    y = y[np.isfinite(y)]
    if y.size == 0 or np.nanstd(y) <= 1e-12:
        return 0.0
    if _is_discrete_target(y):
        _, counts = np.unique(y.astype(np.int64), return_counts=True)
    else:
        quantiles = np.linspace(0.0, 1.0, int(bins) + 1)
        edges = np.unique(np.quantile(y, quantiles))
        if edges.size <= 2:
            counts = np.asarray([y.size], dtype=np.int64)
        else:
            counts, _ = np.histogram(y, bins=edges)
    probs = counts.astype(np.float64) / max(float(np.sum(counts)), 1.0)
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(np.maximum(probs, 1e-12))))


def _mi_matrix(features: np.ndarray, labels: np.ndarray, seed: int = 42) -> np.ndarray:
    x = _as_2d_float(features)
    y = _as_2d_float(labels)
    n = min(x.shape[0], y.shape[0])
    x = x[:n]
    y = y[:n]
    if n < 4 or x.shape[1] == 0 or y.shape[1] == 0:
        return np.zeros((x.shape[1], y.shape[1]), dtype=np.float64)
    xz, _, _ = _standardize(x)
    out = np.zeros((xz.shape[1], y.shape[1]), dtype=np.float64)
    for target_idx in range(y.shape[1]):
        target = y[:, target_idx]
        try:
            if _is_discrete_target(target):
                if mutual_info_classif is None:
                    raise RuntimeError("mutual_info_classif unavailable")
                mi = mutual_info_classif(
                    xz,
                    target.astype(np.int64),
                    discrete_features=False,
                    random_state=int(seed),
                )
            else:
                if mutual_info_regression is None:
                    raise RuntimeError("mutual_info_regression unavailable")
                mi = mutual_info_regression(
                    xz,
                    target.astype(np.float64),
                    discrete_features=False,
                    random_state=int(seed),
                )
            out[:, target_idx] = np.maximum(np.asarray(mi, dtype=np.float64), 0.0)
        except Exception:
            # Conservative fallback: squared correlation is not MI, but it keeps the
            # diagnostic available instead of crashing when sklearn MI fails.
            corr = _correlation_matrix(xz, target.reshape(-1, 1)).reshape(-1)
            out[:, target_idx] = np.maximum(np.nan_to_num(corr, nan=0.0) ** 2, 0.0)
    return out


def _group_mig_from_mi(
    scalar_mi: np.ndarray,
    vector_mi: np.ndarray,
    labels: np.ndarray,
    label_names: Sequence[str],
    residual_mi: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    scalar_best = scalar_mi.max(axis=0) if scalar_mi.size else np.zeros((len(label_names),), dtype=np.float64)
    vector_best = vector_mi.max(axis=0) if vector_mi.size else np.zeros_like(scalar_best)
    if residual_mi is not None and residual_mi.size:
        residual_best = residual_mi.max(axis=0)
    else:
        residual_best = np.zeros_like(scalar_best)
    other_best = np.maximum(vector_best, residual_best)

    labels = _as_2d_float(labels)
    per_label: Dict[str, Dict[str, float]] = {}
    signed_gaps = []
    standard_gaps = []
    scalar_wins = []
    for idx, name in enumerate(label_names):
        h = _target_entropy_nats(labels[:, idx]) if idx < labels.shape[1] else 0.0
        denom = max(h, scalar_best[idx], vector_best[idx], residual_best[idx], 1e-12)
        signed_gap = float((scalar_best[idx] - other_best[idx]) / denom)
        sorted_group_mi = np.sort(np.asarray([scalar_best[idx], vector_best[idx], residual_best[idx]], dtype=np.float64))[::-1]
        standard_gap = float((sorted_group_mi[0] - sorted_group_mi[1]) / denom)
        winner = "emotion_scalar" if scalar_best[idx] >= other_best[idx] else ("meaning_vector" if vector_best[idx] >= residual_best[idx] else "residual")
        per_label[str(name)] = {
            "emotion_scalar_max_mi": float(scalar_best[idx]),
            "meaning_vector_max_mi": float(vector_best[idx]),
            "residual_max_mi": float(residual_best[idx]),
            "target_entropy_nats": float(h),
            "emotion_vs_meaning_mig_signed": signed_gap,
            "standard_group_mig": standard_gap,
            "winning_group": winner,
        }
        signed_gaps.append(signed_gap)
        standard_gaps.append(standard_gap)
        scalar_wins.append(1.0 if winner == "emotion_scalar" else 0.0)

    return {
        "definition": "Branch-level MIG: max MI in emotion-scalar branch minus max MI in meaning-vector/residual branch, normalized by target entropy. It does not require dimensions inside a branch to be individually disentangled.",
        "emotion_vs_meaning_mig_signed_macro": _safe_mean(signed_gaps),
        "emotion_vs_meaning_mig_positive_macro": _safe_mean([max(0.0, v) for v in signed_gaps]),
        "standard_group_mig_macro": _safe_mean(standard_gaps),
        "emotion_scalar_win_rate": _safe_mean(scalar_wins),
        "per_label": per_label,
    }


def _aggregate_group_importance(
    importance: np.ndarray,
    group_slices: Sequence[Tuple[str, slice]],
    normalize_by_group_dim: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    importance = np.asarray(importance, dtype=np.float64)
    if importance.ndim != 2 or importance.size == 0:
        return np.zeros((len(group_slices), 0), dtype=np.float64), {}
    rows = []
    details: Dict[str, Any] = {}
    for name, slc in group_slices:
        block = importance[slc, :]
        if block.size == 0:
            row = np.zeros((importance.shape[1],), dtype=np.float64)
        else:
            row = np.sqrt(np.sum(block ** 2, axis=0))
            if normalize_by_group_dim:
                row = row / math.sqrt(max(block.shape[0], 1))
        rows.append(row)
        details[name] = {
            "num_dimensions": int(block.shape[0]) if block.ndim == 2 else 0,
            "importance_by_label": row.astype(float).tolist(),
            "importance_total": float(np.sum(row)),
        }
    return np.stack(rows, axis=0), details


def classifier_weight_factor_power(model: Any, label_names: Sequence[str]) -> Dict[str, Any]:
    """Inspect linear classifier weights when the classifier exposes them."""

    try:
        weight, bias = model.classifier_linear_weight_and_bias()
    except Exception:
        return {"available": False, "reason": "classifier weights are not exposed for this parameterization"}
    weight_np = weight.detach().cpu().float().numpy()
    num_labels = len(label_names)
    num_heads = int(getattr(model.model_config, "latent_pool_heads", 1))
    expected = num_labels * num_heads
    if weight_np.shape[1] != expected:
        return {"available": False, "reason": f"classifier input dim {weight_np.shape[1]} != labels*heads {expected}"}
    group_power = np.zeros((num_labels, num_labels), dtype=np.float64)
    for out_idx in range(num_labels):
        for factor_idx in range(num_labels):
            start = factor_idx * num_heads
            end = start + num_heads
            group_power[out_idx, factor_idx] = float(np.linalg.norm(weight_np[out_idx, start:end], ord=2))
    diagonal = np.diag(group_power)
    offdiag = group_power.copy()
    np.fill_diagonal(offdiag, 0.0)
    return {
        "available": True,
        "matrix_label_by_factor": group_power.tolist(),
        "diagonal_power_mean": float(diagonal.mean()) if diagonal.size else 0.0,
        "offdiagonal_power_mean": float(offdiag.sum() / max(num_labels * max(num_labels - 1, 1), 1)),
        "diagonal_dominance": float(diagonal.sum() / max(group_power.sum(), 1e-12)),
        "per_label_top_weighted_factor": {
            str(label_names[i]): {
                "factor_index": int(np.argmax(group_power[i])),
                "factor_name": str(label_names[int(np.argmax(group_power[i]))]),
                "power": float(np.max(group_power[i])),
            }
            for i in range(num_labels)
        },
    }


def factor_power_metrics(
    scalar_summary: np.ndarray,
    pooled_scalar_features: Optional[np.ndarray],
    labels: np.ndarray,
    predictions: np.ndarray,
    label_names: Sequence[str],
    ridge_alpha: float = 1.0,
    active_variance_threshold: float = 1e-4,
    classifier_power: Optional[Dict[str, Any]] = None,
    probe_train_indices: Optional[np.ndarray] = None,
    probe_eval_indices: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Legacy scalar-dimension emotion diagnostics.

    This is intentionally not treated as the main disentanglement score. The model
    only needs the scalar branch to hold emotion information; individual scalar
    coordinates do not need to be one-emotion-per-dimension.
    """

    scalar = _as_2d_float(scalar_summary)
    labels = _as_2d_float(labels)
    predictions = _as_2d_float(predictions)
    n = min(scalar.shape[0], labels.shape[0], predictions.shape[0])
    scalar = scalar[:n]
    labels = labels[:n]
    predictions = predictions[:n]
    variances = np.var(scalar, axis=0) if scalar.size else np.zeros((0,), dtype=np.float64)
    variance_total = float(np.sum(variances))
    variance_share = variances / max(variance_total, 1e-12)
    entropy = -float(np.sum(np.where(variance_share > 0, variance_share * np.log(np.maximum(variance_share, 1e-12)), 0.0)))
    effective_factors = float(np.exp(entropy)) if variance_share.size else 0.0

    label_corr = _correlation_matrix(scalar, labels)
    pred_corr = _correlation_matrix(scalar, predictions)
    abs_label_corr = np.abs(label_corr)
    per_label_top = {}
    per_factor_top = {}
    for label_idx, name in enumerate(label_names):
        if abs_label_corr.size:
            factor_idx = int(np.argmax(abs_label_corr[:, label_idx]))
            corr_val = float(label_corr[factor_idx, label_idx])
        else:
            factor_idx = -1
            corr_val = 0.0
        per_label_top[str(name)] = {
            "factor_index": factor_idx,
            "factor_name": str(label_names[factor_idx]) if 0 <= factor_idx < len(label_names) else None,
            "abs_correlation": abs(corr_val),
            "correlation": corr_val,
        }
    for factor_idx in range(scalar.shape[1]):
        if abs_label_corr.size:
            label_idx = int(np.argmax(abs_label_corr[factor_idx]))
            corr_val = float(label_corr[factor_idx, label_idx])
        else:
            label_idx = -1
            corr_val = 0.0
        per_factor_top[str(factor_idx)] = {
            "label_index": label_idx,
            "label_name": str(label_names[label_idx]) if 0 <= label_idx < len(label_names) else None,
            "abs_correlation": abs(corr_val),
            "correlation": corr_val,
        }

    probe = _predictive_metrics_from_features(
        scalar,
        labels,
        label_names=label_names,
        alpha=ridge_alpha,
        train_indices=probe_train_indices,
        eval_indices=probe_eval_indices,
    )
    dci = _dci_from_importance(np.asarray(probe["ridge_importance_matrix"], dtype=np.float64))

    diagonal_alignment = None
    if scalar.shape[1] == len(label_names):
        diag = np.abs(np.diag(label_corr)) if label_corr.size else np.zeros((0,), dtype=np.float64)
        off = np.abs(label_corr.copy()) if label_corr.size else np.zeros((0, 0), dtype=np.float64)
        if off.size:
            np.fill_diagonal(off, 0.0)
        diagonal_alignment = {
            "mean_abs_diagonal_correlation": float(diag.mean()) if diag.size else 0.0,
            "mean_abs_offdiagonal_correlation": float(off.sum() / max(off.size - len(diag), 1)) if off.size else 0.0,
            "diagonal_dominance": float(diag.sum() / max(np.abs(label_corr).sum(), 1e-12)) if label_corr.size else 0.0,
        }

    return {
        "metric_scope": "legacy_scalar_dimension_diagnostics_not_primary_branch_disentanglement",
        "num_scalar_factors": int(scalar.shape[1]),
        "active_scalar_factors": int(np.sum(variances > float(active_variance_threshold))),
        "active_variance_threshold": float(active_variance_threshold),
        "scalar_variance": variances.astype(float).tolist(),
        "scalar_variance_share": variance_share.astype(float).tolist(),
        "effective_num_factors": effective_factors,
        "mean_abs_factor_label_correlation": float(np.mean(abs_label_corr)) if abs_label_corr.size else 0.0,
        "max_abs_factor_label_correlation": float(np.max(abs_label_corr)) if abs_label_corr.size else 0.0,
        "factor_label_correlation_matrix": label_corr.astype(float).tolist(),
        "factor_prediction_correlation_matrix": pred_corr.astype(float).tolist(),
        "per_label_top_factor": per_label_top,
        "per_factor_top_label": per_factor_top,
        "ridge_probe": probe,
        "dci": dci,
        "diagonal_alignment": diagonal_alignment,
        "classifier_weight_power": classifier_power or {"available": False},
    }


def emotion_meaning_split_metrics(
    scalar_summary: np.ndarray,
    vector_summary: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
    label_names: Sequence[str],
    ridge_alpha: float = 1.0,
    residual_summary: Optional[np.ndarray] = None,
    probe_train_indices: Optional[np.ndarray] = None,
    probe_eval_indices: Optional[np.ndarray] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Branch-level emotion-vs-meaning disentanglement diagnostics.

    This is the primary diagnostic for the intended representation: emotion labels
    should be recoverable from the scalar/emotion branch and hard to recover from
    the vector/meaning branch. The metric intentionally does not require dimensions
    inside the scalar or vector branch to be independently disentangled.
    """

    scalar = _as_2d_float(scalar_summary)
    vector = _as_2d_float(vector_summary)
    labels = _as_2d_float(labels)
    predictions = _as_2d_float(predictions)
    n = min(scalar.shape[0], vector.shape[0], labels.shape[0], predictions.shape[0])
    scalar = scalar[:n]
    vector = vector[:n]
    labels = labels[:n]
    predictions = predictions[:n]
    residual = None
    if residual_summary is not None:
        residual = _as_2d_float(residual_summary)[:n]
        if residual.shape[1] == 0:
            residual = None

    scalar_probe = _predictive_metrics_from_features(
        scalar,
        labels,
        label_names=label_names,
        alpha=ridge_alpha,
        train_indices=probe_train_indices,
        eval_indices=probe_eval_indices,
    )
    vector_probe = _predictive_metrics_from_features(
        vector,
        labels,
        label_names=label_names,
        alpha=ridge_alpha,
        train_indices=probe_train_indices,
        eval_indices=probe_eval_indices,
    )
    residual_probe = None
    if residual is not None:
        residual_probe = _predictive_metrics_from_features(
            residual,
            labels,
            label_names=label_names,
            alpha=ridge_alpha,
            train_indices=probe_train_indices,
            eval_indices=probe_eval_indices,
        )

    full_features = np.concatenate([scalar, vector] + ([] if residual is None else [residual]), axis=1)
    full_probe = _predictive_metrics_from_features(
        full_features,
        labels,
        label_names=label_names,
        alpha=ridge_alpha,
        train_indices=probe_train_indices,
        eval_indices=probe_eval_indices,
    )
    group_slices: List[Tuple[str, slice]] = [
        ("emotion_scalar", slice(0, scalar.shape[1])),
        ("meaning_vector", slice(scalar.shape[1], scalar.shape[1] + vector.shape[1])),
    ]
    if residual is not None:
        start = scalar.shape[1] + vector.shape[1]
        group_slices.append(("residual", slice(start, start + residual.shape[1])))
    group_importance, group_importance_details = _aggregate_group_importance(
        np.asarray(full_probe["ridge_importance_matrix"], dtype=np.float64),
        group_slices=group_slices,
        normalize_by_group_dim=True,
    )
    group_dci = _dci_from_importance(group_importance)
    group_names = [name for name, _ in group_slices]
    total_group_importance = float(group_importance.sum())
    scalar_importance = float(group_importance[0].sum()) if group_importance.size else 0.0
    vector_importance = float(group_importance[1].sum()) if group_importance.shape[0] > 1 else 0.0
    residual_importance = float(group_importance[2].sum()) if group_importance.shape[0] > 2 else 0.0

    fit_idx = probe_train_indices if probe_train_indices is not None else np.arange(n)
    fit_idx = np.asarray(fit_idx, dtype=np.int64)
    fit_idx = fit_idx[(fit_idx >= 0) & (fit_idx < n)]
    scalar_mi = _mi_matrix(scalar[fit_idx], labels[fit_idx], seed=seed)
    vector_mi = _mi_matrix(vector[fit_idx], labels[fit_idx], seed=seed)
    residual_mi = _mi_matrix(residual[fit_idx], labels[fit_idx], seed=seed) if residual is not None else None
    group_mig = _group_mig_from_mi(
        scalar_mi=scalar_mi,
        vector_mi=vector_mi,
        residual_mi=residual_mi,
        labels=labels[fit_idx],
        label_names=label_names,
    )

    cross_corr = _correlation_matrix(scalar, vector)
    abs_cross = np.abs(cross_corr)
    scalar_r2 = float(scalar_probe["ridge_r2_macro"])
    vector_r2 = float(vector_probe["ridge_r2_macro"])
    scalar_pearson = float(scalar_probe["ridge_pearson_macro"])
    vector_pearson = float(vector_probe["ridge_pearson_macro"])
    residual_r2 = float(residual_probe["ridge_r2_macro"]) if residual_probe else 0.0
    residual_pearson = float(residual_probe["ridge_pearson_macro"]) if residual_probe else 0.0
    leakage_denom_r2 = max(abs(scalar_r2) + abs(vector_r2) + abs(residual_r2), 1e-12)

    return {
        "metric_scope": "primary_branch_level_emotion_vs_meaning_disentanglement",
        "probe_is_heldout": bool(scalar_probe.get("probe_is_heldout", False)),
        "probe_train_examples": int(scalar_probe.get("probe_train_examples", 0)),
        "probe_eval_examples": int(scalar_probe.get("probe_eval_examples", 0)),
        "scalar_emotion_probe": scalar_probe,
        "vector_emotion_probe": vector_probe,
        "residual_emotion_probe": residual_probe,
        "full_emotion_probe": full_probe,
        "branch_group_names": group_names,
        "branch_group_importance_matrix": group_importance.astype(float).tolist(),
        "branch_group_importance_details": group_importance_details,
        "branch_group_dci": group_dci,
        "branch_group_mig": group_mig,
        "emotion_branch_importance_share": scalar_importance / max(total_group_importance, 1e-12),
        "meaning_branch_emotion_leakage_share": vector_importance / max(total_group_importance, 1e-12),
        "residual_emotion_leakage_share": residual_importance / max(total_group_importance, 1e-12),
        "emotion_in_scalar_r2": scalar_r2,
        "emotion_leakage_vector_r2": vector_r2,
        "emotion_leakage_residual_r2": residual_r2,
        "emotion_in_scalar_pearson": scalar_pearson,
        "emotion_leakage_vector_pearson": vector_pearson,
        "emotion_leakage_residual_pearson": residual_pearson,
        "emotion_meaning_separation_r2": scalar_r2 - max(vector_r2, residual_r2),
        "emotion_meaning_separation_pearson": scalar_pearson - max(vector_pearson, residual_pearson),
        "emotion_leakage_ratio_r2": (abs(vector_r2) + abs(residual_r2)) / leakage_denom_r2,
        "emotion_scalar_share_r2": abs(scalar_r2) / leakage_denom_r2,
        "scalar_vector_mean_abs_correlation": float(abs_cross.mean()) if abs_cross.size else 0.0,
        "scalar_vector_max_abs_correlation": float(abs_cross.max()) if abs_cross.size else 0.0,
        "scalar_vector_correlation_matrix": cross_corr.astype(float).tolist(),
    }


def compute_latent_diagnostics(
    *,
    model: Any,
    scalar_summary: Any,
    vector_summary: Any,
    pooled_scalar_features: Any,
    labels: Any,
    predictions: Any,
    label_names: Sequence[str],
    ridge_alpha: float = 1.0,
    active_variance_threshold: float = 1e-4,
    residual_summary: Optional[Any] = None,
    max_samples: Optional[int] = None,
    seed: int = 42,
    probe_eval_fraction: float = 0.30,
    min_probe_train_examples: int = 64,
) -> Dict[str, Any]:
    """Compute all latent-space diagnostics from collected validation/test features."""

    scalar = _as_2d_float(scalar_summary)
    vector = _as_2d_float(vector_summary)
    pooled = None if pooled_scalar_features is None else _as_2d_float(pooled_scalar_features)
    labels_arr = _as_2d_float(labels)
    preds_arr = _as_2d_float(predictions)
    residual = None if residual_summary is None else _as_2d_float(residual_summary)
    n = min(scalar.shape[0], vector.shape[0], labels_arr.shape[0], preds_arr.shape[0])
    if pooled is not None:
        n = min(n, pooled.shape[0])
    if residual is not None:
        n = min(n, residual.shape[0])

    idx = np.arange(n)
    if max_samples is not None and max_samples > 0 and n > int(max_samples):
        rng = np.random.default_rng(int(seed))
        idx = np.sort(rng.choice(idx, size=int(max_samples), replace=False))

    scalar = scalar[idx]
    vector = vector[idx]
    labels_arr = labels_arr[idx]
    preds_arr = preds_arr[idx]
    pooled = None if pooled is None else pooled[idx]
    residual = None if residual is None else residual[idx]

    train_idx, eval_idx, heldout = _probe_split_indices(
        n=len(idx),
        eval_fraction=probe_eval_fraction,
        min_train=min_probe_train_examples,
        seed=seed,
    )

    classifier_power = classifier_weight_factor_power(model, label_names=label_names)
    return {
        "num_examples_used": int(len(idx)),
        "probe_train_examples": int(len(train_idx)),
        "probe_eval_examples": int(len(eval_idx)),
        "probe_is_heldout": bool(heldout),
        "factor_power": factor_power_metrics(
            scalar_summary=scalar,
            pooled_scalar_features=pooled,
            labels=labels_arr,
            predictions=preds_arr,
            label_names=label_names,
            ridge_alpha=ridge_alpha,
            active_variance_threshold=active_variance_threshold,
            classifier_power=classifier_power,
            probe_train_indices=train_idx,
            probe_eval_indices=eval_idx,
        ),
        "emotion_meaning_split": emotion_meaning_split_metrics(
            scalar_summary=scalar,
            vector_summary=vector,
            residual_summary=residual,
            labels=labels_arr,
            predictions=preds_arr,
            label_names=label_names,
            ridge_alpha=ridge_alpha,
            probe_train_indices=train_idx,
            probe_eval_indices=eval_idx,
            seed=seed,
        ),
    }
