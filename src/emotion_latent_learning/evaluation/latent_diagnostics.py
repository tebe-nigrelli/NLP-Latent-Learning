from __future__ import annotations

from ..utils.common import *
from .regression import safe_pearsonr, safe_spearmanr

# Focused module: FactorVAE factor-power and emotion/meaning split diagnostics.


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
    out_shape = np.take(p.shape, [i for i in range(p.ndim) if i != axis])
    if np.any(denom > 0):
        q = np.divide(p, np.maximum(denom, 1e-12))
    else:
        q = np.zeros_like(p)
    log_base = math.log(max(p.shape[axis], 2))
    ent = -np.sum(np.where(q > 0, q * np.log(np.maximum(q, 1e-12)), 0.0), axis=axis) / log_base
    ent = np.where(np.squeeze(denom, axis=axis) > 0, ent, 1.0)
    return ent.reshape(out_shape) if hasattr(out_shape, "__len__") else ent


def _ridge_fit_predict(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Closed-form ridge regression on standardized X/Y.

    Returns predictions in the original target scale, coefficients in standardized
    space, and non-negative absolute-coefficient importances shaped [n_features, n_targets].
    """

    x = _as_2d_float(x)
    y = _as_2d_float(y)
    n = min(x.shape[0], y.shape[0])
    x = x[:n]
    y = y[:n]
    if n == 0 or x.shape[1] == 0 or y.shape[1] == 0:
        return np.zeros_like(y), np.zeros((x.shape[1], y.shape[1])), np.zeros((x.shape[1], y.shape[1]))
    xz, _, _ = _standardize(x)
    yz, y_mean, y_std = _standardize(y)
    reg = float(alpha) * np.eye(xz.shape[1], dtype=np.float64)
    xtx = xz.T @ xz
    xty = xz.T @ yz
    try:
        coef = np.linalg.solve(xtx + reg, xty)
    except np.linalg.LinAlgError:
        coef = np.linalg.pinv(xtx + reg) @ xty
    pred_z = xz @ coef
    pred = pred_z * y_std.reshape(1, -1) + y_mean.reshape(1, -1)
    importance = np.abs(coef)
    return pred, coef, importance


def _predictive_metrics_from_features(
    x: np.ndarray,
    y: np.ndarray,
    label_names: Sequence[str],
    alpha: float,
) -> Dict[str, Any]:
    x = _as_2d_float(x)
    y = _as_2d_float(y)
    n = min(x.shape[0], y.shape[0])
    x = x[:n]
    y = y[:n]
    pred, coef, importance = _ridge_fit_predict(x, y, alpha=alpha)
    per_label: Dict[str, Dict[str, float]] = {}
    r2_values: List[float] = []
    pearson_values: List[float] = []
    mae_values: List[float] = []
    for idx, name in enumerate(label_names):
        gold = y[:, idx]
        pred_j = pred[:, idx]
        ss_res = float(np.sum((gold - pred_j) ** 2))
        ss_tot = float(np.sum((gold - gold.mean()) ** 2))
        r2 = 0.0 if ss_tot <= 1e-12 else float(1.0 - ss_res / ss_tot)
        pearson = safe_pearsonr(pred_j, gold)
        spearman = safe_spearmanr(pred_j, gold)
        mae = float(np.mean(np.abs(pred_j - gold)))
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
    """DCI-style scores using an importance matrix [factor, target]."""

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
) -> Dict[str, Any]:
    """Measure how strongly scalar FactorVAE dimensions represent emotion factors."""

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

    probe = _predictive_metrics_from_features(scalar, labels, label_names=label_names, alpha=ridge_alpha)
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
) -> Dict[str, Any]:
    """Quantify whether emotion information stays in scalar factors and out of meaning factors.

    Higher scalar probe scores are good; lower vector/residual probe scores mean less emotion
    leakage into the meaning branch. The separation scores report scalar predictive power minus
    vector predictive power, so larger is better.
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

    scalar_probe = _predictive_metrics_from_features(scalar, labels, label_names=label_names, alpha=ridge_alpha)
    vector_probe = _predictive_metrics_from_features(vector, labels, label_names=label_names, alpha=ridge_alpha)
    residual_probe = None
    if residual_summary is not None:
        residual = _as_2d_float(residual_summary)[:n]
        if residual.shape[1] > 0:
            residual_probe = _predictive_metrics_from_features(residual, labels, label_names=label_names, alpha=ridge_alpha)

    cross_corr = _correlation_matrix(scalar, vector)
    abs_cross = np.abs(cross_corr)
    scalar_r2 = float(scalar_probe["ridge_r2_macro"])
    vector_r2 = float(vector_probe["ridge_r2_macro"])
    scalar_pearson = float(scalar_probe["ridge_pearson_macro"])
    vector_pearson = float(vector_probe["ridge_pearson_macro"])
    residual_r2 = float(residual_probe["ridge_r2_macro"]) if residual_probe else 0.0
    residual_pearson = float(residual_probe["ridge_pearson_macro"]) if residual_probe else 0.0
    return {
        "scalar_emotion_probe": scalar_probe,
        "vector_emotion_probe": vector_probe,
        "residual_emotion_probe": residual_probe,
        "emotion_in_scalar_r2": scalar_r2,
        "emotion_leakage_vector_r2": vector_r2,
        "emotion_leakage_residual_r2": residual_r2,
        "emotion_in_scalar_pearson": scalar_pearson,
        "emotion_leakage_vector_pearson": vector_pearson,
        "emotion_leakage_residual_pearson": residual_pearson,
        "emotion_meaning_separation_r2": scalar_r2 - vector_r2,
        "emotion_meaning_separation_pearson": scalar_pearson - vector_pearson,
        "emotion_leakage_ratio_r2": vector_r2 / max(abs(scalar_r2), 1e-12),
        "emotion_scalar_share_r2": scalar_r2 / max(abs(scalar_r2) + abs(vector_r2), 1e-12),
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

    classifier_power = classifier_weight_factor_power(model, label_names=label_names)
    return {
        "num_examples_used": int(len(idx)),
        "factor_power": factor_power_metrics(
            scalar_summary=scalar,
            pooled_scalar_features=pooled,
            labels=labels_arr,
            predictions=preds_arr,
            label_names=label_names,
            ridge_alpha=ridge_alpha,
            active_variance_threshold=active_variance_threshold,
            classifier_power=classifier_power,
        ),
        "emotion_meaning_split": emotion_meaning_split_metrics(
            scalar_summary=scalar,
            vector_summary=vector,
            residual_summary=residual,
            labels=labels_arr,
            predictions=preds_arr,
            label_names=label_names,
            ridge_alpha=ridge_alpha,
        ),
    }
