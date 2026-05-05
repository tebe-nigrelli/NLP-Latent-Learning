from __future__ import annotations

from ..utils.common import *

# Focused module: thresholds.

def compute_data_quantile_thresholds(
    labels: torch.Tensor,
    quantile: float = 0.5,
    mode: str = "per_label",
) -> Any:
    """Compute thresholds as quantiles of the label distribution.
    
    For regression targets (continuous [0,1]), this gives data-driven baselines
    instead of hardcoded values.
    """
    labels_np = labels.detach().cpu().numpy()
    if labels_np.ndim != 2:
        raise ValueError("labels must be 2D tensor of shape [num_examples, num_labels]")
    
    if mode == "global":
        return float(np.quantile(labels_np.reshape(-1), quantile))
    if mode == "per_label":
        thresholds = [float(np.quantile(labels_np[:, i], quantile)) for i in range(labels_np.shape[1])]
        return thresholds
    raise ValueError(f"Unsupported threshold_mode={mode!r}")


def default_threshold(num_labels: int, mode: str) -> Any:
    if mode == "global":
        return 0.5
    if mode == "per_label":
        return [0.5 for _ in range(num_labels)]
    raise ValueError(f"Unsupported threshold_mode={mode!r}")


def threshold_to_numpy(threshold: Any, num_labels: int) -> np.ndarray:
    if isinstance(threshold, torch.Tensor):
        arr = threshold.detach().cpu().numpy().astype(np.float32)
    elif isinstance(threshold, np.ndarray):
        arr = threshold.astype(np.float32)
    elif isinstance(threshold, (list, tuple)):
        arr = np.asarray(threshold, dtype=np.float32)
    else:
        arr = np.asarray([float(threshold)], dtype=np.float32)

    arr = arr.reshape(-1)
    if arr.size == 1:
        return arr
    if arr.size != num_labels:
        raise ValueError(f"Expected threshold of size 1 or {num_labels}, got {arr.size}.")
    return arr


def threshold_to_tensor(threshold: Any, reference: torch.Tensor) -> torch.Tensor:
    arr = threshold_to_numpy(threshold, reference.shape[-1])
    tensor = torch.as_tensor(arr, device=reference.device, dtype=reference.dtype)
    if tensor.numel() == 1:
        return tensor.squeeze(0)
    return tensor


def restore_threshold(threshold: Any, num_labels: int, mode: str) -> Any:
    arr = threshold_to_numpy(threshold, num_labels)
    if mode == "global":
        return float(arr.reshape(-1)[0])
    if mode == "per_label":
        if arr.size == 1:
            return [float(arr[0]) for _ in range(num_labels)]
        return arr.astype(float).tolist()
    raise ValueError(f"Unsupported threshold_mode={mode!r}")


def format_threshold_for_logging(threshold: Any, num_labels: int) -> str:
    arr = threshold_to_numpy(threshold, num_labels)
    if arr.size == 1:
        return f"{float(arr[0]):.3f}"
    return (
        f"per_label(mean={float(arr.mean()):.3f}, "
        f"min={float(arr.min()):.3f}, max={float(arr.max()):.3f})"
    )


def tune_global_threshold_for_micro_f1(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_points: int = 101,
    is_regression: bool = False,
) -> float:
    """Tune threshold for F1 score. If is_regression=True, binarizes at threshold."""
    probs = torch.sigmoid(logits.detach().cpu()).numpy()
    labels_np = labels.detach().cpu().numpy()
    
    if is_regression:
        # For continuous targets, binarize using the optimal threshold
        best_threshold = 0.5
        best_score = -1.0
        for threshold in np.linspace(0.01, 0.99, num_points):
            # Convert both to binary at this threshold
            targets_binary = (labels_np >= threshold).astype(int)
            preds_binary = (probs >= threshold).astype(int)
            score = f1_score(targets_binary, preds_binary, average="micro", zero_division=0)
            if score > best_score:
                best_score = float(score)
                best_threshold = float(threshold)
    else:
        # For binary targets, use standard approach
        targets = (labels_np >= 0.5).astype(int)
        best_threshold = 0.5
        best_score = -1.0
        for threshold in np.linspace(0.01, 0.99, num_points):
            preds = (probs >= threshold).astype(int)
            score = f1_score(targets, preds, average="micro", zero_division=0)
            if score > best_score:
                best_score = float(score)
                best_threshold = float(threshold)
    return best_threshold


def tune_per_label_thresholds_for_f1(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_points: int = 101,
    is_regression: bool = False,
) -> List[float]:
    """Tune per-label thresholds for F1 score. If is_regression=True, binarizes at threshold."""
    probs = torch.sigmoid(logits.detach().cpu()).numpy()
    labels_np = labels.detach().cpu().numpy()
    num_labels = labels_np.shape[1]
    thresholds: List[float] = []

    for label_idx in range(num_labels):
        best_threshold = 0.5
        best_score = -1.0
        y_prob = probs[:, label_idx]
        y_gold = labels_np[:, label_idx]
        
        if is_regression:
            # For continuous targets, find threshold that best separates high vs low
            for threshold in np.linspace(0.01, 0.99, num_points):
                y_true = (y_gold >= threshold).astype(int)
                y_pred = (y_prob >= threshold).astype(int)
                score = f1_score(y_true, y_pred, zero_division=0)
                if score > best_score:
                    best_score = float(score)
                    best_threshold = float(threshold)
        else:
            # For binary targets
            y_true = (y_gold >= 0.5).astype(int)
            for threshold in np.linspace(0.01, 0.99, num_points):
                y_pred = (y_prob >= threshold).astype(int)
                score = f1_score(y_true, y_pred, zero_division=0)
                if score > best_score:
                    best_score = float(score)
                    best_threshold = float(threshold)
        
        thresholds.append(best_threshold)
    return thresholds


def tune_thresholds(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_points: int,
    mode: str,
    is_regression: bool = False,
) -> Any:
    if mode == "global":
        return tune_global_threshold_for_micro_f1(logits=logits, labels=labels, num_points=num_points, is_regression=is_regression)
    if mode == "per_label":
        return tune_per_label_thresholds_for_f1(logits=logits, labels=labels, num_points=num_points, is_regression=is_regression)
    raise ValueError(f"Unsupported threshold_mode={mode!r}")
