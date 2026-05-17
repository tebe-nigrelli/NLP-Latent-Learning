#!/usr/bin/env python3
"""Analyze latent attention-pooling maps for a saved T5 FactorVAE checkpoint.

The model's scalar attention pool returns weights with shape
[batch, scalar_factor, attention_head, input_token]. This script runs an entire
held-out split, summarizes which words/tokens each factor or emotion attends to,
and writes aggregate scientific diagnostics plus a small qualitative highlight file.

Example:
  uv run python scripts/analyze_attention_maps.py \
    --checkpoint factorvae_tokenlevel_clean_artifacts/best_checkpoint.pt \
    --split test \
    --output-dir attention_map_analysis \
    --batch-size 16
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from emotion_latent_learning.config import (
    DataConfig,
    ExperimentConfig,
    LossConfig,
    ModelConfig,
    PromptConfig,
    ScheduleConfig,
)
from emotion_latent_learning.evaluation.thresholds import default_threshold, restore_threshold, threshold_to_numpy
# Import the project loader when available, but use the robust local loader below for
# artifact compatibility across old/new checkpoint namespaces.
try:
    from emotion_latent_learning.training.checkpointing import load_compact_model_state_dict as _project_load_compact_model_state_dict
except Exception:  # pragma: no cover
    _project_load_compact_model_state_dict = None
from emotion_latent_learning.training.runtime import build_data_bundle, build_runtime, get_device


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by", "for",
    "from", "had", "has", "have", "he", "her", "hers", "him", "his", "i", "if", "in",
    "is", "it", "its", "me", "my", "of", "on", "or", "our", "ours", "she", "so", "that",
    "the", "their", "them", "then", "there", "they", "this", "to", "was", "we", "were",
    "what", "when", "where", "which", "who", "will", "with", "you", "your",
}


NUMERIC_FIELDS = [
    "entropy", "norm_entropy", "effective_tokens", "top1_mass", "top3_mass", "top5_mass",
    "expected_position_ratio", "top_position_ratio", "head_js_divergence", "head_top_agreement",
    "attn_scalar_signed_corr", "attn_scalar_abs_corr", "attended_scalar_value",
    "weighted_scalar_value", "mass_on_stopwords", "mass_on_punctuation", "mass_on_numbers",
    "mass_on_content_words", "valid_word_count", "valid_token_count",
]


def eprint(*args: Any) -> None:
    print(*args, file=sys.stderr, flush=True)


def safe_name(value: str, max_len: int = 180) -> str:
    value = re.sub(r"[^A-Za-z0-9_.=-]+", "_", value.strip())
    value = re.sub(r"_+", "_", value).strip("_")
    return value[:max_len] or "run"


def torch_load(path: Path, map_location: Any) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)




def read_json_if_exists(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None:
        return None
    try:
        if path.exists() and path.is_file():
            return json.loads(path.read_text())
    except Exception as exc:
        eprint(f"[warn] could not read JSON {path}: {exc}")
    return None


def first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists() and path.is_file():
            return path
    return None


def discover_config_path(checkpoint_path: Path, explicit_config: Optional[str]) -> Optional[Path]:
    if explicit_config:
        return Path(explicit_config).expanduser()
    parent = checkpoint_path.parent
    return first_existing([
        parent / "config.json",
        parent / "base_train_eval_config.json",
        parent / "train_eval_config.json",
        Path.cwd() / "config.json",
    ])


def discover_threshold_path(checkpoint_path: Path, explicit_threshold_json: Optional[str]) -> Optional[Path]:
    if explicit_threshold_json:
        return Path(explicit_threshold_json).expanduser()
    parent = checkpoint_path.parent
    return first_existing([
        parent / "base_threshold_metrics.json",
        parent / "threshold_metrics.json",
        parent / "base_val_metrics_with_disentanglement.json",
        parent / "base_test_metrics_with_disentanglement.json",
        Path.cwd() / "base_threshold_metrics.json",
        Path.cwd() / "threshold_metrics.json",
        Path.cwd() / "base_val_metrics_with_disentanglement.json",
        Path.cwd() / "base_test_metrics_with_disentanglement.json",
    ])


def checkpoint_configs_payload(checkpoint: Dict[str, Any], config_payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Return config payload from the checkpoint, falling back to sidecar config.json.

    New compact checkpoints include a `configs` dict. Some artifact bundles instead
    ship a sibling `config.json` with the same six sections. This helper accepts both.
    """

    if isinstance(checkpoint.get("configs"), dict) and checkpoint["configs"]:
        return checkpoint["configs"]
    if isinstance(config_payload, dict) and config_payload:
        return config_payload
    return {}


def normalize_state_dict_for_model(model: torch.nn.Module, raw_state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Repair common namespace drift between artifact checkpoints and model classes.

    The training code has existed in both wrapped and unwrapped forms. In one form,
    trainable FactorVAE keys look like `scalar_attention_pool.*`; in another they are
    nested under `emotion_classifier.scalar_attention_pool.*`. This routine builds a
    shape-compatible state dict for the currently instantiated model and drops keys
    that cannot be loaded rather than silently leaving all classifier/pool weights at
    initialization.
    """

    model_state = model.state_dict()
    model_keys = set(model_state.keys())
    candidates: Dict[str, torch.Tensor] = {}

    def add_candidate(key: str, value: torch.Tensor) -> None:
        if key in model_keys and key not in candidates:
            target = model_state[key]
            if hasattr(value, "shape") and hasattr(target, "shape") and tuple(value.shape) == tuple(target.shape):
                candidates[key] = value

    for key, value in raw_state_dict.items():
        if not isinstance(value, torch.Tensor):
            continue
        variants = [key]
        if key.startswith("module."):
            variants.append(key[len("module."):])
        if key.startswith("model."):
            variants.append(key[len("model."):])
        if key.startswith("emotion_classifier."):
            variants.append(key[len("emotion_classifier."):])
        else:
            variants.append("emotion_classifier." + key)
        if key.startswith("classifier_model."):
            variants.append(key[len("classifier_model."):])
        else:
            variants.append("classifier_model." + key)
        # Dedupe while preserving order.
        seen = set()
        for variant in variants:
            if variant in seen:
                continue
            seen.add(variant)
            add_candidate(variant, value)

    return candidates


def load_compatible_model_state_dict(model: torch.nn.Module, checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    raw_state_dict = checkpoint.get("model_state_dict") or checkpoint.get("state_dict") or checkpoint
    if not isinstance(raw_state_dict, dict):
        raise ValueError("Checkpoint does not contain a model_state_dict/state_dict dictionary.")

    loadable = normalize_state_dict_for_model(model, raw_state_dict)
    load_result = model.load_state_dict(loadable, strict=False)
    missing_keys = list(getattr(load_result, "missing_keys", load_result[0] if isinstance(load_result, tuple) else []))
    unexpected_keys = list(getattr(load_result, "unexpected_keys", load_result[1] if isinstance(load_result, tuple) else []))

    ignored_missing_prefixes = (
        "shared_t5",        # base model weights are loaded from HF; compact artifacts keep only LoRA deltas
        "t5_encoder.",
        "t5_decoder.",
    )
    important_missing = [k for k in missing_keys if not k.startswith(ignored_missing_prefixes)]
    important_missing = [k for k in important_missing if k not in loadable]
    skipped_raw = len([k for k, v in raw_state_dict.items() if isinstance(v, torch.Tensor)]) - len(loadable)

    if important_missing:
        eprint(f"[warn] non-frozen keys still missing after compatible load: {important_missing[:12]}")
    if unexpected_keys:
        eprint(f"[warn] unexpected keys after compatible load: {unexpected_keys[:12]}")
    eprint(f"[load] matched_state_tensors={len(loadable)} skipped_or_shape_mismatch={max(skipped_raw, 0)}")
    return {
        "matched_state_tensors": len(loadable),
        "skipped_or_shape_mismatch": max(skipped_raw, 0),
        "important_missing_keys": important_missing[:50],
        "unexpected_keys": unexpected_keys[:50],
    }

def to_serializable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(v) for v in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_serializable(payload), indent=2, sort_keys=True))


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys: List[str] = []
    seen = set()
    preferred = [
        "rank", "label_idx", "label", "factor_idx", "factor_name", "mapped_factors", "mapping_method",
        "num_maps", "gold_positive_maps", "pred_positive_maps", "mean_top1_mass", "mean_norm_entropy",
        "mean_effective_tokens", "mean_head_js_divergence", "mean_head_top_agreement",
        "mean_attn_scalar_abs_corr", "top_words", "top_content_words", "top_tokens",
    ]
    for key in preferred:
        if any(key in row for row in rows) and key not in seen:
            keys.append(key); seen.add(key)
    for row in rows:
        for key in row.keys():
            if key not in seen:
                keys.append(key); seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def dataclass_from_dict(cls: Any, payload: Optional[Dict[str, Any]]) -> Any:
    obj = cls()
    if not payload:
        return obj
    valid = {f.name for f in dataclasses.fields(cls)}
    for key, value in payload.items():
        if key not in valid:
            continue
        default_value = getattr(obj, key, None)
        if isinstance(default_value, tuple) and isinstance(value, list):
            value = tuple(value)
        setattr(obj, key, value)
    return obj


def configs_from_checkpoint(
    checkpoint: Dict[str, Any],
    config_payload: Optional[Dict[str, Any]] = None,
) -> Tuple[DataConfig, PromptConfig, ModelConfig, LossConfig, ScheduleConfig, ExperimentConfig]:
    configs = checkpoint_configs_payload(checkpoint, config_payload)
    return (
        dataclass_from_dict(DataConfig, configs.get("data_config")),
        dataclass_from_dict(PromptConfig, configs.get("prompt_config")),
        dataclass_from_dict(ModelConfig, configs.get("model_config")),
        dataclass_from_dict(LossConfig, configs.get("loss_config")),
        dataclass_from_dict(ScheduleConfig, configs.get("schedule_config")),
        dataclass_from_dict(ExperimentConfig, configs.get("experiment_config")),
    )


class MetricAccumulator:
    def __init__(self) -> None:
        self.n = 0
        self.sums: Dict[str, float] = defaultdict(float)
        self.sumsq: Dict[str, float] = defaultdict(float)
        self.word_counter: Counter[str] = Counter()
        self.content_word_counter: Counter[str] = Counter()
        self.token_counter: Counter[str] = Counter()
        self.category_counter: Counter[str] = Counter()

    def update(self, metrics: Dict[str, float], top_word: str, top_token: str, top_category: str) -> None:
        self.n += 1
        for key in NUMERIC_FIELDS:
            value = metrics.get(key)
            if value is None or not np.isfinite(value):
                continue
            value = float(value)
            self.sums[key] += value
            self.sumsq[key] += value * value
        if top_word:
            self.word_counter[top_word] += 1
            if categorize_word(top_word) == "content":
                self.content_word_counter[top_word] += 1
        if top_token:
            self.token_counter[top_token] += 1
        if top_category:
            self.category_counter[top_category] += 1

    def merge_from(self, other: "MetricAccumulator") -> None:
        self.n += other.n
        for key, value in other.sums.items():
            self.sums[key] += value
        for key, value in other.sumsq.items():
            self.sumsq[key] += value
        self.word_counter.update(other.word_counter)
        self.content_word_counter.update(other.content_word_counter)
        self.token_counter.update(other.token_counter)
        self.category_counter.update(other.category_counter)

    def row(self, prefix: str = "", top_k: int = 12) -> Dict[str, Any]:
        out: Dict[str, Any] = {f"{prefix}num_maps": self.n}
        for key in NUMERIC_FIELDS:
            mean_key = f"{prefix}mean_{key}"
            std_key = f"{prefix}std_{key}"
            if self.n <= 0:
                out[mean_key] = None
                out[std_key] = None
                continue
            mean = self.sums.get(key, 0.0) / self.n
            var = max(0.0, self.sumsq.get(key, 0.0) / self.n - mean * mean)
            out[mean_key] = mean
            out[std_key] = math.sqrt(var)
        out[f"{prefix}top_words"] = format_counter(self.word_counter, top_k=top_k)
        out[f"{prefix}top_content_words"] = format_counter(self.content_word_counter, top_k=top_k)
        out[f"{prefix}top_tokens"] = format_counter(self.token_counter, top_k=top_k)
        out[f"{prefix}top_categories"] = format_counter(self.category_counter, top_k=top_k)
        return out


def format_counter(counter: Counter[str], top_k: int = 12) -> str:
    return " | ".join(f"{key}:{value}" for key, value in counter.most_common(top_k))


def clean_token(token: str) -> str:
    token = token.replace("▁", "").replace("Ġ", "")
    if token.startswith("##"):
        token = token[2:]
    return token.strip()


def categorize_word(word: str) -> str:
    w = word.strip().lower()
    if not w:
        return "empty"
    if all(ch in ".,;:!?()[]{}'\"`-–—…" for ch in w):
        return "punctuation"
    if any(ch.isdigit() for ch in w):
        return "number"
    if w in STOPWORDS:
        return "stopword"
    if len(w) == 1 and not w.isalpha():
        return "punctuation"
    return "content"


def positions_to_words(tokens: Sequence[str], valid_positions: Sequence[int]) -> Tuple[List[str], List[List[int]]]:
    """Group subword tokens into approximate words.

    Handles SentencePiece (▁), GPT/BPE (Ġ), and WordPiece (##) conventions. The
    returned index groups are local to valid_positions, not original sequence positions.
    """

    words: List[str] = []
    groups: List[List[int]] = []
    current = ""
    current_group: List[int] = []

    def flush() -> None:
        nonlocal current, current_group
        if current_group:
            cleaned = current.strip()
            if cleaned:
                words.append(cleaned)
                groups.append(list(current_group))
        current = ""
        current_group = []

    for local_idx, pos in enumerate(valid_positions):
        raw = tokens[pos]
        starts_new = False
        piece = raw
        if raw.startswith("▁") or raw.startswith("Ġ"):
            starts_new = True
            piece = raw[1:]
        elif raw.startswith("##"):
            piece = raw[2:]
        elif not current_group:
            starts_new = True
        elif len(raw) == 1 and re.match(r"\W", raw):
            starts_new = True

        piece = clean_token(piece)
        if not piece:
            continue
        if starts_new and current_group:
            flush()
        current += piece
        current_group.append(local_idx)
    flush()
    return words, groups


def renormalize(prob: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    denom = prob.sum(dim=-1, keepdim=True).clamp_min(eps)
    return prob / denom


def entropy_of_distribution(prob: torch.Tensor) -> float:
    p = prob.clamp_min(1e-12)
    return float(-(p * p.log()).sum().item())


def pearson_corr(x: torch.Tensor, y: torch.Tensor) -> float:
    if x.numel() < 2 or y.numel() < 2:
        return float("nan")
    x = x.float()
    y = y.float()
    x = x - x.mean()
    y = y - y.mean()
    denom = x.norm() * y.norm()
    if float(denom.item()) <= 1e-12:
        return float("nan")
    return float((x * y).sum().div(denom).item())


def js_divergence(p: torch.Tensor, q: torch.Tensor) -> float:
    eps = 1e-12
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    m = 0.5 * (p + q)
    kl_pm = (p * (p / m).log()).sum()
    kl_qm = (q * (q / m).log()).sum()
    return float((0.5 * (kl_pm + kl_qm)).item())


def pairwise_head_metrics(head_maps: torch.Tensor) -> Tuple[float, float]:
    """Return mean JS divergence and top-token agreement among heads."""

    if head_maps.dim() != 2 or head_maps.size(0) < 2:
        return float("nan"), float("nan")
    js_values: List[float] = []
    agree_values: List[float] = []
    top = head_maps.argmax(dim=-1)
    for i in range(head_maps.size(0)):
        for j in range(i + 1, head_maps.size(0)):
            js_values.append(js_divergence(head_maps[i], head_maps[j]))
            agree_values.append(float(top[i].item() == top[j].item()))
    return float(np.mean(js_values)), float(np.mean(agree_values))


def top_mass(prob: torch.Tensor, k: int) -> float:
    if prob.numel() == 0:
        return float("nan")
    k = min(k, prob.numel())
    return float(torch.topk(prob, k=k).values.sum().item())


def describe_attention_map(
    avg_map: torch.Tensor,
    head_maps: torch.Tensor,
    tokens: Sequence[str],
    valid_positions: Sequence[int],
    scalar_values: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    avg_map = renormalize(avg_map.float())
    head_maps = renormalize(head_maps.float())
    num_valid = int(avg_map.numel())
    entropy = entropy_of_distribution(avg_map)
    norm_entropy = entropy / math.log(max(num_valid, 2))
    effective_tokens = math.exp(entropy)
    top_idx = int(avg_map.argmax().item())
    top_pos = int(valid_positions[top_idx])
    top_token = clean_token(tokens[top_pos]) or tokens[top_pos]

    words, word_groups = positions_to_words(tokens=tokens, valid_positions=valid_positions)
    word_masses: List[Tuple[str, float, List[int]]] = []
    for word, group in zip(words, word_groups):
        mass = float(avg_map[torch.tensor(group, dtype=torch.long)].sum().item())
        word_masses.append((word, mass, group))
    if word_masses:
        top_word, top_word_mass, top_word_group = max(word_masses, key=lambda x: x[1])
        top_word_category = categorize_word(top_word)
    else:
        top_word, top_word_mass, top_word_group = top_token, float(avg_map[top_idx].item()), [top_idx]
        top_word_category = categorize_word(top_word)

    local_positions = torch.arange(num_valid, dtype=torch.float32)
    denom = max(num_valid - 1, 1)
    expected_position_ratio = float((avg_map.cpu() * (local_positions / denom)).sum().item())
    top_position_ratio = float(top_idx / denom)
    head_js, head_agree = pairwise_head_metrics(head_maps)

    category_mass = defaultdict(float)
    for word, mass, _group in word_masses:
        category_mass[categorize_word(word)] += mass

    signed_corr = float("nan")
    abs_corr = float("nan")
    attended_scalar_value = float("nan")
    weighted_scalar_value = float("nan")
    if scalar_values is not None and scalar_values.numel() == avg_map.numel():
        scalar_values = scalar_values.detach().float().cpu()
        signed_corr = pearson_corr(avg_map.cpu(), scalar_values)
        abs_corr = pearson_corr(avg_map.cpu(), scalar_values.abs())
        attended_scalar_value = float(scalar_values[top_idx].item())
        weighted_scalar_value = float((avg_map.cpu() * scalar_values).sum().item())

    return {
        "entropy": entropy,
        "norm_entropy": norm_entropy,
        "effective_tokens": effective_tokens,
        "top1_mass": float(avg_map.max().item()),
        "top3_mass": top_mass(avg_map, 3),
        "top5_mass": top_mass(avg_map, 5),
        "expected_position_ratio": expected_position_ratio,
        "top_position_ratio": top_position_ratio,
        "head_js_divergence": head_js,
        "head_top_agreement": head_agree,
        "attn_scalar_signed_corr": signed_corr,
        "attn_scalar_abs_corr": abs_corr,
        "attended_scalar_value": attended_scalar_value,
        "weighted_scalar_value": weighted_scalar_value,
        "mass_on_stopwords": float(category_mass["stopword"]),
        "mass_on_punctuation": float(category_mass["punctuation"]),
        "mass_on_numbers": float(category_mass["number"]),
        "mass_on_content_words": float(category_mass["content"]),
        "valid_word_count": float(len(words)),
        "valid_token_count": float(num_valid),
        "top_token": top_token,
        "top_token_raw": tokens[top_pos],
        "top_word": top_word,
        "top_word_mass": top_word_mass,
        "top_word_category": top_word_category,
        "top_token_position": top_pos,
        "top_word_piece_positions": [int(valid_positions[i]) for i in top_word_group],
    }


def factor_names(num_scalar_factors: int, emotion_names: Sequence[str], classifier_mode: str) -> List[str]:
    num_labels = len(emotion_names)
    if num_scalar_factors == num_labels:
        return list(emotion_names)
    if num_scalar_factors == 2 * num_labels or classifier_mode == "per_emotion_pair_mlp":
        names: List[str] = []
        for label in emotion_names:
            names.append(f"{label}:a")
            names.append(f"{label}:b")
        if len(names) >= num_scalar_factors:
            return names[:num_scalar_factors]
    return [f"scalar_{idx:02d}" for idx in range(num_scalar_factors)]


def infer_label_factor_mapping(
    model: Any,
    emotion_names: Sequence[str],
    top_factors: int,
) -> Tuple[Dict[int, List[int]], str]:
    num_labels = len(emotion_names)
    num_scalars = int(model.num_scalar_factors)
    classifier_mode = str(getattr(model, "classifier_mode", ""))
    if num_scalars == num_labels:
        return {idx: [idx] for idx in range(num_labels)}, "one_scalar_per_label"
    if num_scalars == 2 * num_labels or classifier_mode == "per_emotion_pair_mlp":
        return {idx: [2 * idx, 2 * idx + 1] for idx in range(num_labels)}, "two_scalars_per_label"

    try:
        weight, _bias = model.classifier_linear_weight_and_bias()
        num_heads = int(model.model_config.latent_pool_heads)
        scores = weight.detach().abs().reshape(num_labels, num_scalars, num_heads).mean(dim=-1)
        k = max(1, min(int(top_factors), num_scalars))
        mapping = {idx: torch.topk(scores[idx], k=k).indices.detach().cpu().tolist() for idx in range(num_labels)}
        return mapping, f"classifier_abs_weight_top{ k }"
    except Exception:
        return {idx: list(range(num_scalars)) for idx in range(num_labels)}, "all_scalars_average_unmapped"


def threshold_for_checkpoint(
    checkpoint: Dict[str, Any],
    num_labels: int,
    mode: str,
    threshold_payload: Optional[Dict[str, Any]] = None,
) -> Any:
    threshold = checkpoint.get("threshold")
    if threshold is None and isinstance(threshold_payload, dict):
        threshold = threshold_payload.get("threshold")
    if threshold is None:
        threshold = default_threshold(num_labels=num_labels, mode=mode)
    return restore_threshold(threshold, num_labels=num_labels, mode=mode)


def predicted_binary(probs: torch.Tensor, threshold: Any) -> torch.Tensor:
    arr = threshold_to_numpy(threshold, probs.shape[-1])
    thr = torch.as_tensor(arr, device=probs.device, dtype=probs.dtype)
    if thr.numel() == 1:
        return probs >= thr.reshape(())
    return probs >= thr.reshape(1, -1)


def valid_positions_for_example(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    special_ids: set[int],
    ignore_special_tokens: bool,
) -> List[int]:
    positions: List[int] = []
    for pos, keep in enumerate(attention_mask.detach().cpu().tolist()):
        if not keep:
            continue
        token_id = int(input_ids[pos].item())
        if ignore_special_tokens and token_id in special_ids:
            continue
        positions.append(pos)
    if not positions:
        positions = [pos for pos, keep in enumerate(attention_mask.detach().cpu().tolist()) if keep]
    return positions


def tensor_index(values: torch.Tensor, positions: Sequence[int]) -> torch.Tensor:
    return values[torch.tensor(list(positions), dtype=torch.long, device=values.device)]


def summarize_rows_from_accumulators(
    factor_acc: Dict[int, MetricAccumulator],
    factor_name_list: Sequence[str],
    label_acc: Dict[int, MetricAccumulator],
    label_pos_acc: Dict[int, MetricAccumulator],
    label_neg_acc: Dict[int, MetricAccumulator],
    label_pred_pos_acc: Dict[int, MetricAccumulator],
    emotion_names: Sequence[str],
    label_mapping: Dict[int, List[int]],
    mapping_method: str,
    top_k: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    factor_rows: List[Dict[str, Any]] = []
    for idx, acc in sorted(factor_acc.items()):
        row = {
            "factor_idx": idx,
            "factor_name": factor_name_list[idx] if idx < len(factor_name_list) else f"scalar_{idx:02d}",
            **acc.row(top_k=top_k),
        }
        factor_rows.append(row)
    factor_rows.sort(key=lambda r: (-(r.get("mean_top1_mass") or 0.0), r.get("factor_idx", 0)))
    for rank, row in enumerate(factor_rows, start=1):
        row["rank"] = rank

    label_rows: List[Dict[str, Any]] = []
    for label_idx, label in enumerate(emotion_names):
        row = {
            "label_idx": label_idx,
            "label": label,
            "mapped_factors": " ".join(str(x) for x in label_mapping.get(label_idx, [])),
            "mapping_method": mapping_method,
            **label_acc[label_idx].row(top_k=top_k),
            **label_pos_acc[label_idx].row(prefix="gold_pos_", top_k=top_k),
            **label_neg_acc[label_idx].row(prefix="gold_neg_", top_k=top_k),
            **label_pred_pos_acc[label_idx].row(prefix="pred_pos_", top_k=top_k),
        }
        row["gold_positive_maps"] = label_pos_acc[label_idx].n
        row["pred_positive_maps"] = label_pred_pos_acc[label_idx].n
        label_rows.append(row)
    label_rows.sort(key=lambda r: (-(r.get("mean_top1_mass") or 0.0), r.get("label_idx", 0)))
    for rank, row in enumerate(label_rows, start=1):
        row["rank"] = rank
    return factor_rows, label_rows


def build_runtime_from_checkpoint(
    checkpoint_path: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[Any, Any, Any, Any, Dict[str, Any], Dict[str, Any]]:
    checkpoint = torch_load(checkpoint_path, map_location="cpu")
    config_path = discover_config_path(checkpoint_path, args.config)
    config_payload = read_json_if_exists(config_path)
    if config_path is not None and config_payload is not None:
        eprint(f"[config] found sidecar config={config_path} (used if checkpoint lacks embedded configs)")
    data_config, prompt_config, model_config, loss_config, schedule_config, experiment_config = configs_from_checkpoint(
        checkpoint,
        config_payload=config_payload,
    )

    data_config.eval_batch_size = int(args.batch_size)
    data_config.num_workers = int(args.num_workers)
    if args.max_length is not None:
        data_config.max_length = int(args.max_length)
    experiment_config.output_dir = str(checkpoint_path.parent)
    experiment_config.sample_posterior_eval = False
    # Latent diagnostics are irrelevant here and expensive if imported elsewhere.
    experiment_config.latent_diagnostics_enabled = False

    data = build_data_bundle(data_config=data_config, model_config=model_config, experiment_config=experiment_config)
    runtime = build_runtime(
        data=data,
        data_config=data_config,
        prompt_config=prompt_config,
        model_config=model_config,
        loss_config=loss_config,
        schedule_config=schedule_config,
        experiment_config=experiment_config,
        device=device,
        save_config=False,
    )
    load_report = load_compatible_model_state_dict(runtime.state.model, checkpoint)
    runtime.state.model.eval()
    return runtime, data, model_config, loss_config, checkpoint, load_report


def analyze_checkpoint(checkpoint_path: Path, args: argparse.Namespace, device: torch.device) -> Dict[str, Any]:
    runtime, data, model_config, loss_config, checkpoint, load_report = build_runtime_from_checkpoint(checkpoint_path, args, device)
    model = runtime.state.model
    tokenizer = data.tokenizer
    split_name = args.split
    loader = getattr(data, f"{split_name}_loader")
    threshold_path = discover_threshold_path(checkpoint_path, args.threshold_json)
    threshold_payload = read_json_if_exists(threshold_path)
    if threshold_path is not None and threshold_payload is not None:
        eprint(f"[threshold] found sidecar thresholds={threshold_path} (used if checkpoint lacks embedded threshold)")
    threshold = threshold_for_checkpoint(
        checkpoint,
        num_labels=len(data.emotion_names),
        mode=runtime.ctx.experiment_config.threshold_mode,
        threshold_payload=threshold_payload,
    )
    names = factor_names(
        num_scalar_factors=int(model.num_scalar_factors),
        emotion_names=data.emotion_names,
        classifier_mode=str(getattr(model, "classifier_mode", "")),
    )
    label_mapping, mapping_method = infer_label_factor_mapping(
        model=model,
        emotion_names=data.emotion_names,
        top_factors=args.label_top_factors,
    )
    special_ids = set(int(x) for x in getattr(tokenizer, "all_special_ids", []) or [])

    factor_acc: Dict[int, MetricAccumulator] = defaultdict(MetricAccumulator)
    label_acc: Dict[int, MetricAccumulator] = defaultdict(MetricAccumulator)
    label_pos_acc: Dict[int, MetricAccumulator] = defaultdict(MetricAccumulator)
    label_neg_acc: Dict[int, MetricAccumulator] = defaultdict(MetricAccumulator)
    label_pred_pos_acc: Dict[int, MetricAccumulator] = defaultdict(MetricAccumulator)
    global_acc = MetricAccumulator()

    global_top_words: Counter[str] = Counter()
    example_highlights: List[Dict[str, Any]] = []
    processed = 0
    total_batches = len(loader)
    max_examples = int(args.max_examples or 0)

    eprint(f"[load] checkpoint={checkpoint_path}")
    dataset_attr = 'threshold_tune_dataset' if split_name == 'threshold' else f'{split_name}_dataset'
    eprint(f"[data] split={split_name} examples={len(getattr(data, dataset_attr))} batch_size={args.batch_size}")
    eprint(f"[map] label_factor_mapping={mapping_method}; attention_source={getattr(model_config, 'attention_source', None)}; pooling={getattr(model_config, 'pooling_mode', None)}")

    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader, start=1):
            if max_examples > 0 and processed >= max_examples:
                break
            remaining = None if max_examples <= 0 else max_examples - processed
            if remaining is not None and remaining <= 0:
                break

            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]
            texts = list(batch["texts"])
            if remaining is not None and remaining < input_ids.size(0):
                input_ids = input_ids[:remaining]
                attention_mask = attention_mask[:remaining]
                labels = labels[:remaining]
                texts = texts[:remaining]

            input_ids_device = input_ids.to(device)
            attention_mask_device = attention_mask.to(device)
            out = model(
                input_ids=input_ids_device,
                attention_mask=attention_mask_device,
                labels=None,
                sample_posterior=False,
                classification_weight=0.0,
                recon_weight=0.0,
                kl_weight=0.0,
                residual_scale=float(args.residual_scale),
            )
            probs = torch.sigmoid(out.classification_logits.detach().cpu())
            preds = predicted_binary(probs, threshold=threshold).cpu()
            attn = out.attention_weights.detach().float().cpu()
            scalar_values_all = (out.vae.scalar_mu if args.scalar_value_source == "mu" else out.vae.scalar_z).detach().float().cpu()

            for local_idx in range(input_ids.size(0)):
                global_idx = processed + local_idx
                ids_i = input_ids[local_idx]
                mask_i = attention_mask[local_idx]
                tokens = tokenizer.convert_ids_to_tokens(ids_i.tolist())
                valid_positions = valid_positions_for_example(
                    input_ids=ids_i,
                    attention_mask=mask_i,
                    special_ids=special_ids,
                    ignore_special_tokens=bool(args.ignore_special_tokens),
                )
                if not valid_positions:
                    continue

                label_tensor = labels[local_idx].detach().cpu()
                prob_i = probs[local_idx]
                pred_i = preds[local_idx]
                per_example_best: Optional[Dict[str, Any]] = None

                # Scalar/factor-level maps.
                for factor_idx in range(attn.size(1)):
                    head_maps = attn[local_idx, factor_idx, :, valid_positions]
                    head_maps = renormalize(head_maps)
                    avg_map = renormalize(head_maps.mean(dim=0))
                    scalar_values = scalar_values_all[local_idx, valid_positions, factor_idx]
                    desc = describe_attention_map(
                        avg_map=avg_map,
                        head_maps=head_maps,
                        tokens=tokens,
                        valid_positions=valid_positions,
                        scalar_values=scalar_values,
                    )
                    factor_acc[factor_idx].update(
                        metrics=desc,
                        top_word=str(desc["top_word"]),
                        top_token=str(desc["top_token"]),
                        top_category=str(desc["top_word_category"]),
                    )
                    global_acc.update(
                        metrics=desc,
                        top_word=str(desc["top_word"]),
                        top_token=str(desc["top_token"]),
                        top_category=str(desc["top_word_category"]),
                    )
                    global_top_words[str(desc["top_word"])] += 1
                    if per_example_best is None or float(desc["top1_mass"]) > float(per_example_best["top1_mass"]):
                        per_example_best = {
                            "example_index": global_idx,
                            "kind": "factor",
                            "factor_idx": factor_idx,
                            "factor_name": names[factor_idx] if factor_idx < len(names) else f"scalar_{factor_idx:02d}",
                            "top_word": desc["top_word"],
                            "top_token": desc["top_token"],
                            "top1_mass": desc["top1_mass"],
                            "norm_entropy": desc["norm_entropy"],
                        }

                # Label-level maps, using inferred scalar groups. Useful for "emotion -> word" summaries.
                for label_idx, mapped_factors in label_mapping.items():
                    mapped_factors = [idx for idx in mapped_factors if 0 <= idx < attn.size(1)]
                    if not mapped_factors:
                        continue
                    group_maps = attn[local_idx, mapped_factors, :, :][:, :, valid_positions]
                    flat_heads = group_maps.reshape(-1, len(valid_positions))
                    flat_heads = renormalize(flat_heads)
                    avg_map = renormalize(flat_heads.mean(dim=0))
                    pos_idx = torch.as_tensor(valid_positions, dtype=torch.long)
                    factor_idx_tensor = torch.as_tensor(mapped_factors, dtype=torch.long)
                    scalar_values = (
                        scalar_values_all[local_idx]
                        .index_select(0, pos_idx)
                        .index_select(1, factor_idx_tensor)
                        .mean(dim=-1)
                    )
                    desc = describe_attention_map(
                        avg_map=avg_map,
                        head_maps=flat_heads,
                        tokens=tokens,
                        valid_positions=valid_positions,
                        scalar_values=scalar_values,
                    )
                    label_acc[label_idx].update(desc, str(desc["top_word"]), str(desc["top_token"]), str(desc["top_word_category"]))
                    if float(label_tensor[label_idx].item()) >= args.positive_label_threshold:
                        label_pos_acc[label_idx].update(desc, str(desc["top_word"]), str(desc["top_token"]), str(desc["top_word_category"]))
                    else:
                        label_neg_acc[label_idx].update(desc, str(desc["top_word"]), str(desc["top_token"]), str(desc["top_word_category"]))
                    if bool(pred_i[label_idx].item()):
                        label_pred_pos_acc[label_idx].update(desc, str(desc["top_word"]), str(desc["top_token"]), str(desc["top_word_category"]))

                if per_example_best is not None and len(example_highlights) < args.max_example_rows:
                    gold_labels = [data.emotion_names[i] for i, v in enumerate(label_tensor.tolist()) if float(v) >= args.positive_label_threshold]
                    pred_labels = [data.emotion_names[i] for i, v in enumerate(pred_i.tolist()) if bool(v)]
                    top_prob_idx = int(prob_i.argmax().item()) if prob_i.numel() else -1
                    example_highlights.append({
                        **per_example_best,
                        "input_text": texts[local_idx],
                        "gold_labels": " ".join(gold_labels),
                        "pred_labels": " ".join(pred_labels),
                        "top_pred_label": data.emotion_names[top_prob_idx] if top_prob_idx >= 0 else "",
                        "top_pred_prob": float(prob_i[top_prob_idx].item()) if top_prob_idx >= 0 else None,
                    })

            processed += input_ids.size(0)
            if args.progress_every and (batch_idx % args.progress_every == 0 or batch_idx == total_batches):
                eprint(f"[progress] batch={batch_idx}/{total_batches} examples={processed}")

    factor_rows, label_rows = summarize_rows_from_accumulators(
        factor_acc=factor_acc,
        factor_name_list=names,
        label_acc=label_acc,
        label_pos_acc=label_pos_acc,
        label_neg_acc=label_neg_acc,
        label_pred_pos_acc=label_pred_pos_acc,
        emotion_names=data.emotion_names,
        label_mapping=label_mapping,
        mapping_method=mapping_method,
        top_k=args.top_k_words,
    )

    run_summary = {
        "checkpoint": str(checkpoint_path),
        "split": split_name,
        "num_examples_processed": processed,
        "num_batches": total_batches,
        "dataset_name": data.dataset_name,
        "target_kind": data.target_kind,
        "model_name": getattr(model_config, "model_name", None),
        "attention_source": getattr(model_config, "attention_source", None),
        "pooling_mode": getattr(model_config, "pooling_mode", None),
        "classifier_mode": getattr(model_config, "classifier_mode", None),
        "classifier_parameterization": getattr(model_config, "classifier_parameterization", None),
        "num_scalar_factors": int(model.num_scalar_factors),
        "latent_pool_heads": int(getattr(model_config, "latent_pool_heads", attn.size(2) if 'attn' in locals() else 0)),
        "label_factor_mapping_method": mapping_method,
        "threshold": threshold,
        "load_report": load_report,
        "global_attention": global_acc.row(top_k=args.top_k_words),
        "most_common_top_words": format_counter(global_top_words, top_k=args.top_k_words),
    }

    return {
        "run_summary": run_summary,
        "factor_rows": factor_rows,
        "label_rows": label_rows,
        "example_highlights": example_highlights,
    }


def print_console_summary(results: Dict[str, Any], top_n: int = 8) -> None:
    summary = results["run_summary"]
    factor_rows = results["factor_rows"]
    label_rows = results["label_rows"]
    examples = results["example_highlights"]

    print("\n=== Attention-map analysis ===")
    print(f"checkpoint: {summary['checkpoint']}")
    print(f"split: {summary['split']} | examples processed: {summary['num_examples_processed']}")
    print(f"pooling: {summary['pooling_mode']} | attention_source: {summary['attention_source']} | label map: {summary['label_factor_mapping_method']}")
    print(f"global top words: {summary['most_common_top_words']}")

    print("\nMost concentrated scalar factors:")
    for row in factor_rows[:top_n]:
        print(
            f"  #{row['rank']:02d} {row['factor_name']} "
            f"top1={row.get('mean_top1_mass', 0.0):.3f} "
            f"Hnorm={row.get('mean_norm_entropy', 0.0):.3f} "
            f"eff_tokens={row.get('mean_effective_tokens', 0.0):.2f} "
            f"words=[{row.get('top_content_words') or row.get('top_words')}]")

    print("\nMost concentrated emotion/label maps:")
    for row in label_rows[:top_n]:
        print(
            f"  #{row['rank']:02d} {row['label']} "
            f"top1={row.get('mean_top1_mass', 0.0):.3f} "
            f"Hnorm={row.get('mean_norm_entropy', 0.0):.3f} "
            f"gold+={row.get('gold_positive_maps', 0)} "
            f"words=[{row.get('top_content_words') or row.get('top_words')}]")

    if examples:
        print("\nExample highlights:")
        for row in examples[:min(top_n, len(examples))]:
            text = row["input_text"].replace("\n", " ")
            if len(text) > 110:
                text = text[:107] + "..."
            print(
                f"  ex={row['example_index']} {row['factor_name']} -> '{row['top_word']}' "
                f"mass={row['top1_mass']:.3f}; gold=[{row['gold_labels']}]; pred=[{row['pred_labels']}] :: {text}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a checkpoint on a dataset split and analyze latent attention maps.")
    parser.add_argument("--checkpoint", required=True, help="Path to best_checkpoint.pt/final_checkpoint.pt/checkpoint_epoch_*.pt.")
    parser.add_argument("--config", default=None, help="Optional sidecar config.json. Auto-discovered next to the checkpoint or in cwd when omitted.")
    parser.add_argument("--threshold-json", default=None, help="Optional sidecar threshold/metrics JSON. Auto-discovers base_threshold_metrics.json when omitted.")
    parser.add_argument("--split", choices=["threshold", "val", "test", "train"], default="test", help="Dataset split to analyze. Default: test.")
    parser.add_argument("--output-dir", default="attention_map_analysis", help="Directory for JSON/CSV outputs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Evaluation batch size.")
    parser.add_argument("--num-workers", type=int, default=0, help="Dataloader workers.")
    parser.add_argument("--max-examples", type=int, default=0, help="Debug cap. 0 means run the full split.")
    parser.add_argument("--max-length", type=int, default=None, help="Override tokenizer max_length from checkpoint config.")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="Evaluation device.")
    parser.add_argument("--output-prefix", default=None, help="Optional filename prefix. Defaults to checkpoint stem plus split.")
    parser.add_argument("--residual-scale", type=float, default=0.0, help="Residual/skip scale for forward pass. Attention maps are encoder-side; default 0 is fine.")
    parser.add_argument("--scalar-value-source", choices=["mu", "z"], default="mu", help="Scalar latent values used for attention-latent correlation.")
    parser.add_argument("--positive-label-threshold", type=float, default=0.5, help="Gold label cutoff for positive-vs-negative attention summaries.")
    parser.add_argument("--label-top-factors", type=int, default=3, help="Fallback top classifier-weighted scalar factors per label when mapping is not direct.")
    parser.add_argument("--top-k-words", type=int, default=15, help="Number of top words/tokens to keep in summary cells.")
    parser.add_argument("--max-example-rows", type=int, default=100, help="Number of qualitative example rows to save/print.")
    parser.add_argument("--progress-every", type=int, default=50, help="Print progress every N batches; 0 disables.")
    parser.add_argument("--include-special-tokens", dest="ignore_special_tokens", action="store_false", help="Include special tokens such as </s> in attention analysis.")
    parser.set_defaults(ignore_special_tokens=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).expanduser()
    if not checkpoint_path.exists():
        eprint(f"Checkpoint not found: {checkpoint_path}")
        return 2

    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable.")
        device = torch.device("cuda")
    else:
        device = get_device(prefer_cuda=True)

    results = analyze_checkpoint(checkpoint_path=checkpoint_path, args=args, device=device)

    output_dir = Path(args.output_dir)
    prefix = args.output_prefix or f"{safe_name(checkpoint_path.parent.name)}__{safe_name(checkpoint_path.stem)}__{args.split}"
    write_json(output_dir / f"{prefix}__attention_summary.json", results["run_summary"])
    write_csv(output_dir / f"{prefix}__factor_attention_summary.csv", results["factor_rows"])
    write_csv(output_dir / f"{prefix}__label_attention_summary.csv", results["label_rows"])
    write_csv(output_dir / f"{prefix}__example_attention_highlights.csv", results["example_highlights"])

    print_console_summary(results, top_n=min(8, int(args.top_k_words)))
    print("\nWrote:")
    print(f"  {output_dir / (prefix + '__attention_summary.json')}")
    print(f"  {output_dir / (prefix + '__factor_attention_summary.csv')}")
    print(f"  {output_dir / (prefix + '__label_attention_summary.csv')}")
    print(f"  {output_dir / (prefix + '__example_attention_highlights.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
