#!/usr/bin/env python3
"""Create single-table token-level multi-head attention-map reports.

This is a focused companion to analyze_attention_maps.py. It loads an artifact
checkpoint, samples examples, and writes one HTML table plus one Typst table
showing which tokens each emotion label attends to, with every attention head shown separately.
It does not generate separate per-example HTML pages.

Examples
--------
Dataset examples from the test split:

  uv run python visualize_attention_examples.py \
    --checkpoint config_matched_beta_schedule_ablation_artifacts/.../best_checkpoint.pt \
    --config config_matched_beta_schedule_ablation_artifacts/.../config.json \
    --threshold-json config_matched_beta_schedule_ablation_artifacts/.../base_threshold_metrics.json \
    --split test \
    --emotion gratitude \
    --gold-positive-only \
    --max-examples 12

Specific dataset indices:

  uv run python visualize_attention_examples.py \
    --checkpoint best_checkpoint.pt \
    --example-indices 0,13,42 \
    --labels gratitude love remorse

Custom text examples:

  uv run python visualize_attention_examples.py \
    --checkpoint best_checkpoint.pt \
    --text "Thank you so much, I really appreciate it." \
    --text "I am sorry, I should not have done that." \
    --labels gratitude remorse sadness

Outputs
-------
  attention_example_maps/index.html
  attention_example_maps/index.typ
  attention_example_maps/attention_examples.json
  attention_example_maps/attention_top_tokens.csv
  attention_head_top_tokens.csv
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import html
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

# Make the script work both when the package is installed and when run from the
# repository root with a local ./src layout.
_THIS_FILE = Path(__file__).resolve()
for _candidate in [_THIS_FILE.parent / "src", Path.cwd() / "src"]:
    if _candidate.exists() and str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

from emotion_latent_learning.config import (
    DataConfig,
    ExperimentConfig,
    LossConfig,
    ModelConfig,
    PromptConfig,
    ScheduleConfig,
)
from emotion_latent_learning.evaluation.thresholds import (
    default_threshold,
    restore_threshold,
    threshold_to_numpy,
)
from emotion_latent_learning.training.runtime import build_data_bundle, build_runtime, get_device


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by", "for",
    "from", "had", "has", "have", "he", "her", "hers", "him", "his", "i", "if", "in",
    "is", "it", "its", "me", "my", "of", "on", "or", "our", "ours", "she", "so", "that",
    "the", "their", "them", "then", "there", "they", "this", "to", "was", "we", "were",
    "what", "when", "where", "which", "who", "will", "with", "you", "your",
}

CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif; margin: 24px; line-height: 1.45; color: #111; }
a { color: #0645ad; text-decoration: none; }
a:hover { text-decoration: underline; }
.card { border: 1px solid #ddd; border-radius: 12px; padding: 16px; margin: 16px 0; box-shadow: 0 1px 4px rgba(0,0,0,0.04); }
.meta { color: #555; font-size: 0.92em; }
.textbox { background: #f7f7f7; border-radius: 8px; padding: 12px; margin: 10px 0; white-space: pre-wrap; }
.heatmap { line-height: 2.25; margin: 8px 0 14px; }
.word { display: inline-block; padding: 2px 5px; margin: 2px 1px; border-radius: 5px; border: 1px solid rgba(0,0,0,0.06); }
.token { display: inline-block; padding: 2px 4px; margin: 2px 1px; border-radius: 4px; border: 1px solid rgba(0,0,0,0.05); font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 0.95em; }
.label-block { border-top: 1px solid #eee; padding-top: 12px; margin-top: 14px; }
table { border-collapse: collapse; margin: 8px 0 14px; width: 100%; }
th, td { text-align: left; border-bottom: 1px solid #eee; padding: 6px 8px; vertical-align: top; }
th { background: #fafafa; }
.badge { display: inline-block; border-radius: 999px; padding: 2px 8px; margin: 2px; background: #eee; font-size: 0.88em; }
.badge.pred { background: #e6f4ea; }
.badge.gold { background: #fff4ce; }
.small { font-size: 0.88em; color: #555; }
"""


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
    ])


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


def checkpoint_configs_payload(checkpoint: Dict[str, Any], config_payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if isinstance(checkpoint.get("configs"), dict) and checkpoint["configs"]:
        return checkpoint["configs"]
    if isinstance(config_payload, dict) and config_payload:
        return config_payload
    return {}


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


def normalize_state_dict_for_model(model: torch.nn.Module, raw_state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
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
    result = model.load_state_dict(loadable, strict=False)
    missing_keys = list(getattr(result, "missing_keys", []))
    unexpected_keys = list(getattr(result, "unexpected_keys", []))
    ignored_missing_prefixes = ("shared_t5", "t5_encoder.", "t5_decoder.")
    important_missing = [k for k in missing_keys if not k.startswith(ignored_missing_prefixes)]
    skipped_raw = len([k for k, v in raw_state_dict.items() if isinstance(v, torch.Tensor)]) - len(loadable)
    eprint(f"[load] matched_state_tensors={len(loadable)} skipped_or_shape_mismatch={max(skipped_raw, 0)}")
    if important_missing:
        eprint(f"[warn] important missing keys: {important_missing[:12]}")
    if unexpected_keys:
        eprint(f"[warn] unexpected keys: {unexpected_keys[:12]}")
    return {
        "matched_state_tensors": len(loadable),
        "skipped_or_shape_mismatch": max(skipped_raw, 0),
        "important_missing_keys": important_missing[:50],
        "unexpected_keys": unexpected_keys[:50],
    }


def build_runtime_from_checkpoint(checkpoint_path: Path, args: argparse.Namespace, device: torch.device) -> Tuple[Any, Any, Any, Dict[str, Any], Dict[str, Any]]:
    checkpoint = torch_load(checkpoint_path, map_location="cpu")
    config_path = discover_config_path(checkpoint_path, args.config)
    config_payload = read_json_if_exists(config_path)
    if config_path is not None and config_payload is not None:
        eprint(f"[config] using config={config_path}")
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
    return runtime, data, model_config, checkpoint, load_report


def threshold_for_checkpoint(
    checkpoint: Dict[str, Any],
    num_labels: int,
    mode: str,
    threshold_payload: Optional[Dict[str, Any]],
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


def renormalize(prob: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return prob / prob.sum(dim=-1, keepdim=True).clamp_min(eps)


def entropy(prob: torch.Tensor) -> float:
    p = prob.clamp_min(1e-12)
    return float(-(p * p.log()).sum().item())


def positions_to_words(tokens: Sequence[str], valid_positions: Sequence[int]) -> Tuple[List[str], List[List[int]]]:
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


def factor_names(num_scalar_factors: int, emotion_names: Sequence[str], classifier_mode: str) -> List[str]:
    num_labels = len(emotion_names)
    if num_scalar_factors == num_labels:
        return list(emotion_names)
    if num_scalar_factors == 2 * num_labels or classifier_mode == "per_emotion_pair_mlp":
        names: List[str] = []
        for label in emotion_names:
            names.append(f"{label}:a")
            names.append(f"{label}:b")
        return names[:num_scalar_factors]
    return [f"scalar_{idx:02d}" for idx in range(num_scalar_factors)]


def infer_label_factor_mapping(model: Any, emotion_names: Sequence[str], top_factors: int) -> Tuple[Dict[int, List[int]], str]:
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
        return mapping, f"classifier_abs_weight_top{k}"
    except Exception:
        return {idx: list(range(num_scalars)) for idx in range(num_labels)}, "all_scalars_average_unmapped"


def get_special_ids(tokenizer: Any) -> set[int]:
    ids = set()
    for attr in ["pad_token_id", "eos_token_id", "bos_token_id", "sep_token_id", "cls_token_id", "unk_token_id"]:
        value = getattr(tokenizer, attr, None)
        if value is not None:
            ids.add(int(value))
    try:
        ids.update(int(x) for x in tokenizer.all_special_ids)
    except Exception:
        pass
    return ids


def top_items_from_token_map(
    avg_map: torch.Tensor,
    tokens: Sequence[str],
    valid_positions: Sequence[int],
    top_k: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    avg_map = renormalize(avg_map.detach().float().cpu())
    token_rows: List[Dict[str, Any]] = []
    for local_idx, pos in enumerate(valid_positions):
        raw = tokens[pos]
        token_rows.append({
            "rank": 0,
            "token": clean_token(raw) or raw,
            "raw_token": raw,
            "position": int(pos),
            "mass": float(avg_map[local_idx].item()),
        })
    token_rows.sort(key=lambda row: row["mass"], reverse=True)
    for rank, row in enumerate(token_rows[:top_k], start=1):
        row["rank"] = rank

    words, groups = positions_to_words(tokens=tokens, valid_positions=valid_positions)
    word_rows: List[Dict[str, Any]] = []
    for word, group in zip(words, groups):
        idx = torch.tensor(group, dtype=torch.long)
        mass = float(avg_map[idx].sum().item()) if group else 0.0
        positions = [int(valid_positions[i]) for i in group]
        word_rows.append({
            "rank": 0,
            "word": word,
            "category": categorize_word(word),
            "mass": mass,
            "positions": positions,
        })
    word_rows.sort(key=lambda row: row["mass"], reverse=True)
    for rank, row in enumerate(word_rows[:top_k], start=1):
        row["rank"] = rank
    return token_rows[:top_k], word_rows[:top_k]


def word_heatmap_spans(
    avg_map: torch.Tensor,
    tokens: Sequence[str],
    valid_positions: Sequence[int],
    alpha_scale: float = 1.0,
) -> str:
    avg_map = renormalize(avg_map.detach().float().cpu())
    words, groups = positions_to_words(tokens=tokens, valid_positions=valid_positions)
    if not words:
        return ""
    masses = []
    for group in groups:
        idx = torch.tensor(group, dtype=torch.long)
        masses.append(float(avg_map[idx].sum().item()) if group else 0.0)
    max_mass = max(masses) if masses else 1.0
    max_mass = max(max_mass, 1e-12)
    spans = []
    for word, mass in zip(words, masses):
        intensity = min(1.0, max(0.0, mass / max_mass))
        alpha = 0.08 + 0.82 * intensity * alpha_scale
        title = f"mass={mass:.4f}; category={categorize_word(word)}"
        spans.append(
            f'<span class="word" title="{html.escape(title)}" '
            f'style="background: rgba(255, 165, 0, {alpha:.3f});">{html.escape(word)}</span>'
        )
    return " ".join(spans)


def token_heatmap_spans(avg_map: torch.Tensor, tokens: Sequence[str], valid_positions: Sequence[int]) -> str:
    avg_map = renormalize(avg_map.detach().float().cpu())
    max_mass = float(avg_map.max().item()) if avg_map.numel() else 1.0
    max_mass = max(max_mass, 1e-12)
    spans = []
    for local_idx, pos in enumerate(valid_positions):
        raw = tokens[pos]
        shown = clean_token(raw) or raw
        mass = float(avg_map[local_idx].item())
        alpha = 0.06 + 0.84 * min(1.0, mass / max_mass)
        title = f"raw={raw}; pos={pos}; mass={mass:.4f}"
        spans.append(
            f'<span class="token" title="{html.escape(title)}" '
            f'style="background: rgba(30, 144, 255, {alpha:.3f});">{html.escape(shown)}</span>'
        )
    return " ".join(spans)


def label_attention_map(
    attn: torch.Tensor,
    local_idx: int,
    label_idx: int,
    label_mapping: Dict[int, List[int]],
    valid_positions: Sequence[int],
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    """Return average attention and one map per latent-pool head.

    attention_weights has shape [batch, scalar_factor, head, token]. For a
    label mapped to multiple scalar factors, we average those factors inside
    each head, then renormalize over visible tokens. This keeps the report
    head-specific while still respecting the label-to-factor mapping.
    """
    mapped = [idx for idx in label_mapping.get(label_idx, []) if 0 <= idx < attn.size(1)]
    if not mapped:
        mapped = list(range(attn.size(1)))
    group_maps = attn[local_idx, mapped, :, :][:, :, valid_positions].float()
    # [mapped_factors, heads, visible_tokens] -> normalize each factor/head row.
    factor_head_maps = renormalize(group_maps)
    # [heads, visible_tokens]. Average mapped factors, but keep heads separate.
    per_head_maps = renormalize(factor_head_maps.mean(dim=0))
    # [visible_tokens]. Overall label map used only for top-token summaries.
    avg_map = renormalize(per_head_maps.mean(dim=0))
    return avg_map, per_head_maps, mapped


def label_indices_from_names(labels: Sequence[str], emotion_names: Sequence[str]) -> List[int]:
    name_to_idx = {name.lower(): i for i, name in enumerate(emotion_names)}
    indices = []
    for label in labels:
        key = label.lower().strip()
        if key not in name_to_idx:
            raise ValueError(f"Unknown label '{label}'. Known labels: {', '.join(emotion_names)}")
        indices.append(name_to_idx[key])
    return indices


def select_labels_for_example(
    args: argparse.Namespace,
    emotion_names: Sequence[str],
    probs: torch.Tensor,
    preds: torch.Tensor,
    gold: Optional[torch.Tensor],
) -> List[int]:
    if args.all_labels:
        return list(range(len(emotion_names)))
    selected: List[int] = []
    if args.labels:
        selected.extend(label_indices_from_names(args.labels, emotion_names))
    if gold is not None:
        selected.extend([i for i, v in enumerate(gold.tolist()) if float(v) >= args.positive_label_threshold])
    selected.extend([i for i, v in enumerate(preds.tolist()) if bool(v)])
    if args.top_pred_labels > 0:
        k = min(int(args.top_pred_labels), probs.numel())
        selected.extend(torch.topk(probs, k=k).indices.detach().cpu().tolist())
    # stable dedupe
    seen = set()
    out = []
    for idx in selected:
        if idx not in seen:
            seen.add(idx)
            out.append(int(idx))
    return out[: args.max_labels_per_example]


def model_forward_batch(model: Any, input_ids: torch.Tensor, attention_mask: torch.Tensor, device: torch.device, residual_scale: float) -> Any:
    with torch.no_grad():
        return model(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
            labels=None,
            sample_posterior=False,
            classification_weight=0.0,
            recon_weight=0.0,
            kl_weight=0.0,
            residual_scale=float(residual_scale),
        )


def pad_chunk_for_forward(chunk: Sequence[Dict[str, Any]], tokenizer: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad variable-length examples so they can be batched safely.

    Dataset dataloaders usually return fixed-size padded batches, but after we
    select individual examples from different source batches their sequence
    lengths can differ. Re-padding here avoids torch.stack shape errors.
    """
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = 0

    ids = [torch.as_tensor(ex["input_ids"], dtype=torch.long).view(-1) for ex in chunk]
    masks = [torch.as_tensor(ex["attention_mask"], dtype=torch.long).view(-1) for ex in chunk]

    max_len = max(int(x.numel()) for x in ids)
    padded_ids = torch.full((len(ids), max_len), int(pad_token_id), dtype=torch.long)
    padded_masks = torch.zeros((len(ids), max_len), dtype=torch.long)

    for row, (input_ids_i, mask_i) in enumerate(zip(ids, masks)):
        n = int(input_ids_i.numel())
        padded_ids[row, :n] = input_ids_i
        # Prefer the stored attention mask, but clamp to input length if needed.
        m = min(n, int(mask_i.numel()))
        if m > 0:
            padded_masks[row, :m] = mask_i[:m]
        if m < n:
            padded_masks[row, m:n] = 1

    return padded_ids, padded_masks


def collect_dataset_examples(args: argparse.Namespace, data: Any) -> List[Dict[str, Any]]:
    loader = getattr(data, f"{args.split}_loader")
    requested = parse_indices(args.example_indices)
    requested_set = set(requested) if requested else None
    contains = args.contains.lower() if args.contains else None
    emotion_idx = None
    if args.emotion:
        emotion_idx = label_indices_from_names([args.emotion], data.emotion_names)[0]
    collected: List[Dict[str, Any]] = []
    global_idx = 0
    for batch in loader:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch.get("labels")
        texts = list(batch["texts"])
        for local_idx, text in enumerate(texts):
            keep = True
            if requested_set is not None:
                keep = global_idx in requested_set
            if keep and contains is not None:
                keep = contains in text.lower()
            if keep and emotion_idx is not None and args.gold_positive_only:
                keep = labels is not None and float(labels[local_idx, emotion_idx].item()) >= args.positive_label_threshold
            if keep:
                collected.append({
                    "example_index": global_idx,
                    "text": text,
                    "input_ids": input_ids[local_idx].clone(),
                    "attention_mask": attention_mask[local_idx].clone(),
                    "labels": labels[local_idx].clone() if labels is not None else None,
                })
                if len(collected) >= args.max_examples:
                    return collected
            global_idx += 1
    return collected


def collect_custom_examples(args: argparse.Namespace, tokenizer: Any, max_length: int) -> List[Dict[str, Any]]:
    texts: List[str] = []
    texts.extend(args.text or [])
    if args.text_file:
        path = Path(args.text_file)
        texts.extend([line.strip() for line in path.read_text().splitlines() if line.strip()])
    if not texts:
        return []
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    examples = []
    for i, text in enumerate(texts[: args.max_examples]):
        examples.append({
            "example_index": i,
            "text": text,
            "input_ids": enc["input_ids"][i].clone(),
            "attention_mask": enc["attention_mask"][i].clone(),
            "labels": None,
        })
    return examples


def parse_indices(raw: Optional[str]) -> List[int]:
    if not raw:
        return []
    out: List[int] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start_s, end_s = chunk.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            out.extend(list(range(start, end + 1)))
        else:
            out.append(int(chunk))
    return out


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys: List[str] = []
    seen = set()
    preferred = ["example_index", "label", "label_idx", "prob", "pred", "gold", "rank", "word", "token", "mass", "category", "positions", "text"]
    for key in preferred:
        if any(key in row for row in rows) and key not in seen:
            keys.append(key)
            seen.add(key)
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            clean = dict(row)
            for k, v in list(clean.items()):
                if isinstance(v, (list, tuple)):
                    clean[k] = " ".join(str(x) for x in v)
            writer.writerow(clean)


def html_table(rows: Sequence[Dict[str, Any]], columns: Sequence[str]) -> str:
    if not rows:
        return "<p class='small'>No rows.</p>"
    head = "".join(f"<th>{html.escape(col)}</th>" for col in columns)
    body_parts = []
    for row in rows:
        cells = []
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, float):
                value = f"{value:.4f}"
            elif isinstance(value, (list, tuple)):
                value = " ".join(str(x) for x in value)
            cells.append(f"<td>{html.escape(str(value))}</td>")
        body_parts.append("<tr>" + "".join(cells) + "</tr>")
    return "<table><thead><tr>" + head + "</tr></thead><tbody>" + "\n".join(body_parts) + "</tbody></table>"


def badge_list(items: Sequence[str], cls: str) -> str:
    if not items:
        return "<span class='small'>none</span>"
    return " ".join(f"<span class='badge {cls}'>{html.escape(x)}</span>" for x in items)


# ---------------------------------------------------------------------------
# Single-table HTML + Typst report writer. This section intentionally avoids
# separate per-example HTML files and uses token-level attention only.
# ---------------------------------------------------------------------------

TOKEN_HEATMAP_CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif; margin: 24px; line-height: 1.35; color: #111; }
.card { border: 1px solid #ddd; border-radius: 12px; padding: 16px; margin: 16px 0; box-shadow: 0 1px 4px rgba(0,0,0,0.04); }
.meta { color: #555; font-size: 0.92em; }
.textcell { max-width: 360px; white-space: pre-wrap; }
.top { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 0.86em; }
.small { font-size: 0.86em; color: #555; }
.main-table { border-collapse: collapse; width: 100%; margin: 14px 0; }
.main-table > thead > tr > th, .main-table > tbody > tr > td { text-align: left; border-bottom: 1px solid #e8e8e8; padding: 7px 8px; vertical-align: top; }
.main-table > thead > tr > th { position: sticky; top: 0; background: #fafafa; z-index: 2; }
.main-table > tbody > tr:nth-child(even) { background: #fcfcfc; }
.attn-wrap { max-width: 980px; overflow-x: auto; border: 1px solid #ddd; border-radius: 8px; background: white; }
.attn-table { border-collapse: collapse; width: max-content; margin: 0; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 0.76em; }
.attn-table th, .attn-table td { border: 1px solid #ddd; padding: 2px 4px; text-align: center; vertical-align: middle; min-width: 42px; max-width: 82px; }
.attn-table th { background: #f8fafc; }
.attn-table .head-cell { position: sticky; left: 0; background: #f8fafc; font-weight: 700; z-index: 1; min-width: 46px; }
.attn-table .tok { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 78px; display: block; }
.attn-table .mass { font-size: 0.82em; color: #111; }
.scoreline { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 0.84em; white-space: nowrap; margin: 1px 0; }
.scoreline.gold { background: #fff7d6; }
.scoreline.pred { background: #e6f4ea; }
.scoreline.mismatch { background: #ffe5e5; color: #9f1239; font-weight: 700; border-left: 3px solid #dc2626; padding-left: 4px; }
.thresholds { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 0.80em; max-width: 270px; max-height: 165px; overflow: auto; white-space: normal; }
.threshold { display: inline-block; margin: 1px 4px 1px 0; }
.legend-swatch { display: inline-block; width: 40px; height: 14px; border-radius: 4px; border: 1px solid #ddd; vertical-align: middle; }
"""


def color_for_intensity(intensity: float) -> Tuple[str, float]:
    """Return a warm heatmap hex color and alpha for a normalized intensity."""
    x = min(1.0, max(0.0, float(intensity)))
    lo = (255, 247, 237)  # orange-50
    hi = (194, 65, 12)    # orange-700
    rgb = tuple(int(round(lo[i] + (hi[i] - lo[i]) * x)) for i in range(3))
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}", 0.18 + 0.82 * x


def token_display(raw: str) -> str:
    shown = clean_token(raw) or raw
    if raw.startswith("▁") and not shown.startswith("▁"):
        shown = "▁" + shown
    return shown


def selected_head_indices(per_head_maps: torch.Tensor, max_heads: int = 0) -> List[int]:
    num_heads = int(per_head_maps.size(0)) if per_head_maps.ndim >= 2 else 0
    if max_heads and max_heads > 0:
        num_heads = min(num_heads, int(max_heads))
    return list(range(num_heads))


def token_attention_table_html(
    avg_map: torch.Tensor,
    per_head_maps: torch.Tensor,
    tokens: Sequence[str],
    valid_positions: Sequence[int],
    max_heads: int = 0,
) -> str:
    """Build a real HTML table: columns are tokens, rows are avg + heads."""
    avg_map = renormalize(avg_map.detach().float().cpu())
    per_head_maps = renormalize(per_head_maps.detach().float().cpu()) if per_head_maps.numel() else per_head_maps.detach().float().cpu()
    rows: List[Tuple[str, torch.Tensor]] = [("avg", avg_map)]
    for h in selected_head_indices(per_head_maps, max_heads=max_heads):
        rows.append((f"h{h}", renormalize(per_head_maps[h])))

    header_cells = ['<th class="head-cell">head/token</th>']
    for local_idx, pos in enumerate(valid_positions):
        raw = tokens[pos]
        shown = token_display(raw)
        title = f"raw={raw}; token_index={pos}; local_token_col={local_idx}"
        header_cells.append(f'<th title="{html.escape(title)}"><span class="tok">{html.escape(shown)}</span></th>')

    body_rows: List[str] = []
    for head_name, row_map in rows:
        row_map = renormalize(row_map)
        max_mass = max(float(row_map.max().item()) if row_map.numel() else 0.0, 1e-12)
        cells = [f'<th class="head-cell">{html.escape(head_name)}</th>']
        for local_idx, pos in enumerate(valid_positions):
            raw = tokens[pos]
            shown = token_display(raw)
            mass = float(row_map[local_idx].item()) if local_idx < row_map.numel() else 0.0
            color, _alpha = color_for_intensity(mass / max_mass)
            title = f"head={head_name}; raw={raw}; token_index={pos}; attention_mass={mass:.6f}"
            cells.append(
                f'<td title="{html.escape(title)}" style="background-color:{color};">'
                f'<span class="tok">{html.escape(shown)}</span><br><span class="mass">{mass:.3f}</span></td>'
            )
        body_rows.append("<tr>" + "".join(cells) + "</tr>")

    return '<div class="attn-wrap"><table class="attn-table"><thead><tr>' + "".join(header_cells) + "</tr></thead><tbody>" + "".join(body_rows) + "</tbody></table></div>"


def typst_escape(value: Any) -> str:
    s = str(value)
    replacements = {
        "\\": "\\\\",
        "[": "\\[",
        "]": "\\]",
        "#": "\\#",
        "*": "\\*",
        "_": "\\_",
        "`": "\\`",
        "$": "\\$",
        "<": "\\<",
        ">": "\\>",
    }
    for old, new in replacements.items():
        s = s.replace(old, new)
    return s


def typst_cell(value: str) -> str:
    return f'[{typst_escape(value)}]'


def typst_heat_cell(value: str, color: str) -> str:
    return f'#box(fill: rgb("{color}"), inset: 1.2pt)[#text(font: "DejaVu Sans Mono", size: 5.7pt)[{typst_escape(value)}]]'


def token_attention_table_typst(
    avg_map: torch.Tensor,
    per_head_maps: torch.Tensor,
    tokens: Sequence[str],
    valid_positions: Sequence[int],
    max_heads: int = 0,
    max_typst_tokens: int = 24,
) -> str:
    """Build a compact real Typst table with token columns and head rows."""
    if not valid_positions:
        return "[]"
    display_positions = list(valid_positions[:max_typst_tokens])
    truncated = len(valid_positions) > len(display_positions)
    avg_map = renormalize(avg_map.detach().float().cpu())
    per_head_maps = renormalize(per_head_maps.detach().float().cpu()) if per_head_maps.numel() else per_head_maps.detach().float().cpu()
    rows: List[Tuple[str, torch.Tensor]] = [("avg", avg_map)]
    for h in selected_head_indices(per_head_maps, max_heads=max_heads):
        rows.append((f"h{h}", renormalize(per_head_maps[h])))

    cells: List[str] = [typst_cell("head/token")]
    for pos in display_positions:
        cells.append(typst_cell(token_display(tokens[pos])))
    if truncated:
        cells.append(typst_cell("..."))

    for head_name, row_map in rows:
        row_map = renormalize(row_map)
        max_mass = max(float(row_map.max().item()) if row_map.numel() else 0.0, 1e-12)
        cells.append(typst_cell(head_name))
        for local_idx, _pos in enumerate(display_positions):
            mass = float(row_map[local_idx].item()) if local_idx < row_map.numel() else 0.0
            color, _alpha = color_for_intensity(mass / max_mass)
            cells.append(typst_heat_cell(f"{mass:.2f}", color))
        if truncated:
            cells.append(typst_cell("..."))

    ncols = 1 + len(display_positions) + (1 if truncated else 0)
    joined = ",\n    ".join(cells)
    return f'#table(columns: {ncols}, stroke: 0.25pt, inset: 1.2pt, {joined})'


def threshold_values_list(threshold: Any, num_labels: int) -> List[float]:
    if isinstance(threshold, torch.Tensor):
        arr = threshold.detach().cpu().numpy()
    else:
        arr = np.asarray(threshold)
    if arr.ndim == 0:
        return [float(arr.item())] * num_labels
    arr = arr.reshape(-1)
    if arr.size == 1:
        return [float(arr[0])] * num_labels
    values = [float(x) for x in arr[:num_labels]]
    if len(values) < num_labels:
        values.extend([values[-1] if values else 0.5] * (num_labels - len(values)))
    return values


def score_line_text(name: str, prob: float, threshold: float, pred: bool, gold: Optional[float], positive_threshold: float) -> str:
    gold_text = "NA" if gold is None else ("True" if float(gold) >= positive_threshold else "False")
    return f"p({name})={prob:.3f}; threshold={threshold:.3f}; pred={bool(pred)}; gold={gold_text}"


def gold_pred_score_indices(probs: torch.Tensor, preds: torch.Tensor, gold: Optional[torch.Tensor], positive_threshold: float) -> List[int]:
    """Return only emotions that are predicted positive or gold positive.

    The threshold is shown inline for each returned emotion. For custom unlabeled
    text where nothing is predicted positive, keep one top-probability fallback so
    the report cell is not empty.
    """
    selected = {i for i, v in enumerate(preds.tolist()) if bool(v)}
    if gold is not None:
        selected.update(i for i, v in enumerate(gold.tolist()) if float(v) >= positive_threshold)
    if not selected and gold is None and probs.numel():
        selected.add(int(torch.argmax(probs).item()))
    return sorted(selected)


def emotion_scores_html(
    emotion_names: Sequence[str],
    probs: torch.Tensor,
    preds: torch.Tensor,
    gold: Optional[torch.Tensor],
    thresholds: Sequence[float],
    positive_threshold: float,
) -> Tuple[str, str]:
    indices = gold_pred_score_indices(probs, preds, gold, positive_threshold)
    lines: List[str] = []
    text_lines: List[str] = []
    for idx in indices:
        gold_val = None if gold is None else float(gold[idx].item())
        pred_val = bool(preds[idx].item())
        gold_bool = None if gold_val is None else bool(gold_val >= positive_threshold)
        line = score_line_text(emotion_names[idx], float(probs[idx].item()), float(thresholds[idx]), pred_val, gold_val, positive_threshold)
        cls = "scoreline"
        if gold_bool is not None and pred_val != gold_bool:
            cls += " mismatch"
        else:
            if gold_bool:
                cls += " gold"
            if pred_val:
                cls += " pred"
        lines.append(f'<div class="{cls}">{html.escape(line)}</div>')
        text_lines.append(line)
    return "".join(lines) if lines else "<span class='small'>none</span>", "\n".join(text_lines)


def thresholds_html(emotion_names: Sequence[str], thresholds: Sequence[float]) -> Tuple[str, str]:
    parts_html = []
    parts_text = []
    for name, value in zip(emotion_names, thresholds):
        piece = f"{name}={float(value):.3f}"
        parts_html.append(f'<span class="threshold">{html.escape(piece)}</span>')
        parts_text.append(piece)
    return '<div class="thresholds">' + " ".join(parts_html) + '</div>', "; ".join(parts_text)


def top_token_string(top_tokens: Sequence[Dict[str, Any]], n: int = 6) -> str:
    pieces = []
    for row in list(top_tokens)[:n]:
        raw = str(row.get("raw_token", row.get("token", "")))
        pieces.append(f"{token_display(raw)}:{float(row.get('mass', 0.0)):.3f}")
    return " | ".join(pieces)


def badge_list(items: Sequence[str], cls: str) -> str:
    if not items:
        return "<span class='small'>none</span>"
    return " ".join(f"<span class='badge {cls}'>{html.escape(x)}</span>" for x in items)


def labels_for_request(
    args: argparse.Namespace,
    emotion_names: Sequence[str],
    probs: torch.Tensor,
    preds: torch.Tensor,
    gold: Optional[torch.Tensor],
    explicit_target: Optional[Sequence[int]],
) -> List[int]:
    if explicit_target:
        return [int(x) for x in explicit_target]
    if args.all_labels:
        return list(range(len(emotion_names)))
    if args.labels:
        return label_indices_from_names(args.labels, emotion_names)
    if args.emotion:
        return label_indices_from_names([args.emotion], emotion_names)
    selected: List[int] = []
    if gold is not None:
        selected.extend([i for i, v in enumerate(gold.tolist()) if float(v) >= args.positive_label_threshold])
    selected.extend([i for i, v in enumerate(preds.tolist()) if bool(v)])
    if args.top_pred_labels > 0:
        selected.extend(torch.topk(probs, k=min(args.top_pred_labels, probs.numel())).indices.detach().cpu().tolist())
    seen = set()
    out = []
    for idx in selected:
        if int(idx) not in seen:
            seen.add(int(idx))
            out.append(int(idx))
    return out[: args.max_labels_per_example]


def should_use_balanced_mode(args: argparse.Namespace) -> bool:
    if args.text or args.text_file or args.example_indices or args.contains:
        return False
    if args.no_balanced_emotions:
        return False
    return True


def collect_balanced_dataset_examples(args: argparse.Namespace, data: Any) -> List[Dict[str, Any]]:
    loader = getattr(data, f"{args.split}_loader")
    if args.emotion:
        label_indices = label_indices_from_names([args.emotion], data.emotion_names)
        default_samples = args.max_examples if args.samples_per_emotion is None else args.samples_per_emotion
    elif args.labels:
        label_indices = label_indices_from_names(args.labels, data.emotion_names)
        default_samples = 3 if args.samples_per_emotion is None else args.samples_per_emotion
    else:
        label_indices = list(range(len(data.emotion_names)))
        default_samples = 3 if args.samples_per_emotion is None else args.samples_per_emotion
    samples_per_emotion = max(1, int(default_samples))

    counts = {idx: 0 for idx in label_indices}
    collected: List[Dict[str, Any]] = []
    global_idx = 0
    for batch in loader:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch.get("labels")
        texts = list(batch["texts"])
        if labels is None:
            raise RuntimeError("Balanced emotion sampling requires dataset labels. Use --text/--text-file for custom unlabeled text.")
        for local_idx, text in enumerate(texts):
            for label_idx in label_indices:
                if counts[label_idx] >= samples_per_emotion:
                    continue
                if float(labels[local_idx, label_idx].item()) < args.positive_label_threshold:
                    continue
                counts[label_idx] += 1
                collected.append({
                    "example_index": global_idx,
                    "text": text,
                    "input_ids": input_ids[local_idx].clone(),
                    "attention_mask": attention_mask[local_idx].clone(),
                    "labels": labels[local_idx].clone(),
                    "target_label_indices": [int(label_idx)],
                    "target_label": data.emotion_names[label_idx],
                    "target_rank": counts[label_idx],
                })
            if all(counts[idx] >= samples_per_emotion for idx in label_indices):
                return collected
            global_idx += 1
    missing = {data.emotion_names[i]: samples_per_emotion - counts[i] for i in label_indices if counts[i] < samples_per_emotion}
    if missing:
        eprint(f"[warn] not enough gold-positive examples for some labels: {missing}")
    return collected


def collect_explicit_dataset_examples(args: argparse.Namespace, data: Any) -> List[Dict[str, Any]]:
    examples = collect_dataset_examples(args, data)
    if args.labels:
        target_labels = label_indices_from_names(args.labels, data.emotion_names)
    elif args.emotion:
        target_labels = label_indices_from_names([args.emotion], data.emotion_names)
    elif args.all_labels:
        target_labels = list(range(len(data.emotion_names)))
    else:
        target_labels = None
    for ex in examples:
        if target_labels is not None:
            ex["target_label_indices"] = list(target_labels)
    return examples



def write_single_index_html(path: Path, rows: Sequence[Dict[str, Any]], args: argparse.Namespace, summary: Dict[str, Any]) -> None:
    body_rows = []
    for row in rows:
        text = row["text"]
        if len(text) > args.max_text_chars:
            text = text[: max(0, args.max_text_chars - 3)] + "..."
        body_rows.append(f"""
        <tr>
          <td><b>{html.escape(row['label'])}</b><br><span class="small">sample {html.escape(str(row.get('target_rank', '')))}</span><br><span class="small">factors={html.escape(str(row.get('mapped_factors', [])))}</span></td>
          <td>{int(row['example_index'])}</td>
          <td class="textcell">{html.escape(text)}</td>
          <td>{row['score_lines_html']}</td>
          <td>{row['token_attention_table_html']}</td>
          <td class="top">{html.escape(row['top_tokens_text'])}</td>
        </tr>
        """)
    html_doc = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Token attention maps by head</title><style>{TOKEN_HEATMAP_CSS}</style></head>
<body>
  <h1>Token attention maps by head</h1>
  <div class="card">
    <p><b>Checkpoint:</b> {html.escape(str(args.checkpoint))}</p>
    <p><b>Split:</b> {html.escape(args.split)} | <b>Rows:</b> {len(rows)} | <b>Mapping:</b> {html.escape(summary.get('mapping_method', ''))}</p>
    <p><b>Report style:</b> token-level only; one main table; no separate example pages. Each attention cell is a real table with rows <code>avg</code>, <code>h0</code>, <code>h1</code>, ... for the latent-pool heads. Probability lines show only gold-positive or predicted-positive emotions; false positives and false negatives are red.</p>
    <p class="small"><span class="legend-swatch" style="background-color:#fff7ed"></span> low attention &nbsp;→&nbsp; <span class="legend-swatch" style="background-color:#c2410c"></span> high attention. Hover over an attention cell to see raw token, token index, head, and mass.</p>
  </div>
  <table class="main-table">
    <thead><tr><th>Emotion map</th><th>Idx</th><th>Text</th><th>Gold/predicted probabilities</th><th>Attention by head × token</th><th>Top avg tokens</th></tr></thead>
    <tbody>{''.join(body_rows)}</tbody>
  </table>
</body></html>"""
    path.write_text(html_doc, encoding="utf-8")


def write_typst_report(path: Path, rows: Sequence[Dict[str, Any]], args: argparse.Namespace, summary: Dict[str, Any]) -> None:
    typ_rows: List[str] = []
    for row in rows:
        text = row["text"]
        if len(text) > args.max_text_chars:
            text = text[: max(0, args.max_text_chars - 3)] + "..."
        emotion_meta = f"{row['label']}\nsample {row.get('target_rank', '')}\nfactors={row.get('mapped_factors', [])}"
        typ_rows.extend([
            f'[{typst_escape(emotion_meta)}]',
            f'[{int(row["example_index"])}]',
            f'[{typst_escape(text)}]',
            f'[{typst_escape(row["score_lines_text"])}]',
            f'[{row["token_attention_table_typst"]}]',
            f'[{typst_escape(row["top_tokens_text"])}]',
        ])
    table_cells = ",\n  ".join(typ_rows)
    doc = f"""#set page(paper: "a4", flipped: true, margin: 0.25in)
#set text(size: 5.6pt)

= Token attention maps by head

*Checkpoint:* `{typst_escape(args.checkpoint)}`\\
*Split:* {typst_escape(args.split)} | *Rows:* {len(rows)} | *Mapping:* {typst_escape(summary.get("mapping_method", ""))}\\
Token-level maps only. Each attention cell is a real Typst table: columns are tokens; rows are avg + individual attention heads. Probability lines show only gold-positive or predicted-positive emotions with inline thresholds. Darker orange means larger attention within that head row. Long token sequences are truncated in Typst for page fit; the HTML keeps all visible tokens with horizontal scrolling.

#table(
  columns: (0.65in, 0.25in, 1.75in, 2.05in, 6.55in, 1.15in),
  stroke: 0.25pt,
  inset: 2pt,
  align: horizon,
  table.header([*Emotion*], [*Idx*], [*Text*], [*Gold/pred p*], [*Attention heads × tokens*], [*Top avg tokens*]),
  {table_cells}
)
"""
    path.write_text(doc, encoding="utf-8")


def to_serializable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): to_serializable(v) for k, v in value.items() if not str(k).endswith("_html") and not str(k).endswith("_typst")}
    if isinstance(value, (list, tuple)):
        return [to_serializable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def render_examples(args: argparse.Namespace) -> Dict[str, Any]:
    checkpoint_path = Path(args.checkpoint).expanduser()
    device = get_device(args.device)
    runtime, data, model_config, checkpoint, load_report = build_runtime_from_checkpoint(checkpoint_path, args, device)
    model = runtime.state.model
    tokenizer = data.tokenizer
    special_ids = get_special_ids(tokenizer)

    threshold_path = discover_threshold_path(checkpoint_path, args.threshold_json)
    threshold_payload = read_json_if_exists(threshold_path)
    if threshold_path is not None and threshold_payload is not None:
        eprint(f"[threshold] using threshold-json={threshold_path}")
    threshold = threshold_for_checkpoint(checkpoint, len(data.emotion_names), args.threshold_mode, threshold_payload)
    threshold_values = threshold_values_list(threshold, len(data.emotion_names))

    label_mapping, mapping_method = infer_label_factor_mapping(model, data.emotion_names, args.label_top_factors)
    eprint(f"[map] label_factor_mapping={mapping_method}")

    custom_examples = collect_custom_examples(args, tokenizer, int(getattr(data, "max_length", args.max_length or 128)))
    if custom_examples:
        if args.labels:
            target_labels = label_indices_from_names(args.labels, data.emotion_names)
        elif args.emotion:
            target_labels = label_indices_from_names([args.emotion], data.emotion_names)
        elif args.all_labels:
            target_labels = list(range(len(data.emotion_names)))
        else:
            target_labels = None
        examples = []
        for ex in custom_examples:
            if target_labels is not None:
                ex["target_label_indices"] = list(target_labels)
            examples.append(ex)
    elif should_use_balanced_mode(args):
        examples = collect_balanced_dataset_examples(args, data)
    else:
        examples = collect_explicit_dataset_examples(args, data)

    if not examples:
        raise RuntimeError("No examples matched your selection. Try removing filters or increasing --max-examples/--samples-per-emotion.")
    eprint(f"[examples] selected_requests={len(examples)}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    csv_rows: List[Dict[str, Any]] = []
    head_csv_rows: List[Dict[str, Any]] = []

    for start in range(0, len(examples), args.batch_size):
        chunk = examples[start:start + args.batch_size]
        input_ids, attention_mask = pad_chunk_for_forward(chunk, tokenizer)
        out = model_forward_batch(model, input_ids, attention_mask, device=device, residual_scale=args.residual_scale)
        logits = out.classification_logits.detach().cpu()
        probs = torch.sigmoid(logits)
        preds = predicted_binary(probs, threshold=threshold).cpu()
        attn = out.attention_weights.detach().float().cpu()

        for local_idx, ex in enumerate(chunk):
            ids_i = input_ids[local_idx]
            mask_i = attention_mask[local_idx]
            tokens = tokenizer.convert_ids_to_tokens(ids_i.tolist())
            valid_positions = valid_positions_for_example(
                input_ids=ids_i,
                attention_mask=mask_i,
                special_ids=special_ids,
                ignore_special_tokens=not args.include_special_tokens,
            )
            gold = ex.get("labels")
            prob_i = probs[local_idx]
            pred_i = preds[local_idx]
            explicit_target = ex.get("target_label_indices")
            selected_labels = labels_for_request(args, data.emotion_names, prob_i, pred_i, gold, explicit_target)
            if not selected_labels:
                selected_labels = torch.topk(prob_i, k=min(max(1, args.top_pred_labels), prob_i.numel())).indices.tolist()

            gold_labels = []
            if gold is not None:
                gold_labels = [data.emotion_names[i] for i, v in enumerate(gold.tolist()) if float(v) >= args.positive_label_threshold]
            pred_labels = [data.emotion_names[i] for i, v in enumerate(pred_i.tolist()) if bool(v)]

            for label_idx in selected_labels:
                if args.pred_positive_only and not bool(pred_i[label_idx].item()):
                    continue
                if args.gold_positive_only and gold is not None and float(gold[label_idx].item()) < args.positive_label_threshold:
                    continue

                avg_map, head_maps, mapped_factors = label_attention_map(attn, local_idx, label_idx, label_mapping, valid_positions)
                top_tokens, _top_words_unused = top_items_from_token_map(avg_map, tokens, valid_positions, top_k=args.top_k)
                label_name = data.emotion_names[label_idx]
                gold_value = None if gold is None else float(gold[label_idx].item())
                pred_value = bool(pred_i[label_idx].item())
                prob_value = float(prob_i[label_idx].item())
                ent = entropy(avg_map)
                norm_ent = ent / math.log(max(avg_map.numel(), 2))
                top_text = top_token_string(top_tokens, n=args.top_k)
                score_lines_html, score_lines_text = emotion_scores_html(
                    data.emotion_names, prob_i, pred_i, gold, threshold_values, args.positive_label_threshold
                )
                row = {
                    "row_id": len(rows),
                    "example_index": int(ex["example_index"]),
                    "target_rank": ex.get("target_rank", ""),
                    "text": ex["text"],
                    "gold_labels": gold_labels,
                    "pred_labels": pred_labels,
                    "label_idx": int(label_idx),
                    "label": label_name,
                    "prob": prob_value,
                    "pred": pred_value,
                    "gold": gold_value,
                    "mapped_factors": mapped_factors,
                    "entropy": ent,
                    "norm_entropy": norm_ent,
                    "top1_mass": float(avg_map.max().item()) if avg_map.numel() else 0.0,
                    "top3_mass": float(torch.topk(avg_map, k=min(3, avg_map.numel())).values.sum().item()) if avg_map.numel() else 0.0,
                    "score_lines_html": score_lines_html,
                    "score_lines_text": score_lines_text,
                    "token_attention_table_html": token_attention_table_html(avg_map, head_maps, tokens, valid_positions, max_heads=args.max_heads),
                    "token_attention_table_typst": token_attention_table_typst(avg_map, head_maps, tokens, valid_positions, max_heads=args.max_heads, max_typst_tokens=args.max_typst_tokens),
                    "top_tokens": top_tokens,
                    "top_tokens_text": top_text,
                }
                rows.append(row)
                for token_row in top_tokens:
                    csv_rows.append({
                        "row_id": row["row_id"],
                        "example_index": row["example_index"],
                        "label": label_name,
                        "label_idx": label_idx,
                        "prob": prob_value,
                        "pred": pred_value,
                        "gold": gold_value,
                        "head": "avg",
                        "rank": token_row["rank"],
                        "token": token_row["token"],
                        "raw_token": token_row["raw_token"],
                        "position": token_row["position"],
                        "mass": token_row["mass"],
                        "text": ex["text"],
                    })
                for head_idx in selected_head_indices(head_maps, max_heads=args.max_heads):
                    head_map = renormalize(head_maps[head_idx])
                    head_top_tokens, _unused = top_items_from_token_map(head_map, tokens, valid_positions, top_k=args.top_k)
                    for token_row in head_top_tokens:
                        head_csv_rows.append({
                            "row_id": row["row_id"],
                            "example_index": row["example_index"],
                            "label": label_name,
                            "label_idx": label_idx,
                            "prob": prob_value,
                            "pred": pred_value,
                            "gold": gold_value,
                            "head": f"h{head_idx}",
                            "rank": token_row["rank"],
                            "token": token_row["token"],
                            "raw_token": token_row["raw_token"],
                            "position": token_row["position"],
                            "mass": token_row["mass"],
                            "text": ex["text"],
                        })

    summary = {
        "checkpoint": str(checkpoint_path),
        "output_dir": str(out_dir),
        "num_rows": len(rows),
        "num_requests": len(examples),
        "dataset_name": getattr(data, "dataset_name", None),
        "target_kind": getattr(data, "target_kind", None),
        "emotion_names": list(data.emotion_names),
        "model_name": getattr(model_config, "model_name", None),
        "attention_source": getattr(model_config, "attention_source", None),
        "pooling_mode": getattr(model_config, "pooling_mode", None),
        "classifier_mode": getattr(model_config, "classifier_mode", None),
        "num_scalar_factors": int(model.num_scalar_factors),
        "mapping_method": mapping_method,
        "load_report": load_report,
        "token_based_only": True,
        "all_heads_separately": True,
        "num_attention_heads_rendered": (int(model_config.latent_pool_heads) if not args.max_heads else min(int(model_config.latent_pool_heads), int(args.max_heads))),
        "separate_example_files": False,
    }
    write_single_index_html(out_dir / "index.html", rows, args, summary)
    write_typst_report(out_dir / "index.typ", rows, args, summary)
    write_csv(out_dir / "attention_top_tokens.csv", csv_rows)
    write_csv(out_dir / "attention_head_top_tokens.csv", head_csv_rows)
    json_payload = {"summary": summary, "rows": rows}
    (out_dir / "attention_examples.json").write_text(json.dumps(to_serializable(json_payload), indent=2), encoding="utf-8")
    return json_payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create single-table token-level attention maps in HTML and Typst.")
    parser.add_argument("--checkpoint", required=True, help="Path to best_checkpoint.pt or final_checkpoint.pt.")
    parser.add_argument("--config", default=None, help="Optional sidecar config.json. Auto-discovered beside checkpoint if omitted.")
    parser.add_argument("--threshold-json", default=None, help="Optional base_threshold_metrics.json. Auto-discovered beside checkpoint if omitted.")
    parser.add_argument("--split", choices=["threshold", "val", "test", "train"], default="test")
    parser.add_argument("--output-dir", default="attention_example_maps")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument("--threshold-mode", choices=["global", "per_label"], default="per_label")
    parser.add_argument("--positive-label-threshold", type=float, default=0.5)
    parser.add_argument("--label-top-factors", type=int, default=2, help="Fallback number of factors per label if no direct mapping is available.")

    selection = parser.add_argument_group("example selection")
    selection.add_argument("--samples-per-emotion", type=int, default=None, help="Balanced dataset mode: gold-positive examples per emotion. Default: 3 for all emotions, or --max-examples when --emotion is set.")
    selection.add_argument("--max-examples", type=int, default=8, help="Compatibility cap. In default all-emotion mode, use --samples-per-emotion for the main control.")
    selection.add_argument("--no-balanced-emotions", action="store_true", help="Disable default all-emotions x samples-per-emotion dataset selection.")
    selection.add_argument("--example-indices", default=None, help="Comma/range list, e.g. 0,5,10-15. Dataset mode only.")
    selection.add_argument("--contains", default=None, help="Only dataset examples whose text contains this substring.")
    selection.add_argument("--emotion", default=None, help="Select/render one emotion label, e.g. gratitude.")
    selection.add_argument("--gold-positive-only", action="store_true", help="Only render rows where the displayed emotion is gold-positive.")
    selection.add_argument("--pred-positive-only", action="store_true", help="Only render rows where the displayed emotion is predicted positive.")
    selection.add_argument("--text", action="append", default=[], help="Custom text to visualize. Can be repeated. Bypasses dataset selection.")
    selection.add_argument("--text-file", default=None, help="One custom text per line. Bypasses dataset selection.")

    labels = parser.add_argument_group("labels to display")
    labels.add_argument("--labels", nargs="*", default=[], help="Emotion labels to display or sample from.")
    labels.add_argument("--all-labels", action="store_true", help="For explicit/custom examples, render all 28 label maps per example. Default dataset mode already samples all emotions.")
    labels.add_argument("--top-pred-labels", type=int, default=3, help="For explicit/custom examples with no labels, also show the top N predicted labels.")
    labels.add_argument("--max-labels-per-example", type=int, default=8)

    display = parser.add_argument_group("display")
    display.add_argument("--top-k", type=int, default=8, help="Number of top tokens to list per row.")
    display.add_argument("--max-heads", type=int, default=0, help="Maximum attention heads to render. Default 0 means all heads.")
    display.add_argument("--max-typst-tokens", type=int, default=24, help="Maximum token columns rendered inside each Typst nested table. HTML always keeps all visible tokens with scrolling.")
    display.add_argument("--max-text-chars", type=int, default=220, help="Truncate long text cells in HTML/Typst tables.")
    display.add_argument("--include-special-tokens", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    results = render_examples(args)
    summary = results["summary"]
    out = Path(summary["output_dir"])
    print("\n=== Token attention reports written ===")
    print(f"rows:  {summary['num_rows']}")
    print(f"html:  {out / 'index.html'}")
    print(f"typst: {out / 'index.typ'}")
    print(f"json:  {out / 'attention_examples.json'}")
    print(f"csv:   {out / 'attention_top_tokens.csv'}")
    print(f"heads: {out / 'attention_head_top_tokens.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
