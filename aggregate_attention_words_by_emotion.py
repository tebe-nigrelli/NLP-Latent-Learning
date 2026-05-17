#!/usr/bin/env python3
"""Aggregate token/word attention by gold and/or predicted emotion labels.

This script loads an NLP-Latent-Learning checkpoint, runs a dataset split, and
writes full CSV lists of tokens/words with their average attention conditioned on
emotion-label events such as gold-positive, predicted-positive, true positive,
false positive, and false negative.

The attention used here is the latent scalar-pooling attention returned by the
model, with the same two-scalars-per-label mapping used by the companion
attention-map scripts. Attention is first computed per label and per head:

    attention_weights: [batch, scalar_factor, head, token]

For a label, the mapped scalar factors are averaged within each head, each head
is renormalized over visible tokens, and an additional `avg` head is computed as
the average across heads. Word-level rows merge tokenizer subword pieces and sum
their token attention masses.

Example:

  uv run python aggregate_attention_words_by_emotion.py \
    --checkpoint config_matched_beta_schedule_ablation_artifacts/.../best_checkpoint.pt \
    --config config_matched_beta_schedule_ablation_artifacts/.../config.json \
    --threshold-json config_matched_beta_schedule_ablation_artifacts/.../base_threshold_metrics.json \
    --split test \
    --conditions gold,pred,tp,fp,fn \
    --unit both

Outputs:

  attention_word_condition_stats/word_attention_by_emotion.csv
  attention_word_condition_stats/token_attention_by_emotion.csv
  attention_word_condition_stats/condition_summary.csv
  attention_word_condition_stats/top_words_by_emotion.html
  attention_word_condition_stats/summary.json
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
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

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

CONDITION_ALIASES = {
    "gold": "gold",
    "gold_positive": "gold",
    "pred": "pred",
    "predicted": "pred",
    "pred_positive": "pred",
    "gold_or_pred": "gold_or_pred",
    "or": "gold_or_pred",
    "union": "gold_or_pred",
    "gold_and_pred": "gold_and_pred",
    "and": "gold_and_pred",
    "intersection": "gold_and_pred",
    "tp": "tp",
    "true_positive": "tp",
    "fp": "fp",
    "false_positive": "fp",
    "fn": "fn",
    "false_negative": "fn",
    "tn": "tn",
    "true_negative": "tn",
    "all": "all",
}

DEFAULT_CONDITIONS = ["gold", "pred", "gold_or_pred", "tp", "fp", "fn"]


def eprint(*args: Any) -> None:
    print(*args, file=sys.stderr, flush=True)


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
            return json.loads(path.read_text(encoding="utf-8"))
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


def threshold_values_list(threshold: Any, num_labels: int) -> List[float]:
    values = threshold_to_numpy(threshold, num_labels)
    if getattr(values, "size", 1) == 1:
        return [float(values.reshape(-1)[0])] * num_labels
    return [float(x) for x in values.reshape(-1).tolist()]


def predicted_binary(probs: torch.Tensor, threshold: Any) -> torch.Tensor:
    arr = threshold_to_numpy(threshold, probs.shape[-1])
    thr = torch.as_tensor(arr, device=probs.device, dtype=probs.dtype)
    if thr.numel() == 1:
        return probs >= thr.reshape(())
    return probs >= thr.reshape(1, -1)


def clean_token(token: str) -> str:
    token = token.replace("\u2581", " ")
    token = token.replace("<pad>", "").replace("</s>", "").replace("<unk>", "")
    token = token.strip()
    return token


def token_display(raw: str) -> str:
    cleaned = clean_token(raw)
    return cleaned if cleaned else raw


def normalize_item(value: str, lowercase: bool) -> str:
    value = value.strip()
    if lowercase:
        value = value.lower()
    return value


def categorize_item(value: str) -> str:
    stripped = value.strip()
    if not stripped:
        return "empty"
    if re.fullmatch(r"[\W_]+", stripped):
        return "punctuation"
    if re.fullmatch(r"[0-9]+(?:[.,][0-9]+)?", stripped):
        return "number"
    if stripped.lower() in STOPWORDS:
        return "stopword"
    return "content"


def renormalize(prob: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    denom = prob.sum(dim=-1, keepdim=True).clamp_min(eps)
    return prob / denom


def valid_positions_for_example(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    special_ids: set[int],
    ignore_special_tokens: bool,
) -> List[int]:
    out: List[int] = []
    ids = input_ids.detach().cpu().tolist()
    masks = attention_mask.detach().cpu().tolist()
    for pos, (tok_id, mask_val) in enumerate(zip(ids, masks)):
        if int(mask_val) == 0:
            continue
        if ignore_special_tokens and int(tok_id) in special_ids:
            continue
        out.append(pos)
    return out


def get_special_ids(tokenizer: Any) -> set[int]:
    ids = set()
    for name in ["pad_token_id", "eos_token_id", "bos_token_id", "unk_token_id", "sep_token_id", "cls_token_id"]:
        value = getattr(tokenizer, name, None)
        if value is not None:
            ids.add(int(value))
    try:
        for value in tokenizer.all_special_ids:
            ids.add(int(value))
    except Exception:
        pass
    return ids


def infer_label_factor_mapping(model: Any, emotion_names: Sequence[str], top_factors: int) -> Tuple[Dict[int, List[int]], str]:
    num_labels = len(emotion_names)
    num_factors = int(getattr(model, "num_scalar_factors", 0))
    classifier_mode = str(getattr(model, "classifier_mode", ""))
    if num_factors == 2 * num_labels:
        return {i: [2 * i, 2 * i + 1] for i in range(num_labels)}, "two_scalars_per_label"
    if num_factors == num_labels:
        return {i: [i] for i in range(num_labels)}, "one_scalar_per_label"
    # Fallback: spread contiguous factor blocks over labels.
    out: Dict[int, List[int]] = {}
    width = max(1, int(top_factors))
    if classifier_mode == "per_emotion_pair_mlp" and num_factors >= 2 * num_labels:
        width = 2
    for i in range(num_labels):
        start = min(num_factors - 1, i * width)
        out[i] = list(range(start, min(num_factors, start + width)))
    return out, "contiguous_fallback"


def label_attention_map(
    attn: torch.Tensor,
    local_idx: int,
    label_idx: int,
    label_mapping: Dict[int, List[int]],
    valid_positions: Sequence[int],
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    mapped = [idx for idx in label_mapping.get(label_idx, []) if 0 <= idx < attn.size(1)]
    if not mapped:
        mapped = list(range(attn.size(1)))
    group_maps = attn[local_idx, mapped, :, :][:, :, valid_positions].float()
    factor_head_maps = renormalize(group_maps)
    per_head_maps = renormalize(factor_head_maps.mean(dim=0))
    avg_map = renormalize(per_head_maps.mean(dim=0))
    return avg_map, per_head_maps, mapped


def model_forward_batch(model: Any, batch: Dict[str, Any], device: torch.device, residual_scale: float) -> Any:
    with torch.no_grad():
        return model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            labels=None,
            sample_posterior=False,
            classification_weight=0.0,
            recon_weight=0.0,
            kl_weight=0.0,
            residual_scale=float(residual_scale),
        )


def parse_conditions(raw: str) -> List[str]:
    if not raw:
        return list(DEFAULT_CONDITIONS)
    out: List[str] = []
    for chunk in raw.split(","):
        key = chunk.strip().lower().replace("-", "_")
        if not key:
            continue
        if key not in CONDITION_ALIASES:
            raise ValueError(f"Unknown condition '{chunk}'. Valid: {sorted(CONDITION_ALIASES)}")
        value = CONDITION_ALIASES[key]
        if value not in out:
            out.append(value)
    return out or list(DEFAULT_CONDITIONS)


def condition_value(condition: str, gold: bool, pred: bool) -> bool:
    if condition == "gold":
        return gold
    if condition == "pred":
        return pred
    if condition == "gold_or_pred":
        return gold or pred
    if condition == "gold_and_pred":
        return gold and pred
    if condition == "tp":
        return gold and pred
    if condition == "fp":
        return (not gold) and pred
    if condition == "fn":
        return gold and (not pred)
    if condition == "tn":
        return (not gold) and (not pred)
    if condition == "all":
        return True
    raise ValueError(f"Unhandled condition {condition}")


def iter_word_groups(tokens: Sequence[str], valid_positions: Sequence[int]) -> List[Tuple[str, List[int], List[str]]]:
    """Return merged word groups as (word_text, local_indices, raw_tokens)."""
    groups: List[Tuple[str, List[int], List[str]]] = []
    current_text = ""
    current_indices: List[int] = []
    current_raw: List[str] = []

    def flush() -> None:
        nonlocal current_text, current_indices, current_raw
        text = current_text.strip()
        if text or current_indices:
            groups.append((text if text else "".join(current_raw), list(current_indices), list(current_raw)))
        current_text = ""
        current_indices = []
        current_raw = []

    for local_idx, pos in enumerate(valid_positions):
        raw = tokens[pos]
        starts_word = raw.startswith("\u2581") or not current_indices
        cleaned = clean_token(raw)
        if starts_word:
            flush()
            current_text = cleaned
            current_indices = [local_idx]
            current_raw = [raw]
        else:
            # SentencePiece continuation: append without a space.
            current_text += cleaned
            current_indices.append(local_idx)
            current_raw.append(raw)
    flush()
    return groups


def unit_maps_for_example(
    tokens: Sequence[str],
    valid_positions: Sequence[int],
    avg_map: torch.Tensor,
    per_head_maps: torch.Tensor,
    unit: str,
    lowercase: bool,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Build per-map item masses for one unit.

    Returns item -> {display, raw_forms, category, occurrence_count, head_masses}.
    head_masses has keys avg,h0,h1,... and values summed over occurrences.
    """
    head_names = ["avg"] + [f"h{i}" for i in range(int(per_head_maps.size(0)))]
    out: Dict[str, Dict[str, Any]] = {}

    def add_item(item_text: str, raw_forms: Sequence[str], occurrence_count: int, masses: Dict[str, float]) -> None:
        display = item_text.strip() if item_text.strip() else "<empty>"
        key = normalize_item(display, lowercase=lowercase)
        if not key:
            return
        if key not in out:
            out[key] = {
                "display": display,
                "raw_forms": Counter(),
                "category": categorize_item(display),
                "occurrence_count": 0,
                "head_masses": {name: 0.0 for name in head_names},
            }
        out[key]["occurrence_count"] += int(occurrence_count)
        out[key]["raw_forms"].update(raw_forms)
        for name, value in masses.items():
            out[key]["head_masses"][name] += float(value)

    if unit == "token":
        for local_idx, pos in enumerate(valid_positions):
            raw = tokens[pos]
            text = token_display(raw)
            masses = {"avg": float(avg_map[local_idx].item())}
            for h in range(int(per_head_maps.size(0))):
                masses[f"h{h}"] = float(per_head_maps[h, local_idx].item())
            add_item(text, [raw], 1, masses)
        return out

    if unit == "word":
        for word, local_indices, raw_forms in iter_word_groups(tokens, valid_positions):
            if not local_indices:
                continue
            idx_tensor = torch.as_tensor(local_indices, dtype=torch.long)
            masses = {"avg": float(avg_map.index_select(0, idx_tensor).sum().item())}
            for h in range(int(per_head_maps.size(0))):
                masses[f"h{h}"] = float(per_head_maps[h].index_select(0, idx_tensor).sum().item())
            add_item(word, raw_forms, 1, masses)
        return out

    raise ValueError(f"Unsupported unit {unit}")


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def sample_top_forms(counter: Counter, n: int = 5) -> str:
    return " | ".join(f"{k}:{v}" for k, v in counter.most_common(n))


def make_output_rows(
    stats: Dict[Tuple[str, int, str, str, str], Dict[str, Any]],
    selected_map_counts: Dict[Tuple[str, int], int],
    emotion_names: Sequence[str],
    unit_filter: str,
    min_present_maps: int,
    min_occurrences: int,
    sort_by: str,
    top_k: int,
) -> List[Dict[str, Any]]:
    grouped: DefaultDict[Tuple[str, int, str], List[Dict[str, Any]]] = defaultdict(list)
    for (condition, label_idx, unit, item, head), st in stats.items():
        if unit != unit_filter:
            continue
        present_maps = int(st["present_maps"])
        occurrences = int(st["occurrence_count"])
        if present_maps < min_present_maps or occurrences < min_occurrences:
            continue
        selected_maps = int(selected_map_counts.get((condition, label_idx), 0))
        if selected_maps <= 0:
            continue
        total = float(st["total_attention"])
        row = {
            "condition": condition,
            "emotion": emotion_names[label_idx],
            "label_idx": int(label_idx),
            "unit": unit,
            "head": head,
            "item": item,
            "display": st.get("display", item),
            "category": st.get("category", ""),
            "selected_maps": selected_maps,
            "present_maps": present_maps,
            "present_rate": present_maps / selected_maps,
            "occurrence_count": occurrences,
            "total_attention": total,
            "mean_attention_per_selected_map": total / selected_maps,
            "mean_attention_when_present": total / present_maps if present_maps else 0.0,
            "mean_attention_per_occurrence": total / occurrences if occurrences else 0.0,
            "raw_forms": sample_top_forms(st.get("raw_forms", Counter()), n=8),
        }
        grouped[(condition, label_idx, head)].append(row)

    all_rows: List[Dict[str, Any]] = []
    valid_sort = {
        "mean_attention_per_selected_map",
        "mean_attention_when_present",
        "mean_attention_per_occurrence",
        "total_attention",
        "present_rate",
        "present_maps",
        "occurrence_count",
    }
    sort_key = sort_by if sort_by in valid_sort else "mean_attention_per_selected_map"
    for group_key, rows in grouped.items():
        rows.sort(key=lambda r: (float(r.get(sort_key, 0.0)), float(r.get("total_attention", 0.0))), reverse=True)
        limit = len(rows) if top_k <= 0 else min(top_k, len(rows))
        for rank, row in enumerate(rows[:limit], start=1):
            out = dict(row)
            out["rank"] = rank
            all_rows.append(out)
    all_rows.sort(key=lambda r: (r["condition"], r["label_idx"], r["head"], r["rank"]))
    return all_rows


def write_html_top(path: Path, rows: Sequence[Dict[str, Any]], args: argparse.Namespace, summary: Dict[str, Any]) -> None:
    # HTML is intentionally a compact preview; the full list is in CSV.
    preview_rows = [r for r in rows if r.get("unit") == "word" and r.get("head") == "avg"]
    max_preview = max(0, int(args.html_top_k))
    grouped: DefaultDict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in preview_rows:
        grouped[(str(row["condition"]), str(row["emotion"]))].append(row)

    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>Average attention words by emotion</title>",
        "<style>body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;margin:24px;color:#111} table{border-collapse:collapse;width:100%;margin:12px 0 28px} th,td{border-bottom:1px solid #eee;padding:6px 8px;text-align:left;vertical-align:top} th{background:#fafafa}.meta{color:#555}.bad{color:#9b1c1c}.small{font-size:.9em;color:#555}</style>",
        "</head><body>",
        "<h1>Average attention words by emotion</h1>",
        f"<p class='meta'>Checkpoint: {html.escape(str(summary.get('checkpoint','')))}</p>",
        f"<p class='meta'>Split: {html.escape(str(summary.get('split','')))}; examples processed: {summary.get('examples_processed',0)}; full rows are in CSV.</p>",
    ]
    for condition in summary.get("conditions", []):
        parts.append(f"<h2>Condition: {html.escape(condition)}</h2>")
        for emotion in summary.get("emotion_names", []):
            key = (condition, emotion)
            group = grouped.get(key, [])[:max_preview]
            if not group:
                continue
            parts.append(f"<h3>{html.escape(emotion)}</h3>")
            parts.append("<table><thead><tr><th>rank</th><th>word</th><th>mean attention / selected map</th><th>when present</th><th>present rate</th><th>present maps</th><th>category</th></tr></thead><tbody>")
            for row in group:
                parts.append(
                    "<tr>"
                    f"<td>{int(row['rank'])}</td>"
                    f"<td>{html.escape(str(row['display']))}</td>"
                    f"<td>{float(row['mean_attention_per_selected_map']):.6f}</td>"
                    f"<td>{float(row['mean_attention_when_present']):.6f}</td>"
                    f"<td>{float(row['present_rate']):.3f}</td>"
                    f"<td>{int(row['present_maps'])}/{int(row['selected_maps'])}</td>"
                    f"<td>{html.escape(str(row['category']))}</td>"
                    "</tr>"
                )
            parts.append("</tbody></table>")
    parts.append("</body></html>")
    path.write_text("\n".join(parts), encoding="utf-8")


def aggregate(args: argparse.Namespace) -> Dict[str, Any]:
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

    conditions = parse_conditions(args.conditions)
    units = ["word", "token"] if args.unit == "both" else [args.unit]
    label_mapping, mapping_method = infer_label_factor_mapping(model, data.emotion_names, args.label_top_factors)
    eprint(f"[map] label_factor_mapping={mapping_method}")
    eprint(f"[aggregate] split={args.split} conditions={','.join(conditions)} unit={args.unit}")

    loader = getattr(data, f"{args.split}_loader")
    # stats key: (condition, label_idx, unit, item, head)
    stats: Dict[Tuple[str, int, str, str, str], Dict[str, Any]] = {}
    selected_map_counts: DefaultDict[Tuple[str, int], int] = defaultdict(int)
    condition_confusion: DefaultDict[Tuple[int, str], int] = defaultdict(int)
    label_gold_counts: DefaultDict[int, int] = defaultdict(int)
    label_pred_counts: DefaultDict[int, int] = defaultdict(int)
    examples_processed = 0
    maps_processed = 0

    def ensure_stat(key: Tuple[str, int, str, str, str], display: str, category: str, raw_forms: Counter) -> Dict[str, Any]:
        if key not in stats:
            stats[key] = {
                "display": display,
                "category": category,
                "raw_forms": Counter(),
                "present_maps": 0,
                "occurrence_count": 0,
                "total_attention": 0.0,
            }
        stats[key]["raw_forms"].update(raw_forms)
        return stats[key]

    for batch_idx, batch in enumerate(loader):
        if args.max_examples is not None and examples_processed >= args.max_examples:
            break
        out = model_forward_batch(model, batch, device=device, residual_scale=args.residual_scale)
        logits = out.classification_logits.detach().cpu()
        probs = torch.sigmoid(logits)
        preds = predicted_binary(probs, threshold=threshold).cpu()
        attn = out.attention_weights.detach().float().cpu()

        input_ids = batch["input_ids"].detach().cpu()
        attention_mask = batch["attention_mask"].detach().cpu()
        labels = batch.get("labels")
        if labels is not None:
            labels = labels.detach().cpu()

        batch_size = int(input_ids.size(0))
        for local_idx in range(batch_size):
            if args.max_examples is not None and examples_processed >= args.max_examples:
                break
            tokens = tokenizer.convert_ids_to_tokens(input_ids[local_idx].tolist())
            valid_positions = valid_positions_for_example(
                input_ids=input_ids[local_idx],
                attention_mask=attention_mask[local_idx],
                special_ids=special_ids,
                ignore_special_tokens=not args.include_special_tokens,
            )
            if not valid_positions:
                examples_processed += 1
                continue

            for label_idx in range(len(data.emotion_names)):
                gold_value = False
                if labels is not None:
                    gold_value = bool(float(labels[local_idx, label_idx].item()) >= args.positive_label_threshold)
                pred_value = bool(preds[local_idx, label_idx].item())
                if gold_value:
                    label_gold_counts[label_idx] += 1
                if pred_value:
                    label_pred_counts[label_idx] += 1
                if gold_value and pred_value:
                    condition_confusion[(label_idx, "tp")] += 1
                elif (not gold_value) and pred_value:
                    condition_confusion[(label_idx, "fp")] += 1
                elif gold_value and (not pred_value):
                    condition_confusion[(label_idx, "fn")] += 1
                else:
                    condition_confusion[(label_idx, "tn")] += 1

                active_conditions = [cond for cond in conditions if condition_value(cond, gold_value, pred_value)]
                if not active_conditions:
                    continue

                avg_map, per_head_maps, _mapped = label_attention_map(attn, local_idx, label_idx, label_mapping, valid_positions)
                head_names = []
                if args.heads in {"avg", "both"}:
                    head_names.append("avg")
                if args.heads in {"all", "both"}:
                    head_names.extend([f"h{i}" for i in range(int(per_head_maps.size(0)))])

                unit_item_maps: Dict[str, Dict[str, Dict[str, Any]]] = {}
                for unit in units:
                    unit_item_maps[unit] = unit_maps_for_example(
                        tokens=tokens,
                        valid_positions=valid_positions,
                        avg_map=avg_map,
                        per_head_maps=per_head_maps,
                        unit=unit,
                        lowercase=args.lowercase,
                    )

                for condition in active_conditions:
                    selected_map_counts[(condition, label_idx)] += 1
                    maps_processed += 1
                    for unit, item_map in unit_item_maps.items():
                        for item, payload in item_map.items():
                            for head in head_names:
                                key = (condition, label_idx, unit, item, head)
                                st = ensure_stat(
                                    key,
                                    display=str(payload["display"]),
                                    category=str(payload["category"]),
                                    raw_forms=payload["raw_forms"],
                                )
                                st["present_maps"] += 1
                                st["occurrence_count"] += int(payload["occurrence_count"])
                                st["total_attention"] += float(payload["head_masses"].get(head, 0.0))

            examples_processed += 1

        if args.progress_every and (batch_idx + 1) % int(args.progress_every) == 0:
            eprint(f"[progress] batches={batch_idx + 1} examples={examples_processed} active_maps={maps_processed}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, Any]] = []
    for label_idx, name in enumerate(data.emotion_names):
        for condition in conditions:
            selected = int(selected_map_counts.get((condition, label_idx), 0))
            summary_rows.append({
                "emotion": name,
                "label_idx": label_idx,
                "condition": condition,
                "selected_maps": selected,
                "gold_count": int(label_gold_counts.get(label_idx, 0)),
                "pred_count": int(label_pred_counts.get(label_idx, 0)),
                "tp": int(condition_confusion.get((label_idx, "tp"), 0)),
                "fp": int(condition_confusion.get((label_idx, "fp"), 0)),
                "fn": int(condition_confusion.get((label_idx, "fn"), 0)),
                "tn": int(condition_confusion.get((label_idx, "tn"), 0)),
                "threshold": float(threshold_values[label_idx]),
            })
    write_csv(out_dir / "condition_summary.csv", summary_rows)

    word_rows: List[Dict[str, Any]] = []
    token_rows: List[Dict[str, Any]] = []
    if "word" in units:
        word_rows = make_output_rows(
            stats,
            dict(selected_map_counts),
            data.emotion_names,
            unit_filter="word",
            min_present_maps=args.min_present_maps,
            min_occurrences=args.min_occurrences,
            sort_by=args.sort_by,
            top_k=args.top_k,
        )
        write_csv(out_dir / "word_attention_by_emotion.csv", word_rows)
    if "token" in units:
        token_rows = make_output_rows(
            stats,
            dict(selected_map_counts),
            data.emotion_names,
            unit_filter="token",
            min_present_maps=args.min_present_maps,
            min_occurrences=args.min_occurrences,
            sort_by=args.sort_by,
            top_k=args.top_k,
        )
        write_csv(out_dir / "token_attention_by_emotion.csv", token_rows)

    summary = {
        "checkpoint": str(checkpoint_path),
        "split": args.split,
        "output_dir": str(out_dir),
        "examples_processed": int(examples_processed),
        "active_condition_label_maps": int(maps_processed),
        "conditions": conditions,
        "units": units,
        "heads": args.heads,
        "emotion_names": list(data.emotion_names),
        "num_labels": len(data.emotion_names),
        "num_scalar_factors": int(model.num_scalar_factors),
        "mapping_method": mapping_method,
        "threshold_mode": args.threshold_mode,
        "thresholds": threshold_values,
        "load_report": load_report,
        "rows": {
            "word_attention_by_emotion": len(word_rows),
            "token_attention_by_emotion": len(token_rows),
            "condition_summary": len(summary_rows),
        },
        "notes": {
            "mean_attention_per_selected_map": "Average attention mass assigned to the item over all selected emotion-label maps, counting absent items as zero.",
            "mean_attention_when_present": "Average attention mass assigned to the item only among selected maps where the item appears.",
            "mean_attention_per_occurrence": "Average attention mass per occurrence of the item.",
            "word_unit": "SentencePiece-like subword pieces are merged; word attention is the sum of constituent token attention masses.",
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if word_rows:
        write_html_top(out_dir / "top_words_by_emotion.html", word_rows, args, summary)

    eprint(f"[done] examples_processed={examples_processed}")
    eprint(f"[done] wrote {out_dir / 'condition_summary.csv'}")
    if word_rows:
        eprint(f"[done] wrote {out_dir / 'word_attention_by_emotion.csv'} rows={len(word_rows)}")
    if token_rows:
        eprint(f"[done] wrote {out_dir / 'token_attention_by_emotion.csv'} rows={len(token_rows)}")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate words/tokens by average attention conditioned on gold/predicted emotions.")
    parser.add_argument("--checkpoint", required=True, help="Path to best_checkpoint.pt or final_checkpoint.pt.")
    parser.add_argument("--config", default=None, help="Optional sidecar config.json. Auto-discovered beside checkpoint if omitted.")
    parser.add_argument("--threshold-json", default=None, help="Optional base_threshold_metrics.json. Auto-discovered beside checkpoint if omitted.")
    parser.add_argument("--split", choices=["threshold", "val", "test", "train"], default="test")
    parser.add_argument("--output-dir", default="attention_word_condition_stats")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument("--threshold-mode", choices=["global", "per_label"], default="per_label")
    parser.add_argument("--positive-label-threshold", type=float, default=0.5)
    parser.add_argument("--label-top-factors", type=int, default=2)
    parser.add_argument("--conditions", default=",".join(DEFAULT_CONDITIONS), help="Comma list: gold,pred,gold_or_pred,gold_and_pred,tp,fp,fn,tn,all")
    parser.add_argument("--unit", choices=["word", "token", "both"], default="both", help="Aggregate words, tokens, or both.")
    parser.add_argument("--heads", choices=["avg", "all", "both"], default="both", help="Write average head, individual heads, or both.")
    parser.add_argument("--lowercase", action="store_true", help="Merge case variants such as Thanks/thanks.")
    parser.add_argument("--include-special-tokens", action="store_true", help="Include pad/eos/etc tokens in attention aggregation.")
    parser.add_argument("--max-examples", type=int, default=None, help="Optional debug cap on examples processed.")
    parser.add_argument("--progress-every", type=int, default=25, help="Print progress every N batches. Use 0 to disable.")
    parser.add_argument("--min-present-maps", type=int, default=1, help="Filter CSV rows to items present in at least this many selected maps.")
    parser.add_argument("--min-occurrences", type=int, default=1, help="Filter CSV rows to items with at least this many occurrences.")
    parser.add_argument("--top-k", type=int, default=0, help="Rows per condition/emotion/head/unit. 0 means full list.")
    parser.add_argument("--sort-by", default="mean_attention_per_selected_map", choices=[
        "mean_attention_per_selected_map",
        "mean_attention_when_present",
        "mean_attention_per_occurrence",
        "total_attention",
        "present_rate",
        "present_maps",
        "occurrence_count",
    ])
    parser.add_argument("--html-top-k", type=int, default=25, help="Preview top K avg-head word rows per emotion/condition in HTML.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    aggregate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
