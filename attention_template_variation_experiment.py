#!/usr/bin/env python3
"""Experiment: do latent-pooling attention heads follow one shared template?

This script runs an NLP-Latent-Learning checkpoint over a split, computes one
attention-variation row for every example x emotion map, and summarizes the
results by emotion and by gold/predicted conditions.

For your model, the attention object is the latent-pooling attention, not a
transformer self-attention matrix. For an emotion e and example x, we form

    A_e in R^{H x n}

where H is the number of latent-pool heads and n is the number of visible input
tokens. Each row A_{e,h} is a distribution over tokens. We compute the mean
attention template

    a_bar = (1 / H) sum_h A_{e,h}

and the average divergence from that template

    S_e = (1 / H) sum_h KL(A_{e,h} || a_bar).

Small S_e means heads attend similarly, consistent with a shared lexical/global
pattern. Large S_e means different heads specialize on different tokens.

Outputs:
  attention_template_variation/per_map_attention_variation.csv
  attention_template_variation/condition_summary.csv
  attention_template_variation/emotion_summary.csv
  attention_template_variation/extreme_examples.csv
  attention_template_variation/summary.json
  attention_template_variation/index.html

This script intentionally reuses the artifact-compatible loading and data logic
from aggregate_attention_words_by_emotion.py. Put both scripts in the repo root.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

try:
    from aggregate_attention_words_by_emotion import (
        build_runtime_from_checkpoint,
        clean_token,
        condition_value,
        discover_threshold_path,
        eprint,
        get_special_ids,
        infer_label_factor_mapping,
        label_attention_map,
        model_forward_batch,
        parse_conditions,
        predicted_binary,
        read_json_if_exists,
        threshold_for_checkpoint,
        threshold_values_list,
        token_display,
        valid_positions_for_example,
        write_csv,
    )
    from emotion_latent_learning.training.runtime import get_device
except Exception as exc:  # pragma: no cover - only for user-friendly failure.
    raise SystemExit(
        "Could not import helper functions. Put this file next to "
        "aggregate_attention_words_by_emotion.py in the repo root, then rerun.\n"
        f"Original import error: {exc}"
    )


DEFAULT_CONDITIONS = [
    "all",
    "gold",
    "pred",
    "gold_or_pred",
    "gold_and_pred",
    "tp",
    "fp",
    "fn",
    "tn",
]


def safe_name(text: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in text)


def entropy(p: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    p = p.clamp_min(eps)
    return -(p * p.log()).sum(dim=-1)


def kl_div(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    return (p * (p.log() - q.log())).sum(dim=-1)


def js_pair(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    m = 0.5 * (p + q)
    return 0.5 * kl_div(p, m, eps=eps) + 0.5 * kl_div(q, m, eps=eps)


def gini_coefficient(values: torch.Tensor, eps: float = 1e-12) -> float:
    # Gini of a non-negative vector. For attention, high means concentrated.
    x = values.detach().float().flatten().clamp_min(0)
    n = int(x.numel())
    if n <= 1:
        return 0.0
    total = float(x.sum().item())
    if total <= eps:
        return 0.0
    sorted_x = torch.sort(x).values
    idx = torch.arange(1, n + 1, dtype=sorted_x.dtype)
    gini = (2.0 * (idx * sorted_x).sum() / (n * sorted_x.sum())) - ((n + 1.0) / n)
    return float(gini.item())


def pairwise_mean_js(per_head: torch.Tensor) -> float:
    h = int(per_head.size(0))
    if h <= 1:
        return 0.0
    vals: List[float] = []
    for i in range(h):
        for j in range(i + 1, h):
            vals.append(float(js_pair(per_head[i], per_head[j]).item()))
    return float(sum(vals) / len(vals)) if vals else 0.0


def pairwise_mean_cosine_distance(per_head: torch.Tensor, eps: float = 1e-12) -> float:
    h = int(per_head.size(0))
    if h <= 1:
        return 0.0
    vals: List[float] = []
    norms = per_head.norm(dim=-1).clamp_min(eps)
    for i in range(h):
        for j in range(i + 1, h):
            cos = float((per_head[i] * per_head[j]).sum().item() / (float(norms[i].item()) * float(norms[j].item())))
            vals.append(1.0 - cos)
    return float(sum(vals) / len(vals)) if vals else 0.0


def topk_mass(p: torch.Tensor, k: int) -> float:
    if p.numel() == 0:
        return 0.0
    kk = min(int(k), int(p.numel()))
    return float(torch.topk(p, kk).values.sum().item())


def top_token_strings(tokens: Sequence[str], valid_positions: Sequence[int], att: torch.Tensor, k: int) -> str:
    if att.numel() == 0:
        return ""
    kk = min(int(k), int(att.numel()))
    vals, idxs = torch.topk(att, kk)
    chunks: List[str] = []
    for value, local_idx in zip(vals.tolist(), idxs.tolist()):
        pos = valid_positions[int(local_idx)]
        tok = token_display(tokens[pos])
        chunks.append(f"{tok}:{float(value):.4f}")
    return " | ".join(chunks)


def visible_token_string(tokens: Sequence[str], valid_positions: Sequence[int]) -> str:
    out: List[str] = []
    for pos in valid_positions:
        s = clean_token(tokens[pos])
        if not s:
            s = tokens[pos]
        out.append(s)
    return " ".join(out).replace("  ", " ").strip()


def batch_text_for_example(batch: Dict[str, Any], local_idx: int, tokenizer: Any, input_ids: torch.Tensor) -> str:
    for key in ["text", "texts", "sentence", "sentences"]:
        value = batch.get(key)
        if isinstance(value, (list, tuple)) and local_idx < len(value):
            return str(value[local_idx])
    try:
        return tokenizer.decode(input_ids[local_idx].tolist(), skip_special_tokens=True)
    except Exception:
        return ""


def attention_variation_metrics(
    avg_map: torch.Tensor,
    per_head_maps: torch.Tensor,
    valid_positions: Sequence[int],
    tokens: Sequence[str],
    top_k: int,
) -> Dict[str, Any]:
    h = int(per_head_maps.size(0))
    n = int(avg_map.numel())
    mean_template = avg_map.float()
    head_maps = per_head_maps.float()
    kl_to_mean = kl_div(head_maps, mean_template.unsqueeze(0))
    head_ent = entropy(head_maps)
    mean_ent = float(entropy(mean_template).item())
    max_ent = math.log(max(n, 1)) if n > 1 else 1.0
    head_top = torch.argmax(head_maps, dim=-1).detach().cpu().tolist() if n else []
    mean_top_idx = int(torch.argmax(mean_template).item()) if n else -1
    top_counter = Counter(int(x) for x in head_top)
    top_mode_idx, top_mode_count = top_counter.most_common(1)[0] if top_counter else (-1, 0)
    head_top_agreement_to_mean = (sum(1 for x in head_top if int(x) == mean_top_idx) / h) if h else 0.0
    head_top_mode_share = (top_mode_count / h) if h else 0.0
    head_top_unique = len(top_counter)
    expected_pos = 0.0
    if n > 1:
        ratios = torch.linspace(0.0, 1.0, steps=n, dtype=mean_template.dtype)
        expected_pos = float((mean_template * ratios).sum().item())

    # Interpretable token fields.
    mean_top_token = ""
    top_mode_token = ""
    if 0 <= mean_top_idx < n:
        mean_top_token = token_display(tokens[valid_positions[mean_top_idx]])
    if 0 <= top_mode_idx < n:
        top_mode_token = token_display(tokens[valid_positions[top_mode_idx]])

    return {
        "num_heads": h,
        "valid_tokens": n,
        "head_template_kl_mean": float(kl_to_mean.mean().item()) if h else 0.0,
        "head_template_kl_std": float(kl_to_mean.std(unbiased=False).item()) if h else 0.0,
        "head_template_kl_max": float(kl_to_mean.max().item()) if h else 0.0,
        "head_template_kl_min": float(kl_to_mean.min().item()) if h else 0.0,
        "head_template_js": float(kl_to_mean.mean().item()) if h else 0.0,
        "pairwise_head_js_mean": pairwise_mean_js(head_maps),
        "pairwise_head_cosine_distance_mean": pairwise_mean_cosine_distance(head_maps),
        "mean_attention_entropy": mean_ent,
        "mean_attention_norm_entropy": float(mean_ent / max_ent) if max_ent > 0 else 0.0,
        "head_entropy_mean": float(head_ent.mean().item()) if h else 0.0,
        "head_entropy_std": float(head_ent.std(unbiased=False).item()) if h else 0.0,
        "effective_tokens": float(math.exp(mean_ent)),
        "top1_mass": topk_mass(mean_template, 1),
        "top3_mass": topk_mass(mean_template, 3),
        "top5_mass": topk_mass(mean_template, 5),
        "attention_gini": gini_coefficient(mean_template),
        "expected_position_ratio": expected_pos,
        "top_position_ratio": float(mean_top_idx / max(n - 1, 1)) if n > 1 and mean_top_idx >= 0 else 0.0,
        "mean_top_token": mean_top_token,
        "head_top_mode_token": top_mode_token,
        "head_top_agreement_to_mean": float(head_top_agreement_to_mean),
        "head_top_mode_share": float(head_top_mode_share),
        "head_top_unique_tokens": int(head_top_unique),
        "top_tokens": top_token_strings(tokens, valid_positions, mean_template, top_k),
    }


def row_condition_flags(gold: bool, pred: bool) -> Dict[str, bool]:
    return {
        "all": True,
        "gold": gold,
        "pred": pred,
        "gold_or_pred": gold or pred,
        "gold_and_pred": gold and pred,
        "tp": gold and pred,
        "fp": (not gold) and pred,
        "fn": gold and (not pred),
        "tn": (not gold) and (not pred),
        "correct": gold == pred,
        "incorrect": gold != pred,
    }


def summarize_numeric(rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"maps": len(rows)}
    if not rows:
        for f in fields:
            out[f"{f}_mean"] = None
            out[f"{f}_std"] = None
            out[f"{f}_p10"] = None
            out[f"{f}_p50"] = None
            out[f"{f}_p90"] = None
        return out
    for f in fields:
        vals = sorted(float(r[f]) for r in rows if r.get(f) is not None)
        if not vals:
            out[f"{f}_mean"] = None
            out[f"{f}_std"] = None
            out[f"{f}_p10"] = None
            out[f"{f}_p50"] = None
            out[f"{f}_p90"] = None
            continue
        out[f"{f}_mean"] = float(sum(vals) / len(vals))
        out[f"{f}_std"] = float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0
        def quantile(q: float) -> float:
            if len(vals) == 1:
                return vals[0]
            pos = q * (len(vals) - 1)
            lo = int(math.floor(pos))
            hi = int(math.ceil(pos))
            if lo == hi:
                return vals[lo]
            frac = pos - lo
            return vals[lo] * (1.0 - frac) + vals[hi] * frac
        out[f"{f}_p10"] = float(quantile(0.10))
        out[f"{f}_p50"] = float(quantile(0.50))
        out[f"{f}_p90"] = float(quantile(0.90))
    return out


def make_summary_rows(
    per_map_rows: Sequence[Dict[str, Any]],
    emotion_names: Sequence[str],
    conditions: Sequence[str],
    numeric_fields: Sequence[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    by_condition: DefaultDict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    by_emotion: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in per_map_rows:
        emotion = str(row["emotion"])
        by_emotion[emotion].append(row)
        gold = bool(row["gold"])
        pred = bool(row["pred"])
        flags = row_condition_flags(gold, pred)
        for cond in conditions:
            if flags.get(cond, False):
                by_condition[(emotion, cond)].append(row)

    condition_rows: List[Dict[str, Any]] = []
    for emotion in emotion_names:
        for cond in conditions:
            rows = by_condition.get((emotion, cond), [])
            summary = summarize_numeric(rows, numeric_fields)
            summary.update({
                "emotion": emotion,
                "condition": cond,
                "gold_count": sum(1 for r in by_emotion.get(emotion, []) if bool(r["gold"])),
                "pred_count": sum(1 for r in by_emotion.get(emotion, []) if bool(r["pred"])),
                "tp": sum(1 for r in by_emotion.get(emotion, []) if bool(r["gold"]) and bool(r["pred"])),
                "fp": sum(1 for r in by_emotion.get(emotion, []) if (not bool(r["gold"])) and bool(r["pred"])),
                "fn": sum(1 for r in by_emotion.get(emotion, []) if bool(r["gold"]) and (not bool(r["pred"]))),
                "tn": sum(1 for r in by_emotion.get(emotion, []) if (not bool(r["gold"])) and (not bool(r["pred"]))),
            })
            condition_rows.append(summary)

    emotion_rows: List[Dict[str, Any]] = []
    for emotion in emotion_names:
        rows = by_emotion.get(emotion, [])
        summary = summarize_numeric(rows, numeric_fields)
        summary.update({
            "emotion": emotion,
            "gold_count": sum(1 for r in rows if bool(r["gold"])),
            "pred_count": sum(1 for r in rows if bool(r["pred"])),
            "tp": sum(1 for r in rows if bool(r["gold"]) and bool(r["pred"])),
            "fp": sum(1 for r in rows if (not bool(r["gold"])) and bool(r["pred"])),
            "fn": sum(1 for r in rows if bool(r["gold"]) and (not bool(r["pred"]))),
            "tn": sum(1 for r in rows if (not bool(r["gold"])) and (not bool(r["pred"]))),
        })
        emotion_rows.append(summary)
    return condition_rows, emotion_rows


def make_extreme_rows(
    per_map_rows: Sequence[Dict[str, Any]],
    emotion_names: Sequence[str],
    conditions: Sequence[str],
    metric: str,
    k: int,
) -> List[Dict[str, Any]]:
    grouped: DefaultDict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in per_map_rows:
        flags = row_condition_flags(bool(row["gold"]), bool(row["pred"]))
        for cond in conditions:
            if flags.get(cond, False):
                grouped[(str(row["emotion"]), cond)].append(row)
    out: List[Dict[str, Any]] = []
    for emotion in emotion_names:
        for cond in conditions:
            rows = grouped.get((emotion, cond), [])
            if not rows:
                continue
            sorted_rows = sorted(rows, key=lambda r: float(r.get(metric, 0.0)))
            selections = [("low", r) for r in sorted_rows[:k]] + [("high", r) for r in list(reversed(sorted_rows[-k:]))]
            seen: set[Tuple[str, int]] = set()
            rank_by_side = {"low": 0, "high": 0}
            for side, r in selections:
                key = (side, int(r["global_example_idx"]))
                if key in seen:
                    continue
                seen.add(key)
                rank_by_side[side] += 1
                keep = {
                    "emotion": emotion,
                    "condition": cond,
                    "side": side,
                    "rank": rank_by_side[side],
                    "global_example_idx": r.get("global_example_idx"),
                    "prob": r.get("prob"),
                    "threshold": r.get("threshold"),
                    "gold": r.get("gold"),
                    "pred": r.get("pred"),
                    "head_template_kl_mean": r.get("head_template_kl_mean"),
                    "pairwise_head_js_mean": r.get("pairwise_head_js_mean"),
                    "head_top_agreement_to_mean": r.get("head_top_agreement_to_mean"),
                    "mean_attention_norm_entropy": r.get("mean_attention_norm_entropy"),
                    "top1_mass": r.get("top1_mass"),
                    "top_tokens": r.get("top_tokens"),
                    "text": r.get("text"),
                    "visible_tokens": r.get("visible_tokens"),
                }
                out.append(keep)
    return out


def write_html_report(
    path: Path,
    summary: Dict[str, Any],
    condition_rows: Sequence[Dict[str, Any]],
    extreme_rows: Sequence[Dict[str, Any]],
    conditions_for_preview: Sequence[str],
) -> None:
    metric = "head_template_kl_mean"
    # Preview all emotions for selected conditions.
    cond_set = set(conditions_for_preview)
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>Attention template variation experiment</title>",
        "<style>",
        "body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;margin:24px;color:#111;line-height:1.35}",
        "table{border-collapse:collapse;width:100%;margin:12px 0 28px;font-size:13px}",
        "th,td{border:1px solid #e5e5e5;padding:5px 7px;text-align:left;vertical-align:top}",
        "th{background:#fafafa}.meta{color:#555}.num{text-align:right}.bad{color:#a00}.small{font-size:12px;color:#555}",
        "</style></head><body>",
        "<h1>Attention template variation experiment</h1>",
        f"<p class='meta'>Split: {html.escape(str(summary.get('split')))}; examples: {summary.get('examples_processed')}; maps: {summary.get('maps_processed')}.</p>",
        "<p>This analyzes latent-pooling attention. For each example × emotion, rows are pooling heads and columns are visible tokens. The main score is average KL from each head to the head-mean token template.</p>",
        "<h2>By emotion and condition</h2>",
        "<table><thead><tr><th>emotion</th><th>condition</th><th>maps</th><th>mean KL</th><th>p50 KL</th><th>p90 KL</th><th>mean top1</th><th>head top agreement</th><th>norm entropy</th><th>TP/FP/FN</th></tr></thead><tbody>",
    ]
    for row in condition_rows:
        if row.get("condition") not in cond_set:
            continue
        maps = int(row.get("maps", 0) or 0)
        if maps <= 0:
            continue
        parts.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('emotion')))}</td>"
            f"<td>{html.escape(str(row.get('condition')))}</td>"
            f"<td class='num'>{maps}</td>"
            f"<td class='num'>{float(row.get(metric + '_mean') or 0):.5f}</td>"
            f"<td class='num'>{float(row.get(metric + '_p50') or 0):.5f}</td>"
            f"<td class='num'>{float(row.get(metric + '_p90') or 0):.5f}</td>"
            f"<td class='num'>{float(row.get('top1_mass_mean') or 0):.3f}</td>"
            f"<td class='num'>{float(row.get('head_top_agreement_to_mean_mean') or 0):.3f}</td>"
            f"<td class='num'>{float(row.get('mean_attention_norm_entropy_mean') or 0):.3f}</td>"
            f"<td>{int(row.get('tp',0))}/{int(row.get('fp',0))}/{int(row.get('fn',0))}</td>"
            "</tr>"
        )
    parts.append("</tbody></table>")

    parts.append("<h2>Extreme examples</h2>")
    parts.append("<table><thead><tr><th>emotion</th><th>condition</th><th>side</th><th>KL</th><th>p/threshold</th><th>gold/pred</th><th>top tokens</th><th>text</th></tr></thead><tbody>")
    for row in extreme_rows[:400]:
        parts.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('emotion')))}</td>"
            f"<td>{html.escape(str(row.get('condition')))}</td>"
            f"<td>{html.escape(str(row.get('side')))}</td>"
            f"<td class='num'>{float(row.get('head_template_kl_mean') or 0):.5f}</td>"
            f"<td>p={float(row.get('prob') or 0):.3f}; thr={float(row.get('threshold') or 0):.3f}</td>"
            f"<td>gold={row.get('gold')}; pred={row.get('pred')}</td>"
            f"<td>{html.escape(str(row.get('top_tokens','')))}</td>"
            f"<td>{html.escape(str(row.get('text') or row.get('visible_tokens') or ''))}</td>"
            "</tr>"
        )
    parts.append("</tbody></table></body></html>")
    path.write_text("\n".join(parts), encoding="utf-8")


def run_experiment(args: argparse.Namespace) -> Dict[str, Any]:
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
    label_mapping, mapping_method = infer_label_factor_mapping(model, data.emotion_names, args.label_top_factors)
    eprint(f"[map] label_factor_mapping={mapping_method}")
    eprint(f"[experiment] split={args.split} examples=all emotions={len(data.emotion_names)}")

    loader = getattr(data, f"{args.split}_loader")
    per_map_rows: List[Dict[str, Any]] = []
    examples_processed = 0
    maps_processed = 0

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
            global_idx = examples_processed
            tokens = tokenizer.convert_ids_to_tokens(input_ids[local_idx].tolist())
            valid_positions = valid_positions_for_example(
                input_ids=input_ids[local_idx],
                attention_mask=attention_mask[local_idx],
                special_ids=special_ids,
                ignore_special_tokens=not args.include_special_tokens,
            )
            text = batch_text_for_example(batch, local_idx, tokenizer, input_ids)
            visible_tokens = visible_token_string(tokens, valid_positions)
            if not valid_positions:
                examples_processed += 1
                continue

            for label_idx, emotion in enumerate(data.emotion_names):
                gold_value = False
                if labels is not None:
                    gold_value = bool(float(labels[local_idx, label_idx].item()) >= args.positive_label_threshold)
                pred_value = bool(preds[local_idx, label_idx].item())
                prob = float(probs[local_idx, label_idx].item())
                threshold_value = float(threshold_values[label_idx])

                # Optional speed/output modes. Default is all maps as requested.
                if args.only_active and not (gold_value or pred_value):
                    continue

                avg_map, per_head_maps, mapped = label_attention_map(attn, local_idx, label_idx, label_mapping, valid_positions)
                metrics = attention_variation_metrics(avg_map, per_head_maps, valid_positions, tokens, args.top_k_tokens)
                flags = row_condition_flags(gold_value, pred_value)
                row: Dict[str, Any] = {
                    "global_example_idx": global_idx,
                    "batch_idx": batch_idx,
                    "local_idx": local_idx,
                    "emotion": emotion,
                    "label_idx": label_idx,
                    "gold": bool(gold_value),
                    "pred": bool(pred_value),
                    "correct": bool(gold_value == pred_value),
                    "condition": "tp" if gold_value and pred_value else "fp" if pred_value else "fn" if gold_value else "tn",
                    "prob": prob,
                    "threshold": threshold_value,
                    "mapped_factors": ",".join(str(x) for x in mapped),
                    "text": text,
                    "visible_tokens": visible_tokens,
                }
                for key in ["gold", "pred", "gold_or_pred", "gold_and_pred", "tp", "fp", "fn", "tn", "incorrect"]:
                    row[f"is_{key}"] = bool(flags.get(key, False))
                row.update(metrics)
                per_map_rows.append(row)
                maps_processed += 1

            examples_processed += 1

        if args.progress_every and (batch_idx + 1) % int(args.progress_every) == 0:
            eprint(f"[progress] batches={batch_idx + 1} examples={examples_processed} maps={maps_processed}")

    numeric_fields = [
        "head_template_kl_mean",
        "head_template_kl_std",
        "head_template_kl_max",
        "pairwise_head_js_mean",
        "pairwise_head_cosine_distance_mean",
        "mean_attention_entropy",
        "mean_attention_norm_entropy",
        "head_entropy_mean",
        "head_entropy_std",
        "effective_tokens",
        "top1_mass",
        "top3_mass",
        "top5_mass",
        "attention_gini",
        "expected_position_ratio",
        "top_position_ratio",
        "head_top_agreement_to_mean",
        "head_top_mode_share",
        "head_top_unique_tokens",
    ]
    condition_rows, emotion_rows = make_summary_rows(per_map_rows, data.emotion_names, conditions, numeric_fields)
    extreme_rows = make_extreme_rows(
        per_map_rows,
        data.emotion_names,
        conditions=[c for c in conditions if c in {"gold", "pred", "tp", "fp", "fn", "gold_or_pred"}],
        metric=args.extreme_metric,
        k=args.extreme_k,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_per_map_csv:
        write_csv(out_dir / "per_map_attention_variation.csv", per_map_rows)
    write_csv(out_dir / "condition_summary.csv", condition_rows)
    write_csv(out_dir / "emotion_summary.csv", emotion_rows)
    write_csv(out_dir / "extreme_examples.csv", extreme_rows)

    summary = {
        "checkpoint": str(checkpoint_path),
        "split": args.split,
        "output_dir": str(out_dir),
        "examples_processed": int(examples_processed),
        "maps_processed": int(maps_processed),
        "num_labels": len(data.emotion_names),
        "emotion_names": list(data.emotion_names),
        "conditions": conditions,
        "threshold_mode": args.threshold_mode,
        "thresholds": threshold_values,
        "mapping_method": mapping_method,
        "load_report": load_report,
        "only_active": bool(args.only_active),
        "notes": {
            "head_template_kl_mean": "Mean_h KL(A_h || mean_h A_h). This is Jensen-Shannon divergence across latent-pool heads in nats.",
            "small_values": "Heads follow a similar token-attention template.",
            "large_values": "Heads specialize differently across tokens.",
            "important_caveat": "This is latent-pooling head variation, not transformer query-token self-attention row variation.",
        },
        "rows": {
            "per_map_attention_variation": len(per_map_rows),
            "condition_summary": len(condition_rows),
            "emotion_summary": len(emotion_rows),
            "extreme_examples": len(extreme_rows),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_html_report(
        out_dir / "index.html",
        summary=summary,
        condition_rows=condition_rows,
        extreme_rows=extreme_rows,
        conditions_for_preview=args.html_conditions.split(","),
    )

    eprint(f"[done] examples_processed={examples_processed} maps_processed={maps_processed}")
    if not args.no_per_map_csv:
        eprint(f"[done] wrote {out_dir / 'per_map_attention_variation.csv'}")
    eprint(f"[done] wrote {out_dir / 'condition_summary.csv'}")
    eprint(f"[done] wrote {out_dir / 'emotion_summary.csv'}")
    eprint(f"[done] wrote {out_dir / 'extreme_examples.csv'}")
    eprint(f"[done] wrote {out_dir / 'index.html'}")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Measure latent-pooling attention template/head variation for every example x emotion.")
    p.add_argument("--checkpoint", required=True, help="Path to best_checkpoint.pt or final_checkpoint.pt.")
    p.add_argument("--config", default=None, help="Optional sidecar config.json. Auto-discovered beside checkpoint if omitted.")
    p.add_argument("--threshold-json", default=None, help="Optional base_threshold_metrics.json. Auto-discovered beside checkpoint if omitted.")
    p.add_argument("--split", choices=["threshold", "val", "test", "train"], default="test")
    p.add_argument("--output-dir", default="attention_template_variation")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-length", type=int, default=None)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--residual-scale", type=float, default=1.0)
    p.add_argument("--threshold-mode", choices=["global", "per_label"], default="per_label")
    p.add_argument("--positive-label-threshold", type=float, default=0.5)
    p.add_argument("--label-top-factors", type=int, default=2)
    p.add_argument("--conditions", default=",".join(DEFAULT_CONDITIONS), help="Comma list: all,gold,pred,gold_or_pred,gold_and_pred,tp,fp,fn,tn")
    p.add_argument("--top-k-tokens", type=int, default=8)
    p.add_argument("--include-special-tokens", action="store_true")
    p.add_argument("--max-examples", type=int, default=None, help="Debug cap on examples processed.")
    p.add_argument("--only-active", action="store_true", help="Only compute emotion maps where gold or predicted is true. Default computes all examples x all emotions.")
    p.add_argument("--no-per-map-csv", action="store_true", help="Skip the large per-map CSV and only write summaries.")
    p.add_argument("--progress-every", type=int, default=25)
    p.add_argument("--extreme-k", type=int, default=3, help="Lowest/highest examples per emotion/condition in extreme_examples.csv.")
    p.add_argument("--extreme-metric", default="head_template_kl_mean", choices=[
        "head_template_kl_mean",
        "pairwise_head_js_mean",
        "head_top_agreement_to_mean",
        "mean_attention_norm_entropy",
        "top1_mass",
        "attention_gini",
    ])
    p.add_argument("--html-conditions", default="gold,pred,tp,fp,fn", help="Comma list of condition summaries to preview in index.html.")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    run_experiment(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
