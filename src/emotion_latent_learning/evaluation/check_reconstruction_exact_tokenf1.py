#!/usr/bin/env python3
"""Evaluate reconstruction from saved ablation checkpoints using exact_match and token_f1 only.

Run from the repository root with `uv run python scripts/check_reconstruction_all_models.py`.
The script scans artifact directories, loads each checkpoint using the configs saved in
that checkpoint, reconstructs held-out inputs, and writes aggregate metrics plus
per-example generations.

It evaluates two reconstruction modes by default:
  - full: uses the model's trained residual/skip scale when skip connections exist.
  - latent_only: forces residual_scale=0 so reconstruction must pass through the VAE bottleneck.

This is intentionally checkpoint-driven: no run-specific Python edits should be needed
for base VAE, pair-MLP, adapted-MLP, skip, adversarial, LoRA, or FactorVAE runs, as long
as the current repo code can instantiate the checkpoint's saved config.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from emotion_latent_learning.config import (
    DataConfig,
    ExperimentConfig,
    LossConfig,
    ModelConfig,
    PromptConfig,
    ScheduleConfig,
)
from emotion_latent_learning.evaluation.generation import (
    build_generation_prefix,
    decode_generated_batch,
    normalize_text_for_copy_metric,
)
from emotion_latent_learning.modeling.t5_utils import resolve_decoder_start_token_id_from_module
from emotion_latent_learning.training.checkpointing import load_compact_model_state_dict
from emotion_latent_learning.training.runtime import build_data_bundle, build_runtime, get_device


DEFAULT_ROOT_GLOBS = [
    "base_vae_attention_from_scratch_artifacts",
    "base_vae_joint_attention_from_scratch_artifacts",
    "latent_dim_ablation_artifacts",
    "adapted_mlp_56scalar_ablation_artifacts",
    "pair_mlp_56scalar_ablation_artifacts",
    "skip_connection_ablation_artifacts",
    "skip_decoder_lora_ablation_artifacts",
    "skip_adv_decoder_lora_pooling_ablation_artifacts",
    "skip_adv_beta_schedule_ablation_artifacts",
    "skip_adv_beta_schedule_ablation_artifacts*",
    "skip_decoder_lora_ablation_artifacts*",
    "skip_connection_ablation_artifacts*",
    "factorvae_tc01*",
]

CHECKPOINT_NAMES_DEFAULT = ["best_checkpoint.pt", "final_checkpoint.pt"]


def eprint(*args: Any) -> None:
    print(*args, file=sys.stderr, flush=True)


def safe_name(value: str, max_len: int = 180) -> str:
    value = re.sub(r"[^A-Za-z0-9_.=-]+", "_", value.strip())
    value = re.sub(r"_+", "_", value).strip("_")
    return value[:max_len] or "run"


def read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def dataclass_from_dict(cls: Any, payload: Optional[Dict[str, Any]]) -> Any:
    """Create a config dataclass while tolerating old/new checkpoint fields."""
    obj = cls()
    if not payload:
        return obj
    valid = {f.name for f in dataclasses.fields(cls)}
    for key, value in payload.items():
        if key not in valid:
            continue
        # Keep tuple-valued config fields compatible with JSON list payloads.
        default_value = getattr(obj, key, None)
        if isinstance(default_value, tuple) and isinstance(value, list):
            value = tuple(value)
        setattr(obj, key, value)
    return obj


def configs_from_checkpoint(checkpoint: Dict[str, Any]) -> Tuple[DataConfig, PromptConfig, ModelConfig, LossConfig, ScheduleConfig, ExperimentConfig]:
    configs = checkpoint.get("configs") or {}
    return (
        dataclass_from_dict(DataConfig, configs.get("data_config")),
        dataclass_from_dict(PromptConfig, configs.get("prompt_config")),
        dataclass_from_dict(ModelConfig, configs.get("model_config")),
        dataclass_from_dict(LossConfig, configs.get("loss_config")),
        dataclass_from_dict(ScheduleConfig, configs.get("schedule_config")),
        dataclass_from_dict(ExperimentConfig, configs.get("experiment_config")),
    )


def expand_roots(root_globs: Sequence[str]) -> List[Path]:
    roots: List[Path] = []
    seen = set()
    for pattern in root_globs:
        matches = sorted(Path(".").glob(pattern))
        if not matches and Path(pattern).exists():
            matches = [Path(pattern)]
        for path in matches:
            if path.is_dir():
                resolved = str(path.resolve())
                if resolved not in seen:
                    seen.add(resolved)
                    roots.append(path)
    return roots


def find_checkpoints(roots: Sequence[Path], checkpoint_names: Sequence[str]) -> List[Path]:
    checkpoints: List[Path] = []
    seen = set()
    for root in roots:
        for name in checkpoint_names:
            for path in root.rglob(name):
                if path.is_file():
                    resolved = str(path.resolve())
                    if resolved not in seen:
                        seen.add(resolved)
                        checkpoints.append(path)
    return sorted(checkpoints)


def normalize_lower(text: str) -> str:
    return normalize_text_for_copy_metric(text).lower()


def token_f1(reference_text: str, generated_text: str) -> float:
    ref_tokens = normalize_lower(reference_text).split()
    gen_tokens = normalize_lower(generated_text).split()
    if not ref_tokens and not gen_tokens:
        return 1.0
    if not ref_tokens or not gen_tokens:
        return 0.0
    ref_counter = Counter(ref_tokens)
    gen_counter = Counter(gen_tokens)
    overlap = sum((ref_counter & gen_counter).values())
    if overlap <= 0:
        return 0.0
    precision = overlap / max(len(gen_tokens), 1)
    recall = overlap / max(len(ref_tokens), 1)
    return float(2.0 * precision * recall / max(precision + recall, 1e-12))


def row_metrics(reference: str, generated: str, tokenized_reference: str) -> Dict[str, float]:
    """Return only the two requested reconstruction metrics.

    exact_match is computed after the repository's copy-metric text normalization.
    token_f1 is computed from lowercased normalized whitespace-token overlap.
    tokenized_reference is accepted for API compatibility but is not used.
    """
    del tokenized_reference
    ref_norm = normalize_text_for_copy_metric(reference)
    gen_norm = normalize_text_for_copy_metric(generated)
    return {
        "exact_match": float(ref_norm == gen_norm),
        "token_f1": token_f1(ref_norm, gen_norm),
    }


def aggregate(rows: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    if not rows:
        return {}
    metric_keys = ["exact_match", "token_f1"]
    out: Dict[str, float] = {}
    for key in metric_keys:
        vals = [float(row[key]) for row in rows if key in row]
        out[key] = float(sum(vals) / len(vals)) if vals else 0.0
    out["num_examples"] = float(len(rows))
    return out

def tokenized_reference_texts(tokenizer: Any, input_ids: torch.Tensor) -> List[str]:
    ids = input_ids.detach().cpu()
    return tokenizer.batch_decode(ids, skip_special_tokens=True)


def generate_from_memory(
    model: Any,
    tokenizer: Any,
    decoder_memory: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_config: PromptConfig,
    max_new_tokens: int,
    num_beams: int,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
) -> List[str]:
    decoder_prefix, prompt_prefix_len = build_generation_prefix(
        tokenizer=tokenizer,
        batch_size=decoder_memory.size(0),
        prompt_config=prompt_config,
        device=decoder_memory.device,
    )
    generated_ids = model.t5_decoder.generate_from_memory(
        encoder_hidden_states=decoder_memory,
        encoder_attention_mask=attention_mask,
        decoder_input_ids=decoder_prefix,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        do_sample=False,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
    )
    return decode_generated_batch(tokenizer=tokenizer, generated_ids=generated_ids, prompt_prefix_len=prompt_prefix_len)


def residual_scale_for_mode(mode: str, model_config: ModelConfig, schedule_config: ScheduleConfig) -> float:
    if mode == "latent_only":
        return 0.0
    if mode == "skip_1":
        return 1.0
    if mode == "skip_0p1":
        return 0.1
    if mode == "full":
        if not bool(getattr(model_config, "use_skip_connection", False)):
            return 0.0
        return float(getattr(schedule_config, "residual_scale_end", 1.0))
    try:
        return float(mode)
    except ValueError as exc:
        raise ValueError(f"Unknown reconstruction mode {mode!r}") from exc


def infer_model_type(run_dir: Path, model_config: ModelConfig, loss_config: LossConfig) -> str:
    parts = []
    name = run_dir.name
    for token in ["base", "latent", "adapted", "pair", "skip", "lora", "adv", "beta", "factorvae"]:
        if token in name.lower() or token in str(run_dir.parent).lower():
            parts.append(token)
    parts.append(str(getattr(model_config, "pooling_mode", "unknown_pool")))
    parts.append(str(getattr(model_config, "classifier_mode", "unknown_classifier")))
    parts.append(f"s{getattr(model_config, 'num_scalar_factors', 'na')}")
    parts.append(f"v{getattr(model_config, 'vector_latent_dim', 'na')}")
    if float(getattr(loss_config, "tc_weight", 0.0)) > 0:
        parts.append(f"tc{getattr(loss_config, 'tc_weight')}")
    return "/".join(parts)


def load_summary_metadata(run_dir: Path) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    for filename in ["ablation_result_summary.json", "base_train_eval_summary.json", "base_eval_summary.json"]:
        candidate = run_dir / filename
        if candidate.exists():
            payload = read_json(candidate)
            for key, value in payload.items():
                if isinstance(value, (str, int, float, bool)) or value is None:
                    metadata[f"summary_{key}"] = value
            break
    return metadata


def evaluate_checkpoint(
    checkpoint_path: Path,
    args: argparse.Namespace,
    device: torch.device,
    output_examples_dir: Path,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    eprint(f"[load] {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    data_config, prompt_config, model_config, loss_config, schedule_config, experiment_config = configs_from_checkpoint(checkpoint)

    # Make reconstruction evaluation cheap and deterministic without mutating semantic config.
    data_config.eval_batch_size = int(args.batch_size)
    data_config.num_workers = int(args.num_workers)
    experiment_config.output_dir = str(checkpoint_path.parent)
    experiment_config.sample_posterior_eval = False
    prompt_config.use_prompt = bool(args.use_prompt)
    if args.prompt_text is not None:
        prompt_config.prompt_text = args.prompt_text
    prompt_config.max_decoder_length = max(int(prompt_config.max_decoder_length), int(args.max_new_tokens))

    data = build_data_bundle(
        data_config=data_config,
        model_config=model_config,
        experiment_config=experiment_config,
    )
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

    load_compact_model_state_dict(runtime.state.model, checkpoint["model_state_dict"])
    model = runtime.state.model
    model.eval()

    split_name = args.split
    loader = getattr(data, f"{split_name}_loader")
    run_dir = checkpoint_path.parent
    checkpoint_name = checkpoint_path.name
    model_type = infer_model_type(run_dir, model_config, loss_config)
    metadata = load_summary_metadata(run_dir)

    all_aggregate_rows: List[Dict[str, Any]] = []
    all_example_rows: List[Dict[str, Any]] = []

    for mode in args.modes:
        residual_scale = residual_scale_for_mode(mode, model_config, schedule_config)
        eprint(f"[eval] {checkpoint_path} mode={mode} residual_scale={residual_scale}")
        mode_rows: List[Dict[str, Any]] = []
        seen = 0
        with torch.no_grad():
            for batch in loader:
                if seen >= args.max_examples:
                    break
                remaining = args.max_examples - seen
                texts = list(batch["texts"][:remaining])
                input_ids = batch["input_ids"][:remaining].to(device)
                attention_mask = batch["attention_mask"][:remaining].to(device)
                labels = batch.get("labels")
                labels = labels[:remaining].to(device) if labels is not None else None

                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=None,
                    sample_posterior=False,
                    classification_weight=0.0,
                    recon_weight=0.0,
                    kl_weight=0.0,
                    residual_scale=residual_scale,
                )
                generated = generate_from_memory(
                    model=model,
                    tokenizer=data.tokenizer,
                    decoder_memory=out.decoder_memory,
                    attention_mask=attention_mask,
                    prompt_config=prompt_config,
                    max_new_tokens=args.max_new_tokens,
                    num_beams=args.num_beams,
                    repetition_penalty=args.repetition_penalty,
                    no_repeat_ngram_size=args.no_repeat_ngram_size,
                )
                tokenized_refs = tokenized_reference_texts(data.tokenizer, input_ids)

                for local_idx, (text, tok_ref, gen) in enumerate(zip(texts, tokenized_refs, generated)):
                    metrics = row_metrics(text, gen, tok_ref)
                    example_row: Dict[str, Any] = {
                        "artifact_root": str(run_dir.parents[0]) if len(run_dir.parents) else str(run_dir),
                        "run_dir": str(run_dir),
                        "checkpoint": checkpoint_name,
                        "model_type": model_type,
                        "mode": mode,
                        "residual_scale": residual_scale,
                        "example_index": seen + local_idx,
                        "input_text": text,
                        "tokenized_input_text": tok_ref,
                        "generated_text": gen,
                        **metrics,
                    }
                    mode_rows.append(example_row)
                seen += len(texts)

        agg = aggregate(mode_rows)
        aggregate_row: Dict[str, Any] = {
            "run_dir": str(run_dir),
            "checkpoint": checkpoint_name,
            "model_type": model_type,
            "mode": mode,
            "split": split_name,
            "residual_scale": residual_scale,
            "epoch": checkpoint.get("epoch"),
            "step": checkpoint.get("step"),
            "model_name": getattr(model_config, "model_name", None),
            "pooling_mode": getattr(model_config, "pooling_mode", None),
            "classifier_mode": getattr(model_config, "classifier_mode", None),
            "classifier_parameterization": getattr(model_config, "classifier_parameterization", None),
            "num_scalar_factors": getattr(model_config, "num_scalar_factors", None),
            "vector_latent_dim": getattr(model_config, "vector_latent_dim", None),
            "use_skip_connection": getattr(model_config, "use_skip_connection", None),
            "decoder_conditioning_mode": getattr(model_config, "decoder_conditioning_mode", None),
            "use_lora": getattr(model_config, "use_lora", None),
            "lora_r": getattr(model_config, "lora_r", None),
            "lora_target_modules": " ".join(getattr(model_config, "lora_target_modules", None) or []),
            "tc_weight": getattr(loss_config, "tc_weight", None),
            "copy_weight": getattr(loss_config, "copy_weight", None),
            "vector_adv_weight": getattr(loss_config, "vector_adv_weight", None),
            "residual_adv_weight": getattr(loss_config, "residual_adv_weight", None),
            **metadata,
            **agg,
        }
        all_aggregate_rows.append(aggregate_row)
        all_example_rows.extend(mode_rows)

        if args.save_examples:
            rel = f"{safe_name(str(run_dir))}__{safe_name(checkpoint_name)}__{safe_name(mode)}.csv"
            example_path = output_examples_dir / rel
            write_csv(example_path, mode_rows)
            eprint(f"[write] {example_path}")

    del runtime
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return all_aggregate_rows, all_example_rows


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys: List[str] = []
    seen = set()
    preferred = [
        "run_dir", "checkpoint", "model_type", "mode", "split", "residual_scale",
        "exact_match", "token_f1", "num_examples",
        "pooling_mode", "classifier_mode", "num_scalar_factors", "vector_latent_dim",
        "use_skip_connection", "decoder_conditioning_mode", "use_lora", "lora_r",
        "lora_target_modules", "tc_weight", "copy_weight", "vector_adv_weight", "residual_adv_weight",
    ]
    for key in preferred:
        if any(key in row for row in rows) and key not in seen:
            keys.append(key); seen.add(key)
    for row in rows:
        for key in row.keys():
            if key not in seen:
                keys.append(key); seen.add(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check reconstruction from saved VAE/FactorVAE checkpoints using exact_match and token_f1 only.")
    parser.add_argument("--artifact-roots", nargs="*", default=None, help="Artifact directories to scan. Defaults to known ablation roots.")
    parser.add_argument("--checkpoint-names", nargs="*", default=CHECKPOINT_NAMES_DEFAULT, help="Checkpoint filenames to evaluate.")
    parser.add_argument("--output-dir", default="reconstruction_exact_tokenf1_artifacts", help="Directory for reconstruction summaries.")
    parser.add_argument("--split", choices=["threshold", "val", "test", "train"], default="test", help="Dataset split/loader to reconstruct.")
    parser.add_argument("--max-examples", type=int, default=128, help="Maximum examples per checkpoint/mode.")
    parser.add_argument("--batch-size", type=int, default=8, help="Evaluation batch size.")
    parser.add_argument("--num-workers", type=int, default=0, help="Dataloader workers.")
    parser.add_argument("--modes", nargs="*", default=["full", "latent_only"], help="Reconstruction modes: full, latent_only, skip_1, skip_0p1, or numeric residual_scale.")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Generation max_new_tokens.")
    parser.add_argument("--num-beams", type=int, default=1, help="Greedy when 1; beam search otherwise.")
    parser.add_argument("--repetition-penalty", type=float, default=1.0, help="Use 1.0 for faithful reconstruction checks.")
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0, help="Use 0 for faithful reconstruction checks.")
    parser.add_argument("--use-prompt", action="store_true", help="Use prompt_config.prompt_text as decoder prefix.")
    parser.add_argument("--prompt-text", default=None, help="Override prompt text if --use-prompt is active.")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="Evaluation device.")
    parser.add_argument("--save-examples", action="store_true", help="Write per-checkpoint example CSV files.")
    parser.add_argument("--limit-checkpoints", type=int, default=0, help="Debug: evaluate only first N checkpoints.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root_globs = args.artifact_roots if args.artifact_roots else DEFAULT_ROOT_GLOBS
    roots = expand_roots(root_globs)
    if not roots:
        eprint("No artifact roots found. Pass --artifact-roots explicitly.")
        return 2

    checkpoints = find_checkpoints(roots, args.checkpoint_names)
    if args.limit_checkpoints and args.limit_checkpoints > 0:
        checkpoints = checkpoints[: args.limit_checkpoints]
    if not checkpoints:
        eprint("No checkpoints found under:", ", ".join(str(r) for r in roots))
        return 2

    device = torch.device("cpu") if args.device == "cpu" else get_device(prefer_cuda=(args.device != "cpu"))
    if args.device == "cuda" and device.type != "cuda":
        raise RuntimeError("--device cuda requested but CUDA is unavailable.")

    output_dir = Path(args.output_dir)
    examples_dir = output_dir / "examples"
    output_dir.mkdir(parents=True, exist_ok=True)

    eprint(f"Found {len(checkpoints)} checkpoints under {len(roots)} roots.")
    eprint(f"Device: {device}")

    aggregate_rows: List[Dict[str, Any]] = []
    failed_rows: List[Dict[str, Any]] = []

    for idx, checkpoint_path in enumerate(checkpoints, start=1):
        eprint(f"[{idx}/{len(checkpoints)}] {checkpoint_path}")
        try:
            rows, _examples = evaluate_checkpoint(
                checkpoint_path=checkpoint_path,
                args=args,
                device=device,
                output_examples_dir=examples_dir,
            )
            aggregate_rows.extend(rows)
            write_csv(output_dir / "reconstruction_summary_partial.csv", aggregate_rows)
        except Exception as exc:
            eprint(f"[error] {checkpoint_path}: {exc}")
            failed_rows.append({"checkpoint": str(checkpoint_path), "error": repr(exc)})
            write_csv(output_dir / "reconstruction_failures_partial.csv", failed_rows)
            continue

    write_csv(output_dir / "reconstruction_summary.csv", aggregate_rows)
    write_json(output_dir / "reconstruction_summary.json", aggregate_rows)
    if failed_rows:
        write_csv(output_dir / "reconstruction_failures.csv", failed_rows)
        write_json(output_dir / "reconstruction_failures.json", failed_rows)

    eprint(f"[done] wrote {output_dir / 'reconstruction_summary.csv'}")
    if failed_rows:
        eprint(f"[warn] {len(failed_rows)} checkpoints failed; see reconstruction_failures.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
