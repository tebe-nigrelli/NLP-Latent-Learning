#!/usr/bin/env python3
"""Evaluate latent emotion translation fluency with GPT-2 perplexity only.

Run from the repository root, for example:

  uv run python src/emotion_latent_learning/evaluation/check_translation_gpt2_ppl.py

The script scans saved ablation checkpoints, generates target-emotion translations
using the repository latent-editing code, and reports only GPT-2 perplexity over
the generated translation text. It intentionally does not compute classifier
success, semantic similarity, reconstruction metrics, or any other quality metric.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import re
import sys
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from emotion_latent_learning.config import (
    DataConfig,
    ExperimentConfig,
    LossConfig,
    ModelConfig,
    PromptConfig,
    ScheduleConfig,
)
from emotion_latent_learning.evaluation.editing import (
    make_partial_desired_emotion_target,
    optimize_emotion_latents_for_target,
)
from emotion_latent_learning.evaluation.thresholds import restore_threshold
from emotion_latent_learning.training.checkpointing import load_compact_model_state_dict
from emotion_latent_learning.training.runtime import build_data_bundle, build_runtime, get_device
from emotion_latent_learning.training.schedules import current_loss_weights


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
DEFAULT_TARGET_LABELS = ["joy", "sadness", "anger", "gratitude", "neutral"]


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


def infer_model_type(run_dir: Path, model_config: ModelConfig, loss_config: LossConfig) -> str:
    parts = []
    name = str(run_dir).lower()
    for token in ["base", "latent", "adapted", "pair", "skip", "lora", "adv", "beta", "factorvae"]:
        if token in name:
            parts.append(token)
    parts.append(str(getattr(model_config, "pooling_mode", "unknown_pool")))
    parts.append(str(getattr(model_config, "classifier_mode", "unknown_classifier")))
    parts.append(f"s{getattr(model_config, 'num_scalar_factors', 'na')}")
    parts.append(f"v{getattr(model_config, 'vector_latent_dim', 'na')}")
    if float(getattr(loss_config, "tc_weight", 0.0)) > 0:
        parts.append(f"tc{getattr(loss_config, 'tc_weight')}")
    return "/".join(parts)


class GPT2Perplexity:
    def __init__(self, model_name: str, device: torch.device, max_length: int) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.model.eval()
        self.device = device
        self.max_length = int(max_length)

    @torch.no_grad()
    def __call__(self, text: str) -> float:
        if not text or not text.strip():
            return float("inf")
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        input_ids = encoded["input_ids"].to(self.device)
        if input_ids.numel() <= 1:
            return float("inf")
        attention_mask = encoded.get("attention_mask")
        attention_mask = attention_mask.to(self.device) if attention_mask is not None else None
        labels = input_ids.clone()
        if attention_mask is not None:
            labels = labels.masked_fill(attention_mask == 0, -100)
        out = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = float(out.loss.detach().cpu().item())
        if not math.isfinite(loss):
            return float("inf")
        return float(math.exp(min(loss, 20.0)))


def finite_mean(values: Sequence[float]) -> float:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    return float(sum(finite) / len(finite)) if finite else float("inf")


def finite_median(values: Sequence[float]) -> float:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    return float(median(finite)) if finite else float("inf")


def finite_std(values: Sequence[float]) -> float:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    if len(finite) <= 1:
        return 0.0
    mu = sum(finite) / len(finite)
    return float(math.sqrt(sum((v - mu) ** 2 for v in finite) / (len(finite) - 1)))


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys: List[str] = []
    seen = set()
    preferred = [
        "run_dir", "checkpoint", "model_type", "split", "target_labels", "gpt2_model",
        "gpt2_perplexity_mean", "gpt2_perplexity_median", "gpt2_perplexity_std", "num_examples",
        "pooling_mode", "classifier_mode", "num_scalar_factors", "vector_latent_dim",
        "use_skip_connection", "decoder_conditioning_mode", "use_lora", "lora_r", "lora_target_modules",
        "tc_weight", "copy_weight", "vector_adv_weight", "residual_adv_weight",
    ]
    for key in preferred:
        if any(key in row for row in rows) and key not in seen:
            keys.append(key)
            seen.add(key)
    for row in rows:
        for key in row.keys():
            if key not in seen:
                keys.append(key)
                seen.add(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def evaluate_checkpoint(
    checkpoint_path: Path,
    args: argparse.Namespace,
    device: torch.device,
    ppl: GPT2Perplexity,
    output_examples_dir: Path,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    eprint(f"[load] {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    data_config, prompt_config, model_config, loss_config, schedule_config, experiment_config = configs_from_checkpoint(checkpoint)

    data_config.eval_batch_size = 1
    data_config.num_workers = int(args.num_workers)
    experiment_config.output_dir = str(checkpoint_path.parent)
    experiment_config.sample_posterior_eval = False
    experiment_config.monitor_edit_method = str(args.edit_method)
    experiment_config.monitor_edit_steps = int(args.edit_steps)
    experiment_config.edit_decode_num_beams = int(args.num_beams)
    experiment_config.edit_target_mode = str(args.edit_target_mode)
    if hasattr(experiment_config, "edit_backprop_lr"):
        experiment_config.edit_backprop_lr = float(args.edit_lr)

    prompt_config.max_decoder_length = max(int(prompt_config.max_decoder_length), int(args.max_new_tokens))
    data_config.max_length = max(int(data_config.max_length), int(args.max_new_tokens))

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
    if checkpoint.get("threshold") is not None:
        runtime.state.threshold = restore_threshold(
            checkpoint["threshold"],
            num_labels=len(runtime.ctx.emotion_names),
            mode=runtime.ctx.experiment_config.threshold_mode,
        )
    runtime.state.model.eval()

    target_labels = list(args.target_labels)
    for label in target_labels:
        if label not in runtime.ctx.emotion_names:
            raise ValueError(f"Unknown target label {label!r}; available labels: {runtime.ctx.emotion_names}")

    loader = getattr(data, f"{args.split}_loader")
    weights = current_loss_weights(epoch=int(getattr(schedule_config, "num_epochs", 0)), ctx=runtime.ctx)
    residual_scale = float(args.residual_scale) if args.residual_scale is not None else float(weights.get("residual_scale", 0.0))

    run_dir = checkpoint_path.parent
    checkpoint_name = checkpoint_path.name
    model_type = infer_model_type(run_dir, model_config, loss_config)
    metadata = load_summary_metadata(run_dir)

    example_rows: List[Dict[str, Any]] = []
    perplexities: List[float] = []
    seen = 0

    for batch in loader:
        if seen >= args.max_examples:
            break
        input_ids = batch["input_ids"][:1].to(device)
        attention_mask = batch["attention_mask"][:1].to(device)
        source_text = str(batch["texts"][0])
        target_label = target_labels[seen % len(target_labels)]
        desired_target, desired_mask = make_partial_desired_emotion_target(
            label_names=runtime.ctx.emotion_names,
            positive_labels=[target_label],
            mode=str(args.edit_target_mode),
        )
        edit_result = optimize_emotion_latents_for_target(
            state=runtime.state,
            ctx=runtime.ctx,
            input_ids=input_ids,
            attention_mask=attention_mask,
            desired_target=desired_target,
            desired_mask=desired_mask,
            method=str(args.edit_method),
            steps=int(args.edit_steps),
            lr=float(args.edit_lr),
            emotion_l2_weight=float(args.emotion_l2_weight),
            residual_scale=residual_scale,
        )
        generated_text = str(edit_result.get("generated_text", [""])[0])
        gpt2_ppl = ppl(generated_text)
        perplexities.append(gpt2_ppl)
        example_rows.append({
            "run_dir": str(run_dir),
            "checkpoint": checkpoint_name,
            "model_type": model_type,
            "split": args.split,
            "example_index": seen,
            "target_label": target_label,
            "source_text": source_text,
            "translated_text": generated_text,
            "gpt2_perplexity": gpt2_ppl,
        })
        seen += 1

    summary: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "checkpoint": checkpoint_name,
        "model_type": model_type,
        "split": args.split,
        "target_labels": " ".join(target_labels),
        "gpt2_model": args.gpt2_model,
        "gpt2_perplexity_mean": finite_mean(perplexities),
        "gpt2_perplexity_median": finite_median(perplexities),
        "gpt2_perplexity_std": finite_std(perplexities),
        "num_examples": int(seen),
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
    }

    if args.save_examples:
        example_path = output_examples_dir / f"{safe_name(str(run_dir))}__{safe_name(checkpoint_name)}.csv"
        write_csv(example_path, example_rows)
        eprint(f"[write] {example_path}")

    del runtime
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary, example_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check latent emotion translation using GPT-2 perplexity only.")
    parser.add_argument("--artifact-roots", nargs="*", default=None, help="Artifact directories to scan. Defaults to known ablation roots.")
    parser.add_argument("--checkpoint-names", nargs="*", default=CHECKPOINT_NAMES_DEFAULT, help="Checkpoint filenames to evaluate.")
    parser.add_argument("--output-dir", default="translation_gpt2_ppl_artifacts", help="Directory for GPT-2 perplexity summaries.")
    parser.add_argument("--split", choices=["threshold", "val", "test", "train"], default="test", help="Dataset split/loader to translate.")
    parser.add_argument("--max-examples", type=int, default=32, help="Maximum examples per checkpoint.")
    parser.add_argument("--num-workers", type=int, default=0, help="Dataloader workers.")
    parser.add_argument("--target-labels", nargs="*", default=DEFAULT_TARGET_LABELS, help="Target emotion labels to cycle through.")
    parser.add_argument("--edit-target-mode", default="add_only", choices=["add_only", "rewrite"], help="Latent edit target mode.")
    parser.add_argument("--edit-method", default="backprop", choices=["backprop", "closed_form"], help="Latent edit method.")
    parser.add_argument("--edit-steps", type=int, default=24, help="Backprop edit steps per example.")
    parser.add_argument("--edit-lr", type=float, default=2e-2, help="Backprop edit learning rate.")
    parser.add_argument("--emotion-l2-weight", type=float, default=0.20, help="Backprop edit L2 weight.")
    parser.add_argument("--residual-scale", type=float, default=None, help="Override residual scale for translation generation.")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Generation max_new_tokens via data_config.max_length.")
    parser.add_argument("--num-beams", type=int, default=1, help="Generation beams used by editing decode.")
    parser.add_argument("--gpt2-model", default="gpt2", help="Causal LM used for perplexity.")
    parser.add_argument("--gpt2-max-length", type=int, default=256, help="Max tokens for GPT-2 perplexity scoring.")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="VAE/editing device.")
    parser.add_argument("--gpt2-device", default="auto", choices=["auto", "cuda", "cpu"], help="GPT-2 perplexity device.")
    parser.add_argument("--save-examples", action="store_true", help="Write per-checkpoint translations and GPT-2 perplexities.")
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

    if args.gpt2_device == "cpu":
        ppl_device = torch.device("cpu")
    elif args.gpt2_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--gpt2-device cuda requested but CUDA is unavailable.")
        ppl_device = torch.device("cuda")
    else:
        ppl_device = device

    output_dir = Path(args.output_dir)
    examples_dir = output_dir / "examples"
    output_dir.mkdir(parents=True, exist_ok=True)

    eprint(f"Found {len(checkpoints)} checkpoints under {len(roots)} roots.")
    eprint(f"VAE/edit device: {device}; GPT-2 PPL device: {ppl_device}")
    eprint(f"Loading GPT-2 perplexity model: {args.gpt2_model}")
    ppl = GPT2Perplexity(model_name=args.gpt2_model, device=ppl_device, max_length=args.gpt2_max_length)

    summaries: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    for idx, checkpoint_path in enumerate(checkpoints, start=1):
        eprint(f"[{idx}/{len(checkpoints)}] {checkpoint_path}")
        try:
            summary, _examples = evaluate_checkpoint(
                checkpoint_path=checkpoint_path,
                args=args,
                device=device,
                ppl=ppl,
                output_examples_dir=examples_dir,
            )
            summaries.append(summary)
            write_csv(output_dir / "translation_gpt2_ppl_summary_partial.csv", summaries)
        except Exception as exc:
            eprint(f"[error] {checkpoint_path}: {exc}")
            failures.append({"checkpoint": str(checkpoint_path), "error": repr(exc)})
            write_csv(output_dir / "translation_gpt2_ppl_failures_partial.csv", failures)
            continue

    write_csv(output_dir / "translation_gpt2_ppl_summary.csv", summaries)
    write_json(output_dir / "translation_gpt2_ppl_summary.json", summaries)
    if failures:
        write_csv(output_dir / "translation_gpt2_ppl_failures.csv", failures)
        write_json(output_dir / "translation_gpt2_ppl_failures.json", failures)

    eprint(f"[done] wrote {output_dir / 'translation_gpt2_ppl_summary.csv'}")
    if failures:
        eprint(f"[warn] {len(failures)} checkpoints failed; see translation_gpt2_ppl_failures.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
