from __future__ import annotations

from ..utils.common import *
from ..config import PromptConfig
from .generation import compute_copy_loss_from_memory, generate_text_from_memory, summarize_copy_metrics
from ..training.losses import cross_covariance_penalty, emotion_decoder_sensitivity_loss, factorvae_tc_loss, residual_probe_loss, tc_latents_for_tc, transfer_strength_margin_loss, vector_probe_loss
from ..training.schedules import current_loss_weights
from ..schemas import ExperimentContext, TrainingState
from ..modeling.t5_utils import masked_mean
from .thresholds import format_threshold_for_logging, threshold_to_numpy, tune_thresholds
from .regression import continuous_regression_metrics, semeval_ei_reg_metrics
from .latent_diagnostics import compute_latent_diagnostics
from ..utils import release_memory

# Focused module: metrics.

def multilabel_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    threshold: Any,
    label_names: Sequence[str],
    is_regression: bool = False,
) -> Dict[str, Any]:
    probs_tensor = torch.sigmoid(logits.detach().cpu())
    labels_tensor = labels.detach().cpu()
    probs = probs_tensor.numpy()
    raw_targets = labels_tensor.numpy()
    
    result = {
        "logits": logits.detach().cpu(),
        "labels": labels_tensor,
        "probs": probs_tensor,
    }
    
    # For regression, return only continuous metrics
    if is_regression:
        regression = continuous_regression_metrics(
            predictions=probs,
            targets=raw_targets,
            label_names=label_names,
            high_threshold=None,
            prefix="intensity",
        )
        semeval = semeval_ei_reg_metrics(
            predictions=probs,
            targets=raw_targets,
            label_names=label_names,
            high_threshold=None,
        )
        result.update({**regression, **semeval})
        return result
    
    # For classification: binarize and compute classification metrics
    targets = (raw_targets >= 0.5).astype(int)
    threshold_arr = threshold_to_numpy(threshold, probs.shape[1])
    if threshold_arr.size == 1:
        threshold_value: Any = float(threshold_arr[0])
        threshold_mode = "global"
    else:
        threshold_value = threshold_arr.astype(float).tolist()
        threshold_mode = "per_label"
    
    preds = (probs >= threshold_arr.reshape(1, -1) if threshold_arr.size > 1 else probs >= float(threshold_arr[0])).astype(int)

    def safe_average_precision_micro(y_true: np.ndarray, y_score: np.ndarray) -> float:
        try:
            # Only works with binary targets
            if y_true.max() > 1 or y_true.min() < 0:
                return 0.0
            value = average_precision_score(y_true, y_score, average="micro")
            return float(value) if np.isfinite(value) else 0.0
        except ValueError:
            return 0.0

    def safe_lrap(y_true: np.ndarray, y_score: np.ndarray) -> float:
        try:
            # Only works with binary targets
            if y_true.max() > 1 or y_true.min() < 0:
                return 0.0
            value = label_ranking_average_precision_score(y_true, y_score)
            return float(value) if np.isfinite(value) else 0.0
        except ValueError:
            return 0.0

    regression = continuous_regression_metrics(
        predictions=probs,
        targets=raw_targets,
        label_names=label_names,
        high_threshold=0.5,
        prefix="intensity",
    )
    semeval = semeval_ei_reg_metrics(
        predictions=probs,
        targets=raw_targets,
        label_names=label_names,
        high_threshold=0.5,
    )

    result.update({
        "threshold": threshold_value,
        "threshold_mode": threshold_mode,
        "threshold_mean": float(threshold_arr.mean()),
        "threshold_min": float(threshold_arr.min()),
        "threshold_max": float(threshold_arr.max()),
        **regression,
        **semeval,
        "micro_f1": float(f1_score(targets, preds, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(targets, preds, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(targets, preds, average="weighted", zero_division=0)),
        "subset_accuracy": float(accuracy_score(targets, preds)),
        "hamming_acc": float(1.0 - hamming_loss(targets, preds)),
        "jaccard_micro": float(jaccard_score(targets, preds, average="micro", zero_division=0)),
        "average_precision_micro": safe_average_precision_micro(targets, probs),
        "label_ranking_average_precision": safe_lrap(targets, probs),
        "classification_report_text": classification_report(
            targets,
            preds,
            target_names=list(label_names),
            zero_division=0,
        ),
        "preds": torch.from_numpy(preds),
    })
    
    return result


def evaluate_loader(
    state: TrainingState,
    ctx: ExperimentContext,
    data_loader: DataLoader,
    desc: str,
    tune_threshold: bool,
    weights: Dict[str, float],
) -> Dict[str, Any]:
    model = state.model
    discriminator = state.discriminator
    vector_adversary = state.vector_adversary
    residual_adversary = state.residual_adversary
    model.eval()
    if vector_adversary is not None:
        vector_adversary.eval()
    if residual_adversary is not None:
        residual_adversary.eval()

    total_base_loss = 0.0
    total_total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    total_cls = 0.0
    total_tc = 0.0
    total_copy = 0.0
    total_vec_adv = 0.0
    total_residual_adv = 0.0
    total_sep = 0.0
    total_orth = 0.0
    total_transfer = 0.0
    total_edit_sensitivity = 0.0
    steps = 0

    all_logits = []
    all_labels = []
    all_scalar_summary = []
    all_vector_summary = []
    all_pooled_scalar_features = []
    all_residual_summary = []

    with torch.inference_mode():
        for batch in tqdm(data_loader, desc=desc, leave=False):
            input_ids = batch["input_ids"].to(ctx.device)
            attention_mask = batch["attention_mask"].to(ctx.device)
            labels = batch["labels"].to(ctx.device)
            texts = batch["texts"]

            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                sample_posterior=ctx.experiment_config.sample_posterior_eval,
                classification_weight=weights["classification_weight"],
                recon_weight=weights["recon_weight"],
                kl_weight=weights["kl_weight"],
                residual_scale=weights["residual_scale"],
            )

            tc_loss = out.base_loss.new_zeros(())
            if discriminator is not None and weights["tc_weight"] > 0.0:
                tc_inputs = tc_latents_for_tc(out.vae, attention_mask, ctx.loss_config.tc_subspace, ctx.loss_config.tc_mode)
                tc_loss = factorvae_tc_loss(discriminator, tc_inputs)

            copy_loss = out.base_loss.new_zeros(())
            if weights["copy_weight"] > 0.0:
                copy_loss = compute_copy_loss_from_memory(
                    model=model,
                    tokenizer=ctx.tokenizer,
                    decoder_memory=out.decoder_memory,
                    encoder_attention_mask=attention_mask,
                    texts=texts,
                    prompt_config=ctx.prompt_config,
                    device=ctx.device,
                )

            vector_summary = masked_mean(out.vae.vector_mu, attention_mask)
            scalar_summary = masked_mean(out.vae.scalar_mu, attention_mask)
            residual_summary = masked_mean(out.residual_memory, attention_mask)
            vec_adv_loss = out.base_loss.new_zeros(())
            if vector_adversary is not None:
                _, vec_adv_loss = vector_probe_loss(
                    vector_adversary=vector_adversary,
                    vector_summary=vector_summary,
                    labels=labels,
                    pos_weight=ctx.pos_weight,
                )
            residual_adv_loss = out.base_loss.new_zeros(())
            if residual_adversary is not None:
                _, residual_adv_loss = residual_probe_loss(
                    residual_adversary=residual_adversary,
                    residual_summary=residual_summary,
                    labels=labels,
                    pos_weight=ctx.pos_weight,
                )
            sep_loss = cross_covariance_penalty(scalar_summary, vector_summary)
            orth_loss = model.classifier_regularization_loss()
            transfer_loss = transfer_strength_margin_loss(
                logits=out.classification_logits,
                labels=labels,
                margin=ctx.loss_config.transfer_strength_margin,
            )
            edit_sensitivity_loss = out.base_loss.new_zeros(())
            if weights.get("edit_sensitivity_weight", 0.0) > 0.0:
                zero_scalar = torch.zeros_like(out.vae.scalar_z)
                null_pooled, _ = model.pooled_scalar_tensor_from_latent_parts(
                    scalar_latents=zero_scalar,
                    vector_latents=out.vae.vector_z,
                    t5_encoder_sequence=out.t5_encoder_sequence,
                    attention_mask=attention_mask,
                )
                _, _, decoder_memory_without_emotion = model.decode_from_latent_parts(
                    scalar_latents=zero_scalar,
                    vector_latents=out.vae.vector_z,
                    t5_encoder_sequence=out.t5_encoder_sequence,
                    pooled_scalar_tensor=null_pooled,
                    residual_scale=weights["residual_scale"],
                )
                edit_sensitivity_loss = emotion_decoder_sensitivity_loss(
                    decoder_memory=out.decoder_memory,
                    decoder_memory_without_emotion=decoder_memory_without_emotion,
                    attention_mask=attention_mask,
                    margin=ctx.loss_config.edit_sensitivity_margin,
                )

            total_loss = (
                out.base_loss
                + (weights["tc_weight"] * tc_loss)
                + (weights["copy_weight"] * copy_loss)
                + (weights["vector_adv_weight"] * vec_adv_loss)
                + (weights["residual_adv_weight"] * residual_adv_loss)
                + (weights["vector_sep_weight"] * sep_loss)
                + (weights["orthogonality_weight"] * orth_loss)
                + (weights["transfer_strength_weight"] * transfer_loss)
                + (weights.get("edit_sensitivity_weight", 0.0) * edit_sensitivity_loss)
            )

            all_logits.append(out.classification_logits.detach().cpu())
            all_labels.append(labels.detach().cpu())
            all_scalar_summary.append(scalar_summary.detach().cpu())
            all_vector_summary.append(vector_summary.detach().cpu())
            all_pooled_scalar_features.append(model.flatten_pooled_scalar_tensor(out.pooled_scalar_tensor).detach().cpu())
            all_residual_summary.append(residual_summary.detach().cpu())
            total_base_loss += float(out.base_loss.detach().cpu().item())
            total_total_loss += float(total_loss.detach().cpu().item())
            total_recon += float(out.loss_terms["reconstruction"].item())
            total_kl += float(out.loss_terms["kl"].item())
            total_cls += float(out.loss_terms["classification"].item())
            total_tc += float(tc_loss.detach().cpu().item())
            total_copy += float(copy_loss.detach().cpu().item())
            total_vec_adv += float(vec_adv_loss.detach().cpu().item())
            total_residual_adv += float(residual_adv_loss.detach().cpu().item())
            total_sep += float(sep_loss.detach().cpu().item())
            total_orth += float(orth_loss.detach().cpu().item())
            total_transfer += float(transfer_loss.detach().cpu().item())
            total_edit_sensitivity += float(edit_sensitivity_loss.detach().cpu().item())
            steps += 1

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    
    # Check if regression task from config flag
    is_regression = getattr(ctx.data_config, "is_regression", False)
    
    # For classification tasks, tune thresholds; for regression, skip thresholding
    if tune_threshold and not is_regression:
        threshold = tune_thresholds(
            logits=logits,
            labels=labels,
            num_points=ctx.experiment_config.threshold_grid_points,
            mode=ctx.experiment_config.threshold_mode,
            is_regression=False,
        )
    else:
        threshold = state.threshold if not is_regression else None

    metrics = multilabel_metrics(
        logits=logits,
        labels=labels,
        threshold=threshold,
        label_names=ctx.emotion_names,
        is_regression=is_regression,
    )

    if getattr(ctx.experiment_config, "latent_diagnostics_enabled", True):
        diagnostics = compute_latent_diagnostics(
            model=model,
            scalar_summary=torch.cat(all_scalar_summary, dim=0),
            vector_summary=torch.cat(all_vector_summary, dim=0),
            pooled_scalar_features=torch.cat(all_pooled_scalar_features, dim=0),
            residual_summary=torch.cat(all_residual_summary, dim=0),
            labels=labels,
            predictions=metrics["probs"],
            label_names=ctx.emotion_names,
            ridge_alpha=getattr(ctx.experiment_config, "latent_diagnostics_ridge_alpha", 1.0),
            active_variance_threshold=getattr(ctx.experiment_config, "latent_active_variance_threshold", 1e-4),
            max_samples=getattr(ctx.experiment_config, "latent_diagnostics_max_samples", 4096),
            seed=getattr(ctx.experiment_config, "seed", 42),
            probe_eval_fraction=getattr(ctx.experiment_config, "latent_diagnostics_probe_eval_fraction", 0.30),
            min_probe_train_examples=getattr(ctx.experiment_config, "latent_diagnostics_min_probe_train_examples", 64),
        )
        metrics["latent_diagnostics"] = diagnostics
        metrics["factor_power"] = diagnostics["factor_power"]
        metrics["emotion_meaning_split"] = diagnostics["emotion_meaning_split"]
        metrics["factor_dci_disentanglement"] = diagnostics["factor_power"]["dci"]["dci_disentanglement"]
        metrics["factor_dci_completeness"] = diagnostics["factor_power"]["dci"]["dci_completeness"]
        metrics["branch_dci_completeness"] = diagnostics["emotion_meaning_split"]["branch_group_dci"]["dci_completeness"]
        metrics["branch_mig"] = diagnostics["emotion_meaning_split"]["branch_group_mig"]["emotion_vs_meaning_mig_signed_macro"]
        metrics["emotion_branch_importance_share"] = diagnostics["emotion_meaning_split"]["emotion_branch_importance_share"]
        metrics["meaning_branch_emotion_leakage_share"] = diagnostics["emotion_meaning_split"]["meaning_branch_emotion_leakage_share"]
        metrics["factor_effective_num_factors"] = diagnostics["factor_power"]["effective_num_factors"]
        metrics["factor_active_scalar_factors"] = diagnostics["factor_power"]["active_scalar_factors"]
        metrics["emotion_meaning_separation_r2"] = diagnostics["emotion_meaning_split"]["emotion_meaning_separation_r2"]
        metrics["emotion_meaning_separation_pearson"] = diagnostics["emotion_meaning_split"]["emotion_meaning_separation_pearson"]
        metrics["emotion_leakage_vector_r2"] = diagnostics["emotion_meaning_split"]["emotion_leakage_vector_r2"]
        metrics["emotion_in_scalar_r2"] = diagnostics["emotion_meaning_split"]["emotion_in_scalar_r2"]
        metrics["scalar_vector_mean_abs_correlation"] = diagnostics["emotion_meaning_split"]["scalar_vector_mean_abs_correlation"]
        block_alignment = diagnostics["factor_power"].get("block_alignment")
        if block_alignment is not None:
            metrics["block_assigned_correlation_dominance"] = block_alignment["assigned_block_correlation_dominance"]
            metrics["block_assigned_importance_dominance"] = block_alignment["assigned_block_importance_dominance"]
            metrics["block_top_correlation_match_rate"] = block_alignment["top_correlation_block_match_rate"]
            metrics["scalar_factors_per_label"] = block_alignment["scalar_factors_per_label"]

    metrics.update(
        {
            "avg_base_loss": total_base_loss / max(steps, 1),
            "avg_total_loss": total_total_loss / max(steps, 1),
            "avg_recon": total_recon / max(steps, 1),
            "avg_kl": total_kl / max(steps, 1),
            "avg_cls": total_cls / max(steps, 1),
            "avg_tc": total_tc / max(steps, 1),
            "avg_copy": total_copy / max(steps, 1),
            "avg_vec_adv": total_vec_adv / max(steps, 1),
            "avg_residual_adv": total_residual_adv / max(steps, 1),
            "avg_sep": total_sep / max(steps, 1),
            "avg_orth": total_orth / max(steps, 1),
            "avg_transfer": total_transfer / max(steps, 1),
            "avg_edit_sensitivity": total_edit_sensitivity / max(steps, 1),
            "weighted_recon": weights["recon_weight"] * (total_recon / max(steps, 1)),
            "weighted_kl": weights["kl_weight"] * (total_kl / max(steps, 1)),
            "weighted_cls": weights["classification_weight"] * (total_cls / max(steps, 1)),
            "weighted_tc": weights["tc_weight"] * (total_tc / max(steps, 1)),
            "weighted_copy": weights["copy_weight"] * (total_copy / max(steps, 1)),
            "weighted_vec_adv": weights["vector_adv_weight"] * (total_vec_adv / max(steps, 1)),
            "weighted_residual_adv": weights["residual_adv_weight"] * (total_residual_adv / max(steps, 1)),
            "weighted_sep": weights["vector_sep_weight"] * (total_sep / max(steps, 1)),
            "weighted_orth": weights["orthogonality_weight"] * (total_orth / max(steps, 1)),
            "weighted_transfer": weights["transfer_strength_weight"] * (total_transfer / max(steps, 1)),
            "weighted_edit_sensitivity": weights.get("edit_sensitivity_weight", 0.0) * (total_edit_sensitivity / max(steps, 1)),
            "steps": steps,
        }
    )
    release_memory()
    return metrics


def summarize_eval_metrics(prefix: str, metrics: Dict[str, Any], num_labels: int) -> None:
    threshold_str = format_threshold_for_logging(metrics["threshold"], num_labels)
    print(
        f"{prefix}loss={metrics['avg_total_loss']:.4f} "
        f"raw(recon={metrics['avg_recon']:.4f}, kl={metrics['avg_kl']:.4f}, cls={metrics['avg_cls']:.4f}, tc={metrics['avg_tc']:.4f}, copy={metrics['avg_copy']:.4f}, vec_adv={metrics['avg_vec_adv']:.4f}, res_adv={metrics.get('avg_residual_adv', 0.0):.4f}, sep={metrics['avg_sep']:.4f}, orth={metrics['avg_orth']:.4f}, transfer={metrics['avg_transfer']:.4f}) "
        f"weighted(recon={metrics['weighted_recon']:.4f}, kl={metrics['weighted_kl']:.4f}, cls={metrics['weighted_cls']:.4f}, tc={metrics['weighted_tc']:.4f}, copy={metrics['weighted_copy']:.4f}, vec_adv={metrics['weighted_vec_adv']:.4f}, res_adv={metrics.get('weighted_residual_adv', 0.0):.4f}, sep={metrics['weighted_sep']:.4f}, orth={metrics['weighted_orth']:.4f}, transfer={metrics['weighted_transfer']:.4f}) "
        f"threshold={threshold_str} "
        f"micro_f1={metrics['micro_f1']:.4f} "
        f"macro_f1={metrics['macro_f1']:.4f} "
        f"weighted_f1={metrics['weighted_f1']:.4f} "
        f"subset_acc={metrics['subset_accuracy']:.4f} "
        f"hamming_acc={metrics['hamming_acc']:.4f} "
        f"jaccard_micro={metrics['jaccard_micro']:.4f} "
        f"ap_micro={metrics['average_precision_micro']:.4f} "
        f"lrap={metrics['label_ranking_average_precision']:.4f} "
        f"semval_pearson={metrics.get('semeval_ei_reg_official_score', 0.0):.4f} "
        f"factor_dci={metrics.get('factor_power', {}).get('dci', {}).get('dci_disentanglement', 0.0):.4f} "
        f"branch_dci={metrics.get('branch_dci_completeness', 0.0):.4f} "
        f"branch_mig={metrics.get('branch_mig', 0.0):.4f} "
        f"emo_share={metrics.get('emotion_branch_importance_share', 0.0):.4f} "
        f"leak_share={metrics.get('meaning_branch_emotion_leakage_share', 0.0):.4f} "
        f"split_r2={metrics.get('emotion_meaning_split', {}).get('emotion_meaning_separation_r2', 0.0):.4f}"
    )


def evaluate_prompt_candidates(
    state: TrainingState,
    ctx: ExperimentContext,
    texts: Sequence[str],
    prompt_candidates: Sequence[str],
    use_vae_memory: bool = True,
    max_new_tokens: int = 128,
) -> List[Dict[str, Any]]:
    model = state.model
    model.eval()

    results: List[Dict[str, Any]] = []
    for prompt_text in prompt_candidates:
        candidate_prompt_cfg = PromptConfig(
            use_prompt=bool(prompt_text),
            prompt_text=prompt_text,
            mask_prompt_loss=ctx.prompt_config.mask_prompt_loss,
            max_decoder_length=ctx.prompt_config.max_decoder_length,
        )
        generated_texts: List[str] = []

        for start_idx in range(0, len(texts), ctx.data_config.eval_batch_size):
            batch_texts = list(texts[start_idx : start_idx + ctx.data_config.eval_batch_size])
            encoded = ctx.tokenizer(
                batch_texts,
                max_length=ctx.data_config.max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(ctx.device)
            attention_mask = encoded["attention_mask"].to(ctx.device)

            with torch.no_grad():
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=None,
                    sample_posterior=ctx.experiment_config.sample_posterior_eval,
                    classification_weight=ctx.loss_config.classification_weight,
                    recon_weight=ctx.loss_config.recon_weight,
                    kl_weight=ctx.loss_config.kl_weight,
                    residual_scale=current_loss_weights(ctx.schedule_config.num_epochs, ctx)["residual_scale"],
                )
                memory = out.decoder_memory if use_vae_memory else out.t5_encoder_sequence
                batch_generated = generate_text_from_memory(
                    model=model,
                    tokenizer=ctx.tokenizer,
                    decoder_memory=memory,
                    encoder_attention_mask=attention_mask,
                    prompt_config=candidate_prompt_cfg,
                    max_new_tokens=max_new_tokens,
                    num_beams=1,
                    do_sample=False,
                )
                generated_texts.extend(batch_generated)

        metrics = summarize_copy_metrics(texts, generated_texts)
        results.append(
            {
                "prompt": prompt_text,
                "use_vae_memory": use_vae_memory,
                **metrics,
            }
        )

    return sorted(results, key=lambda row: (row["exact_match"], row["token_f1"], row["edit_similarity"]), reverse=True)
