from __future__ import annotations

from ..utils.common import *
from .generation import generate_text_from_memory
from ..modeling import T5FactorVAEModel
from ..schemas import ExperimentContext, TrainingState
from .thresholds import threshold_to_tensor, threshold_to_numpy

# Focused module: editing.

def make_partial_desired_emotion_target(
    label_names: Sequence[str],
    positive_labels: Sequence[str],
    mode: str = "rewrite",
) -> Tuple[torch.Tensor, torch.Tensor]:
    target = torch.zeros(len(label_names), dtype=torch.float32)
    name_to_idx = {name: idx for idx, name in enumerate(label_names)}
    for name in positive_labels:
        if name not in name_to_idx:
            raise ValueError(f"Unknown emotion label: {name}")
        idx = name_to_idx[name]
        target[idx] = 1.0

    if mode == "rewrite":
        mask = torch.ones(len(label_names), dtype=torch.float32)
    elif mode == "add_only":
        mask = target.clone()
    else:
        raise ValueError(f"Unsupported edit_target_mode={mode!r}")
    return target, mask


def masked_bce_with_logits(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    raw = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    denom = mask.sum().clamp_min(1.0)
    return (raw * mask).sum() / denom


def build_target_logits(
    current_logits: torch.Tensor,
    desired_target: torch.Tensor,
    desired_mask: torch.Tensor,
    margin: float,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Build target logits for editing. Uses threshold to binarize target values."""
    target_logits = current_logits.clone()
    target_logits = torch.where(
        desired_mask.bool(),
        torch.where(desired_target > threshold, torch.full_like(target_logits, margin), torch.full_like(target_logits, -margin)),
        target_logits,
    )
    return target_logits


def solve_min_norm_linear_edit(
    weight: torch.Tensor,
    delta_logits: torch.Tensor,
    ridge: float,
) -> torch.Tensor:
    if delta_logits.ndim == 1:
        delta_logits = delta_logits.unsqueeze(0)
    batch, out_dim = delta_logits.shape
    solve_dtype = torch.float32
    weight_solve = weight.to(dtype=solve_dtype)
    delta_logits_solve = delta_logits.to(dtype=solve_dtype)
    gram = weight_solve @ weight_solve.transpose(0, 1)
    eye = torch.eye(out_dim, device=weight.device, dtype=solve_dtype)
    solve_mat = gram + (ridge * eye)
    edits = []
    for b in range(batch):
        rhs = delta_logits_solve[b]
        coeff = torch.linalg.solve(solve_mat, rhs)
        edits.append(weight_solve.transpose(0, 1) @ coeff)
    return torch.stack(edits, dim=0).to(dtype=weight.dtype)


def jacobian_rows(outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    if outputs.ndim != 2:
        raise ValueError("jacobian_rows expects outputs to be batched rank-2 tensors.")
    if outputs.size(0) != 1 or inputs.size(0) != 1:
        raise ValueError("jacobian_rows currently supports batch size 1.")

    input_shape = tuple(inputs.shape[1:])
    input_dim = int(inputs[0].numel())
    rows = []
    for idx in range(outputs.size(1)):
        grad = torch.autograd.grad(
            outputs[0, idx],
            inputs,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )[0]
        if grad is None:
            grad_row = torch.zeros(input_dim, device=outputs.device, dtype=outputs.dtype)
        else:
            grad_row = grad[0].reshape(-1)
        rows.append(grad_row)
    return torch.stack(rows, dim=0)


def classifier_feature_edit(
    model: T5FactorVAEModel,
    current_features: torch.Tensor,
    target_logits: torch.Tensor,
    ridge: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    current_logits = model.classification_logits_from_features(current_features)
    delta_logits = target_logits - current_logits
    if model.classifier_parameterization == "orthogonal":
        weight, _ = model.classifier_linear_weight_and_bias()
        delta_features = solve_min_norm_linear_edit(weight=weight, delta_logits=delta_logits, ridge=ridge)
        return current_logits.detach(), delta_logits.detach(), delta_features.detach()

    editable_features = current_features.detach().clone().requires_grad_(True)
    logits = model.classification_logits_from_features(editable_features)
    jac = jacobian_rows(logits, editable_features)
    delta_features = solve_min_norm_linear_edit(weight=jac, delta_logits=(target_logits - logits).detach(), ridge=ridge)
    return logits.detach(), (target_logits - logits).detach(), delta_features.detach()


def pooled_feature_preimage_from_scalar_latents(
    model: T5FactorVAEModel,
    scalar_sequence: torch.Tensor,
    vector_sequence: torch.Tensor,
    t5_encoder_sequence: torch.Tensor,
    attention_mask: torch.Tensor,
    pooled_delta_features: torch.Tensor,
    ridge: float,
) -> torch.Tensor:
    edited = []
    for batch_idx in range(scalar_sequence.size(0)):
        scalar_single = scalar_sequence[batch_idx:batch_idx + 1].detach().clone().requires_grad_(True)
        vector_single = vector_sequence[batch_idx:batch_idx + 1].detach()
        encoder_single = t5_encoder_sequence[batch_idx:batch_idx + 1].detach()
        mask_single = attention_mask[batch_idx:batch_idx + 1]
        pooled_single, _ = model.pooled_scalar_tensor_from_latent_parts(
            scalar_latents=scalar_single,
            vector_latents=vector_single,
            t5_encoder_sequence=encoder_single,
            attention_mask=mask_single,
        )
        features_single = model.flatten_pooled_scalar_tensor(pooled_single)
        jac = jacobian_rows(features_single, scalar_single)
        delta_single = solve_min_norm_linear_edit(
            weight=jac,
            delta_logits=pooled_delta_features[batch_idx:batch_idx + 1],
            ridge=ridge,
        )
        edited.append((scalar_single.detach().view(1, -1) + delta_single).view_as(scalar_single))
    return torch.cat(edited, dim=0)


def pooled_to_scalar_preimage(
    attention_weights: torch.Tensor,
    scalar_sequence: torch.Tensor,
    pooled_delta: torch.Tensor,
    ridge: float,
) -> torch.Tensor:
    edited_scalar = scalar_sequence.clone()
    batch_size, seq_len, num_labels = scalar_sequence.shape
    num_heads = pooled_delta.size(-1)
    head_eye = torch.eye(num_heads, device=scalar_sequence.device, dtype=scalar_sequence.dtype)

    for b in range(batch_size):
        for label_idx in range(num_labels):
            A = attention_weights[b, label_idx, :, :].to(dtype=torch.float32)
            delta_p = pooled_delta[b, label_idx, :].to(dtype=torch.float32)
            gram = A @ A.transpose(0, 1)
            coeff = torch.linalg.solve(gram + (ridge * head_eye.to(dtype=torch.float32)), delta_p)
            delta_tokens = A.transpose(0, 1) @ coeff
            edited_scalar[b, :, label_idx] = edited_scalar[b, :, label_idx] + delta_tokens
    return edited_scalar


def optimize_emotion_latents_for_target_backprop(
    state: TrainingState,
    ctx: ExperimentContext,
    input_ids: torch.LongTensor,
    attention_mask: torch.Tensor,
    desired_target: torch.Tensor,
    desired_mask: torch.Tensor,
    steps: int = 80,
    lr: float = 3e-2,
    emotion_l2_weight: float = 5e-3,
    residual_scale: float = 1.0,
) -> Dict[str, Any]:
    model = state.model
    model.eval()

    input_ids = input_ids.to(ctx.device)
    attention_mask = attention_mask.to(ctx.device)
    desired_target = desired_target.to(ctx.device).float().unsqueeze(0)
    desired_mask = desired_mask.to(ctx.device).float().unsqueeze(0)

    with torch.no_grad():
        t5_encoder_sequence, _, _, vae_out = model.encode(
            input_ids=input_ids,
            attention_mask=attention_mask,
            sample_posterior=False,
        )
        emotion_init = vae_out.scalar_mu.detach()
        meaning_fixed = vae_out.vector_mu.detach()

    emotion_edit = nn.Parameter(emotion_init.clone())
    latent_optimizer = torch.optim.Adam([emotion_edit], lr=lr)
    losses: List[float] = []

    for _ in range(steps):
        latent_optimizer.zero_grad(set_to_none=True)

        if model.attention_source == "scalar_only":
            source_sequence = emotion_edit
        elif model.attention_source == "vector_only":
            source_sequence = meaning_fixed
        elif model.attention_source == "latent_full":
            source_sequence = torch.cat([emotion_edit, meaning_fixed], dim=-1)
        else:
            source_sequence = t5_encoder_sequence

        pooled, _ = model.scalar_attention_pool(
            source_sequence=source_sequence,
            scalar_sequence=emotion_edit,
            attention_mask=attention_mask,
        )
        logits = model.classification_logits(pooled)
        target_loss = masked_bce_with_logits(logits, desired_target, desired_mask)
        reg_loss = F.mse_loss(emotion_edit, emotion_init)
        total_loss = target_loss + (emotion_l2_weight * reg_loss)
        total_loss.backward()
        latent_optimizer.step()
        losses.append(float(total_loss.detach().cpu().item()))

    with torch.no_grad():
        if model.attention_source == "scalar_only":
            final_source_sequence = emotion_edit
        elif model.attention_source == "vector_only":
            final_source_sequence = meaning_fixed
        elif model.attention_source == "latent_full":
            final_source_sequence = torch.cat([emotion_edit, meaning_fixed], dim=-1)
        else:
            final_source_sequence = t5_encoder_sequence

        final_pooled = model.scalar_attention_pool(
            source_sequence=final_source_sequence,
            scalar_sequence=emotion_edit,
            attention_mask=attention_mask,
        )[0]
        _, _, decoded_memory = model.decode_from_latent_parts(
            scalar_latents=emotion_edit,
            vector_latents=meaning_fixed,
            t5_encoder_sequence=t5_encoder_sequence,
            pooled_scalar_tensor=final_pooled,
            residual_scale=residual_scale,
        )
        final_logits = model.classification_logits(final_pooled)
        final_probs = torch.sigmoid(final_logits)
        final_preds = (final_probs >= threshold_to_tensor(state.threshold, final_probs)).float()
        generated_text = generate_text_from_memory(
            model=model,
            tokenizer=ctx.tokenizer,
            decoder_memory=decoded_memory,
            encoder_attention_mask=attention_mask,
            prompt_config=ctx.prompt_config,
            max_new_tokens=ctx.data_config.max_length,
            num_beams=1,
            do_sample=False,
        )

    return {
        "edit_method": "backprop",
        "edited_emotion_latents": emotion_edit.detach().cpu(),
        "fixed_meaning_latents": meaning_fixed.detach().cpu(),
        "loss_curve": losses,
        "final_logits": final_logits.detach().cpu(),
        "final_probs": final_probs.detach().cpu(),
        "final_preds": final_preds.detach().cpu(),
        "generated_text": generated_text,
    }


def optimize_emotion_latents_for_target_closed_form(
    state: TrainingState,
    ctx: ExperimentContext,
    input_ids: torch.LongTensor,
    attention_mask: torch.Tensor,
    desired_target: torch.Tensor,
    desired_mask: torch.Tensor,
    force_scale: Optional[float] = None,
    ridge: Optional[float] = None,
    backsolve_ridge: Optional[float] = None,
    target_margin: Optional[float] = None,
    residual_scale: float = 1.0,
) -> Dict[str, Any]:
    model = state.model
    model.eval()

    input_ids = input_ids.to(ctx.device)
    attention_mask = attention_mask.to(ctx.device)
    desired_target = desired_target.to(ctx.device).float().unsqueeze(0)
    desired_mask = desired_mask.to(ctx.device).float().unsqueeze(0)
    force_scale = float(ctx.experiment_config.edit_force_scale if force_scale is None else force_scale)
    ridge = float(ctx.experiment_config.edit_ridge if ridge is None else ridge)
    backsolve_ridge = float(ctx.experiment_config.edit_backsolve_ridge if backsolve_ridge is None else backsolve_ridge)
    target_margin = float(ctx.experiment_config.edit_target_margin if target_margin is None else target_margin)

    with torch.no_grad():
        t5_encoder_sequence, _, pooled_scalar_tensor, vae_out = model.encode(
            input_ids=input_ids,
            attention_mask=attention_mask,
            sample_posterior=False,
        )
        emotion_init = vae_out.scalar_mu.detach()
        meaning_fixed = vae_out.vector_mu.detach()
        current_features = model.flatten_pooled_scalar_tensor(pooled_scalar_tensor)
        current_logits = model.classification_logits(pooled_scalar_tensor)
        # Extract threshold value for use in target logits
        threshold_arr = threshold_to_numpy(state.threshold, current_logits.shape[-1])
        threshold_scalar = float(threshold_arr[0]) if threshold_arr.size == 1 else 0.5  # Default to 0.5 if per-label
        target_logits = build_target_logits(
            current_logits=current_logits,
            desired_target=desired_target,
            desired_mask=desired_mask,
            margin=target_margin,
            threshold=threshold_scalar,
        )

    current_logits_cf, delta_logits, delta_features = classifier_feature_edit(
        model=model,
        current_features=current_features,
        target_logits=target_logits,
        ridge=ridge,
    )
    edited_features = current_features + (force_scale * delta_features)
    pooled_delta_features = edited_features - current_features
    emotion_edit = pooled_feature_preimage_from_scalar_latents(
        model=model,
        scalar_sequence=emotion_init,
        vector_sequence=meaning_fixed,
        t5_encoder_sequence=t5_encoder_sequence,
        attention_mask=attention_mask,
        pooled_delta_features=pooled_delta_features,
        ridge=backsolve_ridge,
    )

    with torch.no_grad():
        final_pooled, _ = model.pooled_scalar_tensor_from_latent_parts(
            scalar_latents=emotion_edit,
            vector_latents=meaning_fixed,
            t5_encoder_sequence=t5_encoder_sequence,
            attention_mask=attention_mask,
        )
        _, _, decoded_memory = model.decode_from_latent_parts(
            scalar_latents=emotion_edit,
            vector_latents=meaning_fixed,
            t5_encoder_sequence=t5_encoder_sequence,
            pooled_scalar_tensor=final_pooled,
            residual_scale=residual_scale,
        )
        final_logits = model.classification_logits(final_pooled)
        final_probs = torch.sigmoid(final_logits)
        final_preds = (final_probs >= threshold_to_tensor(state.threshold, final_probs)).float()
        generated_text = generate_text_from_memory(
            model=model,
            tokenizer=ctx.tokenizer,
            decoder_memory=decoded_memory,
            encoder_attention_mask=attention_mask,
            prompt_config=ctx.prompt_config,
            max_new_tokens=ctx.data_config.max_length,
            num_beams=1,
            do_sample=False,
        )
        objective = masked_bce_with_logits(final_logits, desired_target, desired_mask)

    return {
        "edit_method": "closed_form",
        "edited_emotion_latents": emotion_edit.detach().cpu(),
        "fixed_meaning_latents": meaning_fixed.detach().cpu(),
        "loss_curve": [float(objective.detach().cpu().item())],
        "current_logits": current_logits_cf.detach().cpu(),
        "target_logits": target_logits.detach().cpu(),
        "delta_logits": delta_logits.detach().cpu(),
        "final_logits": final_logits.detach().cpu(),
        "final_probs": final_probs.detach().cpu(),
        "final_preds": final_preds.detach().cpu(),
        "generated_text": generated_text,
    }


def optimize_emotion_latents_for_target(
    state: TrainingState,
    ctx: ExperimentContext,
    input_ids: torch.LongTensor,
    attention_mask: torch.Tensor,
    desired_target: torch.Tensor,
    desired_mask: torch.Tensor,
    steps: int = 80,
    lr: float = 3e-2,
    emotion_l2_weight: float = 5e-3,
    method: Optional[str] = None,
    residual_scale: float = 1.0,
) -> Dict[str, Any]:
    method = ctx.experiment_config.monitor_edit_method if method is None else method
    if method == "closed_form":
        return optimize_emotion_latents_for_target_closed_form(
            state=state,
            ctx=ctx,
            input_ids=input_ids,
            attention_mask=attention_mask,
            desired_target=desired_target,
            desired_mask=desired_mask,
            residual_scale=residual_scale,
        )
    if method == "backprop":
        return optimize_emotion_latents_for_target_backprop(
            state=state,
            ctx=ctx,
            input_ids=input_ids,
            attention_mask=attention_mask,
            desired_target=desired_target,
            desired_mask=desired_mask,
            steps=steps,
            lr=lr,
            emotion_l2_weight=emotion_l2_weight,
            residual_scale=residual_scale,
        )
    raise ValueError(f"Unsupported edit method: {method}")
