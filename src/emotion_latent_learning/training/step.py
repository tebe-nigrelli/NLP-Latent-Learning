from __future__ import annotations

from ..utils.common import *
from ..evaluation.generation import compute_copy_loss_from_memory
from .losses import cross_covariance_penalty, discriminator_loss, factorvae_tc_loss, residual_adversarial_loss, tc_latents_for_tc, transfer_strength_margin_loss, vector_adversarial_loss
from ..evaluation.metrics import multilabel_metrics
from ..schemas import ExperimentContext, TrainingState
from ..modeling.t5_utils import masked_mean, set_requires_grad
from .schedules import copy_loss_should_freeze_vae
from ..utils import release_memory

# Focused module: training.

def train_one_epoch(
    epoch: int,
    state: TrainingState,
    ctx: ExperimentContext,
    data_loader: DataLoader,
    weights: Dict[str, float],
) -> Dict[str, Any]:
    state.model.train()
    if state.vector_adversary is not None:
        state.vector_adversary.train()
    if state.residual_adversary is not None:
        state.residual_adversary.train()
    epoch_total = 0.0
    epoch_base = 0.0
    epoch_recon = 0.0
    epoch_kl = 0.0
    epoch_cls = 0.0
    epoch_tc = 0.0
    epoch_copy = 0.0
    epoch_vec_adv = 0.0
    epoch_residual_adv = 0.0
    epoch_sep = 0.0
    epoch_orth = 0.0
    epoch_transfer = 0.0
    epoch_disc = 0.0
    epoch_logits = []
    epoch_labels = []
    batches = 0

    train_bar = tqdm(data_loader, desc=f"epoch {epoch}/{ctx.schedule_config.num_epochs} [train]", leave=False)
    for batch in train_bar:
        state.optimizer.zero_grad(set_to_none=True)
        if state.vector_adv_optimizer is not None:
            state.vector_adv_optimizer.zero_grad(set_to_none=True)
        if state.residual_adv_optimizer is not None:
            state.residual_adv_optimizer.zero_grad(set_to_none=True)

        input_ids = batch["input_ids"].to(ctx.device)
        attention_mask = batch["attention_mask"].to(ctx.device)
        labels = batch["labels"].to(ctx.device)
        texts = batch["texts"]

        out = state.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            sample_posterior=ctx.experiment_config.sample_posterior_train,
            classification_weight=weights["classification_weight"],
            recon_weight=weights["recon_weight"],
            kl_weight=weights["kl_weight"],
            residual_scale=weights["residual_scale"],
        )

        tc_loss = out.base_loss.new_zeros(())
        tc_for_disc = None
        if state.discriminator is not None and weights["tc_weight"] > 0.0:
            set_requires_grad(state.discriminator, False)
            tc_inputs = tc_latents_for_tc(out.vae, attention_mask, ctx.loss_config.tc_subspace, ctx.loss_config.tc_mode)
            tc_loss = factorvae_tc_loss(state.discriminator, tc_inputs)

        copy_loss = out.base_loss.new_zeros(())
        if weights["copy_weight"] > 0.0:
            copy_loss = compute_copy_loss_from_memory(
                model=state.model,
                tokenizer=ctx.tokenizer,
                decoder_memory=out.decoder_memory,
                encoder_attention_mask=attention_mask,
                texts=texts,
                prompt_config=ctx.prompt_config,
                device=ctx.device,
                detach_decoder_memory=copy_loss_should_freeze_vae(epoch=epoch, ctx=ctx),
            )

        vector_summary = masked_mean(out.vae.vector_mu, attention_mask)
        scalar_summary = masked_mean(out.vae.scalar_mu, attention_mask)
        residual_summary = masked_mean(out.residual_memory, attention_mask)

        vec_adv_loss = out.base_loss.new_zeros(())
        if state.vector_adversary is not None and weights["vector_adv_weight"] > 0.0:
            _, vec_adv_loss = vector_adversarial_loss(
                vector_adversary=state.vector_adversary,
                vector_summary=vector_summary,
                labels=labels,
                pos_weight=ctx.pos_weight,
            )

        residual_adv_loss = out.base_loss.new_zeros(())
        if state.residual_adversary is not None and weights["residual_adv_weight"] > 0.0:
            _, residual_adv_loss = residual_adversarial_loss(
                residual_adversary=state.residual_adversary,
                residual_summary=residual_summary,
                labels=labels,
                pos_weight=ctx.pos_weight,
            )

        sep_loss = out.base_loss.new_zeros(())
        if weights["vector_sep_weight"] > 0.0:
            sep_loss = cross_covariance_penalty(scalar_summary, vector_summary)

        orth_loss = state.model.classifier_regularization_loss()
        transfer_loss = transfer_strength_margin_loss(
            logits=out.classification_logits,
            labels=labels,
            margin=ctx.loss_config.transfer_strength_margin,
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
        )
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(state.model.parameters(), max_norm=ctx.schedule_config.max_grad_norm)
        if state.vector_adversary is not None:
            torch.nn.utils.clip_grad_norm_(state.vector_adversary.parameters(), max_norm=ctx.schedule_config.max_grad_norm)
        if state.residual_adversary is not None:
            torch.nn.utils.clip_grad_norm_(state.residual_adversary.parameters(), max_norm=ctx.schedule_config.max_grad_norm)
        state.optimizer.step()
        if state.vector_adv_optimizer is not None:
            state.vector_adv_optimizer.step()
        if state.residual_adv_optimizer is not None:
            state.residual_adv_optimizer.step()
        state.lr_scheduler.step()

        disc_loss = out.base_loss.new_zeros(())
        if state.discriminator is not None and state.disc_optimizer is not None and weights["tc_weight"] > 0.0:
            set_requires_grad(state.discriminator, True)
            with torch.no_grad():
                disc_forward = state.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    sample_posterior=True,
                    classification_weight=weights["classification_weight"],
                    recon_weight=weights["recon_weight"],
                    kl_weight=weights["kl_weight"],
                    labels=labels,
                    residual_scale=weights["residual_scale"],
                )
                tc_for_disc = tc_latents_for_tc(disc_forward.vae, attention_mask, ctx.loss_config.tc_subspace, ctx.loss_config.tc_mode)

            if tc_for_disc is not None and tc_for_disc.size(0) >= 2:
                state.disc_optimizer.zero_grad(set_to_none=True)
                disc_loss = discriminator_loss(state.discriminator, tc_for_disc)
                disc_loss.backward()
                state.disc_optimizer.step()

        epoch_total += float(total_loss.detach().cpu().item())
        epoch_base += float(out.base_loss.detach().cpu().item())
        epoch_recon += float(out.loss_terms["reconstruction"].item())
        epoch_kl += float(out.loss_terms["kl"].item())
        epoch_cls += float(out.loss_terms["classification"].item())
        epoch_tc += float(tc_loss.detach().cpu().item())
        epoch_copy += float(copy_loss.detach().cpu().item())
        epoch_vec_adv += float(vec_adv_loss.detach().cpu().item())
        epoch_residual_adv += float(residual_adv_loss.detach().cpu().item())
        epoch_sep += float(sep_loss.detach().cpu().item())
        epoch_orth += float(orth_loss.detach().cpu().item())
        epoch_transfer += float(transfer_loss.detach().cpu().item())
        epoch_disc += float(disc_loss.detach().cpu().item())
        epoch_logits.append(out.classification_logits.detach().cpu())
        epoch_labels.append(labels.detach().cpu())
        batches += 1
        state.step += 1

        train_bar.set_postfix(
            loss=f"{epoch_total / max(batches, 1):.4f}",
            cls=f"{epoch_cls / max(batches, 1):.4f}",
            recon=f"{epoch_recon / max(batches, 1):.4f}",
            kl=f"{epoch_kl / max(batches, 1):.4f}",
            tc=f"{epoch_tc / max(batches, 1):.4f}",
            copy=f"{epoch_copy / max(batches, 1):.4f}",
            adv=f"{epoch_vec_adv / max(batches, 1):.4f}",
            radv=f"{epoch_residual_adv / max(batches, 1):.4f}",
            orth=f"{epoch_orth / max(batches, 1):.4f}",
            tr=f"{epoch_transfer / max(batches, 1):.4f}",
            rs=f"{weights['residual_scale']:.3f}",
            lr=f"{state.optimizer.param_groups[0]['lr']:.2e}",
        )

        del out, total_loss, copy_loss, tc_loss, vec_adv_loss, residual_adv_loss, sep_loss, orth_loss, transfer_loss, disc_loss, input_ids, attention_mask, labels
        if tc_for_disc is not None:
            del tc_for_disc

    train_logits = torch.cat(epoch_logits, dim=0)
    train_labels = torch.cat(epoch_labels, dim=0)
    train_metrics = multilabel_metrics(
        logits=train_logits,
        labels=train_labels,
        threshold=state.threshold,
        label_names=ctx.emotion_names,
    )
    train_metrics.update(
        {
            "avg_total_loss": epoch_total / max(batches, 1),
            "avg_base_loss": epoch_base / max(batches, 1),
            "avg_recon": epoch_recon / max(batches, 1),
            "avg_kl": epoch_kl / max(batches, 1),
            "avg_cls": epoch_cls / max(batches, 1),
            "avg_tc": epoch_tc / max(batches, 1),
            "avg_copy": epoch_copy / max(batches, 1),
            "avg_vec_adv": epoch_vec_adv / max(batches, 1),
            "avg_residual_adv": epoch_residual_adv / max(batches, 1),
            "avg_sep": epoch_sep / max(batches, 1),
            "avg_orth": epoch_orth / max(batches, 1),
            "avg_transfer": epoch_transfer / max(batches, 1),
            "avg_disc": epoch_disc / max(batches, 1),
            "weighted_recon": weights["recon_weight"] * (epoch_recon / max(batches, 1)),
            "weighted_kl": weights["kl_weight"] * (epoch_kl / max(batches, 1)),
            "weighted_cls": weights["classification_weight"] * (epoch_cls / max(batches, 1)),
            "weighted_tc": weights["tc_weight"] * (epoch_tc / max(batches, 1)),
            "weighted_copy": weights["copy_weight"] * (epoch_copy / max(batches, 1)),
            "weighted_vec_adv": weights["vector_adv_weight"] * (epoch_vec_adv / max(batches, 1)),
            "weighted_residual_adv": weights["residual_adv_weight"] * (epoch_residual_adv / max(batches, 1)),
            "weighted_sep": weights["vector_sep_weight"] * (epoch_sep / max(batches, 1)),
            "weighted_orth": weights["orthogonality_weight"] * (epoch_orth / max(batches, 1)),
            "weighted_transfer": weights["transfer_strength_weight"] * (epoch_transfer / max(batches, 1)),
        }
    )
    release_memory()
    return train_metrics
