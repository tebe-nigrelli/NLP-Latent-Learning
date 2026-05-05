from __future__ import annotations

from ..utils.common import *
from .display import format_label_names, label_names_from_tensor, predict_emotions_for_text, predicted_names_from_logits, pretty_print_text_block, print_generation_summary
from .editing import make_partial_desired_emotion_target, optimize_emotion_latents_for_target
from .generation import generate_text_from_memory
from ..training.schedules import current_loss_weights
from ..schemas import ExperimentContext, TrainingState

# Focused module: monitoring.

def run_epoch_monitor(
    state: TrainingState,
    ctx: ExperimentContext,
    dataset: Dataset,
    example_indices: Sequence[int],
    edit_targets: Sequence[Sequence[str]],
    epoch: int,
) -> None:
    state.model.eval()
    weights = current_loss_weights(epoch=epoch, ctx=ctx)
    print("\n" + "=" * 120)
    print(f"Epoch {epoch} qualitative monitor on fixed validation examples")
    print("=" * 120)

    for slot, (example_idx, desired_labels) in enumerate(zip(example_indices, edit_targets), start=1):
        row = dataset[example_idx]
        text = row["text"]
        gold_names = label_names_from_tensor(row["labels"], ctx.emotion_names)

        encoded = ctx.tokenizer(
            text,
            max_length=ctx.data_config.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(ctx.device)
        attention_mask = encoded["attention_mask"].to(ctx.device)
        label_tensor = row["labels"].unsqueeze(0).to(ctx.device)

        with torch.no_grad():
            out = state.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=label_tensor,
                sample_posterior=ctx.experiment_config.sample_posterior_eval,
                classification_weight=ctx.loss_config.classification_weight,
                recon_weight=ctx.loss_config.recon_weight,
                kl_weight=ctx.loss_config.kl_weight,
                residual_scale=weights["residual_scale"],
            )
            bypass_text = generate_text_from_memory(
                model=state.model,
                tokenizer=ctx.tokenizer,
                decoder_memory=out.t5_encoder_sequence,
                encoder_attention_mask=attention_mask,
                prompt_config=ctx.prompt_config,
                max_new_tokens=ctx.data_config.max_length,
                num_beams=1,
                do_sample=False,
            )[0]
            vae_text = generate_text_from_memory(
                model=state.model,
                tokenizer=ctx.tokenizer,
                decoder_memory=out.decoder_memory,
                encoder_attention_mask=attention_mask,
                prompt_config=ctx.prompt_config,
                max_new_tokens=ctx.data_config.max_length,
                num_beams=1,
                do_sample=False,
            )[0]

        input_pred_names = predicted_names_from_logits(out.classification_logits[0], ctx.emotion_names, state.threshold)
        bypass_pred_names = predict_emotions_for_text(state, ctx, bypass_text)
        vae_pred_names = predict_emotions_for_text(state, ctx, vae_text)

        desired_target, desired_mask = make_partial_desired_emotion_target(
            label_names=ctx.emotion_names,
            positive_labels=desired_labels,
        )
        edit_result = optimize_emotion_latents_for_target(
            state=state,
            ctx=ctx,
            input_ids=input_ids,
            attention_mask=attention_mask,
            desired_target=desired_target,
            desired_mask=desired_mask,
            steps=ctx.experiment_config.monitor_edit_steps,
            lr=3e-2,
            emotion_l2_weight=5e-3,
            method=ctx.experiment_config.monitor_edit_method,
            residual_scale=weights["residual_scale"],
        )
        edited_text = edit_result["generated_text"][0]
        edited_head_pred_names = predicted_names_from_logits(
            edit_result["final_logits"][0],
            ctx.emotion_names,
            state.threshold,
        )
        edited_generated_pred_names = predict_emotions_for_text(state, ctx, edited_text)

        print("\n" + "-" * 120)
        print(f"Example {slot} | validation index={example_idx}")
        pretty_print_text_block("input text:", text)
        print(f"gold emotions: {format_label_names(gold_names)}")
        print(f"predicted emotions on input: {format_label_names(input_pred_names)}")
        print(
            "loss snapshot: "
            f"recon={float(out.loss_terms['reconstruction'].item()):.4f}, "
            f"kl={float(out.loss_terms['kl'].item()):.4f}, "
            f"cls={float(out.loss_terms['classification'].item()):.4f}"
        )

        print_generation_summary(
            tag="\nBypass path: encoder -> decoder",
            source_text=text,
            generated_text=bypass_text,
            predicted_emotions=bypass_pred_names,
        )
        print_generation_summary(
            tag="\nVAE path: encoder -> token VAE -> decoder",
            source_text=text,
            generated_text=vae_text,
            predicted_emotions=vae_pred_names,
        )

        print("\nEmotion editing")
        print(f"  edit method: {edit_result.get('edit_method', ctx.experiment_config.monitor_edit_method)}")
        print(f"  requested target emotions: {format_label_names(desired_labels)}")
        pretty_print_text_block("  generated edited text:", edited_text)
        print(f"  head predictions after latent edit: {format_label_names(edited_head_pred_names)}")
        print(f"  predicted emotions on edited generated text: {format_label_names(edited_generated_pred_names)}")
        if edit_result['loss_curve']:
            print(f"  final edit objective: {edit_result['loss_curve'][-1]:.4f}")
