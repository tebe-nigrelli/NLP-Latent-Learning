from __future__ import annotations

from ..utils.common import *
from ..config import PromptConfig
from ..modeling import T5FactorVAEModel
from ..modeling.t5_utils import resolve_decoder_start_token_id_from_module, shift_tokens_right_manual

# Focused module: generation.

def prompt_token_ids(tokenizer: AutoTokenizer, prompt_text: str) -> List[int]:
    if not prompt_text:
        return []
    return tokenizer(prompt_text, add_special_tokens=False)["input_ids"]


def build_copy_teacher_forcing_batch(
    texts: Sequence[str],
    tokenizer: AutoTokenizer,
    decoder_start_token_id: int,
    prompt_config: PromptConfig,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer must define pad_token_id.")

    prompt_ids = prompt_token_ids(tokenizer, prompt_config.prompt_text) if prompt_config.use_prompt else []
    eos_id = tokenizer.eos_token_id
    max_len = prompt_config.max_decoder_length

    teacher_force = torch.full((len(texts), max_len), pad_token_id, dtype=torch.long)
    labels = torch.full((len(texts), max_len), -100, dtype=torch.long)
    decoder_attention_mask = torch.zeros((len(texts), max_len), dtype=torch.long)

    for row_idx, text in enumerate(texts):
        target_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        full_ids = list(prompt_ids) + list(target_ids)
        if eos_id is not None:
            full_ids = full_ids + [eos_id]
        full_ids = full_ids[:max_len]
        seq_len = len(full_ids)
        if seq_len == 0:
            continue

        teacher_force[row_idx, :seq_len] = torch.tensor(full_ids, dtype=torch.long)
        labels[row_idx, :seq_len] = torch.tensor(full_ids, dtype=torch.long)
        decoder_attention_mask[row_idx, :seq_len] = 1

        if prompt_config.use_prompt and prompt_config.mask_prompt_loss:
            prompt_len = min(len(prompt_ids), seq_len)
            if prompt_len > 0:
                labels[row_idx, :prompt_len] = -100

    decoder_input_ids = shift_tokens_right_manual(
        input_ids=teacher_force,
        pad_token_id=pad_token_id,
        decoder_start_token_id=decoder_start_token_id,
    )

    return {
        "decoder_input_ids": decoder_input_ids.to(device),
        "decoder_attention_mask": decoder_attention_mask.to(device),
        "labels": labels.to(device),
    }


def build_generation_prefix(
    tokenizer: AutoTokenizer,
    batch_size: int,
    prompt_config: PromptConfig,
    device: torch.device,
) -> Tuple[Optional[torch.LongTensor], int]:
    if not prompt_config.use_prompt:
        return None, 0
    ids = tokenizer(prompt_config.prompt_text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
    return ids.repeat(batch_size, 1), int(ids.size(1))


def decode_generated_batch(
    tokenizer: AutoTokenizer,
    generated_ids: torch.LongTensor,
    prompt_prefix_len: int,
) -> List[str]:
    cpu_ids = generated_ids.detach().cpu()
    if prompt_prefix_len > 0:
        cpu_ids = cpu_ids[:, prompt_prefix_len:]
    return tokenizer.batch_decode(cpu_ids, skip_special_tokens=True)


def compute_copy_loss_from_memory(
    model: T5FactorVAEModel,
    tokenizer: AutoTokenizer,
    decoder_memory: torch.Tensor,
    encoder_attention_mask: torch.Tensor,
    texts: Sequence[str],
    prompt_config: PromptConfig,
    device: torch.device,
    copy_loss_batch_size: int = 1,
    detach_decoder_memory: bool = False,
) -> torch.Tensor:
    """Compute copy loss using micro-batching to reduce peak memory usage.
    
    The full batch can cause CUDA OOM during decoder forward/backward. This function
    chunks the batch into smaller micro-batches and accumulates losses with gradients.
    """
    batch = build_copy_teacher_forcing_batch(
        texts=texts,
        tokenizer=tokenizer,
        decoder_start_token_id=resolve_decoder_start_token_id_from_module(model.t5_decoder),
        prompt_config=prompt_config,
        device=device,
    )
    
    if detach_decoder_memory:
        decoder_memory = decoder_memory.detach()

    batch_size = decoder_memory.size(0)
    total_loss = None
    num_chunks = 0
    
    for i in range(0, batch_size, copy_loss_batch_size):
        end_idx = min(i + copy_loss_batch_size, batch_size)
        chunk_slice = slice(i, end_idx)
        
        # Slice all tensors for this micro-batch
        chunk_memory = decoder_memory[chunk_slice]
        chunk_attention_mask = encoder_attention_mask[chunk_slice] if encoder_attention_mask is not None else None
        chunk_decoder_input_ids = batch["decoder_input_ids"][chunk_slice]
        chunk_decoder_attention_mask = batch["decoder_attention_mask"][chunk_slice]
        chunk_labels = batch["labels"][chunk_slice]
        
        # Forward pass on micro-batch
        outputs = model.t5_decoder.seq2seq_forward(
            encoder_hidden_states=chunk_memory,
            encoder_attention_mask=chunk_attention_mask,
            decoder_input_ids=chunk_decoder_input_ids,
            decoder_attention_mask=chunk_decoder_attention_mask,
            labels=chunk_labels,
        )
        
        if outputs.loss is None:
            raise RuntimeError("Expected copy loss, but model returned None.")
        
        if total_loss is None:
            total_loss = outputs.loss
        else:
            total_loss = total_loss + outputs.loss
        num_chunks += 1
    
    if total_loss is None:
        raise RuntimeError("No chunks processed for copy loss.")
    
    # Average over chunks
    return total_loss / num_chunks


def generate_text_from_memory(
    model: T5FactorVAEModel,
    tokenizer: AutoTokenizer,
    decoder_memory: torch.Tensor,
    encoder_attention_mask: torch.Tensor,
    prompt_config: PromptConfig,
    max_new_tokens: int,
    num_beams: int = 1,
    do_sample: bool = False,
) -> List[str]:
    decoder_prefix, prompt_prefix_len = build_generation_prefix(
        tokenizer=tokenizer,
        batch_size=decoder_memory.size(0),
        prompt_config=prompt_config,
        device=decoder_memory.device,
    )
    generated_ids = model.t5_decoder.generate_from_memory(
        encoder_hidden_states=decoder_memory,
        encoder_attention_mask=encoder_attention_mask,
        decoder_input_ids=decoder_prefix,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        do_sample=do_sample,
    )
    return decode_generated_batch(
        tokenizer=tokenizer,
        generated_ids=generated_ids,
        prompt_prefix_len=prompt_prefix_len,
    )


def normalize_text_for_copy_metric(text: str) -> str:
    return " ".join(str(text).strip().split())


def token_f1_score(reference_text: str, generated_text: str) -> float:
    ref_tokens = normalize_text_for_copy_metric(reference_text).lower().split()
    gen_tokens = normalize_text_for_copy_metric(generated_text).lower().split()
    if not ref_tokens and not gen_tokens:
        return 1.0
    if not ref_tokens or not gen_tokens:
        return 0.0
    ref_counter = Counter(ref_tokens)
    gen_counter = Counter(gen_tokens)
    overlap = sum((ref_counter & gen_counter).values())
    precision = overlap / max(len(gen_tokens), 1)
    recall = overlap / max(len(ref_tokens), 1)
    if (precision + recall) <= 1e-12:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def summarize_copy_metrics(reference_texts: Sequence[str], generated_texts: Sequence[str]) -> Dict[str, float]:
    if len(reference_texts) != len(generated_texts):
        raise ValueError("reference_texts and generated_texts must have the same length.")
    if len(reference_texts) == 0:
        return {"exact_match": 0.0, "token_f1": 0.0, "edit_similarity": 0.0}

    exact = []
    token_f1 = []
    edit_similarity = []
    for reference, generated in zip(reference_texts, generated_texts):
        ref_norm = normalize_text_for_copy_metric(reference)
        gen_norm = normalize_text_for_copy_metric(generated)
        exact.append(1.0 if ref_norm == gen_norm else 0.0)
        token_f1.append(token_f1_score(ref_norm, gen_norm))
        edit_similarity.append(SequenceMatcher(a=ref_norm, b=gen_norm).ratio())

    return {
        "exact_match": float(np.mean(exact)),
        "token_f1": float(np.mean(token_f1)),
        "edit_similarity": float(np.mean(edit_similarity)),
    }
