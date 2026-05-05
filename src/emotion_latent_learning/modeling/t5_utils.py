from __future__ import annotations

from ..utils.common import *

# Focused module: t5_utils.

def resolve_hidden_size_from_config(config: Any) -> int:
    candidates = [
        config,
        getattr(config, "text_config", None),
        getattr(config, "encoder", None),
        getattr(config, "decoder", None),
    ]
    for cfg in candidates:
        if cfg is None:
            continue
        for key in ("hidden_size", "d_model", "model_dim"):
            value = getattr(cfg, key, None)
            if isinstance(value, int):
                return value
    raise ValueError("Could not resolve hidden size from model config.")


def resolve_wrapped_config(module: Any) -> Any:
    candidates = []
    seen = set()

    def add(obj: Any) -> None:
        if obj is None:
            return
        ident = id(obj)
        if ident in seen:
            return
        seen.add(ident)
        candidates.append(obj)

    add(module)
    current = module
    for _ in range(6):
        if current is None:
            break
        add(getattr(current, "model", None))
        add(getattr(current, "base_model", None))
        base_model = getattr(current, "base_model", None)
        if base_model is not None:
            add(getattr(base_model, "model", None))
        current = getattr(current, "model", None) or getattr(current, "base_model", None)

    for obj in candidates:
        cfg = getattr(obj, "config", None)
        if cfg is not None:
            return cfg
    raise AttributeError("Could not resolve config from wrapped model.")


def resolve_decoder_start_token_id_from_module(module: Any) -> int:
    search_roots = []
    seen = set()

    def add(obj: Any) -> None:
        if obj is None:
            return
        ident = id(obj)
        if ident in seen:
            return
        seen.add(ident)
        search_roots.append(obj)

    add(module)
    add(getattr(module, "model", None))
    add(getattr(module, "base_model", None))
    base_model = getattr(module, "base_model", None)
    if base_model is not None:
        add(getattr(base_model, "model", None))

    generation_configs = [getattr(obj, "generation_config", None) for obj in search_roots]
    configs = []
    try:
        configs.append(resolve_wrapped_config(module))
    except AttributeError:
        pass
    configs.extend([cfg for cfg in generation_configs if cfg is not None])

    for cfg in configs:
        start_token_id = getattr(cfg, "decoder_start_token_id", None)
        if start_token_id is None:
            decoder_cfg = getattr(cfg, "decoder", None)
            if decoder_cfg is not None:
                start_token_id = getattr(decoder_cfg, "decoder_start_token_id", None)
        if start_token_id is None:
            start_token_id = getattr(cfg, "bos_token_id", None)
        if start_token_id is None:
            decoder_cfg = getattr(cfg, "decoder", None)
            if decoder_cfg is not None:
                start_token_id = getattr(decoder_cfg, "bos_token_id", None)
        if start_token_id is None:
            start_token_id = getattr(cfg, "pad_token_id", None)
        if start_token_id is None:
            decoder_cfg = getattr(cfg, "decoder", None)
            if decoder_cfg is not None:
                start_token_id = getattr(decoder_cfg, "pad_token_id", None)
        if start_token_id is not None:
            return int(start_token_id)

    raise ValueError("No decoder start token is configured for this model.")


def resolve_lora_target_modules(
    module: nn.Module,
    configured_target_modules: Optional[Sequence[str]],
) -> Tuple[str, ...]:
    if configured_target_modules is not None and len(configured_target_modules) > 0:
        return tuple(configured_target_modules)

    linear_suffixes = {
        name.split(".")[-1]
        for name, submodule in module.named_modules()
        if isinstance(submodule, nn.Linear)
    }
    for pair in (("q_proj", "v_proj"), ("q", "v"), ("query", "value")):
        if all(item in linear_suffixes for item in pair):
            return pair
    raise ValueError(
        "Could not infer LoRA target modules automatically. "
        f"Available linear suffixes include: {sorted(linear_suffixes)[:50]}"
    )


def is_decoder_cross_attention_lora_parameter(name: str) -> bool:
    if "lora_" not in name:
        return False
    lower = name.lower()
    is_decoder = ("decoder.block" in lower) or ("decoder.layers" in lower)
    is_cross = ("encdecattention" in lower) or ("layer.1" in lower) or ("cross_attn" in lower)
    return is_decoder and is_cross


def freeze_non_lora_parameters(module: nn.Module) -> None:
    for name, param in module.named_parameters():
        param.requires_grad = is_decoder_cross_attention_lora_parameter(name)


def set_lora_trainable(module: nn.Module, enabled: bool) -> None:
    for name, param in module.named_parameters():
        if "lora_" in name:
            param.requires_grad = enabled and is_decoder_cross_attention_lora_parameter(name)


def set_requires_grad(module: nn.Module, enabled: bool) -> None:
    for param in module.parameters():
        param.requires_grad = enabled


def shift_tokens_right_manual(
    input_ids: torch.LongTensor,
    pad_token_id: int,
    decoder_start_token_id: int,
) -> torch.LongTensor:
    shifted = input_ids.new_full(input_ids.shape, pad_token_id)
    shifted[:, 0] = decoder_start_token_id
    shifted[:, 1:] = input_ids[:, :-1]
    return shifted


def masked_mean(sequence: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
    if attention_mask is None:
        return sequence.mean(dim=1)
    mask = attention_mask.to(dtype=sequence.dtype).unsqueeze(-1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return (sequence * mask).sum(dim=1) / denom
