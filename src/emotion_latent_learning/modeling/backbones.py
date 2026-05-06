from __future__ import annotations

from ..utils.common import *
from .t5_utils import resolve_decoder_start_token_id_from_module, resolve_hidden_size_from_config

# Focused module: backbones.

class T5EncoderBackbone(nn.Module):
    def __init__(self, shared_model: nn.Module) -> None:
        super().__init__()
        self.shared_model = shared_model
        self.model = shared_model.get_encoder()
        self.hidden_size = resolve_hidden_size_from_config(shared_model.config)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        return outputs.last_hidden_state.float()


class T5DecoderBackbone(nn.Module):
    def __init__(self, shared_model: nn.Module) -> None:
        super().__init__()
        self.model = shared_model
        self.hidden_size = resolve_hidden_size_from_config(shared_model.config)

    @property
    def model_dtype(self) -> torch.dtype:
        first_param = next(self.model.parameters(), None)
        return torch.float32 if first_param is None else first_param.dtype

    @property
    def decoder_start_token_id(self) -> int:
        return resolve_decoder_start_token_id_from_module(self.model)

    def seq2seq_forward(
        self,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor],
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
    ) -> Any:
        return self.model(
            attention_mask=encoder_attention_mask,
            encoder_outputs=BaseModelOutput(last_hidden_state=encoder_hidden_states.to(dtype=self.model_dtype)),
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            use_cache=False,
            return_dict=True,
        )

    def generate_from_memory(
        self,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor],
        decoder_input_ids: Optional[torch.LongTensor],
        max_new_tokens: int,
        num_beams: int = 1,
        do_sample: bool = False,
        repetition_penalty: float = 1.2,
        no_repeat_ngram_size: int = 3,
    ) -> torch.LongTensor:
        return self.model.generate(
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=BaseModelOutput(last_hidden_state=encoder_hidden_states.to(dtype=self.model_dtype)),
            attention_mask=encoder_attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=do_sample,
            decoder_start_token_id=self.decoder_start_token_id,
            pad_token_id=getattr(self.model.config, "pad_token_id", None),
            eos_token_id=getattr(self.model.config, "eos_token_id", None),
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
