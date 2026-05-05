from __future__ import annotations

from ..utils.common import *
from .backbones import T5DecoderBackbone, T5EncoderBackbone
from ..config import ModelConfig
from .pooling import ScalarLatentAttentionPooling
from ..schemas import FactorVAEOutput, T5FactorVAEModelOutput
from .t5_utils import freeze_non_lora_parameters, resolve_lora_target_modules
from .vae import FactorVAEDecoder, FactorVAEEncoder

# Focused module: modeling.

class T5FactorVAEModel(nn.Module):
    def __init__(
        self,
        model_config: ModelConfig,
        num_labels: int,
        pos_weight: Optional[torch.Tensor] = None,
        classification_loss_type: str = "bce",
    ) -> None:
        super().__init__()
        self.model_config = model_config
        self.num_scalar_factors = model_config.num_scalar_factors
        self.vector_latent_dim = model_config.vector_latent_dim
        self.num_labels = num_labels
        self.attention_source = model_config.attention_source
        self.pooling_mode = model_config.pooling_mode
        self.classifier_mode = model_config.classifier_mode
        self.classifier_parameterization = model_config.classifier_parameterization
        self.classifier_uses_mu = model_config.classifier_uses_mu
        self.use_skip_connection = model_config.use_skip_connection
        self.decoder_conditioning_mode = model_config.decoder_conditioning_mode
        self.classification_loss_type = classification_loss_type

        valid_sources = {"scalar_only", "vector_only", "latent_full", "encoder_sequence"}
        if self.attention_source not in valid_sources:
            raise ValueError(f"Unsupported attention_source={self.attention_source}. Expected one of {sorted(valid_sources)}")
        valid_classifier_modes = {"joint_mlp", "per_emotion_mlp"}
        if self.classifier_mode not in valid_classifier_modes:
            raise ValueError(f"Unsupported classifier_mode={self.classifier_mode}. Expected one of {sorted(valid_classifier_modes)}")
        valid_classifier_parameterizations = {"standard", "orthogonal"}
        if self.classifier_parameterization not in valid_classifier_parameterizations:
            raise ValueError(
                f"Unsupported classifier_parameterization={self.classifier_parameterization}. "
                f"Expected one of {sorted(valid_classifier_parameterizations)}"
            )
        valid_decoder_conditioning_modes = {"standard", "split_film"}
        if self.decoder_conditioning_mode not in valid_decoder_conditioning_modes:
            raise ValueError(
                f"Unsupported decoder_conditioning_mode={self.decoder_conditioning_mode}. "
                f"Expected one of {sorted(valid_decoder_conditioning_modes)}"
            )
        if self.classifier_mode == "per_emotion_mlp" and self.num_scalar_factors != num_labels:
            raise ValueError("per_emotion_mlp requires num_scalar_factors == num_labels")

        shared_t5 = AutoModelForSeq2SeqLM.from_pretrained(model_config.model_name)
        self.lora_target_modules = tuple()
        if model_config.use_lora:
            resolved_lora_target_modules = resolve_lora_target_modules(
                shared_t5,
                model_config.lora_target_modules,
            )
            self.lora_target_modules = tuple(resolved_lora_target_modules)
            shared_t5 = get_peft_model(
                shared_t5,
                LoraConfig(
                    task_type=TaskType.SEQ_2_SEQ_LM,
                    r=model_config.lora_r,
                    lora_alpha=model_config.lora_alpha,
                    lora_dropout=model_config.lora_dropout,
                    target_modules=list(resolved_lora_target_modules),
                    bias="none",
                ),
            )
        self.shared_t5 = shared_t5
        self.t5_encoder = T5EncoderBackbone(shared_model=shared_t5)
        self.t5_decoder = T5DecoderBackbone(shared_model=shared_t5)
        self.hidden_size = self.t5_encoder.hidden_size

        self.vae_encoder = FactorVAEEncoder(
            input_dim=self.hidden_size,
            num_scalar_factors=self.num_scalar_factors,
            vector_latent_dim=self.vector_latent_dim,
            hidden_dim=model_config.vae_hidden_dim,
            dropout=model_config.vae_dropout,
        )

        source_dim_map = {
            "scalar_only": self.num_scalar_factors,
            "vector_only": self.vector_latent_dim,
            "latent_full": self.num_scalar_factors + self.vector_latent_dim,
            "encoder_sequence": self.hidden_size,
        }
        self.scalar_attention_pool = ScalarLatentAttentionPooling(
            num_scalar_factors=self.num_scalar_factors,
            source_dim=source_dim_map[self.attention_source],
            num_heads=model_config.latent_pool_heads,
            pooling_mode=model_config.pooling_mode,
            dropout=model_config.vae_dropout,
        )
        self.vae_decoder = FactorVAEDecoder(
            output_dim=self.hidden_size,
            num_scalar_factors=self.num_scalar_factors,
            vector_latent_dim=self.vector_latent_dim,
            hidden_dim=model_config.vae_hidden_dim,
            dropout=model_config.vae_dropout,
        )
        self.vector_branch_dropout = nn.Dropout(model_config.vector_branch_dropout)
        if self.decoder_conditioning_mode == "split_film":
            scalar_hidden = max(64, int(model_config.scalar_decoder_hidden_dim))
            film_hidden = max(64, int(model_config.film_hidden_dim))
            self.vector_memory_decoder = nn.Sequential(
                nn.Linear(self.vector_latent_dim, model_config.vae_hidden_dim),
                nn.GELU(),
                nn.Dropout(model_config.vae_dropout),
                nn.Linear(model_config.vae_hidden_dim, self.hidden_size),
            )
            self.scalar_token_decoder = nn.Sequential(
                nn.Linear(self.num_scalar_factors, scalar_hidden),
                nn.GELU(),
                nn.Dropout(model_config.vae_dropout),
                nn.Linear(scalar_hidden, self.hidden_size),
            )
            classifier_input_dim = self.num_scalar_factors * model_config.latent_pool_heads
            self.scalar_summary_projector = nn.Sequential(
                nn.Linear(classifier_input_dim, film_hidden),
                nn.GELU(),
                nn.Dropout(model_config.vae_dropout),
                nn.Linear(film_hidden, self.hidden_size),
            )
            self.scalar_to_film = nn.Sequential(
                nn.Linear(classifier_input_dim, film_hidden),
                nn.GELU(),
                nn.Dropout(model_config.vae_dropout),
                nn.Linear(film_hidden, 2 * self.hidden_size),
            )
        bottleneck_dim = max(1, min(int(model_config.residual_bottleneck_dim), self.hidden_size))
        self.residual_bottleneck = nn.Sequential(
            nn.Linear(self.hidden_size, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(model_config.residual_bottleneck_dropout),
            nn.Linear(bottleneck_dim, self.hidden_size),
        )
        self.memory_norm = nn.LayerNorm(self.hidden_size)

        classifier_input_dim = self.num_scalar_factors * model_config.latent_pool_heads
        if self.classifier_mode == "joint_mlp":
            if self.classifier_parameterization == "standard":
                self.classifier = nn.Sequential(
                    nn.Dropout(model_config.classifier_dropout),
                    nn.Linear(classifier_input_dim, model_config.joint_mlp_hidden_dim),
                    nn.Dropout(model_config.classifier_dropout),
                    nn.Linear(model_config.joint_mlp_hidden_dim, model_config.joint_mlp_hidden_dim * 2),
                    nn.Dropout(model_config.classifier_dropout),
                    nn.Linear(model_config.joint_mlp_hidden_dim * 2, model_config.joint_mlp_hidden_dim),
                    nn.Dropout(model_config.classifier_dropout),
                    nn.Linear(model_config.joint_mlp_hidden_dim, num_labels),
                )
            else:
                self.classifier = nn.Linear(classifier_input_dim, num_labels)
                nn.init.orthogonal_(self.classifier.weight)
                nn.init.zeros_(self.classifier.bias)
        else:
            if self.classifier_parameterization == "standard":
                self.emotion_classifiers = nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.Dropout(model_config.classifier_dropout),
                            nn.Linear(model_config.latent_pool_heads, model_config.per_emotion_hidden_dim),
                            nn.GELU(),
                            nn.Linear(model_config.per_emotion_hidden_dim, 1),
                        )
                        for _ in range(num_labels)
                    ]
                )
            else:
                self.emotion_classifiers = nn.ModuleList(
                    [
                        nn.Linear(model_config.latent_pool_heads, 1)
                        for _ in range(num_labels)
                    ]
                )
                for head in self.emotion_classifiers:
                    nn.init.normal_(head.weight, mean=0.0, std=1.0 / math.sqrt(max(model_config.latent_pool_heads, 1)))
                    with torch.no_grad():
                        norm = head.weight.norm(p=2, dim=1, keepdim=True).clamp_min(1e-8)
                        head.weight.div_(norm)
                    nn.init.zeros_(head.bias)

        self.classification_loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=None if pos_weight is None else pos_weight.float()
        )
        self.classification_mse_loss_fn = nn.MSELoss(reduction="mean")

        freeze_non_lora_parameters(self.shared_t5)
        self.shared_t5.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.shared_t5.eval()
        return self

    def flatten_pooled_scalar_tensor(self, pooled_scalar_tensor: torch.Tensor) -> torch.Tensor:
        return pooled_scalar_tensor.flatten(start_dim=1)

    def features_to_pooled_scalar_tensor(self, features: torch.Tensor) -> torch.Tensor:
        return features.view(features.size(0), self.num_labels, self.model_config.latent_pool_heads)

    def classification_logits_from_features(self, features: torch.Tensor) -> torch.Tensor:
        if self.classifier_mode == "joint_mlp":
            return self.classifier(features)
        pooled_scalar_tensor = self.features_to_pooled_scalar_tensor(features)
        logits = [
            head(pooled_scalar_tensor[:, idx, :])
            for idx, head in enumerate(self.emotion_classifiers)
        ]
        return torch.cat(logits, dim=-1)

    def pooled_scalar_tensor_from_latent_parts(
        self,
        scalar_latents: torch.Tensor,
        vector_latents: torch.Tensor,
        t5_encoder_sequence: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.attention_source == "scalar_only":
            source_sequence = scalar_latents
        elif self.attention_source == "vector_only":
            source_sequence = vector_latents
        elif self.attention_source == "latent_full":
            source_sequence = torch.cat([scalar_latents, vector_latents], dim=-1)
        else:
            source_sequence = t5_encoder_sequence
        return self.scalar_attention_pool(
            source_sequence=source_sequence,
            scalar_sequence=scalar_latents,
            attention_mask=attention_mask,
        )

    def classifier_linear_weight_and_bias(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.classifier_parameterization != "orthogonal":
            raise ValueError("Linear classifier parameters are only available for orthogonal parameterization.")
        if self.classifier_mode == "joint_mlp":
            return self.classifier.weight, self.classifier.bias

        num_heads = self.model_config.latent_pool_heads
        weight = next(self.parameters()).new_zeros((self.num_labels, self.num_labels * num_heads))
        bias = next(self.parameters()).new_zeros((self.num_labels,))
        for idx, head in enumerate(self.emotion_classifiers):
            start = idx * num_heads
            end = start + num_heads
            weight[idx, start:end] = head.weight.squeeze(0)
            bias[idx] = head.bias.squeeze(0)
        return weight, bias

    def classifier_regularization_loss(self) -> torch.Tensor:
        if self.classifier_parameterization != "orthogonal":
            return next(self.parameters()).new_zeros(())
        weight, _ = self.classifier_linear_weight_and_bias()
        gram = weight @ weight.transpose(0, 1)
        eye = torch.eye(gram.size(0), device=gram.device, dtype=gram.dtype)
        return (gram - eye).pow(2).mean()

    def _attention_source_sequence(
        self,
        vae_out: FactorVAEOutput,
        t5_encoder_sequence: torch.Tensor,
    ) -> torch.Tensor:
        if self.classifier_uses_mu:
            if self.attention_source == "scalar_only":
                return vae_out.scalar_mu
            if self.attention_source == "vector_only":
                return vae_out.vector_mu
            if self.attention_source == "latent_full":
                return vae_out.mu
            return t5_encoder_sequence

        if self.attention_source == "scalar_only":
            return vae_out.scalar_z
        if self.attention_source == "vector_only":
            return vae_out.vector_z
        if self.attention_source == "latent_full":
            return vae_out.z
        return t5_encoder_sequence

    def encode(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor],
        sample_posterior: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, FactorVAEOutput]:
        t5_encoder_sequence = self.t5_encoder(input_ids=input_ids, attention_mask=attention_mask)
        vae_out = self.vae_encoder(t5_encoder_sequence, sample=sample_posterior)
        source_sequence = self._attention_source_sequence(vae_out=vae_out, t5_encoder_sequence=t5_encoder_sequence)
        scalar_sequence = vae_out.scalar_mu if self.classifier_uses_mu else vae_out.scalar_z
        pooled_scalar_tensor, attn_weights = self.scalar_attention_pool(
            source_sequence=source_sequence,
            scalar_sequence=scalar_sequence,
            attention_mask=attention_mask,
        )
        return t5_encoder_sequence, attn_weights, pooled_scalar_tensor, vae_out

    def residual_memory_from_encoder(
        self,
        t5_encoder_sequence: torch.Tensor,
        residual_scale: float = 1.0,
    ) -> torch.Tensor:
        if self.use_skip_connection and abs(float(residual_scale)) > 1e-12:
            return float(residual_scale) * self.residual_bottleneck(t5_encoder_sequence)
        return torch.zeros_like(t5_encoder_sequence)

    def pooled_scalar_fallback(self, scalar_latents: torch.Tensor) -> torch.Tensor:
        mean_scalar = scalar_latents.mean(dim=1)
        return mean_scalar.unsqueeze(-1).expand(-1, -1, self.model_config.latent_pool_heads)

    def decode_from_latent_parts(
        self,
        scalar_latents: torch.Tensor,
        vector_latents: torch.Tensor,
        t5_encoder_sequence: torch.Tensor,
        pooled_scalar_tensor: Optional[torch.Tensor] = None,
        residual_scale: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual_memory = self.residual_memory_from_encoder(
            t5_encoder_sequence=t5_encoder_sequence,
            residual_scale=residual_scale,
        )
        if self.decoder_conditioning_mode == "split_film":
            if pooled_scalar_tensor is None:
                pooled_scalar_tensor = self.pooled_scalar_fallback(scalar_latents)
            vector_memory = self.vector_memory_decoder(vector_latents)
            vector_memory = self.vector_branch_dropout(vector_memory)
            scalar_token_memory = self.scalar_token_decoder(scalar_latents)
            scalar_features = self.flatten_pooled_scalar_tensor(pooled_scalar_tensor)
            summary_bias = self.scalar_summary_projector(scalar_features).unsqueeze(1)
            film_params = self.scalar_to_film(scalar_features)
            gamma, beta = film_params.chunk(2, dim=-1)
            gamma = torch.tanh(gamma).unsqueeze(1)
            beta = beta.unsqueeze(1)
            base_memory = vector_memory + scalar_token_memory + summary_bias
            vae_decoded_sequence = (1.0 + gamma) * base_memory + beta
            decoder_memory = self.memory_norm(vae_decoded_sequence + residual_memory)
            return vae_decoded_sequence, residual_memory, decoder_memory

        z = torch.cat([scalar_latents, vector_latents], dim=-1)
        vae_decoded_sequence = self.vae_decoder(z=z)
        decoder_memory = self.memory_norm(vae_decoded_sequence + residual_memory)
        return vae_decoded_sequence, residual_memory, decoder_memory

    def decode(
        self,
        vae_out: FactorVAEOutput,
        t5_encoder_sequence: torch.Tensor,
        pooled_scalar_tensor: Optional[torch.Tensor] = None,
        residual_scale: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.decode_from_latent_parts(
            scalar_latents=vae_out.scalar_z,
            vector_latents=vae_out.vector_z,
            t5_encoder_sequence=t5_encoder_sequence,
            pooled_scalar_tensor=pooled_scalar_tensor,
            residual_scale=residual_scale,
        )

    def classification_logits(self, pooled_scalar_tensor: torch.Tensor) -> torch.Tensor:
        features = self.flatten_pooled_scalar_tensor(pooled_scalar_tensor)
        return self.classification_logits_from_features(features)

    def base_losses(
        self,
        classification_logits: torch.Tensor,
        t5_encoder_sequence: torch.Tensor,
        vae_decoded_sequence: torch.Tensor,
        vae_out: FactorVAEOutput,
        attention_mask: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
        classification_weight: float,
        recon_weight: float,
        kl_weight: float,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if attention_mask is None:
            recon_loss = F.mse_loss(vae_decoded_sequence, t5_encoder_sequence)
        else:
            mask = attention_mask.to(t5_encoder_sequence.dtype).unsqueeze(-1)
            sq_error = (vae_decoded_sequence - t5_encoder_sequence).pow(2)
            denom = mask.sum().clamp_min(1.0) * sq_error.size(-1)
            recon_loss = (sq_error * mask).sum() / denom

        kl_per_token = -0.5 * (1 + vae_out.logvar - vae_out.mu.pow(2) - vae_out.logvar.exp()).sum(dim=-1)
        if attention_mask is None:
            kl_loss = kl_per_token.mean()
        else:
            mask = attention_mask.to(kl_per_token.dtype)
            kl_loss = (kl_per_token * mask).sum() / mask.sum().clamp_min(1.0)

        total = (recon_weight * recon_loss) + (kl_weight * kl_loss)
        loss_terms = {
            "reconstruction": recon_loss.detach(),
            "kl": kl_loss.detach(),
        }
        if labels is not None:
            if self.classification_loss_type == "mse":
                # For regression targets, use MSE on sigmoid-transformed logits
                probs = torch.sigmoid(classification_logits)
                classification_loss = self.classification_mse_loss_fn(probs, labels.float())
            else:
                # Default: BCE with logits for binary/multi-label classification
                classification_loss = self.classification_loss_fn(classification_logits, labels.float())
            total = total + (classification_weight * classification_loss)
            loss_terms["classification"] = classification_loss.detach()
        return total, loss_terms

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
        sample_posterior: bool,
        classification_weight: float,
        recon_weight: float,
        kl_weight: float,
        residual_scale: float = 1.0,
    ) -> T5FactorVAEModelOutput:
        t5_encoder_sequence, attn_weights, pooled_scalar_tensor, vae_out = self.encode(
            input_ids=input_ids,
            attention_mask=attention_mask,
            sample_posterior=sample_posterior,
        )
        vae_decoded_sequence, residual_memory, decoder_memory = self.decode(
            vae_out=vae_out,
            t5_encoder_sequence=t5_encoder_sequence,
            pooled_scalar_tensor=pooled_scalar_tensor,
            residual_scale=residual_scale,
        )
        classification_logits = self.classification_logits(pooled_scalar_tensor)

        base_loss = None
        loss_terms = None
        if labels is not None:
            base_loss, loss_terms = self.base_losses(
                classification_logits=classification_logits,
                t5_encoder_sequence=t5_encoder_sequence,
                vae_decoded_sequence=vae_decoded_sequence,
                vae_out=vae_out,
                attention_mask=attention_mask,
                labels=labels,
                classification_weight=classification_weight,
                recon_weight=recon_weight,
                kl_weight=kl_weight,
            )

        return T5FactorVAEModelOutput(
            t5_encoder_sequence=t5_encoder_sequence,
            attention_weights=attn_weights,
            pooled_scalar_tensor=pooled_scalar_tensor,
            vae=vae_out,
            vae_decoded_sequence=vae_decoded_sequence,
            residual_memory=residual_memory,
            decoder_memory=decoder_memory,
            classification_logits=classification_logits,
            base_loss=base_loss,
            loss_terms=loss_terms,
        )
