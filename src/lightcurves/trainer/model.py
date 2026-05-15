"""Model building blocks for the light-curve transformer classifier."""

from typing import List, Literal, Optional, Tuple, cast

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AutoModel

from config import ExperimentConfig
from utils import enabled_flux_branches, get_patch_size


class SimpleConfig:
    """
    HF-compatible config with to_dict()
    """
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def to_dict(self):
        return dict(self.__dict__)


class LearnedPositionalEncoding(nn.Module):
    """
    Trainable positional encoding using a simple embedding layer
    """
    def __init__(self, d_model: int, max_position_embeddings: int = 8192, dropout: float = 0.1):
        """
        Create a learned absolute position embedding table
        """
        super().__init__()
        self.d_model = int(d_model)
        self.max_position_embeddings = int(max_position_embeddings)
        self.position_embeddings = nn.Embedding(self.max_position_embeddings, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Add learned positions while keeping padded tokens at zero influence
        """
        batch_size, seq_len, _ = x.shape

        # length check
        if seq_len > self.max_position_embeddings:
            raise ValueError(
                f"Sequence length {seq_len} exceeds MAX_POSITION_EMBEDDINGS={self.max_position_embeddings}. "
                "Increase MAX_POSITION_EMBEDDINGS."
            )

        # embed
        positions = torch.arange(seq_len, device=x.device, dtype=torch.long).unsqueeze(0).expand(batch_size, seq_len)
        pos = self.position_embeddings(positions)
        pos = pos * attention_mask.unsqueeze(-1).to(dtype=x.dtype)
        return self.dropout(x + pos)

def _resolve_pretrained_backbone_name(encoder_type: str, configured_name: Optional[str]) -> str:
    """
    Return pretrained model name: configured value or default for encoder_type
    """
    if configured_name:
        return configured_name
    if encoder_type == "qwen":
        return "Qwen/Qwen2.5-0.5B"
    if encoder_type == "chronos2":
        return "amazon/chronos-2"
    raise ValueError(f"No pretrained backbone available for encoder_type={encoder_type!r}")


def _infer_hidden_size_from_config(cfg) -> int:
    """
    Extract hidden dimension from model config
    """
    text_cfg = getattr(cfg, "text_config", cfg)
    for attr in ("hidden_size", "d_model", "n_embd", "model_dim"):
        value = getattr(text_cfg, attr, None)
        if value is not None:
            return int(value)
    raise RuntimeError("Could not infer hidden size from pretrained backbone config.")


def _load_pretrained_backbone(
        encoder_type: str,
        pretrained_model_name: str,
        use_lora: bool,
        lora_r: int,
        lora_alpha: int,
        lora_dropout: float,
        lora_bias: str,
) -> Tuple[nn.Module, int, bool]:
    """
    Load pretrained Kairos, Qwen, or Chronos-2 backbone.

    If use_lora=True: Apply LoRA fine-tuning with backbone frozen.
    If use_lora=False: Unfreeze backbone for standard fine-tuning.

    Returns:
        (model, hidden_size, use_reg_token)
    """
    if encoder_type == "chronos2":
        try:
            from chronos import Chronos2Pipeline
        except ImportError as exc:
            raise ImportError(
                "Chronos-2 encoder requires `chronos-forecasting>=2.2.0`. "
                "Install with: pip install 'chronos-forecasting>=2.2.0' peft"
            ) from exc

        pipeline = Chronos2Pipeline.from_pretrained(pretrained_model_name)
        backbone = pipeline.model

    else:
        backbone = AutoModel.from_pretrained(
            pretrained_model_name,
            trust_remote_code=True,
            torch_dtype=torch.float32,
        )

    backbone_config = getattr(backbone, "config", None)
    if backbone_config is None:
        raise RuntimeError(f"Loaded backbone {pretrained_model_name!r} does not expose a config object.")

    hidden_size = _infer_hidden_size_from_config(backbone_config)

    chronos_config = getattr(backbone, "chronos_config", None)
    use_reg_token = bool(
        getattr(backbone_config, "use_reg_token", False)
        or getattr(chronos_config, "use_reg_token", False)
    )

    if use_lora:
        # LoRA fine-tuning: freeze backbone and apply LoRA adapters
        if encoder_type == "qwen":
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        elif encoder_type == "chronos2":
            target_modules = ["q", "k", "v", "o"]
        else:
            raise RuntimeError(f"use_lora = True for unsupported encoder type: {encoder_type!r}")

        # LoRA configuration
        lora_config = LoraConfig(
            r=int(lora_r),
            lora_alpha=int(lora_alpha),
            lora_dropout=float(lora_dropout),
            bias=cast(Literal["none", "all", "lora_only"], lora_bias),
            target_modules=target_modules,
            task_type="FEATURE_EXTRACTION",
        )
        backbone = get_peft_model(backbone, lora_config)  # type: ignore[arg-type]
    else:
        # Standard fine-tuning -> unfreeze all backbone parameters
        for p in backbone.parameters():
            p.requires_grad = True

    return backbone, hidden_size, use_reg_token


def _unwrap_peft_model(model: nn.Module) -> nn.Module:
    """
    Return the wrapped base model when model is a PEFT model, otherwise return model
    """
    get_base_model = getattr(model, "get_base_model", None)
    if callable(get_base_model):
        return cast(nn.Module, get_base_model())
    return model


class PatchEmbedding1D(nn.Module):
    """
    Summarize short windows of a sequence into patch tokens.

    Each patch is described by per-channel summary statistics plus a simple
    linear trend estimate with respect to time. This gives the transformer a
    compact representation of local shape without attending to every point.
    """
    def __init__(
            self,
            input_dim: int,
            d_model: int,
            patch_size: int,
            min_valid_points_per_patch: int = 1,
            dropout: float = 0.1,
    ):
        """
        Initialize the patch summarizer and its projection head
        """
        super().__init__()

        # patch validity checks
        if patch_size < 1:
            raise ValueError("patch_size must be >= 1")
        if min_valid_points_per_patch < 1:
            raise ValueError("min_valid_points_per_patch must be >= 1")
        self.input_dim = int(input_dim)
        self.patch_size = int(patch_size)
        self.min_valid_points_per_patch = int(min_valid_points_per_patch)
        patch_feature_dim = 5 * self.input_dim + 1
        self.proj = nn.Sequential(
            nn.Linear(patch_feature_dim, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def _pad_to_patch_multiple(
            self, inputs: torch.Tensor, times: torch.Tensor, attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Pad a batch so the sequence length is divisible by patch_size
        """

        bsz, seq_len, input_dim = inputs.shape
        remainder = seq_len % self.patch_size
        if remainder == 0:
            return inputs, times, attention_mask
        pad_len = self.patch_size - remainder
        inputs_pad = torch.zeros(bsz, pad_len, input_dim, dtype=inputs.dtype, device=inputs.device)
        times_pad = torch.zeros(bsz, pad_len, dtype=times.dtype, device=times.device)
        mask_pad = torch.zeros(bsz, pad_len, dtype=attention_mask.dtype, device=attention_mask.device)
        return torch.cat([inputs, inputs_pad], dim=1), torch.cat([times, times_pad], dim=1), torch.cat(
            [attention_mask, mask_pad], dim=1
        )

    def forward(self, inputs: torch.Tensor, times: torch.Tensor, attention_mask: torch.Tensor):
        """
        Convert a padded variable-length sequence into patch tokens.

        Statistics are computed for valid positions in the patch and converted into one vector that is then embedded into dimension of d_model
        """
        inputs = torch.nan_to_num(inputs, nan=0.0, posinf=0.0, neginf=0.0)
        times = torch.nan_to_num(times, nan=0.0, posinf=0.0, neginf=0.0)
        attention_mask = attention_mask.bool()

        inputs, times, attention_mask = self._pad_to_patch_multiple(inputs, times, attention_mask)

        batch_size, seq_len, input_dim = inputs.shape
        n_patches = seq_len // self.patch_size
        x_view = inputs.view(batch_size, n_patches, self.patch_size, input_dim)
        times_view = times.view(batch_size, n_patches, self.patch_size)
        mask_view = attention_mask.view(batch_size, n_patches, self.patch_size)

        valid_counts = mask_view.sum(dim=2)
        patch_mask = valid_counts >= self.min_valid_points_per_patch
        mask_f = mask_view.unsqueeze(-1).to(dtype=x_view.dtype)
        denominator = valid_counts.clamp_min(1).unsqueeze(-1).to(dtype=x_view.dtype)

        # --- basic statistics ---
        patch_mean = (x_view * mask_f).sum(dim=2) / denominator
        centered = torch.where(mask_view.unsqueeze(-1), x_view - patch_mean.unsqueeze(2), torch.zeros_like(x_view))
        patch_var = (centered ** 2).sum(dim=2) / denominator
        patch_std = torch.sqrt(torch.clamp(patch_var, min=1e-8))

        x_for_min = torch.where(mask_view.unsqueeze(-1), x_view, torch.full_like(x_view, 1e30))
        x_for_max = torch.where(mask_view.unsqueeze(-1), x_view, torch.full_like(x_view, -1e30))
        patch_min = x_for_min.min(dim=2).values
        patch_max = x_for_max.max(dim=2).values
        patch_min = torch.where(patch_mask.unsqueeze(-1), patch_min, torch.zeros_like(patch_min))
        patch_max = torch.where(patch_mask.unsqueeze(-1), patch_max, torch.zeros_like(patch_max))

        valid_fraction = valid_counts.to(dtype=x_view.dtype).unsqueeze(-1) / float(self.patch_size)

        time_mask_f = mask_view.to(dtype=times_view.dtype)
        time_denom = valid_counts.clamp_min(1).to(dtype=times_view.dtype)
        patch_times = (times_view * time_mask_f).sum(dim=2) / time_denom
        patch_times = torch.where(patch_mask, patch_times, torch.zeros_like(patch_times))

        # --- slope computation using cov(input, time) / var(time) ---
        t_mean = patch_times.unsqueeze(-1)
        t_centered = torch.where(mask_view, times_view - t_mean, torch.zeros_like(times_view))
        t_var = torch.clamp((t_centered ** 2).sum(dim=2, keepdim=True), min=1e-8)
        x_centered = torch.where(mask_view.unsqueeze(-1), x_view - patch_mean.unsqueeze(2), torch.zeros_like(x_view))
        cov_xt = (x_centered * t_centered.unsqueeze(-1)).sum(dim=2)
        patch_slope = cov_xt / t_var
        patch_slope = torch.where(patch_mask.unsqueeze(-1), patch_slope, torch.zeros_like(patch_slope))

        # --- serialization into vector and projection ---
        patch_features = torch.cat([patch_mean, patch_std, patch_min, patch_max, patch_slope, valid_fraction], dim=-1)
        patch_features = torch.nan_to_num(patch_features, nan=0.0, posinf=0.0, neginf=0.0)

        patch_tokens = self.dropout(self.norm(self.proj(patch_features)))
        patch_tokens = patch_tokens * patch_mask.unsqueeze(-1).to(patch_tokens.dtype)

        # sanity check
        if not patch_mask.any(dim=1).all():
            bad = (~patch_mask.any(dim=1)).nonzero(as_tuple=False).reshape(-1).tolist()
            raise ValueError(
                "Patch embedding produced samples with no valid patches. "
                f"Bad batch indices: {bad}. Try lowering MIN_VALID_POINTS_PER_PATCH or PATCH_SIZE."
            )

        return patch_tokens, patch_times, patch_mask


class TransformerSequenceBranch(nn.Module):
    """
    Encode time-series via transformer, Qwen, or Chronos-2 backbone

    Special CNN mode is also implemented as a non-transformer baseline for comparison
    """
    def __init__(
            self,
            branch_name: str,
            input_dim: int,
            d_model: int,
            n_heads: int,
            n_layers: int,
            ff_dim: int,
            dropout: float,
            max_position_embeddings: int = 8192,
            pooling_mode: str = "cls_mean",
            use_patch_embedding: bool = False,
            patch_size: int = 1,
            min_valid_points_per_patch: int = 1,
            encoder_type: str = "transformer",
            pretrained_model_name: Optional[str] = None,
            use_lora: bool = True,
            lora_r: int = 8,
            lora_alpha: int = 16,
            lora_dropout: float = 0.1,
            lora_bias: str = "none",
    ):
        """
        Initialize branch encoder based on encoder_type
        """
        super().__init__()
        self.branch_name = branch_name
        self.pooling_mode = pooling_mode
        self.use_patch_embedding = use_patch_embedding
        self.encoder_type = encoder_type.lower().strip()
        self.patch_size = int(patch_size)
        self.output_dim = int(d_model)
        if self.encoder_type not in {"transformer", "qwen", "chronos2", "cnn"}:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")

        # --- basic transformer ---
        if self.encoder_type == "transformer":
            if self.use_patch_embedding:
                self.patch_embed = PatchEmbedding1D(
                    input_dim=input_dim,
                    d_model=d_model,
                    patch_size=self.patch_size,
                    min_valid_points_per_patch=min_valid_points_per_patch,
                    dropout=dropout,
                )
                self.input_proj = None
            else:
                self.patch_embed = None
                self.input_proj = nn.Linear(input_dim, d_model)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            self.pos_encoder = LearnedPositionalEncoding(
                d_model=d_model, max_position_embeddings=max_position_embeddings, dropout=dropout
            )
            self.pre_norm = nn.LayerNorm(d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=ff_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            self.out_norm = nn.LayerNorm(d_model)
            self.dropout = nn.Dropout(dropout)
            nn.init.normal_(self.cls_token, std=0.02)

        # --- pretrained models ---
        elif self.encoder_type in {"qwen", "chronos2"}:
            pretrained_name = _resolve_pretrained_backbone_name(self.encoder_type, pretrained_model_name)
            self.backbone, hidden_size, self.use_reg_token = _load_pretrained_backbone(
                encoder_type=self.encoder_type,
                pretrained_model_name=pretrained_name,
                use_lora=use_lora,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                lora_bias=lora_bias,
            )
            self.output_dim = hidden_size
            self.out_norm = nn.LayerNorm(hidden_size)
            self.dropout = nn.Dropout(dropout)
            self.input_proj = None

            # Qwen uses PatchEmbeddings the same way as the default transformer model
            if self.encoder_type == "qwen":
                self.patch_embed = PatchEmbedding1D(
                    input_dim=input_dim,
                    d_model=hidden_size,
                    patch_size=self.patch_size,
                    min_valid_points_per_patch=min_valid_points_per_patch,
                    dropout=dropout,
                )
                self.qwen_pos_encoder = LearnedPositionalEncoding(
                    d_model=hidden_size, max_position_embeddings=max_position_embeddings, dropout=dropout
                )
                self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
                nn.init.normal_(self.cls_token, std=0.02)
            # Chronos-2 handles its own internal patching/encoding.
            else:
                self.patch_embed = None
                self.qwen_pos_encoder = None
                self.cls_token = None

        else:  # special cnn baseline - 2-layer CNN with stride 3. Use single-channel (primary flux) as input
            self.backbone = None
            self.output_dim = d_model
            self.out_norm = nn.LayerNorm(d_model)
            self.dropout = nn.Dropout(dropout)
            self.input_proj = None
            self.patch_embed = None
            self.qwen_pos_encoder = None
            self.cls_token = None
            self.cnn = nn.Sequential(
                nn.Conv1d(1, 64, kernel_size=5, stride=3, padding=2),
                nn.ReLU(),
                nn.Conv1d(64, 128, kernel_size=5, stride=3, padding=2),
                nn.ReLU(),
            )
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.proj = nn.Linear(128, d_model)

    @staticmethod
    def _primary_signal(inputs: torch.Tensor) -> torch.Tensor:
        """
        Extract the primary flux signal used by pretrained backbones and CNN
        """

        if inputs.dim() == 3:
            signal = inputs[..., 0]
        elif inputs.dim() == 2:
            signal = inputs
        else:
            raise ValueError(f"Expected a 2D or 3D tensor, got shape {tuple(inputs.shape)}")
        return torch.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def masked_mean_pool(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Compute a mean over valid tokens only
        """
        mask_f = mask.unsqueeze(-1).to(x.dtype)
        denom = torch.clamp(mask_f.sum(dim=1), min=1.0)
        return (x * mask_f).sum(dim=1) / denom

    @staticmethod
    def _chronos2_patch_mask(
            point_mask: torch.Tensor,
            context_hidden: torch.Tensor,
            chronos_config,
    ) -> torch.Tensor:
        """
        Create a valid mask for Chronos-2 context tokens.
        
        Since Chronos-2 uses internal patching that we cannot reliably estimate from config,
        we take a conservative approach: all context tokens are treated as valid.
        This is safest because Chronos-2 has already filtered/processed the input internally.
        """
        # all context tokens from Chronos-2 are treated as valid
        return torch.ones(
            context_hidden.size(0),
            context_hidden.size(1),
            dtype=torch.bool,
            device=context_hidden.device,
        )

    def forward(self, inputs: torch.Tensor, times: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Encode a branch and pool it into a single fixed-size representation
        """
        if self.encoder_type == "chronos2":
            signal = self._primary_signal(inputs)
            context_mask = attention_mask.to(dtype=signal.dtype)

            encoded_outputs, _, _, num_context_patches = self.backbone.encode(
                context=signal,
                context_mask=context_mask,
                group_ids=None,
                num_output_patches=1,
                output_attentions=False,
            )

            hidden_states = getattr(encoded_outputs, "last_hidden_state", None)
            if hidden_states is None:
                hidden_states = encoded_outputs[0]

            context_hidden = hidden_states[:, :num_context_patches, :]

            base_model = _unwrap_peft_model(self.backbone)
            chronos_config = getattr(base_model, "chronos_config", None)
            if chronos_config is None:
                # fallback: all returned context tokens are treated as valid.
                patch_mask = torch.ones(
                    context_hidden.size(0),
                    context_hidden.size(1),
                    dtype=torch.bool,
                    device=context_hidden.device,
                )
            else:
                patch_mask = self._chronos2_patch_mask(attention_mask, context_hidden, chronos_config)
                
                # Ensure patch_mask matches context_hidden size (Chronos-2 may have different internal patching)
                if patch_mask.size(1) != context_hidden.size(1):
                    if patch_mask.size(1) < context_hidden.size(1):
                        # Pad if too small
                        pad_size = context_hidden.size(1) - patch_mask.size(1)
                        pad = torch.ones(
                            patch_mask.size(0),
                            pad_size,
                            dtype=torch.bool,
                            device=patch_mask.device,
                        )
                        patch_mask = torch.cat([patch_mask, pad], dim=1)
                    else:
                        # Trim if too large
                        patch_mask = patch_mask[:, :context_hidden.size(1)]

            pooled = self.masked_mean_pool(context_hidden, patch_mask)
            pooled = self.out_norm(pooled)
            return self.dropout(pooled)

        if self.encoder_type == "qwen":
            patch_tokens, _, patch_mask = self.patch_embed(inputs=inputs, times=times, attention_mask=attention_mask)
            patch_tokens = self.qwen_pos_encoder(patch_tokens, attention_mask=patch_mask)

            batch_size = patch_tokens.size(0)
            cls_token = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([patch_tokens, cls_token], dim=1)
            full_mask = torch.cat(
                [patch_mask, torch.ones((batch_size, 1), dtype=torch.bool, device=x.device)],
                dim=1,
            )
            encoded = self.backbone(
                inputs_embeds=x,
                attention_mask=full_mask.long(),
                output_hidden_states=False,
                return_dict=True,
            )
            pooled = encoded.last_hidden_state[:, -1]
            pooled = self.out_norm(pooled)
            return self.dropout(pooled)

        elif self.encoder_type == "cnn":
            # Use the same primary flux signal as Chronos-2 (single channel)
            # inputs: (batch, seq_len, input_dim) or (batch, seq_len)
            signal = self._primary_signal(inputs)
            x = signal.unsqueeze(1)
            x = self.cnn(x)
            x = self.pool(x).squeeze(-1)
            x = self.proj(x)
            x = self.out_norm(x)
            return self.dropout(x)

        if self.use_patch_embedding:
            x, times, attention_mask = self.patch_embed(inputs=inputs, times=times, attention_mask=attention_mask)
        else:
            x = self.input_proj(inputs)

        x = self.pre_norm(self.pos_encoder(x, attention_mask=attention_mask))

        batch_size = x.size(0)
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_token, x], dim=1)

        cls_mask = torch.ones((batch_size, 1), dtype=torch.bool, device=x.device)
        full_mask = torch.cat([cls_mask, attention_mask], dim=1)

        encoded = self.encoder(x, src_key_padding_mask=~full_mask)
        encoded = self.out_norm(encoded)

        cls_vec = encoded[:, 0]
        seq_vec = encoded[:, 1:]

        if self.pooling_mode == "cls":
            pooled = cls_vec
        elif self.pooling_mode == "mean":
            pooled = self.masked_mean_pool(seq_vec, attention_mask)
        elif self.pooling_mode == "cls_mean":
            pooled = 0.5 * (cls_vec + self.masked_mean_pool(seq_vec, attention_mask))
        else:
            raise ValueError(f"Unknown pooling_mode: {self.pooling_mode}")

        return self.dropout(pooled)


class ExtraFeaturesBranch(nn.Module):
    """
    Small 2-layer MLP for scalar non-sequence features
    """
    def __init__(self, input_dim: int, hidden_dim: int, out_dim: int, dropout: float):
        """
        MLP Construction
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """
        Project the extra features into the shared fusion space
        """
        return self.net(x)


class MultiBranchClassifier(nn.Module):
    """
    Fuse all enabled branches and output class logits
    """
    def __init__(
            self,
            cfg: ExperimentConfig,
            n_classes: int,
            branch_input_dim: int,
            d_model: int,
            n_heads: int,
            n_layers: int,
            ff_dim: int,
            dropout: float,
            extra_input_dim: int = 0,
            extra_hidden_dim: int = 128,
            extra_out_dim: int = 128,
    ):
        """
        Build the multimodal fusion classifier from the enabled branches
        """
        super().__init__()
        self.cfg = cfg

        # use the SimpleConfig so the object supports to_dict() (required by HF integrations)
        self.config = SimpleConfig(problem_type="single_label_classification", num_labels=int(n_classes))
        self.flux_branches = enabled_flux_branches(cfg)
        self.use_phase_branch = cfg.USE_PHASE_BRANCH
        self.use_extra_branch = cfg.USE_EXTRA_FEATURES_BRANCH
        if len(self.flux_branches) == 0 and not self.use_phase_branch and not self.use_extra_branch:
            raise ValueError("At least one branch must be enabled.")
        self.encoder_type = cfg.BRANCH_ENCODER_TYPE.lower().strip()
        self.sequence_branches = nn.ModuleDict()

        # --- flux branches ---
        for branch in self.flux_branches:
            self.sequence_branches[branch] = TransformerSequenceBranch(
                branch_name=branch,
                input_dim=branch_input_dim,
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_layers,
                ff_dim=ff_dim,
                dropout=dropout,
                max_position_embeddings=cfg.MAX_POSITION_EMBEDDINGS,
                pooling_mode=cfg.POOLING_MODE,
                use_patch_embedding=cfg.USE_PATCH_EMBEDDING,
                patch_size=get_patch_size(cfg, branch),
                min_valid_points_per_patch=cfg.MIN_VALID_POINTS_PER_PATCH,
                encoder_type=self.encoder_type,
                pretrained_model_name=cfg.BACKBONE_MODEL_NAME,
                use_lora=cfg.USE_LORA,
                lora_r=cfg.LORA_R,
                lora_alpha=cfg.LORA_ALPHA,
                lora_dropout=cfg.LORA_DROPOUT,
                lora_bias=cfg.LORA_BIAS,
            )

        # --- phase branch ---
        self.phase_branch = None
        if self.use_phase_branch:
            self.phase_branch = TransformerSequenceBranch(
                branch_name="phase",
                input_dim=branch_input_dim,
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_layers,
                ff_dim=ff_dim,
                dropout=dropout,
                max_position_embeddings=cfg.MAX_POSITION_EMBEDDINGS,
                pooling_mode=cfg.POOLING_MODE,
                use_patch_embedding=cfg.USE_PATCH_EMBEDDING,
                patch_size=get_patch_size(cfg, "phase"),
                min_valid_points_per_patch=cfg.MIN_VALID_POINTS_PER_PATCH,
                encoder_type=self.encoder_type,
                pretrained_model_name=cfg.BACKBONE_MODEL_NAME,
                use_lora=cfg.USE_LORA,
                lora_r=cfg.LORA_R,
                lora_alpha=cfg.LORA_ALPHA,
                lora_dropout=cfg.LORA_DROPOUT,
                lora_bias=cfg.LORA_BIAS,
            )

        # --- extra branch ---
        self.extra_branch = None
        if self.use_extra_branch:
            self.extra_branch = ExtraFeaturesBranch(
                input_dim=extra_input_dim,
                hidden_dim=extra_hidden_dim,
                out_dim=extra_out_dim,
                dropout=dropout,
            )

        # --- fusion layer using a simple 2 Layer MLP ---
        fusion_dim = sum(branch.output_dim for branch in self.sequence_branches.values())
        if self.use_phase_branch:
            fusion_dim += self.phase_branch.output_dim
        if self.use_extra_branch:
            fusion_dim += extra_out_dim

        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, cfg.FUSION_HIDDEN),
            nn.LayerNorm(cfg.FUSION_HIDDEN),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(cfg.FUSION_HIDDEN, n_classes),
        )

    def forward(self, labels=None, **kwargs):
        """
        Run each enabled branch and fuse the resulting vectors
        """
        branch_vectors = []
        for branch in self.flux_branches:
            vec = self.sequence_branches[branch](
                inputs=kwargs[f"{branch}_inputs"],
                times=kwargs[f"{branch}_times"],
                attention_mask=kwargs[f"{branch}_attention_mask"],
            )
            branch_vectors.append(vec)

        if self.use_phase_branch:
            phase_vec = self.phase_branch(
                inputs=kwargs["phase_inputs"],
                times=kwargs["phase_times"],
                attention_mask=kwargs["phase_attention_mask"],
            )
            branch_vectors.append(phase_vec)

        if self.use_extra_branch:
            branch_vectors.append(self.extra_branch(kwargs["extra_features"]))

        logits = self.fusion(torch.cat(branch_vectors, dim=-1))
        return {"logits": logits}
