# Copyright (c) ModelScope Contributors. All rights reserved.
"""Qwen3.5 -> spatiotemporal latent resampler -> V-RAE native latent.

Implements `qwen_vrae_future_video_model_structure.md`: the context clip and its
instruction go through Qwen3.5, a resampler turns that variable-length context into the
structured latent grid the frozen V-RAE decoder consumes, and the loss compares the
prediction against the cached target latent.

Shapes for the configured setup (40 context frames, 40 target frames, 256x448):

    Qwen video tokens (after 2x2 merge)   20 x  8 x 14 = 2240
    target latent grid                    10 x 16 x 28 = 4480 tokens, D_z = 1024

The two grids share the same patch size, so space matches exactly and time is a clean
2x -- the resampler therefore initializes its queries by upsampling the context rather
than starting from noise.

The model returns its own ``loss``; the trainer takes it directly when ``labels`` stays
in ``inputs`` (swift/trainers/seq2seq_trainer.py), which is why the dataset emits a
placeholder assistant turn and why ``--loss_type`` must not be passed.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from swift.model import Model, ModelGroup, ModelMeta, MultiModelKeys, register_model, register_model_arch
from swift.model.models.qwen import Qwen3_5Loader
from swift.utils import get_env_args, get_logger

logger = get_logger()

MODEL_TYPE = 'qwen3_5_vrae'
TEMPLATE_TYPE = 'qwen3_5_vrae'
MODEL_ARCH = 'qwen3_5_vrae'

# The stock `qwen3_5` arch names modules as `model.language_model`, `model.visual`, ...
# Those paths are turned into a `^`-anchored regex by get_multimodal_target_regex
# (swift/utils/transformers_utils.py), so they have to spell the real parameter names.
# Here the backbone sits one level down under `qwen`, and using the stock arch would
# build a regex that matches nothing -- LoRA would attach to zero modules without any
# error, because deep_getattr still resolves through this class's `model` property.
register_model_arch(
    MultiModelKeys(
        MODEL_ARCH,
        language_model=['qwen.model.language_model', 'qwen.lm_head'],
        aligner='qwen.model.visual.merger',
        vision_tower='qwen.model.visual',
        mlp='qwen.model.language_model.layers.{}.mlp',
    ),
    exist_ok=True,
)


@dataclass
class VRAEPredictorOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    # The trainer reads `outputs.logits` when computing token accuracy
    # (swift/trainers/seq2seq_trainer.py); leaving it None makes it skip that, which is
    # what we want since this head predicts latents rather than tokens.
    logits: Optional[torch.Tensor] = None
    z_pred: Optional[torch.Tensor] = None
    loss_cos: Optional[torch.Tensor] = None
    loss_mse: Optional[torch.Tensor] = None
    loss_temp: Optional[torch.Tensor] = None


class LatentNormalization(nn.Module):
    """Per-channel standardization of V-RAE latents.

    The V-RAE pipeline caches raw, unnormalized latents and normalizes at training time
    (see vrae/training/common/latent_norm.py).  Doing the same here matters for more
    than convention: the cosine term is scale invariant, so without standardization the
    only term carrying magnitude is the down-weighted MSE, and the decoder needs
    magnitude to reconstruct anything.
    """

    def __init__(self, channels: int, stats_path: Optional[str] = None) -> None:
        super().__init__()
        mean = torch.zeros(channels)
        std = torch.ones(channels)
        self.fitted = False
        if stats_path:
            if not Path(stats_path).is_file():
                # Better to stop than to quietly train on raw latents after the run was
                # configured to normalize.
                raise FileNotFoundError(
                    f'LATENT_STATS_PATH points at {stats_path}, which does not exist. Fit it '
                    f'with:\n  python examples/vrae_future_pred/latent_stats.py '
                    f'--cache-root $VPDATA_LATENT_ROOT --out {stats_path}\n'
                    f'or unset LATENT_STATS_PATH to train on raw latents.')
            payload = torch.load(stats_path, map_location='cpu', weights_only=True)
            mean = torch.as_tensor(payload['mean']).flatten().float()
            std = torch.as_tensor(payload['std']).flatten().float()
            if mean.numel() != channels or std.numel() != channels:
                raise ValueError(f'latent stats have {mean.numel()} channels, expected {channels}')
            if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or torch.any(std <= 0):
                raise ValueError(f'latent stats must be finite with positive std: {stats_path}')
            self.fitted = True
            logger.info(f'loaded latent statistics from {stats_path}')
        else:
            logger.warning('no latent statistics given; losses run on raw latents. Fit them with '
                           'examples/vrae_future_pred/latent_stats.py -- the cosine term is scale '
                           'invariant, so unnormalized training under-weights magnitude.')
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

    def normalize(self, latents: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(latents.device, latents.dtype)
        std = self.std.to(latents.device, latents.dtype)
        return (latents - mean) / std

    def denormalize(self, latents: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(latents.device, latents.dtype)
        std = self.std.to(latents.device, latents.dtype)
        return latents * std + mean


class ResamplerBlock(nn.Module):
    """(optional self-attention over queries) -> cross-attention to context -> FFN.

    ``self_attn=False`` is the weak-resampler variant: queries no longer talk to each
    other, so the only place future positions can be reasoned about jointly is inside
    Qwen. That is the point of the ablation -- with the stack left in, a 71 M resampler
    can do the future reasoning itself and the experiment stops measuring what Qwen
    learned.
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 self_attn: bool = True) -> None:
        super().__init__()
        self.norm_self = nn.LayerNorm(dim) if self_attn else None
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True) if self_attn else None
        self.norm_cross_q = nn.LayerNorm(dim)
        self.norm_cross_kv = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        hidden = int(dim * mlp_ratio)
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, queries: torch.Tensor, context: torch.Tensor,
                context_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.self_attn is not None:
            normed = self.norm_self(queries)
            queries = queries + self.self_attn(normed, normed, normed, need_weights=False)[0]
        queries = queries + self.cross_attn(
            self.norm_cross_q(queries),
            self.norm_cross_kv(context),
            self.norm_cross_kv(context),
            key_padding_mask=context_mask,
            need_weights=False,
        )[0]
        return queries + self.ffn(self.norm_ffn(queries))


class SpatiotemporalLatentResampler(nn.Module):
    """Map Qwen context onto the ``T_c x H_f x W_f`` latent grid.

    Each query keeps an explicit spatiotemporal identity, per section 4 of the design
    doc: ``q = q_base + e_t(t) + e_h(h) + e_w(w)``.
    """

    def __init__(
        self,
        context_dim: int,
        dim: int,
        grid: Tuple[int, int, int],
        depth: int = 4,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        self_attn: bool = True,
    ) -> None:
        super().__init__()
        self.grid = grid
        chunks, height, width = grid
        self.num_queries = chunks * height * width
        self.context_proj = nn.Sequential(nn.LayerNorm(context_dim), nn.Linear(context_dim, dim))
        self.query_base = nn.Parameter(torch.zeros(1, 1, dim))
        self.time_embed = nn.Parameter(torch.zeros(chunks, dim))
        self.height_embed = nn.Parameter(torch.zeros(height, dim))
        self.width_embed = nn.Parameter(torch.zeros(width, dim))
        self.blocks = nn.ModuleList(
            [ResamplerBlock(dim, num_heads, mlp_ratio, self_attn=self_attn) for _ in range(depth)])
        self.norm_out = nn.LayerNorm(dim)
        for parameter in (self.query_base, self.time_embed, self.height_embed, self.width_embed):
            nn.init.trunc_normal_(parameter, std=0.02)

    def build_queries(self, batch_size: int, device, dtype) -> torch.Tensor:
        chunks, height, width = self.grid
        time = self.time_embed[:, None, None, :]
        rows = self.height_embed[None, :, None, :]
        cols = self.width_embed[None, None, :, :]
        queries = self.query_base.view(1, 1, 1, -1) + time + rows + cols
        queries = queries.reshape(1, self.num_queries, -1)
        return queries.expand(batch_size, -1, -1).to(device=device, dtype=dtype)

    def forward(self, context: torch.Tensor, context_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        context = self.context_proj(context)
        queries = self.build_queries(context.shape[0], context.device, context.dtype)
        for block in self.blocks:
            queries = block(queries, context, context_mask)
        return self.norm_out(queries)


class LatentHead(nn.Module):
    """Deliberately shallow, per section 5: the head only changes coordinates."""

    def __init__(self, dim: int, latent_dim: int) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(dim) if hasattr(nn, 'RMSNorm') else nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, latent_dim)
        nn.init.zeros_(self.proj.bias)
        nn.init.trunc_normal_(self.proj.weight, std=0.02)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.proj(self.norm(hidden))


class QwenVRAEFuturePredictorConfig(PretrainedConfig):
    model_type = MODEL_TYPE

    def __init__(
        self,
        latent_chunks: int = 10,
        latent_height: int = 16,
        latent_width: int = 28,
        latent_dim: int = 1024,
        resampler_dim: int = 1024,
        resampler_depth: int = 4,
        resampler_heads: int = 16,
        resampler_self_attn: bool = True,
        lambda_mse: float = 0.1,
        lambda_temporal: float = 0.1,
        latent_stats_path: Optional[str] = None,
        **kwargs,
    ) -> None:
        self.latent_chunks = latent_chunks
        self.latent_height = latent_height
        self.latent_width = latent_width
        self.latent_dim = latent_dim
        self.resampler_dim = resampler_dim
        self.resampler_depth = resampler_depth
        self.resampler_heads = resampler_heads
        self.resampler_self_attn = resampler_self_attn
        self.lambda_mse = lambda_mse
        self.lambda_temporal = lambda_temporal
        self.latent_stats_path = latent_stats_path
        super().__init__(**kwargs)


class QwenVRAEFuturePredictor(PreTrainedModel):
    """Qwen3.5 backbone with a latent-prediction head bolted on."""

    config_class = QwenVRAEFuturePredictorConfig
    supports_gradient_checkpointing = True
    _supports_flash_attn = True
    _supports_sdpa = True

    def __init__(self, qwen: PreTrainedModel, config: QwenVRAEFuturePredictorConfig) -> None:
        # Reuse the backbone's own config for everything transformers introspects
        # (dtype, gradient checkpointing, device placement), and keep ours alongside.
        super().__init__(qwen.config)
        self.qwen = qwen
        self.vrae_config = config
        context_dim = self._text_hidden_size(qwen.config)
        self.resampler = SpatiotemporalLatentResampler(
            context_dim=context_dim,
            dim=config.resampler_dim,
            grid=(config.latent_chunks, config.latent_height, config.latent_width),
            depth=config.resampler_depth,
            num_heads=config.resampler_heads,
            self_attn=config.resampler_self_attn,
        )
        self.latent_head = LatentHead(config.resampler_dim, config.latent_dim)
        self.latent_norm = LatentNormalization(config.latent_dim, config.latent_stats_path)
        trainable = sum(p.numel() for p in self.resampler.parameters()) + \
            sum(p.numel() for p in self.latent_head.parameters())
        logger.info(f'resampler queries: {self.resampler.num_queries} '
                    f'(grid {config.latent_chunks}x{config.latent_height}x{config.latent_width}), '
                    f'context_dim {context_dim} -> {config.resampler_dim}, D_z {config.latent_dim}, '
                    f'depth {config.resampler_depth}, self_attn {config.resampler_self_attn}, '
                    f'resampler+head params {trainable / 1e6:.2f}M')

    @staticmethod
    def _text_hidden_size(qwen_config) -> int:
        text_config = getattr(qwen_config, 'text_config', None)
        size = getattr(text_config, 'hidden_size', None) if text_config is not None else None
        return int(size or qwen_config.hidden_size)

    # -- Delegation so swift/peft/transformers see the backbone's structure ----------
    def get_input_embeddings(self):
        return self.qwen.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.qwen.set_input_embeddings(value)

    def gradient_checkpointing_enable(self, **kwargs):
        self.qwen.gradient_checkpointing_enable(**kwargs)

    def prepare_inputs_for_generation(self, *args, **kwargs):
        # PeftModelForCausalLM.__init__ reads this off the base model even when nothing
        # will ever call generate() on it.
        return self.qwen.prepare_inputs_for_generation(*args, **kwargs)

    def get_output_embeddings(self):
        return self.qwen.get_output_embeddings()

    @property
    def visual(self):
        return getattr(self.qwen, 'visual', None)

    @property
    def model(self):
        return getattr(self.qwen, 'model', self.qwen)

    # -- Loss -----------------------------------------------------------------------
    def latent_losses(self, z_pred: torch.Tensor, z_target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """L = L_cos + lambda_mse * L_MSE + lambda_temporal * L_temp, on normalized latents."""

        pred = self.latent_norm.normalize(z_pred.float())
        target = self.latent_norm.normalize(z_target.float())
        loss_cos = (1.0 - F.cosine_similarity(pred, target, dim=-1)).mean()
        loss_mse = F.mse_loss(pred, target)
        if pred.shape[1] > 1:
            loss_temp = (pred.diff(dim=1) - target.diff(dim=1)).abs().mean()
        else:
            loss_temp = pred.new_zeros(())
        total = (loss_cos + self.vrae_config.lambda_mse * loss_mse
                 + self.vrae_config.lambda_temporal * loss_temp)
        return {'loss': total, 'loss_cos': loss_cos, 'loss_mse': loss_mse, 'loss_temp': loss_temp}

    def forward(
        self,
        target_latent: Optional[torch.Tensor] = None,
        grid_hw: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> VRAEPredictorOutput:
        # `labels` is the placeholder assistant turn; the backbone must not spend
        # compute on a cross-entropy we discard.
        kwargs.pop('compute_loss_func', None)
        # The lm_head is 4096 x 248320, so materializing logits over the full ~2.3k-token
        # sequence costs about 1.1 GB per rank -- all of it discarded, since this model
        # scores latents rather than tokens. Ask for a single position instead.
        # Qwen3_5ForConditionalGeneration.forward slices only the lm_head *input*
        # (`hidden_states[:, slice_indices, :]`), so the `output_hidden_states` tuple
        # read below still covers the whole sequence.
        kwargs['logits_to_keep'] = 1
        # Override rather than pass alongside: the caller already puts
        # output_hidden_states / use_cache in kwargs, and passing them again as explicit
        # keywords raises "got multiple values for keyword argument".
        kwargs.update({'output_hidden_states': True, 'use_cache': False})
        outputs = self.qwen(attention_mask=attention_mask, **kwargs)
        hidden = outputs.hidden_states[-1]

        context_mask = None
        if attention_mask is not None and attention_mask.dim() == 2:
            # MultiheadAttention wants True where a key must be ignored.
            context_mask = attention_mask == 0
            if context_mask.shape[1] != hidden.shape[1]:
                context_mask = None

        resampled = self.resampler(hidden, context_mask)
        z_pred = self.latent_head(resampled)
        chunks = self.vrae_config.latent_chunks
        tokens = self.vrae_config.latent_height * self.vrae_config.latent_width
        z_pred = z_pred.view(z_pred.shape[0], chunks, tokens, self.vrae_config.latent_dim)

        if target_latent is None:
            return VRAEPredictorOutput(z_pred=z_pred)
        if target_latent.shape[1:] != z_pred.shape[1:]:
            raise ValueError(f'target latent {tuple(target_latent.shape)} does not match prediction '
                             f'{tuple(z_pred.shape)}; check --image-size and --num-frames of the cache')
        losses = self.latent_losses(z_pred, target_latent.to(z_pred.device))
        return VRAEPredictorOutput(z_pred=z_pred, **losses)


class Qwen35VRAELoader(Qwen3_5Loader):
    """Load the released Qwen3.5 weights, then wrap them with the latent predictor."""

    def get_model(self, model_dir: str, config, processor, model_kwargs) -> PreTrainedModel:
        qwen = super().get_model(model_dir, config, processor, model_kwargs)
        predictor_config = QwenVRAEFuturePredictorConfig(
            latent_chunks=get_env_args('latent_chunks', int, 10),
            latent_height=get_env_args('latent_height', int, 16),
            latent_width=get_env_args('latent_width', int, 28),
            latent_dim=get_env_args('latent_dim', int, 1024),
            resampler_dim=get_env_args('resampler_dim', int, 1024),
            resampler_depth=get_env_args('resampler_depth', int, 4),
            resampler_heads=get_env_args('resampler_heads', int, 16),
            resampler_self_attn=bool(get_env_args('resampler_self_attn', int, 1)),
            lambda_mse=get_env_args('lambda_mse', float, 0.1),
            lambda_temporal=get_env_args('lambda_temporal', float, 0.1),
            latent_stats_path=os.environ.get('LATENT_STATS_PATH') or None,
        )
        model = QwenVRAEFuturePredictor(qwen, predictor_config)
        # Device as well as dtype: these two modules are constructed on CPU after
        # device_map has already placed the backbone. Under swift training the Trainer
        # moves the whole model and the omission is invisible, but any script that calls
        # get_model_processor and uses the model directly hits "weight is on cpu,
        # different from other tensors on cuda:0" in the resampler's first LayerNorm.
        reference = next(qwen.parameters())
        dtype = getattr(qwen, 'dtype', None) or reference.dtype
        model.resampler.to(device=reference.device, dtype=dtype)
        model.latent_head.to(device=reference.device, dtype=dtype)
        model.latent_norm.to(device=reference.device)
        return model


register_model(
    ModelMeta(
        MODEL_TYPE,
        [ModelGroup([Model('Qwen/Qwen3.5-9B', 'Qwen/Qwen3.5-9B')], TEMPLATE_TYPE)],
        Qwen35VRAELoader,
        model_arch=MODEL_ARCH,
        architectures=['Qwen3_5ForConditionalGeneration'],
        requires=['transformers>=4.57', 'qwen_vl_utils>=0.0.14'],
        tags=['vision', 'video', 'vrae'],
        is_multimodal=True,
    ),
    exist_ok=True,
)
