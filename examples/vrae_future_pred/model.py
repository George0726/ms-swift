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
    rollout_metrics: Optional[Dict[str, torch.Tensor]] = None


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

    def project_context(self, context: torch.Tensor) -> torch.Tensor:
        """The step-invariant half of the rollout, hoisted so it runs once.

        ``context_proj`` is a LayerNorm + Linear over the whole Qwen sequence
        ([B, ~2543, 4096] -> 1024). Calling ``forward`` once per rollout step used to
        redo it every step against the same hidden states: about a quarter of the
        predictor's per-step cost, plus a retained copy per step for the backward pass.
        """
        return self.context_proj(context)

    def forward(self, context: torch.Tensor, context_mask: Optional[torch.Tensor] = None,
                feedback: Optional[torch.Tensor] = None, *, projected: bool = False,
                step: Optional[int] = None) -> torch.Tensor:
        if not projected:
            context = self.project_context(context)
        queries = self.build_queries(context.shape[0], context.device, context.dtype)
        # `step` is the future time index during a rollout. Without it the transition is
        # time-invariant -- the queries are identical at every step and the only thing
        # that changes is the feedback, so on a near-static clip the model cannot tell
        # step 1 from step 9 and has no way to know how much motion to apply.
        if step is not None:
            queries = queries + self.step_embed[step].to(dtype=queries.dtype)
        if feedback is not None:
            queries = queries + self.feedback_proj(feedback.to(context.dtype))
        for block in self.blocks:
            queries = block(queries, context, context_mask)
        return self.norm_out(queries)


class StructuredFutureEmbeddings(nn.Module):
    """Continuous ``(time, row, column)`` positions consumed by Qwen itself."""

    def __init__(self, dim: int, grid: Tuple[int, int, int]) -> None:
        super().__init__()
        self.grid = grid
        chunks, height, width = grid
        self.num_positions = chunks * height * width
        self.base = nn.Parameter(torch.zeros(1, 1, dim))
        self.time_embed = nn.Parameter(torch.zeros(chunks, dim))
        self.height_embed = nn.Parameter(torch.zeros(height, dim))
        self.width_embed = nn.Parameter(torch.zeros(width, dim))
        for parameter in (self.base, self.time_embed, self.height_embed, self.width_embed):
            nn.init.trunc_normal_(parameter, std=0.02)

    def forward(self, batch_size: int, *, device, dtype) -> torch.Tensor:
        time = self.time_embed[:, None, None, :]
        rows = self.height_embed[None, :, None, :]
        cols = self.width_embed[None, None, :, :]
        positions = self.base.view(1, 1, 1, -1) + time + rows + cols
        positions = positions.reshape(1, self.num_positions, -1)
        return positions.expand(batch_size, -1, -1).to(device=device, dtype=dtype)


class ContextSliceExpansion(nn.Module):
    """Map context video-token hidden states onto one future latent chunk, in place.

    Adds nothing to Qwen's input. The geometry closes exactly, measured on trunk0:

        Qwen video tokens   video_grid_thw [20,16,28], spatial_merge_size 2
                            -> 20 x 8 x 14 = 2240 positions
        V-RAE target        10 x 16 x 28   = 4480 tokens

    2240 x 2 = 4480. Time is 2x finer on the Qwen side (20 context slices vs 10 future
    chunks) and space 2x coarser (Qwen merged each 2x2 block of 16-px patches, and those
    patches are exactly the V-RAE grid). So future chunk k reads context slices
    {s*k ... s*k+s-1} and every merged cell is expanded back into the 2x2 block it
    covers -- the inverse of the vision merge, not an arbitrary upsample.

    Note the pairing is by index, not by physics: slice 2k is that far into the past,
    chunk k that far into the future. It is monotone and gives every chunk a distinct
    conditioning (which is what stops the transition from being time-invariant); what it
    *means* is left to the model. The within-cell assignment of the m*m outputs is free
    -- `proj` can permute its own output groups -- so only the coarse placement of cell
    (h, w) onto block (m*h, m*w) has to be right, and that follows from the merged tokens
    being row-major over (t, h, w).
    """

    def __init__(self, context_dim: int, dim: int, slices_per_chunk: int, merge_size: int,
                 merge_w: Optional[int] = None) -> None:
        super().__init__()
        self.slices_per_chunk = slices_per_chunk
        self.merge_size = merge_size
        # Rectangular factors so the same module serves the in-sequence latent tokens,
        # whose per-chunk grid need not be a square fraction of the V-RAE grid.
        self.merge_h = merge_size
        self.merge_w = merge_size if merge_w is None else merge_w
        fan_in = context_dim * slices_per_chunk
        self.norm = nn.LayerNorm(fan_in)
        self.proj = nn.Linear(fan_in, dim * self.merge_h * self.merge_w)

    def forward(self, cells: torch.Tensor) -> torch.Tensor:
        """[B, slices, H_q, W_q, D_ctx] -> [B, H_q*m * W_q*m, dim], row-major over (h, w)."""
        batch, slices, height, width, _ = cells.shape
        if slices != self.slices_per_chunk:
            raise ValueError(f'expected {self.slices_per_chunk} context slices per chunk, got {slices}')
        # Fold the slices into the feature dimension rather than averaging them: which
        # slice a hidden state came from is information, and the factor is small (2).
        folded = cells.permute(0, 2, 3, 1, 4).reshape(batch, height, width, -1)
        out = self.proj(self.norm(folded))
        out = out.view(batch, height, width, self.merge_h, self.merge_w, -1)
        # (h, i, w, j) flattens to (h*f_h + i) * (W*f_w) + (w*f_w + j), i.e. row-major on
        # the full V-RAE grid -- the order target_latent's 448 tokens are stored in.
        out = out.permute(0, 1, 3, 2, 4, 5)
        return out.reshape(batch, height * self.merge_h * width * self.merge_w, -1)


class InterleavedLatentTokens(nn.Module):
    """Latent positions that live *inside* Qwen's sequence and are generated by it.

    This is the Future-L1 mechanism (arXiv:2606.05769) adapted to a V-RAE target: the
    hidden state produced at a latent position is what predicts that chunk, and the
    previous chunk's latent is fed in as the *input embedding* of the next positions, so
    the recurrence runs through all 32 layers and stays in the KV cache. That is strictly
    deeper than folding feedback into a head after the backbone, which is what the
    `direct` and `slice` modes do.

    Training costs one forward, not ten: the fed-back value is the ground-truth previous
    chunk (teacher forcing), so every chunk's positions can sit in the same causal
    sequence and be trained in a single pass -- ordinary autoregressive training. Only
    true predicted feedback (eval, inference) has to run the steps sequentially.

    Token budget is the knob that matters. The paper sweeps span length 2..64 and finds
    *shorter is better* (best at 4, degrading by ~6 points at 64), while the V-RAE target
    needs 448 positions per chunk -- so the sequence carries a coarse `seq_h x seq_w` grid
    and an expansion recovers the full grid. Fewer sequence tokens means a larger
    expansion: 112 per chunk costs 16.8 M and 1120 added tokens, 16 per chunk costs 117 M
    and 160 added tokens.
    """

    def __init__(self, context_dim: int, latent_dim: int, dim: int, chunks: int,
                 seq_grid: Tuple[int, int], latent_grid: Tuple[int, int],
                 feedback_source: str = 'latent') -> None:
        super().__init__()
        seq_h, seq_w = seq_grid
        latent_h, latent_w = latent_grid
        if latent_h % seq_h or latent_w % seq_w:
            raise ValueError(f'latent grid {latent_h}x{latent_w} must be divisible by the '
                             f'in-sequence grid {seq_h}x{seq_w}')
        self.chunks = chunks
        self.seq_grid = (seq_h, seq_w)
        self.latent_grid = (latent_grid[0], latent_grid[1])
        self.pool = (latent_h // seq_h, latent_w // seq_w)
        self.tokens_per_chunk = seq_h * seq_w
        # Identity of each latent position, exactly as the direct mode builds it: the
        # sequence position already carries order through RoPE, this carries which chunk
        # and which cell within it.
        self.positions = StructuredFutureEmbeddings(context_dim, (chunks, seq_h, seq_w))
        # Feed-in: previous chunk latent, pooled onto the coarse grid, into Qwen's space.
        # Used by feedback_source='latent' at every step, and by 'hidden' for step 0 only
        # -- there is no previous hidden state then, and without this the model would be
        # asked to predict a delta on Z_0 without ever being told what Z_0 is.
        self.norm_in = nn.LayerNorm(latent_dim)
        self.in_proj = nn.Linear(latent_dim, context_dim)
        nn.init.trunc_normal_(self.in_proj.weight, std=0.02)
        nn.init.zeros_(self.in_proj.bias)
        # feedback_source='hidden' only: a learned opening block, and a norm on the fed
        # back hidden states (Qwen's input side normally sees token embeddings, and a raw
        # output hidden state is not drawn from that distribution).
        #
        # Built only for that variant, on purpose. Creating them unconditionally puts dead
        # weights in every 'latent' checkpoint and makes checkpoints from before/after the
        # change fail to load with a bare KeyError from PEFT's strict lookup.
        self.initial_query = None
        self.norm_feedback = None
        if feedback_source == 'hidden':
            self.initial_query = nn.Parameter(torch.zeros(1, self.tokens_per_chunk, context_dim))
            nn.init.trunc_normal_(self.initial_query, std=0.02)
            self.norm_feedback = nn.LayerNorm(context_dim)
        # Read-out: hidden state at each coarse cell -> the pool_h x pool_w block of
        # V-RAE positions it stands for.
        self.expand = ContextSliceExpansion(context_dim, dim, slices_per_chunk=1,
                                            merge_size=self.pool[0], merge_w=self.pool[1])

    def pool_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """[B, H*W, D_z] -> [B, seq_h*seq_w, D_z] by averaging each block."""
        batch = latent.shape[0]
        latent_h, latent_w = self.latent_grid
        seq_h, seq_w = self.seq_grid
        pool_h, pool_w = self.pool
        grid = latent.view(batch, latent_h, latent_w, -1)
        grid = grid.view(batch, seq_h, pool_h, seq_w, pool_w, -1)
        return grid.mean(dim=(2, 4)).reshape(batch, seq_h * seq_w, -1)

    def input_embeddings(self, previous: torch.Tensor, chunk: int, *, dtype) -> torch.Tensor:
        """Input embeddings for chunk `chunk`, conditioned on the previous chunk latent."""
        # Pool in fp32 for the averaging, then hand the module its own parameter dtype.
        # Feeding an fp32 tensor to a bf16 LayerNorm raises "expected scalar type Float
        # but found BFloat16" -- training hid it because the forward runs under autocast,
        # so it only surfaced in the plain-inference path (decode_eval).
        pooled = self.pool_latent(previous.float()).to(self.norm_in.weight.dtype)
        fed = self.in_proj(self.norm_in(pooled)).to(dtype)
        ident = self.positions(previous.shape[0], device=previous.device, dtype=dtype)
        start = chunk * self.tokens_per_chunk
        return fed + ident[:, start:start + self.tokens_per_chunk]

    def _identity(self, batch: int, chunk: int, *, device, dtype) -> torch.Tensor:
        ident = self.positions(batch, device=device, dtype=dtype)
        start = chunk * self.tokens_per_chunk
        return ident[:, start:start + self.tokens_per_chunk]

    def initial_embeddings(self, observed: torch.Tensor, *, dtype) -> torch.Tensor:
        """Step 0 for feedback_source='hidden': learned queries plus the observed chunk.

        The observed latent has to enter here. The residual is taken on Z_0, so a model
        that never sees Z_0 is being asked for a delta on an unknown quantity -- the one
        real bug in the obvious version of this loop.
        """
        pooled = self.pool_latent(observed.float()).to(self.norm_in.weight.dtype)
        fed = self.in_proj(self.norm_in(pooled)).to(dtype)
        query = self.initial_query.to(dtype).expand(observed.shape[0], -1, -1)
        return query + fed + self._identity(observed.shape[0], 0, device=observed.device, dtype=dtype)

    def hidden_embeddings(self, hidden: torch.Tensor, chunk: int, *, dtype) -> torch.Tensor:
        """Steps 1.. for feedback_source='hidden': the previous block's hidden states."""
        fed = self.norm_feedback(hidden.to(self.norm_feedback.weight.dtype)).to(dtype)
        return fed + self._identity(hidden.shape[0], chunk, device=hidden.device, dtype=dtype)

    def read_out(self, hidden: torch.Tensor) -> torch.Tensor:
        """[B, tokens_per_chunk, D_ctx] -> [B, H*W, dim]."""
        seq_h, seq_w = self.seq_grid
        cells = hidden.view(hidden.shape[0], 1, seq_h, seq_w, hidden.shape[-1])
        return self.expand(cells)


class LatentHead(nn.Module):
    """Deliberately shallow, per section 5: the head only changes coordinates."""

    # The residual rollout wants this head to start out doing almost nothing, so that the
    # rollout begins as the identity -- i.e. as the persistence baseline -- rather than at
    # the dataset mean. RMSNorm hands the projection unit-variance input, so the output
    # delta has std ~ init_std * sqrt(dim): the 0.02 default lands at ~0.5, the same order
    # as the latents themselves, which would mean adding half a latent of noise per step.
    #
    # Small rather than exactly zero, though. Measured on real trunk0 latents, a delta of
    # std 0.03 moves the starting loss by 0.0006 (0.4403 -> 0.4409), so there is nothing to
    # buy by going to zero -- and zero would make the gradient w.r.t. everything upstream
    # of this projection exactly zero on the first step, since it arrives multiplied by
    # these weights. RESIDUAL_INIT_STD=1e-3 gives delta std ~0.03: indistinguishable
    # starting loss, gradients alive everywhere from step 0.
    RESIDUAL_INIT_STD = 1e-3

    def __init__(self, dim: int, latent_dim: int, residual: bool = False) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(dim) if hasattr(nn, 'RMSNorm') else nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, latent_dim)
        nn.init.zeros_(self.proj.bias)
        nn.init.trunc_normal_(self.proj.weight,
                              std=self.RESIDUAL_INIT_STD if residual else 0.02)

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
        predictor_mode: str = 'resampler',
        prediction_mode: str = 'one_shot',
        rollout_steps: Optional[int] = None,
        rollout_weight: float = 1.0,
        bptt_depth: Optional[int] = None,
        residual_feedback: bool = True,
        ar_time_embed: bool = False,
        context_slices: int = 20,
        latent_seq_h: int = 8,
        latent_seq_w: int = 14,
        feedback_source: str = 'latent',
        **kwargs,
    ) -> None:
        # Two orthogonal axes whose names are one letter apart, so spell out which is
        # which wherever they appear: `predictor_mode` is WHERE the future positions live
        # (outside Qwen in a resampler, or packed into Qwen's own sequence);
        # `prediction_mode` is whether all horizons are emitted at once or rolled out.
        if predictor_mode not in ('resampler', 'direct', 'slice', 'interleave'):
            raise ValueError(f'predictor_mode must be resampler, direct, slice or interleave, '
                             f'got {predictor_mode!r}')
        if feedback_source not in ('latent', 'hidden'):
            raise ValueError(f'feedback_source must be latent or hidden, got {feedback_source!r}')
        if feedback_source == 'hidden' and rollout_weight is not None and rollout_weight < 1:
            # Teacher forcing needs a ground-truth value to feed. With hidden-state
            # feedback that would be a ground-truth *hidden state*, which does not exist:
            # our target lives in V-RAE space, not the backbone's. (The paper can do it
            # because its target is the backbone's own vision embedding.) So this variant
            # is predicted-feedback only, and pays one forward per chunk.
            raise ValueError(
                'feedback_source=hidden cannot be teacher forced: there is no ground-truth '
                'hidden state to feed, because the target is a V-RAE latent rather than a '
                'backbone embedding. Use ROLLOUT_WEIGHT=1 (and a small BPTT_DEPTH), or '
                'feedback_source=latent.')
        if predictor_mode == 'interleave' and prediction_mode != 'autoregressive':
            # The whole point of putting latent positions in the sequence is that the
            # previous chunk conditions the next one. Without a rollout it would just be
            # `direct` with a coarser grid.
            raise ValueError('predictor_mode=interleave requires prediction_mode=autoregressive')
        if prediction_mode not in ('one_shot', 'autoregressive', 'paired_residual'):
            raise ValueError('prediction_mode must be one_shot, autoregressive or '
                             'paired_residual')
        if prediction_mode == 'paired_residual' and rollout_weight not in (0, 0.0):
            # There is no rollout here: every chunk is predicted from its own context
            # chunk in one pass, so a rollout weight has nothing to weight.
            raise ValueError('prediction_mode=paired_residual has no rollout; leave '
                             'rollout_weight at 0')
        rollout_steps = latent_chunks if rollout_steps is None else rollout_steps
        if not 1 <= rollout_steps <= latent_chunks:
            raise ValueError('rollout_steps must be between 1 and latent_chunks')
        if not 0 <= rollout_weight <= 1:
            raise ValueError('rollout_weight must be in [0, 1]')
        bptt_depth = rollout_steps if bptt_depth is None else bptt_depth
        if bptt_depth < 1:
            raise ValueError('bptt_depth must be >= 1 (1 = feedback treated as a constant)')
        if min(latent_chunks, latent_height, latent_width, latent_dim, resampler_dim, resampler_depth,
               resampler_heads) < 1 or resampler_dim % resampler_heads:
            raise ValueError('positive geometry/depth required; resampler_dim must divide by heads')
        self.predictor_mode = predictor_mode
        self.prediction_mode = prediction_mode
        self.rollout_steps = rollout_steps
        self.rollout_weight = rollout_weight
        self.bptt_depth = bptt_depth
        self.residual_feedback = residual_feedback
        self.ar_time_embed = ar_time_embed
        # Qwen video temporal positions over the context clip: 40 frames / temporal_patch
        # 2 = 20 for this cache. Only `slice` mode uses it, and forward asserts it against
        # the actual video_grid_thw rather than trusting it.
        self.context_slices = context_slices
        # Per-chunk grid of latent positions that actually enter Qwen's sequence, for
        # `interleave`. 8x14 = 112 per chunk adds 1120 tokens and needs a 16.8 M
        # expansion; 4x4 = 16 adds 160 and needs 117 M. The paper this follows sweeps span
        # length and finds shorter better, so the knob is worth an ablation either way.
        self.latent_seq_h = latent_seq_h
        self.latent_seq_w = latent_seq_w
        # What gets fed back into the sequence at the next chunk's positions.
        #   latent  the predicted V-RAE latent, pooled and projected back to D_qwen.
        #           Teacher-forceable (the ground-truth latent exists), so training is a
        #           single forward -- but the feedback passes two lossy bottlenecks.
        #   hidden  the previous block's hidden states, straight back in. This is what
        #           Future-L1 does, no bottleneck -- but not teacher-forceable here, so
        #           it costs one forward per chunk.
        self.feedback_source = feedback_source
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
        self.config.vrae_predictor = config.to_dict()
        context_dim = self._text_hidden_size(qwen.config)
        rollout = config.prediction_mode == 'autoregressive'
        # An edit is not an extrapolation: target chunk k is the *same* 4 frames as
        # context chunk k, edited. So the residual base is the corresponding chunk and
        # nothing chains -- one pass, every chunk at once, like one_shot.
        paired = config.prediction_mode == 'paired_residual'
        # A rollout emits one chunk per step, so its query grid is a single time slice;
        # one_shot emits every horizon at once and needs the full T_c x H x W grid.
        grid = (config.latent_chunks, config.latent_height, config.latent_width)
        self.resampler = None
        self.future_embeddings = None
        self.slice_expand = None
        self.interleave = None
        if config.predictor_mode == 'interleave':
            self.interleave = InterleavedLatentTokens(
                context_dim=context_dim,
                latent_dim=config.latent_dim,
                dim=config.resampler_dim,
                chunks=config.latent_chunks,
                seq_grid=(config.latent_seq_h, config.latent_seq_w),
                latent_grid=(config.latent_height, config.latent_width),
                feedback_source=config.feedback_source,
            )
            head_dim = config.resampler_dim
        elif config.predictor_mode == 'resampler':
            self.resampler = SpatiotemporalLatentResampler(
                context_dim=context_dim,
                dim=config.resampler_dim,
                grid=(1, config.latent_height, config.latent_width) if rollout else grid,
                depth=config.resampler_depth,
                num_heads=config.resampler_heads,
                self_attn=config.resampler_self_attn,
            )
            head_dim = config.resampler_dim
        elif config.predictor_mode == 'slice':
            # Nothing is added to Qwen's input. The cost of that, stated plainly: the
            # backbone does no future-specific computation at all -- chunk k is
            # conditioned on hidden states computed to represent the *past*, and the
            # whole "translate past into future" job falls to this module.
            #
            # Built eagerly, never lazily: `modules_to_save` wraps modules when the model
            # is constructed, so a module created on the first forward would train and
            # never be written to the checkpoint. The geometry therefore comes from
            # config here and is validated against the real video_grid_thw in forward.
            if config.context_slices % config.latent_chunks:
                raise ValueError(
                    f'context_slices={config.context_slices} must be a multiple of '
                    f'latent_chunks={config.latent_chunks} to pair slices with chunks')
            merge = int(getattr(getattr(qwen.config, 'vision_config', None),
                                'spatial_merge_size', 2) or 2)
            self.slice_expand = ContextSliceExpansion(
                context_dim=context_dim,
                dim=config.resampler_dim,
                slices_per_chunk=config.context_slices // config.latent_chunks,
                merge_size=merge,
            )
            head_dim = config.resampler_dim
        else:
            self.future_embeddings = StructuredFutureEmbeddings(context_dim, grid)
            head_dim = context_dim
        # Small-init only where the head predicts a *delta*: it makes the model very
        # nearly the identity before training, which puts the starting loss at the
        # persistence baseline instead of at the dataset mean. For paired_residual the
        # identity is "emit the source unchanged" -- exactly the baseline an edit model
        # has to beat, so the run starts on the gate instead of spending most of its
        # budget relearning it.
        self.latent_head = LatentHead(head_dim, config.latent_dim,
                                      residual=(rollout or paired) and config.residual_feedback)
        if rollout:
            # The feedback projection is attached to whichever module `modules_to_save`
            # already keeps -- the resampler, or the head in direct mode. PEFT copies
            # those wholesale (they are modules, not deltas fused into a weight), so a
            # projection parked anywhere else would train and never be saved.
            #
            # Small init, not the default: the feedback is a normalized latent (std ~1),
            # so a default-init Linear produces an injection an order of magnitude larger
            # than the thing it is added to, drowning it at step 0.
            # `interleave` needs no feedback projection: feeding the previous chunk in as
            # an input embedding *is* its feedback, and that lives in `in_proj`.
            host = {'resampler': self.resampler, 'slice': self.slice_expand,
                    'interleave': None}.get(config.predictor_mode, self.latent_head)
        if rollout and host is not None:
            host.feedback_proj = nn.Linear(config.latent_dim, head_dim)
            nn.init.trunc_normal_(host.feedback_proj.weight, std=0.02)
            nn.init.zeros_(host.feedback_proj.bias)
            # Only the resampler needs a step embedding. In direct mode the conditioning
            # is Qwen's own hidden state at chunk k, which already carries that chunk's
            # time identity from future_embeddings.time_embed.
            if config.ar_time_embed and config.predictor_mode == 'resampler':
                self.resampler.step_embed = nn.Parameter(torch.zeros(config.latent_chunks,
                                                                     config.resampler_dim))
                nn.init.trunc_normal_(self.resampler.step_embed, std=0.02)
        self.latent_norm = LatentNormalization(config.latent_dim, config.latent_stats_path)
        predictor = (self.resampler or self.future_embeddings or self.slice_expand
                     or self.interleave)
        trainable = sum(p.numel() for p in predictor.parameters()) + \
            sum(p.numel() for p in self.latent_head.parameters())
        logger.info(f'future predictor mode {config.predictor_mode}, '
                    f'prediction_mode {config.prediction_mode}, grid '
                    f'{config.latent_chunks}x{config.latent_height}x{config.latent_width}, '
                    f'Qwen dim {context_dim}, D_z {config.latent_dim}, '
                    f'predictor+head params {trainable / 1e6:.2f}M')
        if rollout:
            logger.info(f'rollout_steps={config.rollout_steps} (horizons trained/scored), '
                        f'bptt_depth={config.bptt_depth}, rollout_weight={config.rollout_weight}, '
                        f'residual_feedback={config.residual_feedback}, '
                        f'ar_time_embed={config.ar_time_embed}, '
                        f'feedback_source={config.feedback_source}')
            if config.rollout_steps != config.latent_chunks:
                # Cheap to miss and it silently invalidates every comparison: fewer
                # horizons is an easier problem, so the loss drops for a reason that has
                # nothing to do with the model.
                logger.warning(
                    f'rollout_steps={config.rollout_steps} < latent_chunks='
                    f'{config.latent_chunks}: this run scores only the first '
                    f'{config.rollout_steps} horizons, so its loss is NOT comparable to a '
                    f'full-horizon run. Use BPTT_DEPTH to shorten the gradient chain '
                    f'instead, which keeps all {config.latent_chunks} horizons scored.')

    def _append_future_positions(
        self, inputs_embeds: torch.Tensor, attention_mask: Optional[torch.Tensor],
        future: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Insert extra positions directly after every sample's valid context.

        `future` defaults to the `direct` mode's learned future-position embeddings;
        `interleave` passes its own per-chunk input embeddings instead. Packing after the
        *valid* context (not after the padded length) is what keeps the appended block
        contiguous and causally after the context for every sample in the batch.
        """
        if inputs_embeds.ndim != 3:
            raise ValueError(f'direct predictor expects [B,L,D] inputs_embeds, got {tuple(inputs_embeds.shape)}')
        batch, _, dim = inputs_embeds.shape
        if attention_mask is None:
            attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=inputs_embeds.device)
        if attention_mask.ndim != 2:
            raise ValueError('direct predictor currently requires a 2-D attention_mask')
        valid = attention_mask.bool()
        lengths = valid.sum(dim=1)
        if future is None:
            future = self.future_embeddings(batch, device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        num_future = future.shape[1]
        total_length = int(lengths.max().item()) + num_future
        packed = inputs_embeds.new_zeros(batch, total_length, dim)
        packed_mask = attention_mask.new_zeros(batch, total_length)
        future_mask = torch.zeros(batch, total_length, dtype=torch.bool, device=inputs_embeds.device)
        for index in range(batch):
            context = inputs_embeds[index, valid[index]]
            length = context.shape[0]
            packed[index, :length] = context
            packed[index, length:length + num_future] = future[index]
            packed_mask[index, :length + num_future] = 1
            future_mask[index, length:length + num_future] = True
        return packed, packed_mask, future_mask

    def _rollout_output(self, roll: Optional[torch.Tensor], teacher: Optional[torch.Tensor],
                        target: torch.Tensor, weight: float,
                        reference: torch.Tensor) -> VRAEPredictorOutput:
        """Mix the branch losses and record the per-horizon breakdown.

        Shared by every rollout path so the logged metrics mean the same thing whatever
        the predictor mode. `horizon_N_loss` is what shows whether error grows with the
        distance predicted or stays flat.
        """
        cfg = self.vrae_config
        z_pred = roll if roll is not None else teacher
        branches = [(w, z) for w, z in ((weight, roll), (1 - weight, teacher))
                    if z is not None and w > 0]
        losses: Dict[str, torch.Tensor] = {}
        metrics: Dict[str, torch.Tensor] = {}
        for w, z in branches:
            terms = self.latent_losses(z, target)
            metrics['loss_rollout' if z is roll else 'loss_teacher'] = terms['loss'].detach()
            for name, value in terms.items():
                losses[name] = losses.get(name, 0) + w * value
        for step in range(cfg.rollout_steps):
            metrics[f'horizon_{step + 1}_loss'] = self.latent_losses(
                z_pred[:, step:step + 1], target[:, step:step + 1])['loss'].detach()
        metrics['rollout_weight'] = reference.new_tensor(weight)
        return VRAEPredictorOutput(z_pred=z_pred, rollout_metrics=metrics, **losses)

    def _interleave_predict(self, kwargs: Dict, attention_mask: Optional[torch.Tensor],
                            observed: torch.Tensor, target: Optional[torch.Tensor],
                            teacher_forcing: bool) -> torch.Tensor:
        """Run the in-sequence latent rollout and return [B, steps, H*W, D_z].

        Teacher forcing is **one** forward: the fed-back value is ground truth, so all the
        chunks' positions can share a single causal sequence -- ordinary autoregressive
        training. Predicted feedback cannot, because chunk k's input does not exist until
        chunk k-1 has been decoded, so it costs one forward per step. That asymmetry is
        why the paper trains with teacher forcing and leaves predicted trajectories to RL.
        """
        cfg = self.vrae_config
        module = self.interleave
        steps = cfg.rollout_steps
        base_embeds = kwargs['inputs_embeds']
        dtype = base_embeds.dtype

        def run(fed: torch.Tensor, blocks: int):
            call = dict(kwargs)
            packed, mask, latent_mask = self._append_future_positions(base_embeds, attention_mask, fed)
            call['inputs_embeds'] = packed
            call.pop('position_ids', None)
            call.pop('mm_token_type_ids', None)
            out = self.qwen(attention_mask=mask, **call)
            hidden = out.hidden_states[-1]
            per = module.tokens_per_chunk
            return hidden[latent_mask].view(hidden.shape[0], blocks, per, hidden.shape[-1])

        if teacher_forcing:
            if target is None:
                raise ValueError('teacher forcing needs target_latent')
            if cfg.feedback_source == 'hidden':
                raise ValueError('feedback_source=hidden has no ground-truth hidden state '
                                 'to teacher force with')
            previous = [observed] + [target[:, k] for k in range(steps - 1)]
            fed = torch.cat([module.input_embeddings(previous[k], k, dtype=dtype)
                             for k in range(steps)], dim=1)
            hidden = run(fed, steps)
            deltas = [self.latent_head(module.read_out(hidden[:, k])) for k in range(steps)]
            stacked = [previous[k] + deltas[k] if cfg.residual_feedback else deltas[k]
                       for k in range(steps)]
            return torch.stack(stacked, dim=1)

        # Predicted feedback: grow the sequence one chunk at a time. One backbone forward
        # per chunk -- unavoidable, since chunk k's input does not exist until chunk k-1
        # has been decoded.
        use_hidden = cfg.feedback_source == 'hidden'
        predictions: List[torch.Tensor] = []
        previous = observed
        fed_blocks: List[torch.Tensor] = []
        attached = 0
        for step in range(steps):
            if step == 0:
                block = (module.initial_embeddings(observed, dtype=dtype) if use_hidden
                         else module.input_embeddings(observed, 0, dtype=dtype))
            elif use_hidden:
                block = module.hidden_embeddings(last_hidden, step, dtype=dtype)
            else:
                block = module.input_embeddings(previous, step, dtype=dtype)
            fed_blocks.append(block)
            hidden = run(torch.cat(fed_blocks, dim=1), step + 1)
            last_hidden = hidden[:, step]
            delta = self.latent_head(module.read_out(last_hidden))
            current = previous + delta if cfg.residual_feedback else delta
            predictions.append(current)
            previous = current
            attached += 1
            if attached >= cfg.bptt_depth:
                # Without this the graph holds every step's activations at once, which at
                # ~25 GiB per forward does not fit for ten steps.
                previous = previous.detach()
                fed_blocks = [b.detach() for b in fed_blocks]
                if use_hidden:
                    last_hidden = last_hidden.detach()
                attached = 0
        return torch.stack(predictions, dim=1)

    def _gather_context_slices(self, hidden: torch.Tensor, video_token_mask: Optional[torch.Tensor],
                               video_grid_thw: Optional[torch.Tensor]) -> torch.Tensor:
        """Qwen's video-token hidden states, reshaped to [B, chunks, slices, H_q, W_q, D].

        Nothing was appended to the input, so these are the only future-bearing states
        available. The geometry is checked against the real `video_grid_thw` rather than
        assumed: a silent mismatch here would scramble the spatial correspondence between
        the prediction and the target, and still train.
        """
        cfg = self.vrae_config
        if video_token_mask is None:
            raise ValueError(
                'slice predictor requires video_token_mask from the template; it is built in '
                'Qwen35VRAETemplate._post_encode, which is the last place input_ids exist')
        mask = video_token_mask.to(hidden.device).bool()
        if mask.shape != hidden.shape[:2]:
            raise ValueError(f'video_token_mask {tuple(mask.shape)} does not match the '
                             f'sequence {tuple(hidden.shape[:2])}')
        counts = mask.sum(dim=1)
        if not bool((counts == counts[0]).all()):
            raise ValueError(f'samples in one batch have different video token counts '
                             f'({counts.tolist()}); slice mode needs a fixed frame count '
                             'and resolution per batch')
        merge = self.slice_expand.merge_size
        slices_per_chunk = self.slice_expand.slices_per_chunk
        expected_slices = cfg.latent_chunks * slices_per_chunk
        height, width = cfg.latent_height // merge, cfg.latent_width // merge
        expected = expected_slices * height * width
        if int(counts[0]) != expected:
            raise ValueError(
                f'{int(counts[0])} video tokens but slice mode expects {expected} '
                f'({expected_slices} temporal slices x {height}x{width} merged cells, from '
                f'context_slices={cfg.context_slices}, latent grid '
                f'{cfg.latent_height}x{cfg.latent_width}, spatial_merge_size={merge}). '
                'Set CONTEXT_SLICES to nframes/temporal_patch_size for this cache.')
        if video_grid_thw is not None:
            grid = video_grid_thw.reshape(-1, 3)[0].tolist()
            if grid != [expected_slices, cfg.latent_height, cfg.latent_width]:
                raise ValueError(
                    f'video_grid_thw {grid} disagrees with the configured geometry '
                    f'[{expected_slices}, {cfg.latent_height}, {cfg.latent_width}]')
        batch, _, dim = hidden.shape
        video = hidden[mask].view(batch, expected_slices, height, width, dim)
        return video.view(batch, cfg.latent_chunks, slices_per_chunk, height, width, dim)

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

    def _rollout_conditioning(self, hidden: torch.Tensor) -> torch.Tensor:
        """Everything the rollout can compute once, before the first step.

        direct: Qwen's own hidden states at the future positions, [B, T_c, H*W, D_qwen].
        They are already there -- one forward produced all ten chunks -- so the rollout
        costs one backbone pass, not ten.

        resampler: the projected context, [B, L_ctx, resampler_dim]. Step-invariant, and
        re-projecting it per step was about a quarter of the per-step cost.
        """
        if self.vrae_config.predictor_mode in ('direct', 'slice'):
            # Already shaped [B, chunks, ...] by forward; nothing is step-invariant to
            # hoist here beyond what Qwen's single pass already produced.
            return hidden
        return self.resampler.project_context(hidden)

    def _transition(self, conditioning: torch.Tensor, context_mask: Optional[torch.Tensor],
                    feedback: torch.Tensor, step: int) -> torch.Tensor:
        """One chunk of the rollout: the delta, or the chunk itself if not residual."""
        mode = self.vrae_config.predictor_mode
        if mode == 'direct':
            # No resampler and no cross-attention on this path. Chunk k's conditioning is
            # the hidden state Qwen produced at its own future positions, which already
            # went through all 32 layers and already carries that chunk's time identity,
            # so the step only has to fold in the previous latent.
            hidden = conditioning[:, step] + self.latent_head.feedback_proj(
                feedback.to(conditioning.dtype))
            return self.latent_head(hidden)
        if mode == 'slice':
            # conditioning: [B, chunks, slices, H_q, W_q, D_ctx] -- Qwen's own video-token
            # hidden states, nothing appended to its input. The expansion turns chunk k's
            # paired context slices into that chunk's 448 latent positions.
            expanded = self.slice_expand(conditioning[:, step])
            hidden = expanded + self.slice_expand.feedback_proj(feedback.to(expanded.dtype))
            return self.latent_head(hidden)
        return self.latent_head(self.resampler(
            conditioning, context_mask, feedback=feedback, projected=True,
            step=step if self.vrae_config.ar_time_embed else None))

    def predict_sequence(self, hidden, context_mask, observed, target=None, teacher_forcing=False):
        """Shared one-chunk transition, applied `rollout_steps` times.

        With ``residual_feedback`` the step predicts a delta on the previous chunk, so a
        zero-initialised head makes the whole rollout the identity -- i.e. exactly the
        persistence baseline -- before any training. Without it the step predicts the
        chunk outright and starts from the dataset mean instead, which is where the
        one-shot model starts and the reason feeding the observed chunk in bought nothing.

        ``bptt_depth`` truncates backpropagation through the feedback chain: the chain is
        detached every ``d`` steps, so no gradient path spans more than ``d`` transitions.
        d >= rollout_steps never detaches (full BPTT); d == 1 treats each feedback as a
        constant input. Teacher forcing feeds ground truth and is depth-1 by construction.
        """
        cfg = self.vrae_config
        conditioning = self._rollout_conditioning(hidden)
        predictions = []
        previous = observed
        attached = 0
        for step in range(cfg.rollout_steps):
            feedback = self.latent_norm.normalize(previous.float())
            delta = self._transition(conditioning, context_mask, feedback, step)
            current = previous + delta if cfg.residual_feedback else delta
            predictions.append(current)
            if teacher_forcing:
                previous = target[:, step]
                continue
            previous = current
            attached += 1
            if attached >= cfg.bptt_depth:
                previous = previous.detach()
                attached = 0
        return torch.stack(predictions, dim=1)

    def forward(
        self,
        target_latent: Optional[torch.Tensor] = None,
        grid_hw: Optional[torch.Tensor] = None,
        context_latent: Optional[torch.Tensor] = None,
        video_token_mask: Optional[torch.Tensor] = None,
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
        cfg = self.vrae_config
        if cfg.predictor_mode == 'interleave':
            if kwargs.get('inputs_embeds') is None:
                raise ValueError('interleave predictor requires inputs_embeds from the template')
            tokens = cfg.latent_height * cfg.latent_width
            if (context_latent is None or context_latent.ndim != 4
                    or tuple(context_latent.shape[2:]) != (tokens, cfg.latent_dim)):
                raise ValueError('interleave mode requires context_latent [B, T>=1, H*W, D]')
            observed = context_latent[:, -1].to(kwargs['inputs_embeds'].device)
            if target_latent is not None:
                target_latent = target_latent.to(observed.device)
            target = None if target_latent is None else target_latent[:, :cfg.rollout_steps]
            weight = cfg.rollout_weight if self.training else 1.0
            roll = (self._interleave_predict(kwargs, attention_mask, observed, target, False)
                    if weight > 0 or target is None else None)
            if target is None:
                return VRAEPredictorOutput(z_pred=roll)
            teacher = (self._interleave_predict(kwargs, attention_mask, observed, target, True)
                       if weight < 1 else None)
            return self._rollout_output(roll, teacher, target, weight, observed)
        future_mask = None
        if self.vrae_config.predictor_mode == 'direct':
            inputs_embeds = kwargs.get('inputs_embeds')
            if inputs_embeds is None:
                raise ValueError('direct predictor requires inputs_embeds from the multimodal template')
            kwargs['inputs_embeds'], attention_mask, future_mask = self._append_future_positions(
                inputs_embeds, attention_mask)
            kwargs.pop('position_ids', None)
            kwargs.pop('mm_token_type_ids', None)
        outputs = self.qwen(attention_mask=attention_mask, **kwargs)
        hidden = outputs.hidden_states[-1]

        context_mask = None
        if attention_mask is not None and attention_mask.dim() == 2:
            # MultiheadAttention wants True where a key must be ignored.
            context_mask = attention_mask == 0
            if context_mask.shape[1] != hidden.shape[1]:
                context_mask = None

        cfg = self.vrae_config
        tokens = cfg.latent_height * cfg.latent_width
        if grid_hw is not None and not torch.all(
                grid_hw.to(hidden.device) == hidden.new_tensor([cfg.latent_height, cfg.latent_width])):
            raise ValueError('grid_hw does not match configured latent spatial grid')
        if target_latent is not None:
            expected = (hidden.shape[0], cfg.latent_chunks, tokens, cfg.latent_dim)
            if tuple(target_latent.shape) != expected:
                raise ValueError(f'target latent {tuple(target_latent.shape)} != {expected}')
            target_latent = target_latent.to(hidden.device)
        # In direct mode the future positions are part of the sequence Qwen just ran, so
        # one forward already produced every chunk's conditioning. Split it per chunk
        # here: one_shot flattens it straight into the head, and the rollout indexes
        # chunk k at step k -- which is why the rollout needs no second backbone pass.
        if cfg.predictor_mode == 'direct':
            hidden = hidden[future_mask].view(
                hidden.shape[0], cfg.latent_chunks, tokens, hidden.shape[-1])
        elif cfg.predictor_mode == 'slice':
            hidden = self._gather_context_slices(hidden, video_token_mask,
                                                 kwargs.get('video_grid_thw'))
        if cfg.prediction_mode in ('one_shot', 'paired_residual'):
            if cfg.predictor_mode == 'direct':
                z_pred = self.latent_head(hidden.flatten(1, 2))
            elif cfg.predictor_mode == 'slice':
                # Every chunk at once: fold chunks into the batch so the same expansion
                # runs per chunk, exactly as the rollout applies it per step (minus the
                # feedback). This is the A0' arm -- no rollout, no observed chunk fed in.
                batch, chunks = hidden.shape[:2]
                flat = self.slice_expand(hidden.reshape(batch * chunks, *hidden.shape[2:]))
                z_pred = self.latent_head(flat).view(batch, chunks * tokens, cfg.latent_dim)
            else:
                z_pred = self.latent_head(self.resampler(hidden, context_mask))
            z_pred = z_pred.view(z_pred.shape[0], cfg.latent_chunks, tokens, cfg.latent_dim)
            if cfg.prediction_mode == 'paired_residual' and cfg.residual_feedback:
                # Chunk k on chunk k, so a chunk-count mismatch between context and
                # target must stop the run: broadcasting or truncating here would train
                # against a silently misaligned pair.
                if (context_latent is None or context_latent.ndim != 4
                        or tuple(context_latent.shape[1:]) != (cfg.latent_chunks, tokens,
                                                               cfg.latent_dim)
                        or context_latent.shape[0] != z_pred.shape[0]):
                    raise ValueError(
                        'prediction_mode=paired_residual needs context_latent '
                        f'[B, {cfg.latent_chunks}, {tokens}, {cfg.latent_dim}] aligned '
                        'chunk-for-chunk with the target; got '
                        f'{None if context_latent is None else tuple(context_latent.shape)}')
                z_pred = context_latent.to(z_pred.device, z_pred.dtype) + z_pred
            losses = {} if target_latent is None else self.latent_losses(z_pred, target_latent)
            return VRAEPredictorOutput(z_pred=z_pred, **losses)

        if (context_latent is None or context_latent.ndim != 4 or context_latent.shape[1] < 1
                or context_latent.shape[0] != hidden.shape[0]
                or tuple(context_latent.shape[2:]) != (tokens, cfg.latent_dim)):
            raise ValueError('autoregressive mode requires context_latent [B, T>=1, H*W, D]; '
                             'rebuild JSONL with first_half/context_latent_path')
        observed = context_latent[:, -1].to(hidden.device)
        target = None if target_latent is None else target_latent[:, :cfg.rollout_steps]
        # Evaluation always rolls out, including a model trained with weight=0 (B).
        weight = cfg.rollout_weight if self.training else 1.0
        roll = self.predict_sequence(hidden, context_mask, observed) if weight > 0 or target is None else None
        if target is None:
            return VRAEPredictorOutput(z_pred=roll)
        teacher = self.predict_sequence(hidden, context_mask, observed, target, True) if weight < 1 else None
        return self._rollout_output(roll, teacher, target, weight, hidden)



class Qwen35VRAELoader(Qwen3_5Loader):
    """Load the released Qwen3.5 weights, then wrap them with the latent predictor."""

    def get_model(self, model_dir: str, config, processor, model_kwargs) -> PreTrainedModel:
        qwen = super().get_model(model_dir, config, processor, model_kwargs)
        saved = getattr(config, 'vrae_predictor', {}) or {}
        def setting(name, kind, default):
            return get_env_args(name, kind, saved.get(name, default))
        predictor_config = QwenVRAEFuturePredictorConfig(
            # FUTURE_PREDICTOR_MODE, not PREDICTOR_MODE: the env name predates this
            # config field and node-7 has trained runs plus eval scripts using it.
            predictor_mode=(os.environ.get('FUTURE_PREDICTOR_MODE')
                            or saved.get('predictor_mode') or 'resampler').lower(),
            prediction_mode=setting('prediction_mode', str, 'one_shot'),
            rollout_steps=setting('rollout_steps', int, setting('latent_chunks', int, 10)),
            rollout_weight=setting('rollout_weight', float, 1.0),
            bptt_depth=setting('bptt_depth', int, setting('latent_chunks', int, 10)),
            residual_feedback=bool(setting('residual_feedback', int, 1)),
            ar_time_embed=bool(setting('ar_time_embed', int, 0)),
            context_slices=setting('context_slices', int, 20),
            latent_seq_h=setting('latent_seq_h', int, 8),
            latent_seq_w=setting('latent_seq_w', int, 14),
            feedback_source=setting('interleave_feedback', str, 'latent'),
            latent_chunks=setting('latent_chunks', int, 10),
            latent_height=setting('latent_height', int, 16),
            latent_width=setting('latent_width', int, 28),
            latent_dim=setting('latent_dim', int, 1024),
            resampler_dim=setting('resampler_dim', int, 1024),
            resampler_depth=setting('resampler_depth', int, 4),
            resampler_heads=setting('resampler_heads', int, 16),
            resampler_self_attn=bool(setting('resampler_self_attn', int, 1)),
            lambda_mse=setting('lambda_mse', float, 0.1),
            lambda_temporal=setting('lambda_temporal', float, 0.1),
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
        if model.resampler is not None:
            model.resampler.to(device=reference.device, dtype=dtype)
        if model.future_embeddings is not None:
            model.future_embeddings.to(device=reference.device, dtype=dtype)
        if model.slice_expand is not None:
            model.slice_expand.to(device=reference.device, dtype=dtype)
        if model.interleave is not None:
            model.interleave.to(device=reference.device, dtype=dtype)
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
