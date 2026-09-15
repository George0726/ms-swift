# Copyright (c) ModelScope Contributors. All rights reserved.
"""Template carrying the cached V-RAE target latent alongside the Qwen3.5 video inputs.

Three overrides matter here:

``_encode``
    Loads the target latent named by the row's ``target_latent_path``.  That key reached
    us through ``StdTemplateInputs.extra_kwargs``, which collects every dataset column
    the dataclass does not declare (swift/template/template_inputs.py).

``_data_collator``
    ``Template._data_collator`` (swift/template/base.py) builds its result from a fixed
    key whitelist, so a tensor added in ``_encode`` is dropped unless it is stacked here.
    ``Qwen2AudioTemplate._data_collator`` handles ``input_features`` the same way.

``compute_sft_loss``
    The default implementation (swift/template/base.py) rescales the model loss by
    ``(labels[:, 1:] != -100).sum() / num_items_in_batch``.  That is correct for
    per-token cross entropy and wrong for a latent regression loss, so this template
    returns the model output untouched.
"""


from typing import Any, Dict, List, Optional

import torch

from swift.template import StdTemplateInputs, register_template
from swift.template.templates.qwen import Qwen3_5Template, QwenTemplateMeta

TEMPLATE_TYPE = 'qwen3_5_vrae'


class Qwen35VRAETemplate(Qwen3_5Template):
    """Qwen3.5 video template plus a ``target_latent`` tensor per sample."""

    # Costs a factor of world_size * gradient_accumulation_steps on the effective learning
    # rate if left at the default, silently. Two places assume the model's loss is a sum
    # already divided by the *global* token count:
    #   * seq2seq_trainer.py:243 multiplies it by num_processes, to undo DDP's gradient
    #     averaging;
    #   * trainer.py:1934 then skips `loss /= gradient_accumulation_steps`, so accumulated
    #     micro-steps sum instead of average.
    # `latent_losses` returns a plain mean over the local batch, so both just over-count.
    #
    # This has to live on the template, not on the model: swift ignores what transformers
    # infers from the forward signature (trainer.py:495, which would see `**kwargs` and
    # conclude True) and hardcodes `self.model_accepts_loss_kwargs = True` in
    # seq2seq_trainer.py:31, leaving this attribute as the only override -- the same hook
    # Qwen3TTSTemplate uses. Setting it on the model has no effect at all.
    #
    # Note the reported loss changes scale once this is set: a curve from before reads
    # world_size times high (the 6-GPU overfit logged 0.149 for a real 0.0248).
    model_accepts_loss_kwargs = False

    @staticmethod
    def _load_target_latent(path: str) -> torch.Tensor:
        payload = torch.load(path, map_location='cpu', weights_only=True)
        latent = payload['latent'] if isinstance(payload, dict) else payload
        if not isinstance(latent, torch.Tensor) or latent.ndim != 3:
            raise ValueError(f'target latent must be a [chunks, tokens, channels] tensor: {path}')
        if not torch.isfinite(latent).all():
            raise ValueError(f'target latent contains non-finite values: {path}')
        return latent

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        extra = getattr(inputs, 'extra_kwargs', None) or {}
        path = extra.get('target_latent_path')
        if path:
            encoded['target_latent'] = self._load_target_latent(str(path))
            grid_hw = extra.get('grid_hw')
            if grid_hw is not None:
                encoded['grid_hw'] = [int(grid_hw[0]), int(grid_hw[1])]
        return encoded

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = super()._data_collator(batch, padding_to=padding_to)
        latents = [b['target_latent'] for b in batch if b.get('target_latent') is not None]
        if latents:
            shapes = {tuple(latent.shape) for latent in latents}
            if len(shapes) > 1:
                raise ValueError(f'target latents in one batch must share a shape, got {sorted(shapes)}')
            res['target_latent'] = torch.stack(latents)
            grid_hw = next((b.get('grid_hw') for b in batch if b.get('grid_hw') is not None), None)
            if grid_hw is not None:
                res['grid_hw'] = torch.tensor(grid_hw, dtype=torch.long)
        return res

    def _post_encode(self, model, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # `Qwen2VLTemplate._post_encode` returns only {'inputs_embeds': ...}, and
        # `Template.pre_forward_hook` (swift/template/base.py) then restores just a fixed
        # whitelist of keys from the pre-hook kwargs.  Anything else -- `target_latent`
        # included -- never reaches the model, so carry it across explicitly.  This is the
        # third key whitelist on the path, after the collator's gather_keys and the
        # dataset loader's remove_useless_columns.
        res = super()._post_encode(model, inputs)
        for key in ('target_latent', 'grid_hw'):
            if key in inputs and key not in res:
                res[key] = inputs[key]
        return res

    # The three terms the model already computes but nothing logs: only the combined
    # `loss` reaches the trainer's log line, and the combination hides which half of the
    # objective is stuck.  A run plateauing on loss_cos ~ 1 is predicting a zero vector
    # (no direction learned at all); one with loss_cos near 0 and loss_mse large has the
    # direction and not the magnitude, which points at the latent statistics rather than
    # at the resampler.  Worth the four lines during an overfit test.
    LOSS_TERMS = ('loss_cos', 'loss_mse', 'loss_temp')

    def compute_sft_loss(self, model, inputs: Dict[str, Any], num_items_in_batch: Optional[int] = None,
                         trainer=None):
        # The model owns the loss; do not apply the token-count rescaling that the base
        # class uses for cross entropy.
        if inputs.get('target_latent') is None:
            return super().compute_sft_loss(model, inputs, num_items_in_batch=num_items_in_batch, trainer=trainer)
        outputs = model(**inputs)
        if trainer is not None:
            # SwiftMixin.compute_custom_metrics all_gathers the key set before reducing,
            # so adding these is DDP-safe as long as every rank adds the same keys --
            # which it does, since target_latent is present on every rank or none.
            metrics = trainer.custom_metrics['train' if trainer.model.training else 'eval']
            for name in self.LOSS_TERMS:
                value = getattr(outputs, name, None)
                if value is not None:
                    metrics[name].update(value)
        return outputs


register_template(QwenTemplateMeta(TEMPLATE_TYPE, template_cls=Qwen35VRAETemplate))


# Frame count and resolution are NOT controlled from here.  `Qwen2VLTemplate.replace_tag`
# merges each row's `chat_template_kwargs` into the `qwen_vl_utils.fetch_video` call, so
# `nframes` / `resized_height` / `resized_width` are set per sample in dataset.py.  Env
# vars like NFRAMES belong to other templates (Ovis2) and have no effect on this path:
# leaving the defaults gives FPS=2.0 sampling (40 frames -> 4) and a smart_resize pixel
# budget that lands on 288x480 instead of 256x448.
