#!/usr/bin/env python3
# Copyright (c) ModelScope Contributors. All rights reserved.
"""Checks that do not need the 19 GB backbone weights.

Covers the failure modes that are silent at training time:

* the custom columns reaching the template,
* ``target_latent`` surviving the collator whitelist,
* ``labels`` still present (the trainer indexes it unconditionally),
* the video processor keeping all 40 context frames instead of resampling to 4,
* the loss being exactly zero when the prediction equals the target.

    python examples/vrae_future_pred/verify.py --cache-root <latent cache>
"""

import argparse
import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
for candidate in (str(HERE.parent.parent), str(HERE)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

QWEN_PATH = os.environ.get('QWEN35_PATH', '/GPFS/ComfyUI_models/LLM/Qwen3.5-9B')


def check_checkpoint(checkpoint: Path) -> int:  # noqa: C901
    """Assert a finished run actually trained the new modules.

    Worth its own check because the failure is silent: with `--trainable_parameters`
    instead of `--modules_to_save`, swift trains LoRA only (activate_parameters runs
    just for tuner_type='full', and its startswith match would miss the
    `base_model.model.` prefix PEFT adds anyway).  Training completes, the loss falls,
    grad_norm looks healthy -- but the resampler stays at its random init and the loss
    stalls at the value for predicting zero.
    """
    from safetensors.torch import load_file

    from swift.utils import get_env_args

    adapter = checkpoint / 'adapter_model.safetensors'
    if not adapter.is_file():
        print(f'  [FAIL] {adapter} not found')
        return 1
    state = load_file(str(adapter))
    total = sum(value.numel() for value in state.values())
    print(f'{adapter}\n  {len(state)} tensors, {total / 1e6:.2f} M params')

    # Expectations are built from the live config rather than hardcoded, because the
    # resampler's size is a knob: RESAMPLER_DEPTH / RESAMPLER_SELF_ATTN / RESAMPLER_DIM
    # move it between ~17 M and ~72 M. A literal here turns every ablation into a false
    # FAIL, which is worse than no check -- it trains the reader to ignore the output.
    # Set the same env vars this check runs under as the run being checked.
    import model as vrae_model
    config = vrae_model.QwenVRAEFuturePredictorConfig(
        latent_chunks=get_env_args('latent_chunks', int, 10),
        latent_height=get_env_args('latent_height', int, 16),
        latent_width=get_env_args('latent_width', int, 28),
        latent_dim=get_env_args('latent_dim', int, 1024),
        resampler_dim=get_env_args('resampler_dim', int, 1024),
        resampler_depth=get_env_args('resampler_depth', int, 4),
        resampler_heads=get_env_args('resampler_heads', int, 16),
        resampler_self_attn=bool(get_env_args('resampler_self_attn', int, 1)),
    )
    reference = vrae_model.SpatiotemporalLatentResampler(
        context_dim=4096, dim=config.resampler_dim,
        grid=(config.latent_chunks, config.latent_height, config.latent_width),
        depth=config.resampler_depth, num_heads=config.resampler_heads,
        self_attn=config.resampler_self_attn)
    expected = {
        'resampler': sum(t.numel() for t in reference.parameters()) / 1e6,
        'latent_head': sum(t.numel() for t in vrae_model.LatentHead(
            config.resampler_dim, config.latent_dim).parameters()) / 1e6,
    }
    print(f'  config: depth {config.resampler_depth}, self_attn {config.resampler_self_attn}, '
          f'dim {config.resampler_dim}')

    failures = []
    for name, expected_m in expected.items():
        hits = [key for key in state if name in key]
        count = sum(state[key].numel() for key in hits) / 1e6
        ok = count > expected_m * 0.9
        print(f'  [{"PASS" if ok else "FAIL"}] {name}: {len(hits)} tensors, {count:.2f} M '
              f'(expected ~{expected_m:.2f} M)')
        if not ok:
            failures.append(name)
    if failures:
        print(f'\n{failures} absent from the checkpoint: the run trained LoRA only. '
              f'Use --modules_to_save resampler latent_head, not --trainable_parameters.')
        return 1
    print('\nthe new modules were trained')
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-root', required=True)
    parser.add_argument('--qwen', default=QWEN_PATH)
    parser.add_argument(
        '--check-checkpoint',
        default=None,
        help='path to a checkpoint-N directory: assert the resampler and latent head '
        'were actually trained, not just LoRA')
    args = parser.parse_args()

    if args.check_checkpoint:
        return check_checkpoint(Path(args.check_checkpoint))

    root = Path(args.cache_root).expanduser().resolve()
    os.environ['VPDATA_LATENT_ROOT'] = str(root)

    import dataset as vrae_dataset  # noqa: E402  (registers the dataset)
    import template as vrae_template  # noqa: E402  (registers the template)
    import model as vrae_model  # noqa: E402  (registers the model type)
    from swift.dataset import load_dataset  # noqa: E402
    from swift.model import get_model_processor  # noqa: E402
    from swift.template import get_template  # noqa: E402

    vrae_dataset.register(root)
    failures = []

    def check(name: str, condition: bool, detail: str = '') -> None:
        print(f'  [{"PASS" if condition else "FAIL"}] {name}' + (f' -- {detail}' if detail else ''))
        if not condition:
            failures.append(name)

    # ---- 1. dataset columns ----------------------------------------------------
    print('\n1. dataset pipeline')
    train = load_dataset([vrae_dataset.DATASET_NAME], remove_unused_columns=False)[0]
    row = train[0]
    check('target_latent_path present', 'target_latent_path' in row)
    check('videos present', 'videos' in row)
    check('grid_hw present', 'grid_hw' in row, str(row.get('grid_hw')))

    # ---- 2. template encode + collator ----------------------------------------
    print('\n2. template encode / collator')
    # load_model=False gives the processor with the `model_info` that get_template needs,
    # without pulling in the 19 GB backbone.
    _, processor = get_model_processor(args.qwen, model_type=vrae_model.MODEL_TYPE, load_model=False)
    tpl = get_template(processor, template_type=vrae_template.TEMPLATE_TYPE, max_length=8192)
    tpl.set_mode('train')
    encoded = tpl.encode(dict(row))
    check('encoded target_latent', encoded.get('target_latent') is not None,
          str(tuple(encoded['target_latent'].shape)) if encoded.get('target_latent') is not None else 'missing')
    check('encoded labels', encoded.get('labels') is not None)

    batch = tpl.data_collator([encoded])
    latent = batch.get('target_latent')
    check('collated target_latent', latent is not None,
          str(tuple(latent.shape)) if latent is not None else 'dropped by the collator whitelist')
    # The trainer does `labels = inputs['labels']` even when the model returns its own
    # loss, so losing this key turns into a KeyError mid-training.
    check('collated labels', batch.get('labels') is not None)

    expected = (1, 10, 448, 1024)
    if latent is not None:
        check('target_latent shape', tuple(latent.shape) == expected,
              f'{tuple(latent.shape)} vs expected {expected}')

    # ---- 3. context frames not resampled ---------------------------------------
    print('\n3. video grid (all 40 context frames)')
    thw = batch.get('video_grid_thw')
    if thw is None:
        check('video_grid_thw present', False, 'no video tokens in the batch')
    else:
        grid = thw[0].tolist() if hasattr(thw, 'tolist') else list(thw[0])
        check(
            'video_grid_thw == [20, 16, 28]', grid == [20, 16, 28],
            f'{grid} -- a temporal 2 means fetch_video resampled 40 frames to 4, and a '
            f'spatial 18x30 means smart_resize picked its own size. Both come from the '
            f"row's chat_template_kwargs (nframes / resized_height / resized_width) in "
            f'dataset.py; rebuild the jsonl with --build after changing them.')

    # ---- 4. loss is zero at the optimum ----------------------------------------
    print('\n4. loss at z_pred == z_target')
    config = vrae_model.QwenVRAEFuturePredictorConfig(latent_stats_path=os.environ.get('LATENT_STATS_PATH'))
    norm = vrae_model.LatentNormalization(config.latent_dim, config.latent_stats_path)

    class _Stub:
        latent_norm = norm
        vrae_config = config
        latent_losses = vrae_model.QwenVRAEFuturePredictor.latent_losses

    target = torch.randn(1, 10, 448, 1024)
    same = _Stub.latent_losses(_Stub, target.clone(), target)
    check('loss_cos ~ 0', abs(same['loss_cos'].item()) < 1e-5, f"{same['loss_cos'].item():.3e}")
    check('loss_mse ~ 0', abs(same['loss_mse'].item()) < 1e-9, f"{same['loss_mse'].item():.3e}")
    check('loss_temp ~ 0', abs(same['loss_temp'].item()) < 1e-9, f"{same['loss_temp'].item():.3e}")
    worse = _Stub.latent_losses(_Stub, torch.randn_like(target), target)
    check('loss > 0 for a wrong prediction', worse['loss'].item() > 0.5, f"{worse['loss'].item():.4f}")

    # ---- 5. resampler shape ------------------------------------------------------
    print('\n5. resampler / head shapes')
    resampler = vrae_model.SpatiotemporalLatentResampler(
        context_dim=4096, dim=256, grid=(config.latent_chunks, config.latent_height, config.latent_width),
        depth=1, num_heads=8)
    head = vrae_model.LatentHead(256, config.latent_dim)
    context = torch.randn(1, 37, 4096)
    out = head(resampler(context))
    check('resampler query count', resampler.num_queries == 4480, str(resampler.num_queries))
    check('head output', tuple(out.shape) == (1, 4480, 1024), str(tuple(out.shape)))

    print(f'\n{"all checks passed" if not failures else f"{len(failures)} FAILED: {failures}"}')
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
