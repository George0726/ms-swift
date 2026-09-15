#!/usr/bin/env python3
# Copyright (c) ModelScope Contributors. All rights reserved.
"""Fit per-channel statistics over the training split's target latents.

The latent cache stores raw, unnormalized latents (``normalized: false``), matching the
convention of the V-RAE pipeline, which normalizes at training time.  Reusing V-RAE's
own ``DistributedLatentStats`` keeps the statistics numerically identical to what the
rest of that pipeline produces (float64 accumulation, non-affine finalization).

    python examples/vrae_future_pred/latent_stats.py \
        --cache-root /data1/qirui/Datasets/vpdata_future_latents/trunk0 \
        --out ckpts/latent_stats/vjepa_vpdata_40future.pt
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List

import torch

DEFAULT_VRAE_ROOT = '/data1/qirui/V-RAE'


def load_vrae_stats_classes(vrae_root: str):
    src = Path(vrae_root).expanduser().resolve() / 'src'
    if not src.is_dir():
        raise SystemExit(f'V-RAE source not found at {src}; pass --vrae-root')
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from vrae.training.common.latent_norm import DistributedLatentStats  # noqa: E402
    return DistributedLatentStats


def target_paths(cache_root: Path, split: str) -> List[Path]:
    """Prefer the split jsonl so statistics only ever see training samples."""

    jsonl = cache_root / 'swift_jsonl' / f'{split}.jsonl'
    if jsonl.is_file():
        paths = []
        with jsonl.open('r', encoding='utf-8') as handle:
            for line in handle:
                row = json.loads(line)
                paths.append(Path(row['target_latent_path']))
        return paths
    raise SystemExit(f'{jsonl} not found; build the splits with dataset.py --build first. '
                     f'Fitting statistics over every sample would leak the validation split.')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-root', required=True)
    parser.add_argument('--split', default='train')
    parser.add_argument('--out', required=True)
    parser.add_argument('--vrae-root', default=DEFAULT_VRAE_ROOT)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()

    DistributedLatentStats = load_vrae_stats_classes(args.vrae_root)
    cache_root = Path(args.cache_root).expanduser().resolve()
    paths = target_paths(cache_root, args.split)
    if args.limit:
        paths = paths[:args.limit]
    if not paths:
        raise SystemExit('no target latents to fit')

    accumulator = None
    channels = None
    for position, path in enumerate(paths, start=1):
        payload = torch.load(path, map_location='cpu', weights_only=True)
        latent = payload['latent'] if isinstance(payload, dict) else payload
        if channels is None:
            channels = int(latent.shape[-1])
            accumulator = DistributedLatentStats(channels, device=args.device)
            print(f'fitting over {len(paths)} samples, latent {tuple(latent.shape)}, '
                  f'channels {channels}', flush=True)
        elif int(latent.shape[-1]) != channels:
            raise SystemExit(f'{path} has {latent.shape[-1]} channels, expected {channels}')
        # DistributedLatentStats.update expects tokens [B,T,N,C]; one sample at a time.
        accumulator.update(latent.unsqueeze(0).float().to(args.device))
        if position % 200 == 0 or position == len(paths):
            print(f'  [{position}/{len(paths)}]', flush=True)

    normalizer = accumulator.finalize(metadata={
        'dataset': 'vpdata_inpaint',
        'split': args.split,
        'scope': 'last_half',
        'cache_root': str(cache_root),
        'num_samples': len(paths),
        'clean_latent': True,
        'normalized': False,
    })
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    saved = normalizer.save(out)
    mean, std = normalizer.mean, normalizer.std
    print(f'\nsaved {saved}\n'
          f'  mean: min {mean.min():+.4f} max {mean.max():+.4f} avg {mean.mean():+.4f}\n'
          f'  std : min {std.min():.4f} max {std.max():.4f} avg {std.mean():.4f}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
