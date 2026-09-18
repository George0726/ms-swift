#!/usr/bin/env python3
# Copyright (c) ModelScope Contributors. All rights reserved.
"""Score the copy-the-context baselines on exactly the rows a model was scored on.

`eval_latent.py` prints the *zero*-prediction baseline, which only sets the scale.  The
baseline the research question turns on is persistence -- "the future looks like the last
thing you saw" -- and it has to be measured on the same rows, with the same latent
statistics and the same loss, or the ratio against a model number means nothing.

Two variants, both from the plan's Sec 2.4:

* ``chunkwise``   pred[i] = context[i]           (whole context shifted forward)
* ``last_chunk``  pred[i] = context[-1]          (freeze the final observed chunk)

No Qwen, no GPU: this reads the cached .pt pairs only.

    python examples/vrae_future_pred/persistence_baseline.py \
        --cache-root /data1/qirui/Datasets/vpdata_future_latents/trunk0 \
        --split train --limit 300
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

DEFAULT_STATS = '/data1/qirui/ms-swift/ckpts/latent_stats/vjepa_vpdata_40future.pt'


def load_latent(path: str) -> torch.Tensor:
    payload = torch.load(path, map_location='cpu', weights_only=False)
    latent = payload['latent'] if isinstance(payload, dict) else payload
    return torch.as_tensor(latent).float()


def latent_losses(pred: torch.Tensor, target: torch.Tensor, mean, std, l_mse=0.1, l_temp=0.1):
    """Byte-for-byte the objective in model.py:latent_losses, on a single sample."""
    pred = (pred - mean) / std
    target = (target - mean) / std
    loss_cos = (1.0 - F.cosine_similarity(pred, target, dim=-1)).mean()
    loss_mse = F.mse_loss(pred, target)
    # dim 0 is the chunk axis for a single unbatched sample [chunks, hw, channels].
    loss_temp = (pred.diff(dim=0) - target.diff(dim=0)).abs().mean() if pred.shape[0] > 1 \
        else pred.new_zeros(())
    total = loss_cos + l_mse * loss_mse + l_temp * loss_temp
    return {'loss': total.item(), 'loss_cos': loss_cos.item(),
            'loss_mse': loss_mse.item(), 'loss_temp': loss_temp.item()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--cache-root', required=True)
    parser.add_argument('--split', default='train')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--random', type=int, default=None, metavar='SEED',
                        help='sample --limit rows at random instead of taking the first N; '
                             '--limit alone takes the first N, which is not a random sample')
    parser.add_argument('--latent-stats', default=DEFAULT_STATS)
    parser.add_argument('--json-out', default=None)
    args = parser.parse_args()

    jsonl = Path(args.cache_root).expanduser().resolve() / 'swift_jsonl' / f'{args.split}.jsonl'
    rows = [json.loads(line) for line in jsonl.open('r', encoding='utf-8') if line.strip()]
    if args.random is not None and args.limit:
        generator = torch.Generator().manual_seed(args.random)
        picked = torch.randperm(len(rows), generator=generator)[:args.limit].tolist()
        rows = [rows[i] for i in sorted(picked)]
    elif args.limit:
        rows = rows[:args.limit]

    payload = torch.load(args.latent_stats, map_location='cpu', weights_only=True)
    mean = torch.as_tensor(payload['mean']).flatten().float()
    std = torch.as_tensor(payload['std']).flatten().float()

    variants = {'chunkwise': lambda ctx: ctx,
                'last_chunk': lambda ctx: ctx[-1:].expand_as(ctx).contiguous()}
    totals = {name: [] for name in variants}
    totals['dataset_mean'] = []
    per_sample = []

    for position, row in enumerate(rows, start=1):
        context = load_latent(row['context_latent_path'])
        target = load_latent(row['target_latent_path'])
        if context.shape != target.shape:
            raise SystemExit(f"{row.get('sample_id')}: context {tuple(context.shape)} != "
                             f"target {tuple(target.shape)}")
        entry = {'sample_id': row.get('sample_id', f'row{position}')}
        for name, make in variants.items():
            scored = latent_losses(make(context), target, mean, std)
            totals[name].append(scored['loss'])
            entry[name] = round(scored['loss'], 4)
        # predicting the fitted mean is the normalized zero vector: the "learned nothing"
        # level, printed so the persistence numbers have a ceiling to sit under.
        scored = latent_losses(mean.expand_as(target), target, mean, std)
        totals['dataset_mean'].append(scored['loss'])
        entry['dataset_mean'] = round(scored['loss'], 4)
        per_sample.append(entry)
        if position % 25 == 0 or position == len(rows):
            print(f'  [{position}/{len(rows)}]', flush=True)

    print(f'\n{len(rows)} samples from {jsonl}')
    for name, values in totals.items():
        tensor = torch.tensor(values)
        print(f'  {name:13s} mean {tensor.mean():.4f}  sd {tensor.std():.4f}  '
              f'min {tensor.min():.4f}  max {tensor.max():.4f}')
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(
            {'jsonl': str(jsonl), 'n': len(rows),
             'mean': {k: torch.tensor(v).mean().item() for k, v in totals.items()},
             'per_sample': per_sample}, indent=1), encoding='utf-8')
        print(f'wrote {args.json_out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
