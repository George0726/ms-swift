#!/usr/bin/env python3
# Copyright (c) ModelScope Contributors. All rights reserved.
"""Score a trained checkpoint in latent space, one sample at a time.

This is the check that closes the overfit test, and it is not redundant with the training
loss.  Three things can make a falling training curve mean less than it looks:

* the logged ``loss`` is not the model's loss.  With ``**kwargs`` in forward,
  ``model_accepts_loss_kwargs`` comes out True (trainer.py:499), and
  ``seq2seq_trainer.py:243`` then multiplies by ``num_processes`` -- a contract meant for
  a loss already divided by the *global* token count, not for a mean over the local
  batch.  A 6-GPU run reports 6x the real number.
* training runs with dropout and in train mode; nothing guarantees the same weights score
  the same way under ``eval()``.
* nothing so far has proved the *checkpoint* holds what trained.  ``verify.py
  --check-checkpoint`` counts resampler tensors, which catches an absent module but not a
  broken one.

So: load the adapter from disk, run the same template and collator training used, in eval
mode under no_grad, and compare against the run's own curve. Numbers that match mean the
pipeline round-trips. On the overfit split they should be near zero; the zero-prediction
baseline is printed alongside because that is the value a model which learned nothing
lands on (~1.19 with fitted statistics), and it is the only way to read the scale.

    # close the overfit gate
    python examples/vrae_future_pred/eval_latent.py \
        --cache-root /data1/qirui/Datasets/vpdata_future_latents/trunk0/_overfit6 \
        --adapters output/_overfit6/v0-20260913-182548/checkpoint-1500

    # held-out split after a real run
    python examples/vrae_future_pred/eval_latent.py \
        --cache-root /data1/qirui/Datasets/vpdata_future_latents/trunk0 \
        --adapters output/vrae_future_pred_trunk0/vN/checkpoint-XXXX --split val
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import torch

HERE = Path(__file__).resolve().parent
for candidate in (str(HERE.parent.parent), str(HERE)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

QWEN_PATH = os.environ.get('QWEN35_PATH', '/GPFS/ComfyUI_models/LLM/Qwen3.5-9B')
DEFAULT_STATS = '/data1/qirui/ms-swift/ckpts/latent_stats/vjepa_vpdata_40future.pt'


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--cache-root', required=True,
                        help='root holding swift_jsonl/<split>.jsonl; for an overfit run this is '
                        'the _overfitN directory overfit.sh created')
    parser.add_argument('--adapters', required=True, help='checkpoint-N directory')
    parser.add_argument('--split', default='train')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--model', default=QWEN_PATH)
    parser.add_argument('--device', default='cuda:0')
    # The statistics are part of the objective, not of the model: latent_losses normalizes
    # both prediction and target with them, so scoring with statistics other than the ones
    # the run trained under produces a number that cannot be compared to its curve.
    parser.add_argument('--latent-stats', default=os.environ.get('LATENT_STATS_PATH') or DEFAULT_STATS)
    args = parser.parse_args()

    cache_root = Path(args.cache_root).expanduser().resolve()
    jsonl = cache_root / 'swift_jsonl' / f'{args.split}.jsonl'
    if not jsonl.is_file():
        raise SystemExit(f'{jsonl} not found')
    if not Path(args.latent_stats).is_file():
        raise SystemExit(f'--latent-stats {args.latent_stats} not found')

    # Both are read from the environment: dataset.py registers from VPDATA_LATENT_ROOT at
    # import time, and Qwen35VRAELoader.get_model reads LATENT_STATS_PATH when it builds
    # the predictor. Set them before the imports below, not after.
    os.environ['VPDATA_LATENT_ROOT'] = str(cache_root)
    os.environ['LATENT_STATS_PATH'] = str(args.latent_stats)

    import dataset as vrae_dataset  # noqa: F401,E402  (registers the dataset)
    import model as vrae_model  # noqa: F401,E402  (registers the model type)
    import template as vrae_template  # noqa: F401,E402  (registers the template)
    from swift.model import get_model_processor  # noqa: E402
    from swift.template import get_template  # noqa: E402
    from swift.tuners import Swift  # noqa: E402

    vrae_dataset.register(cache_root)
    device = torch.device(args.device)

    rows: List[Dict] = []
    with jsonl.open('r', encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if args.limit:
        rows = rows[:args.limit]
    print(f'{len(rows)} samples from {jsonl}')
    print(f'latent stats: {args.latent_stats}')
    print(f'adapters    : {args.adapters}\n', flush=True)

    base, processor = get_model_processor(args.model, model_type=vrae_model.MODEL_TYPE,
                                         torch_dtype=torch.bfloat16, device_map=args.device)
    model = Swift.from_pretrained(base, args.adapters)
    # device_map placed the backbone, but Swift.from_pretrained materializes the adapter on
    # CPU -- including the modules_to_save copies of resampler and latent_head, which are
    # whole modules rather than deltas fused into an existing weight. Without this the
    # first LayerNorm inside the resampler fails with weight on cpu / input on cuda.
    model = model.to(device)
    model.eval()

    template = get_template(processor, template_type=vrae_template.TEMPLATE_TYPE, max_length=8192)
    # 'train' mode, deliberately, on a model in eval(): Qwen2VLTemplate._post_encode returns
    # `inputs` untouched when not training, which would hand the backbone input_ids plus
    # pixel_values instead of the inputs_embeds the training path built. Same inputs as
    # training is the whole point -- eval() and no_grad are what make this an evaluation.
    template.set_mode('train')
    template.register_post_encode_hook([model])

    results = []
    for position, row in enumerate(rows, start=1):
        encoded = template.encode(dict(row))
        batch = template.data_collator([encoded])
        batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        target = batch['target_latent']
        with torch.no_grad():
            out = model(**batch)

        z_pred = out.z_pred.float()
        # Per-chunk cosine over the 10 latent chunks: chunk 0 is the nearest future and
        # chunk 9 the furthest, so a rising error across chunks is the signature of a model
        # that only extrapolates a little way past the context.
        norm = vrae_model.LatentNormalization(1024, args.latent_stats).to(device)
        pred_n, target_n = norm.normalize(z_pred), norm.normalize(target.float())
        per_chunk = (1.0 - torch.nn.functional.cosine_similarity(pred_n, target_n, dim=-1)).mean(dim=(0, 2))

        entry = {
            'sample_id': row.get('sample_id', f'row{position}'),
            'loss': out.loss.item(),
            'loss_cos': out.loss_cos.item(),
            'loss_mse': out.loss_mse.item(),
            'loss_temp': out.loss_temp.item(),
            'cos_per_chunk': [round(v, 4) for v in per_chunk.tolist()],
        }
        # What the same objective scores when the prediction is the zero vector. Printed
        # because the loss has no natural unit: 1.19 here is "learned nothing" and the
        # exact value shifts with the statistics, so it has to be measured, not recalled.
        zero = vrae_model.QwenVRAEFuturePredictor.latent_losses(
            type('S', (), {'latent_norm': norm, 'vrae_config': model.model.vrae_config})(),
            torch.zeros_like(z_pred), target.float())
        entry['zero_baseline'] = zero['loss'].item()
        results.append(entry)
        print(f"  [{position}/{len(rows)}] {entry['sample_id']}  loss {entry['loss']:.4f}  "
              f"(cos {entry['loss_cos']:.4f}  mse {entry['loss_mse']:.4f}  "
              f"temp {entry['loss_temp']:.4f})  zero-baseline {entry['zero_baseline']:.4f}",
              flush=True)

    n = len(results)
    mean = {k: sum(r[k] for r in results) / n for k in ('loss', 'loss_cos', 'loss_mse', 'loss_temp',
                                                        'zero_baseline')}
    chunks = [sum(r['cos_per_chunk'][i] for r in results) / n for i in range(len(results[0]['cos_per_chunk']))]
    print(f'\n{n} samples'
          f"\n  loss           {mean['loss']:.4f}"
          f"\n  loss_cos       {mean['loss_cos']:.4f}   (cosine similarity {1 - mean['loss_cos']:.4f})"
          f"\n  loss_mse       {mean['loss_mse']:.4f}"
          f"\n  loss_temp      {mean['loss_temp']:.4f}"
          f"\n  zero baseline  {mean['zero_baseline']:.4f}   "
          f"({mean['zero_baseline'] / max(mean['loss'], 1e-9):.1f}x the model's loss)"
          f"\n  1-cos by chunk (near future -> far): "
          f"{' '.join(f'{v:.3f}' for v in chunks)}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
