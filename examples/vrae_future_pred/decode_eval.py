#!/usr/bin/env python3
# Copyright (c) ModelScope Contributors. All rights reserved.
"""Decode predicted latents through the frozen V-RAE decoder and compare three paths.

Section 9 of the design doc asks for exactly this comparison, because it is what makes
a bad result diagnosable:

    ground truth      the real last 40 frames
    oracle            Z_target -> V-RAE decoder        (the ceiling: 28.0 dB measured)
    prediction        Z_pred   -> V-RAE decoder

Oracle good / prediction bad points at the latent predictor.  Oracle already bad means
V-RAE itself is the bottleneck on this domain and Qwen is not to blame.

    python examples/vrae_future_pred/decode_eval.py \
        --cache-root <latent cache> --adapters output/vrae_future_pred/checkpoint-xxx \
        --out outputs/vrae_future_eval --limit 4
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

import torch

# Running this as a script puts *this* directory on sys.path, not the repo root, so
# `import swift` inside load_predictor() fails unless PYTHONPATH happened to be set --
# and it fails only after the 19 GB backbone path is already reached. Same two entries
# eval_latent.py adds, for the same reason.
HERE = Path(__file__).resolve().parent
for candidate in (str(HERE.parent.parent), str(HERE)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

DEFAULT_VRAE_ROOT = '/data1/qirui/V-RAE'


def load_vrae(vrae_root: Path, image_size, device, variant: str = 'vjepa'):
    """Build the frozen V-RAE at the runtime geometry of the cache."""

    for candidate in (vrae_root, vrae_root / 'src'):
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
    import sampling as reference_sampling
    from vrae.checkpoint import load_checkpoint
    from vrae.models.adapter import VRAELatentAdapter
    from vrae.models.autoencoder import VRAE
    from vrae.paths import ProjectPaths

    expected_encoder, checkpoint_path = reference_sampling.VARIANTS[variant]
    payload = load_checkpoint(checkpoint_path, map_location='cpu', mmap=True)
    config = dict(reference_sampling.prepare_model_config(payload['resolved_config'], expected_encoder))
    config['data'] = {**dict(config.get('data', {})), 'image_size': list(image_size)}
    model = VRAE.from_config(config, project_paths=ProjectPaths(project_root=vrae_root))
    model.load_state_dict(payload['model'], strict=True)
    reference_sampling.load_ema_weights(model, payload)
    del payload
    model = model.requires_grad_(False).eval().to(device)
    return model, VRAELatentAdapter(model, model.metadata(), precision='bf16'), reference_sampling


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    return (-10 * torch.log10(((a - b)**2).mean())).item()


def metadata_dir_for(row: dict) -> Path:
    """Locate the cache's metadata/ from a row, not from --cache-root.

    They are not the same directory for an overfit run: overfit.sh builds a root holding
    nothing but swift_jsonl/, and the rows point at the real cache by absolute path
    (<cache>/latents/last_half/<stem>.pt). Deriving the metadata directory from the
    target instead of from --cache-root makes this script work against both.
    """
    return Path(row['target_latent_path']).resolve().parents[2] / 'metadata'


def metadata_for(row: dict) -> dict:
    path = metadata_dir_for(row) / (Path(row['target_latent_path']).stem + '.json')
    return json.loads(path.read_text(encoding='utf-8'))


def load_predictor(model_path: str, adapters: str, latent_stats: str, device):
    """Qwen3.5 + the trained resampler, wired exactly as training ran it."""

    # Imported before load_vrae() inserts the V-RAE roots at sys.path[0]: these three are
    # imported by bare name (`import model`), so binding them first keeps a future
    # top-level model.py / dataset.py in the V-RAE tree from shadowing them.
    import dataset as vrae_dataset  # noqa: F401
    import model as vrae_model
    import template as vrae_template
    from swift.model import get_model_processor
    from swift.template import get_template
    from swift.tuners import Swift

    os.environ['LATENT_STATS_PATH'] = str(latent_stats)
    base, processor = get_model_processor(model_path, model_type=vrae_model.MODEL_TYPE,
                                         torch_dtype=torch.bfloat16, device_map=str(device))
    predictor = Swift.from_pretrained(base, adapters)
    # Swift.from_pretrained materializes the adapter on CPU, modules_to_save copies of the
    # resampler and latent head included -- they are whole modules, not deltas fused into
    # a placed weight.
    predictor = predictor.to(device).eval()

    template = get_template(processor, template_type=vrae_template.TEMPLATE_TYPE, max_length=8192)
    # 'train' mode on a model in eval(): Qwen2VLTemplate._post_encode returns `inputs`
    # untouched when not training, which hands the backbone input_ids + pixel_values
    # instead of the inputs_embeds the training path built.
    template.set_mode('train')
    template.register_post_encode_hook([predictor])
    return predictor, template


@torch.no_grad()
def predict_latent(predictor, template, row: dict, device) -> torch.Tensor:
    """Return z_pred as [chunks, tokens, channels] in raw latent space.

    Raw, not normalized: latent_losses standardizes both sides internally and
    LatentNormalization.denormalize is never called, so the head's output is already in
    the space the V-RAE decoder consumes.
    """
    batch = template.data_collator([template.encode(dict(row))])
    batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
    return predictor(**batch).z_pred[0].float()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-root', required=True)
    parser.add_argument('--split', default='val')
    parser.add_argument('--adapters', default=None,
                        help='trained checkpoint; omit to evaluate the oracle path only')
    parser.add_argument('--out', default='outputs/vrae_future_eval')
    parser.add_argument('--limit', type=int, default=4)
    parser.add_argument('--vrae-root', default=DEFAULT_VRAE_ROOT)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--model', default=os.environ.get('QWEN35_PATH', '/GPFS/ComfyUI_models/LLM/Qwen3.5-9B'))
    parser.add_argument('--latent-stats',
                        default=os.environ.get('LATENT_STATS_PATH')
                        or '/data1/qirui/ms-swift/ckpts/latent_stats/vjepa_vpdata_40future.pt')
    args = parser.parse_args()

    cache_root = Path(args.cache_root).expanduser().resolve()
    vrae_root = Path(args.vrae_root).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    device = torch.device(args.device)

    rows = []
    jsonl = cache_root / 'swift_jsonl' / f'{args.split}.jsonl'
    with jsonl.open('r', encoding='utf-8') as handle:
        for line in handle:
            rows.append(json.loads(line))
    rows = rows[:args.limit]
    if not rows:
        raise SystemExit(f'{jsonl} has no rows')

    sample_meta = metadata_for(rows[0])
    image_size = tuple(sample_meta['image_size'])
    grid_h, grid_w = sample_meta['grid_height'], sample_meta['grid_width']
    half_frames = sample_meta['half_frames']
    print(f'geometry: image_size {image_size}, grid {grid_h}x{grid_w}, '
          f'half_frames {half_frames}', flush=True)

    vrae, adapter, reference_sampling = load_vrae(vrae_root, image_size, device)
    from vrae.data import VideoReader, resize_video, uint8_to_float

    predictor = template = None
    if args.adapters:
        # Optional so the oracle ceiling can be measured on its own, without the 19 GB
        # backbone: oracle already bad means V-RAE is the bottleneck on this domain and
        # there is no point reading the prediction panel yet.
        if not Path(args.latent_stats).is_file():
            raise SystemExit(f'--latent-stats {args.latent_stats} not found')
        print(f'loading predictor from {args.adapters}', flush=True)
        predictor, template = load_predictor(args.model, args.adapters, args.latent_stats, device)

    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for row in rows:
        sample_id = row['sample_id']
        target = torch.load(row['target_latent_path'], map_location='cpu', weights_only=True)['latent']
        meta = metadata_for(row)

        reader = VideoReader(meta['source']['video_path'], backend='auto', num_threads=1,
                            seek_mode='exact')
        start = meta['source']['window_start_frame'] + half_frames
        frames = uint8_to_float(reader.get_frames(range(start, start + half_frames)))
        ground_truth = resize_video(frames, image_size, mode='bicubic').clamp_(0, 1)

        with torch.no_grad():
            grid = adapter.tokens_to_grid(
                target.unsqueeze(0).float().to(device), height=grid_h, width=grid_w)
            oracle = adapter.decode_grid(grid)[0].clamp_(0, 1).cpu()

        panels = [ground_truth, oracle]
        labels = ['Ground Truth', 'Oracle V-RAE']
        row_summary = {'sample_id': sample_id, 'oracle_psnr': psnr(ground_truth, oracle)}

        if predictor is not None:
            z_pred = predict_latent(predictor, template, row, device)
            with torch.no_grad():
                grid = adapter.tokens_to_grid(z_pred.unsqueeze(0), height=grid_h, width=grid_w)
                prediction = adapter.decode_grid(grid)[0].clamp_(0, 1).cpu()
            panels.append(prediction)
            labels.append('Prediction')
            row_summary['pred_psnr'] = psnr(ground_truth, prediction)
            # Prediction against the oracle rather than against the frames isolates the
            # latent error: the gap to ground truth also contains V-RAE's own
            # reconstruction loss, which the predictor cannot be blamed for.
            row_summary['pred_vs_oracle_psnr'] = psnr(oracle, prediction)

        video = torch.cat(panels, dim=-1)
        path = out_dir / f'{sample_id}-compare.mp4'
        reference_sampling.write_video(path, video, float(meta['source'].get('fps') or 24.0))
        detail = f'oracle {row_summary["oracle_psnr"]:.2f} dB'
        if 'pred_psnr' in row_summary:
            detail += (f'  pred {row_summary["pred_psnr"]:.2f} dB'
                       f'  pred-vs-oracle {row_summary["pred_vs_oracle_psnr"]:.2f} dB')
        print(f'{sample_id}: {detail}  ({" | ".join(labels)}) -> {path}', flush=True)
        summary.append(row_summary)

    def mean_of(key):
        values = [item[key] for item in summary if key in item]
        return sum(values) / len(values) if values else None

    report = f'\n{len(summary)} samples, mean oracle PSNR {mean_of("oracle_psnr"):.2f} dB'
    if mean_of('pred_psnr') is not None:
        report += (f', prediction {mean_of("pred_psnr"):.2f} dB'
                   f', prediction vs oracle {mean_of("pred_vs_oracle_psnr"):.2f} dB'
                   f'\n  the oracle number is the ceiling: no predictor can beat decoding the '
                   f'target latent itself.')
    print(report)
    (out_dir / 'summary.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
