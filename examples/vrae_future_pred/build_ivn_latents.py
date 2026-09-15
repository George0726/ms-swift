#!/usr/bin/env python3
# Copyright (c) ModelScope Contributors. All rights reserved.
"""Encode the same clips with InternVideo-Next, as a second prediction target.

This exists to answer the one question the V-RAE runs cannot: *is the target space the
problem?*  A V-RAE-native run that loses to a copy-the-context baseline has two possible
causes -- the predictor, or a latent space whose future is not determined by its past.
Re-running the identical experiment against a different target space separates them,
which is Phase 1 of the design notes.

Deliberately NOT reproduced from the V-RAE pipeline:

* the decoder. InternVideo-Next s2 ships an encoder only (the Stage-1 diffusion decoder
  is not in the released checkpoint), so there is no oracle reconstruction and no PSNR
  here -- ``decode_eval.py`` cannot be pointed at this cache. Latent metrics and the
  persistence baseline are the whole deliverable.
* the context geometry. What Qwen reads stays at the V-RAE setting (40 frames,
  256x448) and only the *target* changes, so the probe varies one thing. The manifest
  records that separation via ``context_image_size`` / ``context_num_frames``, which
  dataset.py reads.

Layout mirrors the V-RAE cache so dataset.py and template.py work unchanged:

    <out>/latents/first_half/<index>-<id>.pt   context half  (persistence baseline)
    <out>/latents/last_half/<index>-<id>.pt    future half   (prediction target)
    <out>/metadata/<index>-<id>.json
    <out>/clips/context -> <vrae cache>/clips/context     (symlink, not a copy)
    <out>/manifest.json

    # 8-way shard, then merge
    for r in 0 1 2 3 4 5 6 7; do
      CUDA_VISIBLE_DEVICES=$r python build_ivn_latents.py --shard $r 8 &
    done; wait
    python build_ivn_latents.py --merge
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List

import torch

DEFAULT_VRAE_CACHE = '/data1/qirui/vpdata_future_latents/trunk0'
DEFAULT_OUT = '/data1/qirui/vpdata_future_latents/ivn_trunk0'
DEFAULT_MODEL = '/data1/qirui/internvideo/ckpts/ivn_large'
DEFAULT_VRAE_ROOT = '/data1/qirui/V-RAE'
# From preprocessor_config.json of the released checkpoint.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SCHEMA_VERSION = 1


def atomic_save(path: Path, payload) -> None:
    """Never leave a truncated .pt behind: a killed shard would otherwise poison the
    cache with a file that loads as garbage instead of failing loudly."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(payload, tmp)
    tmp.replace(path)


def load_encoder(model_dir: str, device, dtype=torch.bfloat16):
    from transformers import AutoConfig, AutoModel
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_dir, trust_remote_code=True, dtype=dtype)
    model = model.requires_grad_(False).eval().to(device)
    mc = dict(config.model_config)
    grid = mc['img_size'] // mc['patch_size']
    chunks = mc['num_frames'] // mc['tubelet_size']
    return model, {
        'img_size': int(mc['img_size']),
        'patch_size': int(mc['patch_size']),
        'num_frames': int(mc['num_frames']),
        'tubelet_size': int(mc['tubelet_size']),
        'embed_dim': int(mc['embed_dim']),
        'grid': (int(grid), int(grid)),
        'chunks': int(chunks),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--vrae-cache', default=DEFAULT_VRAE_CACHE,
                        help='drives the sample list, source video paths and window offsets')
    parser.add_argument('--out', default=DEFAULT_OUT)
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--vrae-root', default=DEFAULT_VRAE_ROOT, help='for its video reader')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--shard', type=int, nargs=2, default=(0, 1), metavar=('RANK', 'WORLD'))
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--merge', action='store_true',
                        help='combine manifest.rank*.json into manifest.json and exit')
    # squash keeps the full field of view, matching what V-RAE encoded, so the two target
    # spaces describe the same future. center_crop is closer to how InternVideo-Next was
    # trained but discards the sides of a 16:9 frame -- a different future, which would
    # confound the comparison this cache exists to make.
    parser.add_argument('--resize-mode', default='squash', choices=('squash', 'center_crop'))
    args = parser.parse_args()

    vrae_cache = Path(args.vrae_cache).expanduser().resolve()
    out_root = Path(args.out).expanduser().resolve()
    rank, world = args.shard

    if args.merge:
        records, seen = [], set()
        for part in sorted(out_root.glob('manifest.rank*.json')):
            payload = json.loads(part.read_text(encoding='utf-8'))
            for record in payload['records']:
                if record['sample_id'] not in seen:
                    seen.add(record['sample_id'])
                    records.append(record)
            base = payload
        records.sort(key=lambda item: int(item['index']))
        base['records'] = records
        base['num_samples'] = len(records)
        base.pop('shard', None)
        (out_root / 'manifest.json').write_text(json.dumps(base, indent=2) + '\n', encoding='utf-8')
        print(f'merged {len(records)} records -> {out_root / "manifest.json"}')
        return 0

    for candidate in (args.vrae_root, str(Path(args.vrae_root) / 'src')):
        if candidate not in sys.path:
            sys.path.insert(0, candidate)
    from vrae.data import VideoReader, resize_video, uint8_to_float

    source = json.loads((vrae_cache / 'manifest.json').read_text(encoding='utf-8'))
    records_in = source['records']
    if args.limit:
        records_in = records_in[:args.limit]
    assigned = [r for i, r in enumerate(records_in) if i % world == rank]
    print(f'{len(records_in)} samples, {len(assigned)} on shard {rank}/{world}', flush=True)

    device = torch.device(args.device)
    model, geom = load_encoder(args.model, device)
    print(f'InternVideo-Next: {geom}', flush=True)
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1, 1)
    size = geom['img_size']
    # Uniform subsample of the half's frames down to what the encoder takes.
    half_frames = int(source['half_frames'])
    index = torch.linspace(0, half_frames - 1, geom['num_frames']).round().long().tolist()
    print(f'frame subsample {half_frames} -> {geom["num_frames"]}: {index}', flush=True)

    def prepare(frames: torch.Tensor) -> torch.Tensor:
        """[T,C,H,W] uint8-derived float in [0,1] -> normalized [1,C,T,size,size]."""
        if args.resize_mode == 'center_crop':
            scale = size / min(frames.shape[-2:])
            height = max(size, int(round(frames.shape[-2] * scale)))
            width = max(size, int(round(frames.shape[-1] * scale)))
            frames = resize_video(frames, (height, width), mode='bicubic')
            top = (frames.shape[-2] - size) // 2
            left = (frames.shape[-1] - size) // 2
            frames = frames[..., top:top + size, left:left + size]
        else:
            frames = resize_video(frames, (size, size), mode='bicubic')
        video = frames.clamp_(0, 1).permute(1, 0, 2, 3).unsqueeze(0).to(device)
        return (video - mean) / std

    manifest_records: List[dict] = []
    skipped: List[dict] = []
    written = reused = 0
    import time
    started = time.time()

    for position, record in enumerate(assigned, start=1):
        sample_id = str(record['sample_id'])
        stem = Path(record['last_half']).stem
        paths = {name: out_root / 'latents' / name / f'{stem}.pt'
                 for name in ('first_half', 'last_half')}
        meta_out = out_root / 'metadata' / f'{stem}.json'

        if not args.overwrite and meta_out.is_file() and all(p.is_file() for p in paths.values()):
            reused += 1
        else:
            meta = json.loads((vrae_cache / record['metadata_path']).read_text(encoding='utf-8'))
            src = meta['source']
            try:
                reader = VideoReader(src['video_path'], backend='auto', num_threads=1,
                                     seek_mode='exact')
                starts = {'first_half': int(src['window_start_frame']),
                          'last_half': int(src['window_start_frame']) + half_frames}
                latents = {}
                for name, start in starts.items():
                    raw = uint8_to_float(reader.get_frames([start + i for i in index]))
                    with torch.no_grad():
                        feats = model.extract_features(prepare(raw))
                    grid_h, grid_w = geom['grid']
                    latents[name] = feats[0].reshape(
                        geom['chunks'], grid_h * grid_w, geom['embed_dim']).to(torch.float16).cpu()
            except Exception as error:  # a single unreadable video must not kill the shard
                skipped.append({'sample_id': sample_id, 'reason': f'{type(error).__name__}: {error}'})
                print(f'  [skip] {sample_id}: {type(error).__name__}: {error}', flush=True)
                continue

            for name, tensor in latents.items():
                atomic_save(paths[name], {'sample_id': sample_id, 'half': name,
                                          'latent': tensor, 'caption': record.get('caption', '')})
            meta_out.parent.mkdir(parents=True, exist_ok=True)
            meta_out.write_text(json.dumps({
                'schema_version': SCHEMA_VERSION,
                'dataset': 'vpdata_inpaint',
                'sample_id': sample_id,
                'index': int(record['index']),
                'caption': record.get('caption', ''),
                'encoder': 'internvideo_next_s2_large',
                'normalized': False,
                'num_frames': geom['num_frames'],
                'half_frames': half_frames,
                'frame_subsample': index,
                'image_size': [size, size],
                'resize_mode': args.resize_mode,
                'grid_height': geom['grid'][0],
                'grid_width': geom['grid'][1],
                'patch_size': geom['patch_size'],
                'latent': {'shape': [geom['chunks'], geom['grid'][0] * geom['grid'][1],
                                     geom['embed_dim']],
                           'dtype': 'float16',
                           'layout': '[chunks, height*width, channels]'},
                # Carried over so decode-side tools and the persistence baseline can find
                # the source frames without reaching back into the V-RAE cache.
                'source': src,
            }, indent=2) + '\n', encoding='utf-8')
            written += 1

        entry = {
            'sample_id': sample_id,
            'index': int(record['index']),
            'metadata_path': f'metadata/{stem}.json',
            'shape': [geom['chunks'], geom['grid'][0] * geom['grid'][1], geom['embed_dim']],
            'caption': record.get('caption', ''),
            'first_half': f'latents/first_half/{stem}.pt',
            'last_half': f'latents/last_half/{stem}.pt',
        }
        # The context clip Qwen reads is the V-RAE cache's, by absolute path: this cache
        # changes the target, not the input.
        if 'context_clip' in record:
            entry['context_clip'] = str(vrae_cache / record['context_clip'])
        manifest_records.append(entry)

        if position == 1 or position % 25 == 0 or position == len(assigned):
            rate = position / max(time.time() - started, 1e-6)
            print(f'  [{position}/{len(assigned)}] {sample_id}  {rate:.2f}/s  '
                  f'eta {(len(assigned) - position) / max(rate, 1e-9) / 60:.1f} min', flush=True)

    manifest_records.sort(key=lambda item: int(item['index']))
    suffix = '' if world == 1 else f'.rank{rank}'
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / f'manifest{suffix}.json').write_text(json.dumps({
        'schema_version': SCHEMA_VERSION,
        'dataset': 'vpdata_inpaint',
        'encoder': 'internvideo_next_s2_large',
        'encoder_path': str(Path(args.model).resolve()),
        'vrae_cache': str(vrae_cache),
        'num_samples': len(manifest_records),
        'shard': {'rank': rank, 'world': world},
        'normalized': False,
        'num_frames': geom['num_frames'],
        'half_frames': half_frames,
        'frame_subsample': index,
        'image_size': [size, size],
        'resize_mode': args.resize_mode,
        'patch_size': geom['patch_size'],
        # dataset.py reads these three: the grid it cannot derive (patch 14 at 224 gives
        # 16, not 224//16), and the context geometry it must not take from image_size.
        'latent_grid': list(geom['grid']),
        'context_image_size': list(source['image_size']),
        'context_num_frames': int(source['half_frames']),
        'latent_dtype': 'float16',
        'latent_shape': [geom['chunks'], geom['grid'][0] * geom['grid'][1], geom['embed_dim']],
        'latent_layout': '[chunks, height*width, channels]',
        'records': manifest_records,
        'skipped': skipped,
    }, indent=2) + '\n', encoding='utf-8')

    # The clips live once, in the V-RAE cache; a symlink keeps decode-side tools that
    # expect <root>/clips/context working without duplicating 1.4 GB of mp4.
    clips = out_root / 'clips'
    clips.mkdir(parents=True, exist_ok=True)
    if not (clips / 'context').exists():
        os.symlink(vrae_cache / 'clips' / 'context', clips / 'context')

    print(f'\ndone in {(time.time() - started) / 60:.1f} min: {written} encoded, '
          f'{reused} reused, {len(skipped)} skipped\n'
          f'  latent shape : {[geom["chunks"], geom["grid"][0] * geom["grid"][1], geom["embed_dim"]]}\n'
          f'  manifest     : {out_root / f"manifest{suffix}.json"}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
