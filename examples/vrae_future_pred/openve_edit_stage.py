#!/usr/bin/env python3
# Copyright (c) ModelScope Contributors. All rights reserved.
"""Stage openVE edit pairs as 80-frame clips the existing V-RAE builder can encode as-is.

The builder caches ``first_half`` / ``last_half`` of a single 80-frame window.  An edit
sample is two *different* videos, so instead of teaching the builder about pairs, we hand
it a clip whose first half is the source and whose second half is the edited result:

    frames 0..39   source[0:40]     -> first_half latent  = context  (what Qwen reads)
    frames 40..79  edited[0:40]     -> last_half  latent  = target   (what it predicts)

Everything downstream -- ``build_vpdata_latents.py``, ``dataset.py``, ``template.py``,
``decode_eval.py`` -- then works unchanged, and ``decode_eval.py``'s ground-truth read
(frames 40..79 of this file) lands on the edited frames by construction.

Both halves are re-encoded identically, so the compression does not bias source against
target.  Pair geometry is verified per clip: openVE pairs are frame-aligned, and a
mismatch here would silently produce a target that is not the edit of the context.

    python examples/vrae_future_pred/openve_edit_stage.py \
        --pairs /data1/qirui/Datasets/openVE/edit_pairs.json \
        --out /data1/qirui/Datasets/openVE/edit_stage --workers 32
"""

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HALF = 40


def probe(path: str):
    out = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
         'stream=width,height,nb_frames', '-of', 'csv=p=0', path],
        capture_output=True, text=True).stdout.strip().split(',')
    try:
        return int(out[0]), int(out[1]), int(out[2])
    except (ValueError, IndexError):
        return None


def stage_one(triplet, out_videos: Path, crf: int, overwrite: bool):
    sample_id = triplet['sample_id']
    dst = out_videos / f'{sample_id}.mp4'
    txt = out_videos / f'{sample_id}.txt'
    if dst.is_file() and txt.is_file() and not overwrite:
        return sample_id, 'skip'
    source, target = probe(triplet['source_video']), probe(triplet['target_video'])
    if source is None or target is None:
        return sample_id, 'unreadable'
    if source[:2] != target[:2]:
        return sample_id, f'geometry {source[:2]} vs {target[:2]}'
    if min(source[2], target[2]) < HALF:
        return sample_id, f'too short {source[2]}/{target[2]}'
    # trim by frame count, not by time: a clip at 23.976 fps and one at 25 must both
    # contribute exactly 40 frames or the halves stop lining up chunk for chunk.
    graph = (f'[0:v]trim=start_frame=0:end_frame={HALF},setpts=N/FRAME_RATE/TB[a];'
             f'[1:v]trim=start_frame=0:end_frame={HALF},setpts=N/FRAME_RATE/TB[b];'
             f'[a][b]concat=n=2:v=1:a=0[v]')
    tmp = dst.with_suffix('.tmp.mp4')
    done = subprocess.run(
        ['ffmpeg', '-y', '-v', 'error', '-i', triplet['source_video'], '-i',
         triplet['target_video'], '-filter_complex', graph, '-map', '[v]',
         '-frames:v', str(HALF * 2), '-c:v', 'libx264', '-crf', str(crf),
         '-pix_fmt', 'yuv420p', '-an', str(tmp)],
        capture_output=True, text=True)
    if done.returncode != 0 or not tmp.is_file():
        return sample_id, f'ffmpeg failed: {done.stderr.strip()[:120]}'
    staged = probe(str(tmp))
    if staged is None or staged[2] != HALF * 2:
        tmp.unlink(missing_ok=True)
        return sample_id, f'staged {staged[2] if staged else "?"} frames, want {HALF * 2}'
    tmp.replace(dst)
    txt.write_text(triplet['instruction'] + '\n', encoding='utf-8')
    return sample_id, 'ok'


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pairs', default='/data1/qirui/Datasets/openVE/edit_pairs.json')
    parser.add_argument('--out', default='/data1/qirui/Datasets/openVE/edit_stage')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--workers', type=int, default=32)
    parser.add_argument('--crf', type=int, default=16)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    triplets = json.loads(Path(args.pairs).read_text(encoding='utf-8'))
    if args.limit:
        triplets = triplets[:args.limit]
    out_videos = Path(args.out) / 'videos'
    out_videos.mkdir(parents=True, exist_ok=True)

    counts, failures = {}, []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(stage_one, t, out_videos, args.crf, args.overwrite)
                   for t in triplets]
        for position, future in enumerate(futures, start=1):
            sample_id, status = future.result()
            key = status if status in ('ok', 'skip') else status.split()[0]
            counts[key] = counts.get(key, 0) + 1
            if key not in ('ok', 'skip'):
                failures.append((sample_id, status))
            if position % 250 == 0 or position == len(futures):
                print(f'  [{position}/{len(futures)}] {counts}', flush=True)

    print(f'\nstaged into {out_videos}')
    for key, value in sorted(counts.items()):
        print(f'  {key:12s} {value}')
    if failures:
        report = Path(args.out) / 'staging_failures.json'
        report.write_text(json.dumps(failures, indent=1, ensure_ascii=False) + '\n',
                          encoding='utf-8')
        print(f'  {len(failures)} failures listed in {report}')
    return 0 if counts.get('ok', 0) or counts.get('skip', 0) else 1


if __name__ == '__main__':
    raise SystemExit(main())
