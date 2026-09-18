#!/usr/bin/env python3
# Copyright (c) ModelScope Contributors. All rights reserved.
"""Select (source video, target video, instruction) triplets from openVE background_change.

The csv is the only place the pairing lives, and only part of the pool is on disk, so the
usable dataset is the intersection -- not the csv and not the directory.  Emitting that
intersection as a manifest is what makes an edit experiment reproducible: the latent
builder, the jsonl and any later eval all read the same file.

Why per-pair selection matters: `videos_1000/` was sampled per *clip*, so all 1000 entries
are edited videos and only 15 of them have their source on disk.  Selecting by pair instead
is the whole difference between an edit task and a future-prediction task.

    python examples/vrae_future_pred/openve_edit_pairs.py \
        --root /data1/qirui/Datasets/openVE --out /data1/qirui/Datasets/openVE/edit_pairs.json
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', default='/data1/qirui/Datasets/openVE')
    parser.add_argument('--csv', default=None, help='default: <root>/background_change.csv')
    parser.add_argument('--pool', default=None,
                        help='directory holding the mp4 pool; default: '
                             '<root>/extracted/background_change_new')
    parser.add_argument('--out', default=None, help='default: <root>/edit_pairs.json')
    parser.add_argument('--limit', type=int, default=None, help='keep only the first N triplets')
    parser.add_argument('--one-per-target', action='store_true',
                        help='an edited clip can appear with several (source, prompt) rows; keep '
                             'only its first, so no target latent is scored twice')
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    pool = Path(args.pool) if args.pool else root / 'extracted' / 'background_change_new'
    csv_path = Path(args.csv) if args.csv else root / 'background_change.csv'
    out_path = Path(args.out) if args.out else root / 'edit_pairs.json'
    if not pool.is_dir():
        raise SystemExit(f'{pool} is not a directory')

    on_disk = {name[:-4] for name in os.listdir(pool) if name.endswith('.mp4')}
    print(f'{len(on_disk)} mp4 in {pool}', flush=True)

    triplets = []
    seen_targets = set()
    rows = missing_target = missing_source = 0
    with csv_path.open(newline='', encoding='utf-8') as handle:
        reader = csv.reader(handle, delimiter=';')
        header = next(reader)
        if header[:3] != ['video', 'prompt', 'original_video']:
            raise SystemExit(f'unexpected csv header: {header}')
        for record in reader:
            if len(record) < 3:
                continue
            rows += 1
            target_id = os.path.basename(record[0])[:-4]
            source_id = os.path.basename(record[2])[:-4]
            instruction = record[1].strip()
            if target_id not in on_disk:
                missing_target += 1
                continue
            if source_id not in on_disk:
                missing_source += 1
                continue
            if args.one_per_target and target_id in seen_targets:
                continue
            seen_targets.add(target_id)
            triplets.append({
                'sample_id': f'{source_id}__{target_id}',
                'source_video': str(pool / f'{source_id}.mp4'),
                'target_video': str(pool / f'{target_id}.mp4'),
                'instruction': instruction,
                'source_id': source_id,
                'target_id': target_id,
            })
            if args.limit and len(triplets) >= args.limit:
                break

    print(f'csv rows                 {rows}\n'
          f'  edited clip not on disk {missing_target}\n'
          f'  source clip not on disk {missing_source}\n'
          f'usable triplets          {len(triplets)}\n'
          f'  distinct targets       {len(seen_targets)}')
    out_path.write_text(json.dumps(triplets, indent=1, ensure_ascii=False) + '\n',
                        encoding='utf-8')
    print(f'wrote {out_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
