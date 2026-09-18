# Copyright (c) ModelScope Contributors. All rights reserved.
"""VPData future-prediction dataset: context clip + instruction -> V-RAE target latent.

The rows produced here carry two non-standard keys, ``target_latent_path`` and
``grid_hw``.  Getting them all the way to the model needs three things to line up:

1. A registered ``DatasetMeta`` so this module's ``preprocess_func`` runs.  Passing the
   jsonl path straight to ``--dataset`` instead picks the default ``AutoPreprocessor``,
   which rebuilds each row from a fixed set of fields and silently drops both keys.
2. ``--remove_unused_columns false``.  ``DatasetLoader._load_dataset_path``
   (swift/dataset/loader.py) ends with ``RowPreprocessor.remove_useless_columns``,
   which keeps only ``RowPreprocessor.standard_keys``; the flag defaults to *true*.
   Note this is a different switch from the HF ``TrainingArguments.remove_unused_columns``
   that ``sft_args.py`` already forces off -- same name, different effect.
3. ``StdTemplateInputs.from_dict`` (swift/template/template_inputs.py), which collects
   every column the dataclass does not declare into ``extra_kwargs``.  That is where
   ``Qwen35VRAETemplate._encode`` reads them.

Build the jsonl splits first, then train against the registered dataset name:

    python examples/vrae_future_pred/dataset.py --cache-root <latent cache> --build
    swift sft --dataset vpdata_future_pred --remove_unused_columns false ...
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from swift.dataset import DatasetMeta, RowPreprocessor, load_dataset, register_dataset

DEFAULT_CACHE_ROOT = '/data1/qirui/Datasets/vpdata_future_latents/trunk0'
# Used when --no-caption builds the text-free ablation: the model then has to predict
# the future from the context frames alone.
FALLBACK_INSTRUCTION = 'Predict how this video continues over the next 40 frames.'
DATASET_NAME = 'vpdata_future_pred'
# The val split needs a name of its own: `--val_dataset <path>/val.jsonl` would route
# through AutoPreprocessor and drop target_latent_path exactly as it does for train,
# leaving compute_sft_loss with no target to score.
VAL_DATASET_NAME = 'vpdata_future_pred_val'
SEED = 3407


def cache_root() -> Path:
    return Path(os.environ.get('VPDATA_LATENT_ROOT', DEFAULT_CACHE_ROOT)).expanduser().resolve()


def jsonl_dir(root: Optional[Path] = None) -> Path:
    return (root or cache_root()) / 'swift_jsonl'


def source_video_id(sample_id: str) -> str:
    """Group clips coming from the same source video.

    VPData sample ids look like ``000005000001.0_2`` and ``000005000001.0_3``: the
    trailing ``_<n>`` is the clip/mask index, and both of those come from the single
    source video ``000005000001.0.mp4``.  Splitting on sample id would therefore leak
    frames of one video across train and validation, so the split key drops that suffix.
    """
    return sample_id.rsplit('_', 1)[0]


def build_jsonl(
    root: Path,
    *,
    val_ratio: float = 0.1,
    use_caption: bool = True,
    out_dir: Optional[Path] = None,
) -> Dict[str, Path]:
    """Turn the latent cache manifest into train/val jsonl files."""

    import random

    manifest_path = root / 'manifest.json'
    with manifest_path.open('r', encoding='utf-8') as handle:
        manifest = json.load(handle)
    records = manifest['records']
    if not records:
        raise ValueError(f'{manifest_path} has no records')
    image_height, image_width = (int(value) for value in manifest['image_size'])
    num_frames = int(manifest['half_frames'])
    # The latent grid: taken from the manifest when it says so, otherwise derived from
    # image_size and the patch size. Not hardcoded to 16 any more -- an InternVideo-Next
    # target cache is patch 14 at 224x224, where 224//16 = 14 would disagree with the
    # actual 16x16 grid and trip the token-count check below with a confusing message.
    latent_grid = manifest.get('latent_grid')
    if latent_grid:
        grid_height, grid_width = (int(value) for value in latent_grid)
    else:
        patch_size = int(manifest.get('patch_size', 16))
        grid_height, grid_width = image_height // patch_size, image_width // patch_size
    expected_tokens = grid_height * grid_width

    # What Qwen sees of the *context* is independent of which latent space the target
    # lives in, and has to stay fixed across target spaces or the predictability probe
    # compares two different inputs as well as two different targets. So these come from
    # their own manifest keys, defaulting to the target geometry for a V-RAE cache where
    # the two coincide.
    context_height, context_width = (int(v) for v in manifest.get('context_image_size',
                                                                  [image_height, image_width]))
    context_frames = int(manifest.get('context_num_frames', num_frames))
    latent_shape = manifest.get('latent_shape')
    if latent_shape and int(latent_shape[1]) != expected_tokens:
        raise ValueError(f'manifest latent_shape {latent_shape} disagrees with image_size '
                         f'{[image_height, image_width]} -> {expected_tokens} tokens')

    missing: List[str] = []
    rows: List[Dict[str, Any]] = []
    for record in records:
        if 'last_half' not in record:
            missing.append(f"{record['sample_id']}: no last_half latent")
            continue
        if 'context_clip' not in record:
            missing.append(f"{record['sample_id']}: no context clip")
            continue
        clip = root / record['context_clip']
        target = root / record['last_half']
        if not clip.is_file() or not target.is_file():
            missing.append(f"{record['sample_id']}: referenced file absent")
            continue
        instruction = (record.get('caption') or '').strip() if use_caption else ''
        rows.append({
            'sample_id': record['sample_id'],
            'video_id': source_video_id(record['sample_id']),
            'messages': [
                {
                    'role': 'user',
                    'content': f'<video>{instruction or FALLBACK_INSTRUCTION}'
                },
                # The assistant turn exists only so the template emits a `labels`
                # tensor.  The trainer reads `inputs['labels']` unconditionally
                # (swift/trainers/seq2seq_trainer.py) even when the model returns its
                # own loss, and the model ignores these labels entirely.
                {
                    'role': 'assistant',
                    'content': ''
                },
            ],
            'videos': [str(clip)],
            'context_latent_path': str(root / record['first_half']) if record.get('first_half') else None,
            'target_latent_path': str(target),
            'grid_hw': [grid_height, grid_width],
            # Qwen2VLTemplate.replace_tag merges chat_template_kwargs straight into the
            # qwen_vl_utils fetch_video call (swift/template/templates/qwen.py), and
            # chat_template_kwargs is one of RowPreprocessor.standard_keys, so it also
            # survives column pruning. These three keys pin what Qwen actually sees:
            #   nframes        - without it fetch_video samples at FPS=2.0, turning the
            #                    40-frame clip into ~4 frames
            #   resized_*      - without them smart_resize picks its own size from a
            #                    pixel budget and lands on 288x480 instead of 256x448
            'chat_template_kwargs': {
                'nframes': context_frames,
                'resized_height': context_height,
                'resized_width': context_width,
            },
        })

    if missing:
        print(f'skipped {len(missing)} records (first: {missing[:3]})')
    if not rows:
        raise ValueError('no usable rows; did you build the cache with --context-clip?')

    # Split by source video, never by sample, so two clips of one video cannot land on
    # both sides of the split.
    video_ids = sorted({row['video_id'] for row in rows})
    random.Random(SEED).shuffle(video_ids)
    num_val = max(1, int(round(len(video_ids) * val_ratio))) if val_ratio > 0 else 0
    val_ids = set(video_ids[:num_val])

    splits = {'train': [], 'val': []}
    for row in rows:
        splits['val' if row['video_id'] in val_ids else 'train'].append(row)

    target_dir = out_dir or jsonl_dir(root)
    target_dir.mkdir(parents=True, exist_ok=True)
    written: Dict[str, Path] = {}
    for split, split_rows in splits.items():
        if not split_rows:
            continue
        path = target_dir / f'{split}.jsonl'
        with path.open('w', encoding='utf-8') as handle:
            for row in split_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + '\n')
        written[split] = path
        print(f'{split}: {len(split_rows)} samples, '
              f'{len({r["video_id"] for r in split_rows})} source videos -> {path}')
    return written


class VRAEFuturePreprocessor(RowPreprocessor):
    """Pass the row through, keeping the two custom keys intact."""

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        target = row.get('target_latent_path')
        videos = row.get('videos')
        if not target or not videos:
            return None
        return {
            'messages': row['messages'],
            'videos': videos,
            'target_latent_path': target,
            'context_latent_path': row.get('context_latent_path'),
            'grid_hw': row.get('grid_hw'),
            # Drives qwen_vl_utils.fetch_video; losing it silently changes the frames
            # and resolution Qwen sees relative to what V-RAE encoded.
            'chat_template_kwargs': row.get('chat_template_kwargs') or {},
        }


def register(root: Optional[Path] = None) -> None:
    """(Re)register both splits for one cache root.

    Registration happens at import time from ``VPDATA_LATENT_ROOT``; call this again
    when pointing at a different cache (``--cache-root``).

    Both splits get a registered name rather than being passed as jsonl paths, because
    the custom columns only survive the loader when this module's ``preprocess_func``
    runs -- see the module docstring.  ``val.jsonl`` is allowed to be absent (a cache
    built with ``--val-ratio 0`` has none); the name then simply fails to load if a run
    asks for it.
    """
    for dataset_name, split in ((DATASET_NAME, 'train'), (VAL_DATASET_NAME, 'val')):
        register_dataset(
            DatasetMeta(
                dataset_path=str(jsonl_dir(root) / f'{split}.jsonl'),
                dataset_name=dataset_name,
                preprocess_func=VRAEFuturePreprocessor(),
                tags=['video', 'future-prediction', 'vrae'],
            ),
            exist_ok=True,
        )


register()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-root', default=str(cache_root()))
    parser.add_argument('--val-ratio', type=float, default=0.1)
    parser.add_argument('--caption', action=argparse.BooleanOptionalAction, default=True,
                        help='--no-caption builds the text-free ablation split')
    parser.add_argument('--out-dir', default=None)
    parser.add_argument('--build', action='store_true', help='write the jsonl splits')
    parser.add_argument('--show', action='store_true', help='load the dataset and print row 0')
    args = parser.parse_args()

    root = Path(args.cache_root).expanduser().resolve()
    os.environ['VPDATA_LATENT_ROOT'] = str(root)
    register(root)
    if args.build or not (jsonl_dir(root) / 'train.jsonl').is_file():
        build_jsonl(
            root,
            val_ratio=args.val_ratio,
            use_caption=args.caption,
            out_dir=None if args.out_dir is None else Path(args.out_dir),
        )
    if args.show:
        # Load by the registered name, not the jsonl path, and keep the extra columns.
        dataset = load_dataset([DATASET_NAME], remove_unused_columns=False)[0]
        print(f'dataset: {dataset}')
        row = dataset[0]
        for key, value in row.items():
            text = str(value)
            print(f'  {key} = {text[:110]}{"..." if len(text) > 110 else ""}')
        for key in ('target_latent_path', 'videos', 'messages'):
            assert key in row, (f'{key} did not survive the dataset pipeline; '
                                f'load by the registered name {DATASET_NAME!r} and pass '
                                f'remove_unused_columns=False')
        print('\ncustom keys survived the pipeline')


if __name__ == '__main__':
    main()
