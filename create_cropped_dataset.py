#!/usr/bin/env python3
"""Create cropped RGB hand images from annotations into a folder structure similar to annotations.

Output layout: data/<result_name>/<split>/<label>/*.jpg

Usage examples:
  uv run scripts/create_cropped_dataset.py --result-name crops --max-per-class 200
  uv run scripts/create_cropped_dataset.py --result-name crops --max-per-class 0 --exclude-labels "" --overwrite

By default the script excludes the label 'no_gesture'.
"""
from pathlib import Path
import json
import argparse
from typing import Optional, Set
from PIL import Image
import logging

# tqdm optional
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **_k):
        return x

_LOG = logging.getLogger("create_crops")
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}


def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMAGE_EXTS


def resolve_image_path(images_root: Path, image_ref) -> Optional[Path]:
    if not image_ref:
        return None
    if isinstance(image_ref, list):
        image_ref = image_ref[0] if image_ref else None
    if not image_ref:
        return None
    p = Path(image_ref)
    if p.exists():
        return p
    candidate = images_root / image_ref
    if candidate.exists():
        return candidate
    name = Path(image_ref).name
    found = list(images_root.rglob(name))
    if found:
        return found[0]
    stem = Path(image_ref).stem
    found = list(images_root.rglob(stem + '.*'))
    if found:
        return found[0]
    return None


def crop_and_save(img_path: Path, bbox, out_path: Path) -> bool:
    try:
        im = Image.open(img_path).convert('RGB')
    except Exception as e:
        _LOG.debug('failed to open image %s: %s', img_path, e)
        return False
    w, h = im.size
    try:
        x, y, bw, bh = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    except Exception:
        return False
    if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 < bw <= 1.0 and 0.0 < bh <= 1.0:
        left = int(x * w)
        top = int(y * h)
        right = left + int(bw * w)
        bottom = top + int(bh * h)
    else:
        left = int(x)
        top = int(y)
        right = left + int(bw)
        bottom = top + int(bh)
    left = max(0, min(left, w - 1))
    top = max(0, min(top, h - 1))
    right = max(0, min(right, w))
    bottom = max(0, min(bottom, h))
    if right <= left or bottom <= top:
        return False
    crop = im.crop((left, top, right, bottom))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        crop.save(out_path)
        return True
    except Exception as e:
        _LOG.debug('failed to save crop %s: %s', out_path, e)
        return False


def parse_items_from_json(doc):
    items = []
    if isinstance(doc, dict):
        values = list(doc.values())
        if values and isinstance(values[0], dict) and any(('bboxes' in v or 'hand_landmarks' in v or 'labels' in v) for v in values if isinstance(v, dict)):
            for k, v in doc.items():
                if isinstance(v, dict):
                    vv = dict(v)
                    vv['__id'] = k
                    items.append(vv)
        else:
            for k in ('annotations', 'samples', 'items', 'images'):
                if k in doc and isinstance(doc[k], list):
                    items = list(doc[k])
                    break
            if not items:
                items = [doc]
    elif isinstance(doc, list):
        items = list(doc)
    return items


def process_split(images_root: Path, ann_dir: Path, out_root: Path, max_per_class: int = 0, exclude_labels: Optional[Set[str]] = None):
    """Process one split's annotation directory and create crops under out_root/<split>/<label>/

    exclude_labels: optional set of lowercase labels to skip entirely.
    """
    ann_files = sorted([p for p in ann_dir.iterdir() if p.suffix.lower() == '.json'])
    counts = {}
    for jf in tqdm(ann_files, desc=f"ann_files ({ann_dir.name})"):
        try:
            doc = json.loads(jf.read_text())
        except Exception as e:
            _LOG.warning('failed to load %s: %s', jf, e)
            continue
        items = parse_items_from_json(doc)
        # precompute non-excluded label set present in this JSON so we can stop early
        labels_in_file = set()
        for it0 in items:
            bbs0 = it0.get('bboxes') or it0.get('boxes') or None
            labs0 = it0.get('labels') or it0.get('hand_labels') or None
            if bbs0 and isinstance(bbs0, list):
                for j, _ in enumerate(bbs0):
                    if isinstance(labs0, list) and len(labs0) == len(bbs0):
                        lab = str(labs0[j])
                    elif isinstance(labs0, list) and len(labs0) == 1:
                        lab = str(labs0[0])
                    elif isinstance(labs0, str):
                        lab = str(labs0)
                    else:
                        lab = jf.stem
                    if exclude_labels and lab.lower() in exclude_labels:
                        continue
                    labels_in_file.add(lab)
        for it in tqdm(items, desc=f"items {jf.name}", leave=False):
            # if max_per_class is set and every non-excluded label present in this file has reached the limit,
            # skip remaining items and move to the next JSON file
            if max_per_class and labels_in_file:
                all_reached = True
                for lbl in labels_in_file:
                    if counts.get(lbl, 0) < max_per_class:
                        all_reached = False
                        break
                if all_reached:
                    _LOG.info('all labels in %s reached max_per_class=%d; skipping remaining items', jf.name, max_per_class)
                    break
            # resolve image
            image_keys = [k for k in it.keys() if k.lower() in ('image', 'file_name', 'image_path', 'path')]
            image_ref = None
            for k in image_keys:
                image_ref = it.get(k)
                if image_ref:
                    break
            if isinstance(image_ref, list):
                image_ref = image_ref[0] if image_ref else None
            if not image_ref and '__id' in it:
                image_ref = it.get('__id')
            img_path = resolve_image_path(images_root, image_ref)
            if not img_path:
                continue
            bboxes = it.get('bboxes') or it.get('boxes') or None
            labels = it.get('labels') or it.get('hand_labels') or None
            if bboxes and isinstance(bboxes, list):
                for i, box in enumerate(bboxes):
                    if isinstance(labels, list) and len(labels) == len(bboxes):
                        label_name = labels[i]
                    elif isinstance(labels, list) and len(labels) == 1:
                        label_name = labels[0]
                    elif isinstance(labels, str):
                        label_name = labels
                    else:
                        label_name = jf.stem
                    label_name = str(label_name)
                    # skip excluded labels if provided
                    if exclude_labels and label_name.lower() in exclude_labels:
                        continue
                    # respect max per class
                    cnt = counts.get(label_name, 0)
                    if max_per_class > 0 and cnt >= max_per_class:
                        continue
                    # output path follows: out_root/<split>/<label_name>/*.jpg ; jf.stem used as split-local id
                    out_path = out_root / ann_dir.name / label_name / f"{img_path.stem}__{i}.jpg"
                    ok = crop_and_save(img_path, box, out_path)
                    if ok:
                        counts[label_name] = cnt + 1
    return counts


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--result-name', default='crops', help='name of the result folder under data (data/<result-name>/...)')
    p.add_argument('--data-dir', default='data', help='root data directory')
    p.add_argument('--max-per-class', type=int, default=200, help='max crops per class per split (0 = no limit)')
    p.add_argument('--splits', default='train,val,test', help='comma-separated splits to process')
    p.add_argument('--images-root', default='data/images', help='images root to resolve image paths')
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--exclude-labels', default='no_gesture', help='comma-separated labels to exclude (case-insensitive)')
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    images_root = Path(args.images_root)
    out_root = data_dir / args.result_name
    splits = [s.strip() for s in args.splits.split(',') if s.strip()]

    exclude_labels_set = {lbl.strip().lower() for lbl in args.exclude_labels.split(',') if lbl.strip()} if args.exclude_labels else set()

    logging.basicConfig(level=logging.INFO)

    for s in splits:
        ann_dir = data_dir / 'annotations' / s
        if not ann_dir.exists():
            _LOG.info('annotations dir not found for split %s: %s', s, ann_dir)
            continue
        if args.overwrite and (out_root / s).exists():
            import shutil
            shutil.rmtree(out_root / s)
        (out_root / s).mkdir(parents=True, exist_ok=True)
        _LOG.info('processing split %s -> %s', s, out_root / s)
        counts = process_split(images_root, ann_dir, out_root, max_per_class=args.max_per_class, exclude_labels=exclude_labels_set)
        _LOG.info('split %s: produced %d classes', s, len(counts))
        for k, v in sorted(counts.items(), key=lambda x: -x[1])[:20]:
            _LOG.info('  %s: %d', k, v)

    print('done')


if __name__ == '__main__':
    main()
