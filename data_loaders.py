"""Flexible data loaders for the hand-gestures project.

Provides:
- RGBImageDataset: simple dataset that walks `data/images/<class>/*` and returns PIL images + label indices.
- LandmarksDataset: loads landmarks/features from per-class JSON files under `data/annotations/<split>/`.

The code is written to be robust to a few annotation schema variants. It will try to auto-detect common keys
('landmarks', 'keypoints', 'hand_landmarks', 'pose') and image filename keys ('image','file_name','path').
If torch is available the dataset will optionally return tensors; otherwise it returns numpy arrays.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from PIL import Image
import numpy as np

try:
    from torch.utils.data import Dataset
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False
    class Dataset:  # type: ignore
        pass

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.gif'}
_LOG = logging.getLogger("data_loaders")


def is_image_file(p: Path) -> bool:
    return p.suffix.lower() in IMAGE_EXTS


def count_images(root: str) -> int:
    """Count image files recursively in root (matches common extensions)."""
    n = 0
    for dirpath, _, files in os.walk(root):
        for fn in files:
            if Path(fn).suffix.lower() in IMAGE_EXTS:
                n += 1
    return n


class RGBImageDataset(Dataset):
    """Dataset that reads images organized by class folders.

    Args:
        root: path to `data/images` containing subfolders for each class.
        transform: optional callable applied to a PIL.Image and returning the final sample.
        extensions: optional set of extensions to accept.
    Returns (image, label_index) where image is a PIL.Image or a torch.Tensor (if torchvision is used
    in the transform) / numpy array otherwise.
    """

    def __init__(self, root: str, transform: Optional[Callable] = None, extensions: Optional[set] = None):
        self.root = Path(root)
        self.transform = transform
        self.extensions = extensions or IMAGE_EXTS

        classes = [p.name for p in self.root.iterdir() if p.is_dir()]
        classes.sort()
        self.class_to_idx = {c: i for i, c in enumerate(classes)}

        self.samples: List[Tuple[Path, int]] = []
        for cls in classes:
            d = self.root / cls
            for p in d.rglob("*"):
                if p.is_file() and p.suffix.lower() in self.extensions:
                    self.samples.append((p, self.class_to_idx[cls]))

        _LOG.info("RGBImageDataset: found %d samples across %d classes", len(self.samples), len(classes))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            return self.transform(img), label
        # default: return numpy array HWC uint8
        return np.array(img), label


class LandmarksDataset(Dataset):
    """Load landmarks/features from JSON annotation files.

    Expected usage:
      annotations_dir = 'data/annotations/train'
      ds = LandmarksDataset(annotations_dir, images_root='data/images', normalize=True)

    This loader is flexible: it will scan all .json files in the provided annotations_dir, and try to
    extract per-sample landmarks and an optional image path and label (label defaults to the basename of the
    JSON file when per-class files are used).
    """

    def __init__(
        self,
        annotations_dir: str,
        images_root: Optional[str] = None,
        normalize: bool = False,
        transform: Optional[Callable] = None,
    ):
        self.annotations_dir = Path(annotations_dir)
        self.images_root = Path(images_root) if images_root else None
        self.normalize = normalize
        self.transform = transform

        self.samples: List[Dict[str, Any]] = []
        self.label_to_idx: Dict[str, int] = {}

        self._load_annotations()
        _LOG.info("LandmarksDataset: loaded %d samples from %s", len(self.samples), self.annotations_dir)

    def _load_annotations(self) -> None:
        json_files = [p for p in self.annotations_dir.iterdir() if p.suffix.lower() == '.json']
        json_files.sort()
        for jf in json_files:
            try:
                with jf.open('r') as f:
                    doc = json.load(f)
            except Exception as e:
                _LOG.warning("failed to load %s: %s", jf, e)
                continue

            # Determine label: for per-class jsons we use the filename
            # If the file is a per-class file (one big mapping id->sample),
            # convert to a list of sample dicts.
            items: List[Dict[str, Any]] = []
            if isinstance(doc, dict):
                # case A: dict is mapping id -> sample (your provided example)
                # detect by checking whether values are dicts that contain 'hand_landmarks' or similar
                values = list(doc.values())
                if values and isinstance(values[0], dict):
                    # if any value has hand_landmarks or labels, treat as per-id mapping
                    if any(('hand_landmarks' in v or 'labels' in v or 'landmarks' in v) for v in values if isinstance(v, dict)):
                        items = values
                # case B: dict contains a top-level list under common keys
                if not items:
                    for k in ('annotations', 'samples', 'items', 'images'):
                        if k in doc and isinstance(doc[k], list):
                            items = doc[k]
                            break
            elif isinstance(doc, list):
                items = doc

            # fallback: if the file contains a single sample dict (not a mapping), keep it as one item
            if not items and isinstance(doc, dict):
                items = [doc]

            # For each item, handle potentially multiple hands in 'hand_landmarks'
            for it in items:
                # determine a default label name (filename stem) in case per-hand labels are not provided
                default_label = jf.stem

                # collect lists if present
                hands = it.get('hand_landmarks') or it.get('hand_keypoints') or it.get('keypoints') or None
                labels = it.get('labels') or it.get('hand_labels') or None
                bboxes = it.get('bboxes') or it.get('boxes') or None

                if hands and isinstance(hands, list) and len(hands) > 0:
                    # iterate each detected hand and pair with label (if available)
                    for i, hand in enumerate(hands):
                        try:
                            lm = np.array(hand, dtype=float)
                            if lm.ndim > 1:
                                lm = lm.reshape(-1)
                        except Exception:
                            _LOG.debug("skipping invalid hand landmarks entry in %s", jf)
                            continue

                        # determine label_name taking into account possible length mismatches
                        label_name = default_label
                        if isinstance(labels, list):
                            if len(labels) == len(hands):
                                label_name = labels[i]
                            elif len(labels) == 1:
                                # single label provided for all hands
                                label_name = labels[0]
                            else:
                                # mismatched lengths; try to pick if available, else fallback
                                if i < len(labels):
                                    label_name = labels[i]
                                else:
                                    _LOG.debug("labels length (%d) != hands (%d) in %s; using default label", len(labels), len(hands), jf)
                                    label_name = default_label
                        elif isinstance(labels, str):
                            label_name = labels

                        if label_name not in self.label_to_idx:
                            self.label_to_idx[label_name] = len(self.label_to_idx)

                        sample: Dict[str, Any] = {
                            'landmarks': lm.astype(np.float32),
                            'label': self.label_to_idx[label_name],
                            'label_name': label_name,
                            'image_path': None,
                        }

                        # attach bbox if available and well-formed (support single bbox for all hands)
                        if bboxes and isinstance(bboxes, list):
                            if len(bboxes) == len(hands) and i < len(bboxes):
                                sample['bbox'] = bboxes[i]
                            elif len(bboxes) == 1:
                                sample['bbox'] = bboxes[0]

                        # try to resolve image path if present
                        image_keys = [k for k in it.keys() if k.lower() in ('image', 'file_name', 'image_path', 'path')]
                        image_path = None
                        for k in image_keys:
                            image_path = it.get(k)
                            if image_path:
                                break
                        if isinstance(image_path, list):
                            image_path = image_path[0] if image_path else None
                        if image_path and self.images_root:
                            candidate = (self.images_root / image_path)
                            if not candidate.exists():
                                candidates = list(self.images_root.rglob(Path(image_path).name))
                                candidate = candidates[0] if candidates else candidate
                            sample['image_path'] = str(candidate)

                        self.samples.append(sample)
                    continue

                # fallback: try parsing single-sample item using existing helper
                sample = self._parse_item(it, jf.stem)
                if sample is not None:
                    self.samples.append(sample)

    def _parse_item(self, it: Dict[str, Any], label: str) -> Optional[Dict[str, Any]]:
        # find landmarks field
        landmark_keys = [k for k in it.keys() if k.lower() in ('landmarks', 'keypoints', 'hand_landmarks', 'hand_keypoints', 'pose')]
        landmarks = None
        for k in landmark_keys:
            landmarks = it.get(k)
            if landmarks:
                break

        # try also numeric-only lists (some annotations are directly lists)
        if landmarks is None:
            # if dict has numeric sequence under 'points' or 'kp'
            for k in ('points', 'kp'):
                if k in it:
                    landmarks = it[k]
                    break

        if landmarks is None:
            # maybe the item itself is a list
            if isinstance(it, list):
                landmarks = it

        if landmarks is None:
            _LOG.debug("no landmarks found in item; skipping")
            return None

        # flatten landmarks
        lm = np.array(landmarks, dtype=float)
        if lm.ndim > 1:
            lm = lm.reshape(-1)

        # image path detection
        image_keys = [k for k in it.keys() if k.lower() in ('image', 'file_name', 'image_path', 'path')]
        image_path = None
        for k in image_keys:
            image_path = it.get(k)
            if image_path:
                break

        if isinstance(image_path, list):
            # sometimes stored as list
            image_path = image_path[0] if image_path else None

        if image_path and self.images_root:
            candidate = (self.images_root / image_path)
            if not candidate.exists():
                # try basename search
                candidates = list(self.images_root.rglob(Path(image_path).name))
                candidate = candidates[0] if candidates else candidate
            image_path = str(candidate)

        # normalization: if requested and image exists, scale by width/height
        if self.normalize and image_path:
            try:
                im = Image.open(image_path)
                w, h = im.size
                # assume landmarks alternate x,y
                xy = lm.reshape(-1, 2)
                xy[:, 0] = xy[:, 0] / float(w)
                xy[:, 1] = xy[:, 1] / float(h)
                lm = xy.reshape(-1)
            except Exception:
                # maybe landmarks are already normalized
                pass

        sample = {
            'landmarks': lm.astype(np.float32),
            'label': self.label_to_idx[label],
            'label_name': label,
            'image_path': image_path,
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        # return (landmarks, label) simple tuple for most training code
        return s['landmarks'], s['label']


if __name__ == '__main__':
    # quick local sanity run when executed directly
    logging.basicConfig(level=logging.INFO)
    # small runner; no pprint used

    root = Path('data/images')
    ann = Path('data/annotations/train')
    
    print("\n=====================Images=====================")
    print('images count:', count_images(str(root)))
    ds = RGBImageDataset(str(root))
    print('classes:', ds.class_to_idx)
    if len(ds) > 0:
        im, lbl = ds[0]
        print('sample image size:', type(im), 'label:', lbl)

    print("\n=====================Landmarks=====================")
    lds = LandmarksDataset(str(ann), images_root=str(root), normalize=False)
    print('landmarks samples:', len(lds))
    # print class mapping for landmarks dataset
    print('landmark classes (label_name -> idx):', lds.label_to_idx)
    print('num landmark classes:', len(lds.label_to_idx))
    if len(lds) > 0:
        lm, lbl = lds[0]
        # label index -> try to read label_name from the underlying sample if available
        label_name = None
        try:
            label_name = lds.samples[0].get('label_name')
        except Exception:
            label_name = None
        print('landmark shape:', lm.shape, 'label_idx:', lbl, 'label_name:', label_name)
