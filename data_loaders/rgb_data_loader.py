from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


@dataclass
class ImageSample:
    path: str
    label_index: int
    label_str: str

    def to_dict(self) -> Dict[str, object]:
        return {"path": self.path, "label": self.label_index, "label_str": self.label_str}


class RGBImageDataset(Dataset):
    """PyTorch Dataset for RGB cropped images.

    Expects directory layout:
      data/crops/{split}/{label}/*.jpg|png|jpeg

    Parameters
    ----------
    crops_root: str
        Root directory containing `train`, `val`, `test` folders of label subfolders.
    split: str
        One of `train`, `val`, or `test`.
    transform: Optional[Callable]
        Callable applied to the PIL Image or a torch.Tensor representing the image.
        If None, images are converted to float32 tensors in range [0, 1] with shape (C, H, W).
    label_mapping: Optional[Dict[str, int]]
        Predefined mapping from label name to integer index. If None the mapping is inferred
        from the label folder names (sorted alphabetically for determinism).
    extensions: Sequence[str]
        Accepted image file extensions (case-insensitive).
    """

    def __init__(
        self,
        crops_root: str = "data/crops",
        split: str = "train",
        transform: Optional[Callable] = None,
        label_mapping: Optional[Dict[str, int]] = None,
        extensions: Sequence[str] = (".jpg", ".jpeg", ".png", ".bmp"),
    ) -> None:
        self.crops_root = crops_root
        self.split = split
        self.transform = transform
        self.extensions = tuple(e.lower() for e in extensions)

        split_dir = os.path.join(crops_root, split)
        if not os.path.isdir(split_dir):
            raise ValueError(f"Split directory does not exist: {split_dir}")

        # Discover label folders
        label_dirs = [d for d in sorted(os.listdir(split_dir)) if os.path.isdir(os.path.join(split_dir, d))]
        if not label_dirs:
            raise ValueError(f"No label subdirectories found in {split_dir}")

        samples: List[ImageSample] = []
        labels_encountered: set[str] = set()

        # Walk each label folder and collect image paths
        for lbl in label_dirs:
            lbl_dir = os.path.join(split_dir, lbl)
            for fname in sorted(os.listdir(lbl_dir)):
                _, ext = os.path.splitext(fname)
                if ext.lower() in self.extensions:
                    path = os.path.join(lbl_dir, fname)
                    samples.append(ImageSample(path=path, label_index=-1, label_str=lbl))
                    labels_encountered.add(lbl)

        if not samples:
            raise ValueError(f"No images found under {split_dir} with extensions {self.extensions}")

        # Build or validate label mapping
        if label_mapping is None:
            self.label_to_index = {lbl: i for i, lbl in enumerate(sorted(labels_encountered))}
        else:
            self.label_to_index = label_mapping
            missing = labels_encountered - set(label_mapping.keys())
            if missing:
                raise ValueError(f"Provided label_mapping missing labels present in data: {missing}")

        # Assign indices into samples
        for s in samples:
            s.label_index = self.label_to_index[s.label_str]

        self._samples = samples

    def __len__(self) -> int:  # type: ignore[override]
        return len(self._samples)

    def _load_image(self, path: str) -> torch.Tensor:
        # Open image, ensure RGB, convert to float32 tensor CHW in [0,1]
        with Image.open(path) as im:
            im = im.convert("RGB")
            if self.transform is not None:
                out = self.transform(im)
                # If transform returns a PIL Image convert again
                if isinstance(out, Image.Image):
                    arr = np.asarray(out, dtype=np.float32) / 255.0
                    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
                    return tensor
                if isinstance(out, torch.Tensor):
                    return out
                # If transform returned a numpy array
                if isinstance(out, np.ndarray):
                    tensor = torch.from_numpy(out.astype(np.float32)).permute(2, 0, 1).contiguous()
                    return tensor
                # Otherwise try to coerce to tensor
                arr = np.asarray(out, dtype=np.float32)
                if arr.ndim == 3:
                    arr = arr / 255.0
                    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
                    return tensor
                raise RuntimeError("Unsupported transform return type for image")

            # Default conversion
            arr = np.asarray(im, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
            return tensor

    def __getitem__(self, idx: int) -> Dict[str, object]:  # type: ignore[override]
        s = self._samples[idx]
        try:
            img = self._load_image(s.path)
        except Exception as e:
            raise RuntimeError(f"Failed to load image {s.path}: {e}") from e

        return {"image": img, "label": s.label_index, "label_str": s.label_str, "path": s.path}

    @property
    def num_classes(self) -> int:
        return len(self.label_to_index)

    def class_names(self) -> List[str]:
        return [lbl for lbl, _ in sorted(self.label_to_index.items(), key=lambda x: x[1])]

    def __repr__(self) -> str:
        return (
            f"RGBImageDataset(split={self.split!r}, size={len(self)}, num_classes={self.num_classes})"
        )


if __name__ == "__main__":
    print("=== Sanity Check: RGBImageDataset ===")
    ds = RGBImageDataset(split="train")
    print(ds)
    print(f"Total samples: {len(ds)}  | Classes ({ds.num_classes}): {ds.class_names()}")
    assert len(ds) > 0, "Dataset is empty – check crops path."
    first = ds[0]
    print("First sample:", first["path"]) 
    print("Label:", first["label_str"], "->", first["label"]) 
    img = first["image"]
    print("Image tensor shape (C,H,W):", tuple(img.shape))
    print("Min/Max:", float(img.min()), float(img.max()))
    assert img.min() >= 0.0 and img.max() <= 1.0, "Image tensor not in [0,1]"
    print("=== Sanity Check Complete ===")
