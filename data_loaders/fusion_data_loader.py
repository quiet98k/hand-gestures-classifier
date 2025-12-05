"""Fusion Dataset that provides both RGB images and graph landmarks for each sample.

This module provides `FusionDataset`, which combines data from both RGBImageDataset
and LandmarksGraphDataset for multi-modal fusion models.

Usage:
    from data_loaders import FusionDataset
    ds = FusionDataset(split='train', image_size=128)
    sample = ds[0]
    # sample['image']: (3, H, W) RGB image tensor
    # sample['graph']: PyG Data object with .x, .edge_index
    # sample['label']: integer label
"""
from __future__ import annotations

import json
import os
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

try:
    from torch_geometric.data import Data as PyGData
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    PyGData = None  # type: ignore

from .landmark_graph_data_loader import HAND_SKELETON_EDGES, build_hand_edge_index


class FusionDataset(Dataset):
    """Dataset that provides both RGB images and graph landmarks for fusion models.

    Each sample contains:
    - image: (C, H, W) RGB image tensor
    - graph: PyG Data object with node features and edge connectivity
    - label: integer class label
    - label_str: string class label

    The dataset matches samples by their file identifiers, ensuring that each
    sample has both modalities available.

    Parameters
    ----------
    annotations_root : str
        Root directory containing train/val/test folders of JSON annotation files.
    crops_root : str
        Root directory containing train/val/test folders of cropped images.
    split : str
        One of 'train', 'val', or 'test'.
    image_size : int
        Size to resize images to (image_size x image_size).
    image_transform : Optional[Callable]
        Additional transform to apply to images after resizing.
    graph_transform : Optional[Callable]
        Transform to apply to PyG Data objects.
    exclude_labels : Sequence[str]
        Labels to exclude (e.g., 'no_gesture').
    """

    def __init__(
        self,
        annotations_root: str = "data/annotations",
        crops_root: str = "data/crops",
        split: str = "train",
        image_size: int = 128,
        image_transform: Optional[Callable] = None,
        graph_transform: Optional[Callable] = None,
        exclude_labels: Sequence[str] = ("no_gesture",),
    ) -> None:
        if not HAS_PYG:
            raise ImportError(
                "torch_geometric is required for FusionDataset. "
                "Install with: pip install torch_geometric"
            )

        self.annotations_root = annotations_root
        self.crops_root = crops_root
        self.split = split
        self.image_size = image_size
        self.image_transform = image_transform
        self.graph_transform = graph_transform
        self.exclude_labels = set(exclude_labels)

        split_ann_dir = os.path.join(annotations_root, split)
        split_crops_dir = os.path.join(crops_root, split)

        if not os.path.isdir(split_ann_dir):
            raise ValueError(f"Annotations directory does not exist: {split_ann_dir}")
        if not os.path.isdir(split_crops_dir):
            raise ValueError(f"Crops directory does not exist: {split_crops_dir}")

        # Build edge_index for hand skeleton (shared across all samples)
        self.edge_index = build_hand_edge_index(num_nodes=21, bidirectional=True)

        # Load all annotations and match with image files
        samples: List[Dict] = []
        labels_encountered: set[str] = set()

        # Get all label directories from crops
        label_dirs = [
            d for d in sorted(os.listdir(split_crops_dir))
            if os.path.isdir(os.path.join(split_crops_dir, d))
        ]

        # For each label, load annotations and find matching images
        for label_str in label_dirs:
            if label_str in self.exclude_labels:
                continue

            label_crop_dir = os.path.join(split_crops_dir, label_str)
            label_ann_file = os.path.join(split_ann_dir, f"{label_str}.json")

            if not os.path.exists(label_ann_file):
                continue

            # Load annotation file
            with open(label_ann_file, "r", encoding="utf-8") as f:
                ann_data = json.load(f)

            # Get all image files in this label's crop folder
            image_files = {}
            for fname in os.listdir(label_crop_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    # Extract base name (without extension) as key
                    base_name = os.path.splitext(fname)[0]
                    image_files[base_name] = os.path.join(label_crop_dir, fname)

            # Match annotations with images
            for obj_id, obj in ann_data.items():
                bboxes = obj.get("bboxes", [])
                labels = obj.get("labels", [])
                hand_landmarks = obj.get("hand_landmarks", [])

                num_hands = min(len(bboxes), len(labels), len(hand_landmarks))

                for hand_idx in range(num_hands):
                    lbl = labels[hand_idx]
                    if lbl != label_str or lbl in self.exclude_labels:
                        continue

                    landmarks_list = hand_landmarks[hand_idx]
                    if not landmarks_list:
                        continue

                    # Generate possible image file names
                    # Common patterns: obj_id, obj_id_hand_idx, etc.
                    possible_keys = [
                        obj_id,
                        f"{obj_id}_{hand_idx}",
                        f"{obj_id}_hand{hand_idx}",
                    ]

                    img_path = None
                    for key in possible_keys:
                        if key in image_files:
                            img_path = image_files[key]
                            break

                    # If no exact match, try partial matching
                    if img_path is None:
                        for base_name, path in image_files.items():
                            if obj_id in base_name:
                                img_path = path
                                break

                    if img_path is None:
                        continue

                    lm_tensor = torch.tensor(landmarks_list, dtype=torch.float32)
                    if lm_tensor.dim() == 1:
                        lm_tensor = lm_tensor.view(-1, 2)

                    samples.append({
                        "image_path": img_path,
                        "landmarks": lm_tensor,
                        "label_str": label_str,
                    })
                    labels_encountered.add(label_str)

        if not samples:
            # Fallback: Create samples by iterating through images and finding landmarks
            samples = self._fallback_load(split_ann_dir, split_crops_dir, label_dirs)
            for s in samples:
                labels_encountered.add(s["label_str"])

        if not samples:
            raise ValueError(f"No matched samples found for split {split}")

        # Build label mapping
        self.label_to_index = {lbl: i for i, lbl in enumerate(sorted(labels_encountered))}

        # Assign label indices
        for s in samples:
            s["label"] = self.label_to_index[s["label_str"]]

        self._samples = samples

    def _fallback_load(
        self, ann_dir: str, crops_dir: str, label_dirs: List[str]
    ) -> List[Dict]:
        """Fallback loading strategy: pair images with landmarks by index order."""
        samples = []

        for label_str in label_dirs:
            if label_str in self.exclude_labels:
                continue

            label_crop_dir = os.path.join(crops_dir, label_str)
            label_ann_file = os.path.join(ann_dir, f"{label_str}.json")

            if not os.path.exists(label_ann_file):
                continue

            # Get sorted image files
            image_files = sorted([
                os.path.join(label_crop_dir, f)
                for f in os.listdir(label_crop_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
            ])

            # Load all landmarks from annotation file
            with open(label_ann_file, "r", encoding="utf-8") as f:
                ann_data = json.load(f)

            landmarks_list = []
            for obj_id, obj in sorted(ann_data.items()):
                hand_landmarks = obj.get("hand_landmarks", [])
                labels = obj.get("labels", [])
                for hand_idx, lm in enumerate(hand_landmarks):
                    if hand_idx < len(labels) and labels[hand_idx] == label_str:
                        if lm:
                            landmarks_list.append(lm)

            # Pair by index
            num_pairs = min(len(image_files), len(landmarks_list))
            for i in range(num_pairs):
                lm_tensor = torch.tensor(landmarks_list[i], dtype=torch.float32)
                if lm_tensor.dim() == 1:
                    lm_tensor = lm_tensor.view(-1, 2)
                samples.append({
                    "image_path": image_files[i],
                    "landmarks": lm_tensor,
                    "label_str": label_str,
                })

        return samples

    def _load_image(self, path: str) -> torch.Tensor:
        """Load and preprocess an image."""
        with Image.open(path) as im:
            im = im.convert("RGB")
            im = im.resize((self.image_size, self.image_size), Image.BILINEAR)

            if self.image_transform is not None:
                out = self.image_transform(im)
                if isinstance(out, torch.Tensor):
                    return out
                if isinstance(out, Image.Image):
                    im = out

            arr = np.asarray(im, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
            return tensor

    def _build_graph(self, landmarks: torch.Tensor) -> PyGData:
        """Build a PyG Data object from landmarks."""
        # Ensure landmarks is (21, 2)
        if landmarks.dim() == 1:
            landmarks = landmarks.view(-1, 2)

        data = PyGData(
            x=landmarks,
            edge_index=self.edge_index.clone(),
        )

        if self.graph_transform is not None:
            data = self.graph_transform(data)

        return data

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Dict:
        s = self._samples[idx]

        # Load image
        image = self._load_image(s["image_path"])

        # Build graph
        graph = self._build_graph(s["landmarks"])

        return {
            "image": image,
            "graph": graph,
            "label": s["label"],
            "label_str": s["label_str"],
        }

    @property
    def num_classes(self) -> int:
        return len(self.label_to_index)

    @property
    def num_node_features(self) -> int:
        return 2  # x, y coordinates

    def class_names(self) -> List[str]:
        return [lbl for lbl, _ in sorted(self.label_to_index.items(), key=lambda x: x[1])]

    def __repr__(self) -> str:
        return (
            f"FusionDataset(split={self.split!r}, size={len(self)}, "
            f"num_classes={self.num_classes}, image_size={self.image_size})"
        )


def fusion_collate_fn(batch: List[Dict]) -> Dict:
    """Custom collate function for FusionDataset.

    Batches images normally, and uses PyG Batch for graphs.
    """
    from torch_geometric.data import Batch

    images = torch.stack([item["image"] for item in batch], dim=0)
    graphs = Batch.from_data_list([item["graph"] for item in batch])
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    label_strs = [item["label_str"] for item in batch]

    return {
        "image": images,
        "graph": graphs,
        "label": labels,
        "label_str": label_strs,
    }


if __name__ == "__main__":
    print("=== Sanity Check: FusionDataset ===")
    ds = FusionDataset(split="train", image_size=128)
    print(ds)
    print(f"Total samples: {len(ds)}  | Classes ({ds.num_classes}): {ds.class_names()}")

    if len(ds) > 0:
        sample = ds[0]
        print(f"Image shape: {sample['image'].shape}")
        print(f"Graph x shape: {sample['graph'].x.shape}")
        print(f"Graph edge_index shape: {sample['graph'].edge_index.shape}")
        print(f"Label: {sample['label_str']} -> {sample['label']}")

        # Test collate function
        from torch.utils.data import DataLoader
        loader = DataLoader(ds, batch_size=4, collate_fn=fusion_collate_fn)
        batch = next(iter(loader))
        print(f"\nBatch image shape: {batch['image'].shape}")
        print(f"Batch graph x shape: {batch['graph'].x.shape}")
        print(f"Batch labels: {batch['label']}")

    print("=== Sanity Check Complete ===")
