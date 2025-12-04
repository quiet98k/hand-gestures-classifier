"""Graph-based dataset for hand gesture landmarks, suitable for GNN models.

This module provides `LandmarksGraphDataset`, which loads hand landmark data
and returns PyTorch Geometric `Data` objects with:
- Node features: 2D normalized landmark coordinates (21 nodes for MediaPipe hands)
- Edge index: Hand skeleton connectivity (bi-directional edges)
- Label: Gesture class index

Usage:
    from data_loaders import LandmarksGraphDataset
    ds = LandmarksGraphDataset(split='train')
    data = ds[0]  # PyG Data object
    # data.x: (21, 2) node features
    # data.edge_index: (2, num_edges) edge connectivity
    # data.y: scalar label
"""
from __future__ import annotations

import json
import os
from typing import Callable, List, Sequence, Dict, Optional, Tuple

import torch
from torch.utils.data import Dataset

# Optional: try to import torch_geometric Data class
try:
    from torch_geometric.data import Data as PyGData
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    PyGData = None  # type: ignore


# MediaPipe Hands skeleton connectivity (21 landmarks, 0-indexed)
# Each tuple is (src, dst); we'll create bidirectional edges
HAND_SKELETON_EDGES: List[Tuple[int, int]] = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index finger
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle finger
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Ring finger
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Palm connections (optional, for denser graph)
    (5, 9), (9, 13), (13, 17),
]


def build_hand_edge_index(num_nodes: int = 21, bidirectional: bool = True) -> torch.Tensor:
    """Build edge_index tensor for hand skeleton graph.
    
    Returns:
        edge_index: (2, num_edges) LongTensor
    """
    edges = []
    for src, dst in HAND_SKELETON_EDGES:
        if src < num_nodes and dst < num_nodes:
            edges.append((src, dst))
            if bidirectional:
                edges.append((dst, src))
    
    if not edges:
        # Fallback: fully connected if no skeleton edges apply
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    edges.append((i, j))
    
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index


class LandmarksGraphDataset(Dataset):
    """PyTorch Geometric-compatible Dataset for hand gesture landmarks.

    Each sample is a `torch_geometric.data.Data` object with:
    - x: (num_nodes, 2) node feature matrix (normalized landmark coordinates)
    - edge_index: (2, num_edges) edge connectivity based on hand skeleton
    - y: scalar label (gesture class index)
    - Additional attributes: bbox, label_str

    Parameters
    ----------
    annotations_root : str
        Root directory containing `train`, `val`, `test` folders of JSON files.
    split : str
        One of `train`, `val`, or `test`.
    exclude_labels : Sequence[str]
        Labels to exclude entirely (e.g., 'no_gesture').
    transform : Optional[Callable]
        Optional callable applied to the Data object after construction.
        Useful for PyG transforms (e.g., NormalizeFeatures, AddSelfLoops).
    pre_transform : Optional[Callable]
        Applied once during dataset construction (cached).
    label_mapping : Optional[Dict[str, int]]
        Predefined mapping from label string to integer index.
    include_bbox_features : bool
        If True, append bbox (x, y, w, h) as additional node features.
    bidirectional_edges : bool
        If True, edges are bidirectional (default for undirected GNNs).
    """

    def __init__(
        self,
        annotations_root: str = "data/annotations",
        split: str = "train",
        exclude_labels: Sequence[str] = ("no_gesture",),
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        label_mapping: Optional[Dict[str, int]] = None,
        include_bbox_features: bool = False,
        bidirectional_edges: bool = True,
    ) -> None:
        if not HAS_PYG:
            raise ImportError(
                "torch_geometric is required for LandmarksGraphDataset. "
                "Install it with: pip install torch_geometric"
            )
        
        self.annotations_root = annotations_root
        self.split = split
        self.exclude_labels = set(exclude_labels)
        self.transform = transform
        self.pre_transform = pre_transform
        self.include_bbox_features = include_bbox_features
        self.bidirectional_edges = bidirectional_edges

        split_dir = os.path.join(annotations_root, split)
        if not os.path.isdir(split_dir):
            raise ValueError(f"Split directory does not exist: {split_dir}")

        json_files = [
            os.path.join(split_dir, f)
            for f in os.listdir(split_dir)
            if f.endswith(".json")
        ]
        if not json_files:
            raise ValueError(f"No annotation JSON files found in {split_dir}")

        # Raw sample storage before conversion to Data objects
        raw_samples: List[Dict] = []
        labels_encountered: set[str] = set()
        skipped_empty = 0

        for jf in sorted(json_files):
            with open(jf, "r", encoding="utf-8") as fh:
                try:
                    data = json.load(fh)
                except json.JSONDecodeError as e:
                    raise RuntimeError(f"Failed to parse {jf}: {e}") from e

            for obj_id, obj in data.items():
                bboxes = obj.get("bboxes", [])
                labels = obj.get("labels", [])
                hand_landmarks = obj.get("hand_landmarks", [])

                num_hands = min(len(bboxes), len(labels), len(hand_landmarks))
                if num_hands == 0:
                    continue

                for hand_idx in range(num_hands):
                    label_str = labels[hand_idx]
                    if label_str in self.exclude_labels:
                        continue
                    
                    landmarks_list = hand_landmarks[hand_idx]
                    bbox_list = bboxes[hand_idx]

                    lm_tensor = torch.tensor(landmarks_list, dtype=torch.float32)
                    if lm_tensor.numel() == 0:
                        skipped_empty += 1
                        continue
                    
                    bbox_tensor = torch.tensor(bbox_list, dtype=torch.float32)

                    raw_samples.append({
                        "landmarks": lm_tensor,
                        "bbox": bbox_tensor,
                        "label_str": label_str,
                    })
                    labels_encountered.add(label_str)

        self._skipped_empty = skipped_empty

        # Build or validate label mapping
        if label_mapping is None:
            self.label_to_index = {lbl: i for i, lbl in enumerate(sorted(labels_encountered))}
        else:
            self.label_to_index = label_mapping
            missing = labels_encountered - set(label_mapping.keys())
            if missing:
                raise ValueError(
                    f"Provided label_mapping missing labels present in data: {missing}"
                )

        # Convert raw samples to PyG Data objects
        self._data_list: List[PyGData] = []
        for raw in raw_samples:
            data_obj = self._to_pyg_data(raw)
            if self.pre_transform is not None:
                data_obj = self.pre_transform(data_obj)
            self._data_list.append(data_obj)

    def _to_pyg_data(self, raw: Dict) -> PyGData:
        """Convert raw sample dict to PyG Data object."""
        lm = raw["landmarks"]  # (N, 2) or (N*2,)
        bbox = raw["bbox"]     # (4,)
        label_str = raw["label_str"]
        label_idx = self.label_to_index[label_str]

        # Ensure landmarks are (N, 2)
        if lm.ndim == 1:
            lm = lm.view(-1, 2)
        
        num_nodes = lm.shape[0]

        # Normalize landmarks relative to bbox
        bx, by, bw, bh = bbox
        bw = bw if bw != 0 else 1e-6
        bh = bh if bh != 0 else 1e-6
        lm = lm.clone()
        lm[:, 0] = (lm[:, 0] - bx) / bw
        lm[:, 1] = (lm[:, 1] - by) / bh
        lm = lm.clamp(0.0, 1.0)

        # Node features: normalized (x, y) coordinates
        x = lm  # (num_nodes, 2)

        # Optionally append bbox as repeated feature per node
        if self.include_bbox_features:
            bbox_expanded = bbox.unsqueeze(0).expand(num_nodes, -1)  # (num_nodes, 4)
            x = torch.cat([x, bbox_expanded], dim=1)  # (num_nodes, 6)

        # Build edge index based on hand skeleton
        edge_index = build_hand_edge_index(num_nodes, bidirectional=self.bidirectional_edges)

        # Create PyG Data object
        data = PyGData(
            x=x,
            edge_index=edge_index,
            y=torch.tensor(label_idx, dtype=torch.long),
            bbox=bbox,
            label_str=label_str,
        )
        return data

    def __len__(self) -> int:
        return len(self._data_list)

    def __getitem__(self, idx: int) -> PyGData:
        data = self._data_list[idx]
        if self.transform is not None:
            data = self.transform(data)
        return data

    @property
    def num_classes(self) -> int:
        return len(self.label_to_index)

    @property
    def num_node_features(self) -> int:
        """Number of features per node."""
        if len(self._data_list) == 0:
            return 2 + (4 if self.include_bbox_features else 0)
        return self._data_list[0].x.shape[1]

    def class_names(self) -> List[str]:
        return [lbl for lbl, _ in sorted(self.label_to_index.items(), key=lambda x: x[1])]

    def __repr__(self) -> str:
        return (
            f"LandmarksGraphDataset(split={self.split!r}, size={len(self)}, "
            f"num_classes={self.num_classes}, num_node_features={self.num_node_features})"
        )


# Run with: uv run data_loaders/landmark_graph_data_loader.py
if __name__ == "__main__":
    print("=== Sanity Check: LandmarksGraphDataset ===")
    
    try:
        ds = LandmarksGraphDataset(split="train")
        print(ds)
        
        if getattr(ds, "_skipped_empty", 0) > 0:
            print(f"Skipped {ds._skipped_empty} samples with empty landmarks")
        
        print(f"Total samples: {len(ds)}  | Classes ({ds.num_classes}): {ds.class_names()}")

        assert len(ds) > 0, "Dataset is empty – check annotation path."

        # Inspect first sample
        first = ds[0]
        print(f"\nFirst sample:")
        print(f"  x (node features) shape: {first.x.shape}")
        print(f"  edge_index shape: {first.edge_index.shape}")
        print(f"  num_edges: {first.edge_index.shape[1]}")
        print(f"  y (label): {first.y.item()}")
        print(f"  label_str: {first.label_str}")
        print(f"  bbox: {first.bbox.tolist()}")
        
        # Check node feature bounds
        print(f"  x min/max: {first.x.min().item():.4f} / {first.x.max().item():.4f}")
        assert first.x.min().item() >= -1e-6, "Node features below 0"
        assert first.x.max().item() <= 1.0 + 1e-6, "Node features above 1"

        # Test with PyG DataLoader
        try:
            from torch_geometric.loader import DataLoader as PyGDataLoader
            loader = PyGDataLoader(ds, batch_size=32, shuffle=True)
            batch = next(iter(loader))
            print(f"\nBatched sample:")
            print(f"  batch.x shape: {batch.x.shape}")
            print(f"  batch.edge_index shape: {batch.edge_index.shape}")
            print(f"  batch.y shape: {batch.y.shape}")
            print(f"  batch.batch shape: {batch.batch.shape}")
        except ImportError:
            print("\nSkipping DataLoader test (torch_geometric.loader not available)")

        print("\n=== Sanity Check Complete ===")
        
    except ImportError as e:
        print(f"Cannot run sanity check: {e}")
        print("Install torch_geometric to use this dataset.")
