"""Data loading utilities for hand gesture landmark annotations.

This module implements a simple PyTorch Dataset that reads landmark-only
annotations from JSON files under `data/annotations/{train,val,test}`.

Each JSON file contains a dict of annotation objects keyed by an ID. Example
object structure (irrelevant keys omitted):

{
	"some-uuid": {
		"bboxes": [[x, y, w, h], ...],          # one per detected hand
		"labels": ["gesture_label", ...],      # one per detected hand
		"hand_landmarks": [                     # one list per hand
			[[x0, y0], [x1, y1], ...],            # landmarks for hand 0
			[[x0, y0], [x1, y1], ...]             # landmarks for hand 1
		]
	}
}

Dataset item granularity: one hand per sample (NOT one person). If an object
has two hands we create two samples. Any label in `exclude_labels` is skipped.

Returned sample dict contains (landmarks are normalized to the bounding box
and clamped to [0,1]):
	- landmarks: FloatTensor shape (N_landmarks, 2) each coordinate in [0,1]
	  relative to its hand bbox (x,y offset removed, width/height scaled)
	- bbox: FloatTensor shape (4,) in [x, y, w, h] normalized to image dims
	- label: int class index
	- label_str: original string label

Keep it intentionally minimal / non-opinionated so you can extend later.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Callable, List, Sequence, Dict, Optional

import torch
from torch.utils.data import Dataset, DataLoader


@dataclass
class LandmarkSample:
	"""Structured representation for a single hand sample.

	This is an internal convenience container; externally we expose a
	dictionary for compatibility with default PyTorch DataLoader collation.
	"""
	landmarks: torch.Tensor  # (N, 2)
	bbox: torch.Tensor       # (4,)
	label_index: int
	label_str: str

	def to_dict(self) -> Dict[str, torch.Tensor | int | str]:
		return {
			"landmarks": self.landmarks,
			"bbox": self.bbox,
			"label": self.label_index,
			"label_str": self.label_str,
		}


class LandmarksDataset(Dataset):
	"""PyTorch Dataset for hand gesture landmarks.

	Parameters
	----------
	annotations_root : str
		Root directory containing `train`, `val`, `test` folders of JSON files.
	split : str
		One of `train`, `val`, or `test`.
	exclude_labels : Sequence[str]
		Labels to exclude entirely (e.g., 'no_gesture').
	transform : Optional[Callable]
		Optional callable applied to the landmarks tensor ONLY
		(signature: fn(torch.Tensor) -> torch.Tensor). Use for basic
		normalization / augmentation that doesn't need the bbox/label.
	label_mapping : Optional[Dict[str, int]]
		Predefined mapping from label string to integer index. If None, the
		mapping is inferred from data (sorted alphabetically for determinism).
	flatten : bool
		If True, returned `landmarks` tensor is flattened to shape (N*2,) for
		simple MLP style models.
	"""

	def __init__(
		self,
		annotations_root: str = "data/annotations",
		split: str = "train",
		exclude_labels: Sequence[str] = ("no_gesture",),
		transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
		label_mapping: Optional[Dict[str, int]] = None,
		flatten: bool = False,
	) -> None:
		self.annotations_root = annotations_root
		self.split = split
		self.exclude_labels = set(exclude_labels)
		self.transform = transform
		self.flatten = flatten

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

		samples: List[LandmarkSample] = []
		labels_encountered: set[str] = set()

		for jf in sorted(json_files):
			with open(jf, "r", encoding="utf-8") as fh:
				try:
					data = json.load(fh)
				except json.JSONDecodeError as e:
					raise RuntimeError(f"Failed to parse {jf}: {e}") from e

			# Each top-level key is a person/object instance
			for obj_id, obj in data.items():
				bboxes = obj.get("bboxes", [])
				labels = obj.get("labels", [])
				hand_landmarks = obj.get("hand_landmarks", [])

				# Basic consistency check; skip malformed entries
				num_hands = min(len(bboxes), len(labels), len(hand_landmarks))
				if num_hands == 0:
					continue

				for hand_idx in range(num_hands):
					label_str = labels[hand_idx]
					if label_str in self.exclude_labels:
						continue
					landmarks_list = hand_landmarks[hand_idx]
					bbox_list = bboxes[hand_idx]

					# Convert to tensors (float32)
					lm_tensor = torch.tensor(landmarks_list, dtype=torch.float32)
					bbox_tensor = torch.tensor(bbox_list, dtype=torch.float32)

					sample = LandmarkSample(
						landmarks=lm_tensor,
						bbox=bbox_tensor,
						label_index=-1,  # placeholder; assigned after mapping
						label_str=label_str,
					)
					samples.append(sample)
					labels_encountered.add(label_str)

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

		# Assign indices
		for s in samples:
			s.label_index = self.label_to_index[s.label_str]

		self._samples = samples

	def __len__(self) -> int:  # type: ignore[override]
		return len(self._samples)

	def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | int | str]:  # type: ignore[override]
		sample = self._samples[idx]
		lm = sample.landmarks
		bbox = sample.bbox  # [x, y, w, h] already normalized to image dims
		# Normalize landmarks relative to bbox: ( (x - bx)/bw, (y - by)/bh )
		if lm.ndim == 2 and lm.shape[1] >= 2:
			bx, by, bw, bh = bbox
			# Avoid division by zero in malformed boxes
			bw = bw if bw != 0 else 1e-6
			bh = bh if bh != 0 else 1e-6
			lm = lm.clone()
			lm[:, 0] = (lm[:, 0] - bx) / bw
			lm[:, 1] = (lm[:, 1] - by) / bh
			lm.clamp_(0.0, 1.0)
		if self.transform is not None:
			lm = self.transform(lm)
		if self.flatten:
			lm = lm.flatten()
		out = {
			"landmarks": lm,
			"bbox": bbox,
			"label": sample.label_index,
			"label_str": sample.label_str,
		}
		return out

	@property
	def num_classes(self) -> int:
		return len(self.label_to_index)

	def class_names(self) -> List[str]:
		return [lbl for lbl, _ in sorted(self.label_to_index.items(), key=lambda x: x[1])]

	def __repr__(self) -> str:
		return (
			f"LandmarksDataset(split={self.split!r}, size={len(self)}, "
			f"num_classes={self.num_classes}, flatten={self.flatten})"
		)




if __name__ == "__main__":
	print("=== Sanity Check: LandmarksDataset ===")
	ds = LandmarksDataset(split="train")
	print(ds)
	print(f"Total samples: {len(ds)}  | Classes ({ds.num_classes}): {ds.class_names()}")

	# Assert at least one sample exists
	assert len(ds) > 0, "Dataset is empty – check annotation path."

	# Grab first sample
	first = ds[0]
	print("First sample label:", first["label_str"], "index:", first["label"])
	print("First sample landmarks shape:", first["landmarks"].shape)
	print("First sample bbox:", first["bbox"].tolist())
	print("Landmark min/max (per coord):", first["landmarks"].min().item(), first["landmarks"].max().item())
	assert 0.0 <= first["landmarks"].min().item() - 1e-6, "Landmarks below 0 after normalization"  # tolerance
	assert first["landmarks"].max().item() <= 1.0 + 1e-6, "Landmarks above 1 after normalization"

	# Basic shape expectation (assumption: 21 landmarks like MediaPipe Hands)
	if first["landmarks"].ndim == 2:
		n_pts = first["landmarks"].shape[0]
		if n_pts not in (21,):  # adjust if your landmark count differs
			print(f"[WARN] Unexpected landmark count: {n_pts}")

	# DataLoader batch sanity (keep simple, no custom collate)
	loader = DataLoader(ds, batch_size=4, shuffle=True)
	batch = next(iter(loader))
	# batch is a dict of tensors / lists
	lm_batch = batch["landmarks"]
	if isinstance(lm_batch, torch.Tensor):
		print("Batch landmarks tensor shape:", lm_batch.shape)
	else:
		print("Batch landmarks list length:", len(lm_batch))
	print("Batch labels:", batch["label"])

	# Flatten demo
	ds_flat = LandmarksDataset(split="train", flatten=True)
	flat_sample = ds_flat[0]["landmarks"]
	print("Flattened landmarks vector length:", flat_sample.shape[0])
	print("Flat sample min/max:", flat_sample.min().item(), flat_sample.max().item())
	print("=== Sanity Check Complete ===")

