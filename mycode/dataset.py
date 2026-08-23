from __future__ import annotations

import json
import os
from typing import Dict, Optional

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

from config import cfg


CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)


def official_base_name(dataset_name: str, captions_per_image: int, min_word_freq: int) -> str:
    return f"{dataset_name}_{captions_per_image}_cap_per_img_{min_word_freq}_min_word_freq"


def clip_transform(image_size: int = 224):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ]
    )


class ChangeCaptionDataset(Dataset):
    """LEVIR-CC official_rsicc_style reader.

    Returns train items as:
        img_a, img_b, caption, caplen, changeflag

    Returns val/test items as:
        img_a, img_b, caption, caplen, changeflag, all_captions, image_id
    """

    def __init__(
        self,
        data_root: str,
        split: str,
        transform=None,
        dataset_name: str = cfg.data.dataset_name,
        captions_per_image: int = cfg.data.captions_per_image,
        min_word_freq: int = cfg.data.min_word_freq,
    ) -> None:
        split = split.lower()
        if split not in {"train", "val", "test"}:
            raise ValueError(f"split must be train/val/test, got {split}")

        self.data_root = data_root
        self.split = split
        self.cpi = captions_per_image
        self.transform = transform or clip_transform(cfg.data.image_size)

        processed_dir = os.path.join(data_root, "processed", "official_rsicc_style")
        base = official_base_name(dataset_name, captions_per_image, min_word_freq)
        split_upper = split.upper()

        self.h5_path = os.path.join(processed_dir, f"{split_upper}_IMAGES_{base}.hdf5")
        self.captions_path = os.path.join(processed_dir, f"{split_upper}_CAPTIONS_{base}.json")
        self.caplens_path = os.path.join(processed_dir, f"{split_upper}_CAPLENS_{base}.json")
        self.metadata_path = os.path.join(processed_dir, f"{split_upper}_METADATA_{base}.json")
        self.wordmap_path = os.path.join(processed_dir, f"WORDMAP_{base}.json")

        for path in [self.h5_path, self.captions_path, self.caplens_path, self.metadata_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(path)

        self.h5_file: Optional[h5py.File] = None
        with open(self.captions_path, "r", encoding="utf-8") as f:
            self.captions = json.load(f)
        with open(self.caplens_path, "r", encoding="utf-8") as f:
            self.caplens = json.load(f)
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

    def _images(self):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, "r")
        return self.h5_file["images"]

    def __len__(self) -> int:
        return len(self.captions)

    def _load_pair(self, image_idx: int):
        pair = np.asarray(self._images()[image_idx], dtype=np.uint8)
        img_a = Image.fromarray(np.transpose(pair[0], (1, 2, 0))).convert("RGB")
        img_b = Image.fromarray(np.transpose(pair[1], (1, 2, 0))).convert("RGB")
        return self.transform(img_a), self.transform(img_b)

    def __getitem__(self, idx: int):
        image_idx = idx // self.cpi
        img_a, img_b = self._load_pair(image_idx)
        caption = torch.tensor(self.captions[idx], dtype=torch.long)
        caplen = int(self.caplens[idx])
        changeflag = int(self.metadata[image_idx].get("changeflag", 1))

        if self.split == "train":
            return img_a, img_b, caption, caplen, changeflag

        start = image_idx * self.cpi
        end = start + self.cpi
        all_captions = torch.tensor(self.captions[start:end], dtype=torch.long)
        image_id = self.metadata[image_idx].get("filename", str(image_idx))
        return img_a, img_b, caption, caplen, changeflag, all_captions, image_id

    def close(self) -> None:
        if self.h5_file is not None:
            self.h5_file.close()
            self.h5_file = None


class ImageLevelChangeDataset(Dataset):
    """Image-level view used by the binary pretraining stage."""

    def __init__(self, caption_dataset: ChangeCaptionDataset) -> None:
        self.base = caption_dataset

    def __len__(self) -> int:
        return len(self.base.metadata)

    def __getitem__(self, image_idx: int):
        img_a, img_b = self.base._load_pair(image_idx)
        changeflag = int(self.base.metadata[image_idx].get("changeflag", 1))
        image_id = self.base.metadata[image_idx].get("filename", str(image_idx))
        return img_a, img_b, changeflag, image_id


class BinaryPretrainDataset11Style(Dataset):
    """Image-level binary dataset using the baseline-11 tensor resize pipeline."""

    def __init__(
        self,
        data_root: str,
        split: str,
        dataset_name: str = cfg.data.dataset_name,
        captions_per_image: int = cfg.data.captions_per_image,
        min_word_freq: int = cfg.data.min_word_freq,
        image_size: int = cfg.data.image_size,
    ) -> None:
        split = split.lower()
        if split not in {"train", "val", "test"}:
            raise ValueError(f"split must be train/val/test, got {split}")
        self.data_root = data_root
        self.split = split
        self.image_size = int(image_size)
        self.h5_file: Optional[h5py.File] = None

        processed_dir = os.path.join(data_root, "processed", "official_rsicc_style")
        base = official_base_name(dataset_name, captions_per_image, min_word_freq)
        split_upper = split.upper()
        self.h5_path = os.path.join(processed_dir, f"{split_upper}_IMAGES_{base}.hdf5")
        self.metadata_path = os.path.join(processed_dir, f"{split_upper}_METADATA_{base}.json")
        for path in [self.h5_path, self.metadata_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(path)
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

    def _images(self):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, "r")
        return self.h5_file["images"]

    def __len__(self) -> int:
        return len(self.metadata)

    @staticmethod
    def _resize_tensor(x: torch.Tensor, size: int) -> torch.Tensor:
        if x.shape[-2:] == (size, size):
            return x
        return TF.resize(x, [size, size], antialias=True)

    @staticmethod
    def _clip_normalize(x: torch.Tensor) -> torch.Tensor:
        mean = CLIP_MEAN.to(dtype=x.dtype)
        std = CLIP_STD.to(dtype=x.dtype)
        return (x - mean) / std

    def __getitem__(self, image_idx: int):
        meta = self.metadata[image_idx]
        pair = torch.from_numpy(self._images()[image_idx]).float() / 255.0
        img_a = self._clip_normalize(self._resize_tensor(pair[0], self.image_size))
        img_b = self._clip_normalize(self._resize_tensor(pair[1], self.image_size))
        changeflag = int(meta.get("changeflag", 1))
        image_id = meta.get("filename", str(image_idx))
        return img_a, img_b, torch.tensor(changeflag, dtype=torch.long), image_id

    def close(self) -> None:
        if self.h5_file is not None:
            self.h5_file.close()
            self.h5_file = None
