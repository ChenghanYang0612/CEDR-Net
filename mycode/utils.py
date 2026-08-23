from __future__ import annotations

import json
import os
import random
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch


PAD_TOKEN = "<pad>"
BOS_TOKEN = "<start>"
EOS_TOKEN = "<end>"
UNK_TOKEN = "<unk>"

PAD_IDX = 0
BOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_vocab(path: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    with open(path, "r", encoding="utf-8") as f:
        word2idx = json.load(f)
    idx2word = {int(v): k for k, v in word2idx.items()}
    return word2idx, idx2word


def decode_caption(ids: Iterable[int], idx2word: Dict[int, str]) -> str:
    words: List[str] = []
    for idx in ids:
        word = idx2word.get(int(idx), UNK_TOKEN)
        if word in (PAD_TOKEN, BOS_TOKEN):
            continue
        if word == EOS_TOKEN:
            break
        words.append(word)
    return " ".join(words)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / max(1, self.count)


def save_json(obj, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
