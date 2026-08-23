"""Public configuration preview for CEDR-Net.

Only the repository paths and interface-level fields needed to inspect the
released data, training, and evaluation code are provided here. The exact paper
configuration is withheld during peer review.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Paths:
    project_root: str = os.environ.get("PROJECT_ROOT", str(PROJECT_ROOT))
    data_root: str = os.environ.get("DATA_ROOT", "data/LEVIR-CC")
    remoteclip_b32: str = os.environ.get(
        "REMOTECLIP_CKPT", "checkpoint/remoteclip/RemoteCLIP-ViT-B-32.pt"
    )
    qwen_default: str = os.environ.get("QWEN_MODEL", "checkpoint/Qwen/Qwen3-1.7B")
    output_root: str = os.environ.get("OUTPUT_ROOT", "result")
    # Optional: only required when running the caption stage without binary pretraining.
    binary_init_ckpt: str = os.environ.get("BINARY_INIT_CKPT", "")


@dataclass
class DataConfig:
    dataset_name: str = "LEVIR_CC"
    captions_per_image: int = 5
    min_word_freq: int = 5
    max_len: int = 50
    image_size: int = 224


@dataclass
class ModelConfig:
    # Interface-level dimensions only. The complete architecture is withheld.
    clip_arch: str = "ViT-B-32"
    clip_dim: int = 512
    binary_semantic_dim: int = 256
    binary_attn_heads: int = 8
    num_visual_tokens: int = 16
    mode_dim: int = 256
    controller_mode: str = "dual"
    controller_condition_mode: str = "full"
    routing_mode: str = "confidence"
    controller_residual_scale: float = 0.15


@dataclass
class TrainConfig:
    # Preview defaults are not the complete paper experiment configuration.
    seed: int = 42
    epochs: int = 20
    batch_size: int = 8
    eval_batch_size: int = 1
    num_workers: int = 4
    lr: float = 1e-4
    binary_lr: float = 1e-5
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    log_interval: int = 20
    run_tag: str = "qwen3_plain"
    qwen_name: str = "Qwen3-1.7B"
    qwen_prompt_style: str = "plain"
    qwen_system_prompt: str = ""
    lambda_caption: float = 1.0
    lambda_binary: float = 0.2
    alpha_max: float = 1.0
    progressive_warmup_epochs: int = 5
    route_threshold: float = 0.5
    val_metric_every: int = 1
    early_stop_patience: int = 3
    final_eval_split: str = "test"
    max_answer_tokens: int = 64
    max_new_tokens: int = 48
    dtype: str = "auto"
    limit_train: int = 0
    limit_val: int = 0
    limit_val_gen: int = 0
    limit_test: int = 0


@dataclass
class BinaryPretrainConfig:
    epochs: int = 5
    batch_size: int = 24
    eval_batch_size: int = 64
    num_workers: int = 0
    lr: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip: float = 5.0
    min_epochs: int = 5
    early_stop_patience: int = 0
    early_stop_metric: str = "val_loss"
    select_metric: str = "val_loss"
    test_eval: bool = True
    deterministic: bool = False
    gamma: float = 1.8
    alpha_change: float = 0.58
    alpha_nochange: float = 0.42
    ce_warmup_epochs: int = 3
    classifier_dropout: float = 0.2


@dataclass
class TwoStageConfig:
    workflow: str = "two_stage"
    run_group: str = "cedrnet_preview"
    seeds: list[int] = field(default_factory=lambda: [42])
    conda_env: str = "cedrnet"
    gpu_id: str = "0"


@dataclass
class Config:
    paths: Paths = field(default_factory=Paths)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    binary_pretrain: BinaryPretrainConfig = field(default_factory=BinaryPretrainConfig)
    two_stage: TwoStageConfig = field(default_factory=TwoStageConfig)


cfg = Config()
