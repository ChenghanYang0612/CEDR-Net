from __future__ import annotations

import argparse
import importlib.machinery
import os
import random
import shutil
import sys
import time
import types
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from dataset import BinaryPretrainDataset11Style, ChangeCaptionDataset
from evaluate import build_eval_subset, evaluate as eval_generation, print_scores, save_results
from model import ChangeAwareHiddenControllerQwen
from prompting import DEFAULT_PROMPT, make_answer_batch, make_prompt_batch, slice_prompt_batch
from utils import AverageMeter, PAD_IDX, decode_caption, ensure_dir, load_vocab, save_json, set_seed


PROMPT_USER = DEFAULT_PROMPT

def parse_args():
    parser = argparse.ArgumentParser(description="Train baseline 18 joint binary-caption-controller with progressive binary evidence")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--dataset-name", default="LEVIR_CC")
    parser.add_argument("--captions-per-image", type=int, default=5)
    parser.add_argument("--min-word-freq", type=int, default=5)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--encoder-arch", default="ViT-B-32")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--qwen-model", required=True)
    parser.add_argument("--qwen-prompt-style", default="plain", choices=["plain", "chat", "auto"])
    parser.add_argument("--qwen-system-prompt", default="")
    parser.add_argument("--max-prompt-tokens", type=int, default=128)
    parser.add_argument("--train-stage", default="caption", choices=["binary", "caption"])
    parser.add_argument("--binary-init-ckpt", default="")
    parser.add_argument("--binary-pretrain-output-dir", default="")
    parser.add_argument("--binary-pretrain-epochs", type=int, default=5)
    parser.add_argument("--binary-pretrain-batch-size", type=int, default=24)
    parser.add_argument("--binary-pretrain-eval-batch-size", type=int, default=64)
    parser.add_argument("--binary-pretrain-num-workers", type=int, default=0)
    parser.add_argument("--binary-pretrain-lr", type=float, default=1e-4)
    parser.add_argument("--binary-pretrain-weight-decay", type=float, default=1e-4)
    parser.add_argument("--binary-pretrain-grad-clip", type=float, default=5.0)
    parser.add_argument("--binary-pretrain-min-epochs", type=int, default=5)
    parser.add_argument("--binary-pretrain-early-stop-patience", type=int, default=0)
    parser.add_argument(
        "--binary-pretrain-early-stop-metric",
        default="val_loss",
        choices=["val_loss"],
    )
    parser.add_argument(
        "--binary-pretrain-select-metric",
        default="val_loss",
        choices=["val_loss"],
    )
    parser.add_argument("--binary-pretrain-test-eval", action="store_true", help="evaluate the selected binary checkpoint on test split before caption training")
    parser.add_argument("--binary-deterministic", action="store_true", help="enable deterministic CUDA/DataLoader settings for binary pretraining")
    parser.add_argument("--encoder-lora-rank", type=int, default=8)
    parser.add_argument("--encoder-lora-alpha", type=float, default=16.0)
    parser.add_argument("--binary-semantic-dim", type=int, default=256)
    parser.add_argument("--binary-attn-heads", type=int, default=8)
    parser.add_argument("--binary-classifier-dropout", type=float, default=0.2)
    parser.add_argument("--num-visual-tokens", type=int, default=16)
    parser.add_argument("--mode-dim", type=int, default=256)
    parser.add_argument("--controller-mode", default="dual", choices=["none", "nochange_only", "change_only", "shared", "dual"])
    parser.add_argument("--controller-condition-mode", default="full", choices=["full", "route_only"])
    parser.add_argument("--routing-mode", default="confidence", choices=["hard", "confidence"])
    parser.add_argument("--disable-binary-cross-attn", action="store_true")
    parser.add_argument("--controller-residual-scale", type=float, default=0.15)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-name", default="change_caption_main")
    parser.add_argument("--binary-gamma", type=float, default=1.8)
    parser.add_argument("--binary-alpha-change", type=float, default=0.58)
    parser.add_argument("--binary-alpha-nochange", type=float, default=0.42)
    parser.add_argument("--binary-ce-warmup-epochs", type=int, default=3, help="use cross entropy for the first N binary pretrain epochs, then switch to focal loss")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--binary-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--lambda-caption", type=float, default=1.0)
    parser.add_argument("--lambda-binary", type=float, default=0.2)
    parser.add_argument("--alpha-max", type=float, default=1.0)
    parser.add_argument("--progressive-warmup-epochs", type=int, default=5)
    parser.add_argument("--route-threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-val", type=int, default=0)
    parser.add_argument("--limit-val-gen", type=int, default=0, help="image-level limit for val generation metrics; 0 uses --limit-val/full val")
    parser.add_argument("--limit-test", type=int, default=0)
    parser.add_argument("--val-metric-every", type=int, default=1)
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--disable-val-gen", action="store_true")
    parser.add_argument("--disable-final-eval", action="store_true")
    parser.add_argument("--final-eval-split", default="test", choices=["val", "test"])
    parser.add_argument("--max-answer-tokens", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--transformers-path", default="")
    parser.add_argument("--allow-remote-model", action="store_true")
    return parser.parse_args()


def maybe_use_transformers_path(path: str) -> None:
    if not path:
        return
    if not os.path.isdir(path):
        raise FileNotFoundError(path)
    sys.path.insert(0, path)
    if "deepspeed" not in sys.modules:
        fake_deepspeed = types.ModuleType("deepspeed")
        fake_deepspeed.__spec__ = importlib.machinery.ModuleSpec("deepspeed", loader=None)
        fake_deepspeed.zero = types.SimpleNamespace()
        sys.modules["deepspeed"] = fake_deepspeed
    print(f"using local transformers path: {path}")


def resolve_dtype(name: str, device: torch.device):
    if name == "float32" or device.type != "cuda":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def caption_ids_to_text(caption_ids: Iterable[int], idx2word: dict[int, str]) -> str:
    text = decode_caption(caption_ids, idx2word)
    return text.strip() or "There is no obvious change."


def make_qwen_batch(tokenizer, answers: list[str], device: torch.device, args):
    prompt_input_ids = make_prompt_batch(
        tokenizer,
        len(answers),
        device,
        prompt_style=args.qwen_prompt_style,
        instruction=PROMPT_USER,
        system_prompt=args.qwen_system_prompt,
        max_prompt_tokens=args.max_prompt_tokens,
    )
    answer_input_ids, labels = make_answer_batch(tokenizer, answers, device, args.max_answer_tokens)
    return prompt_input_ids, answer_input_ids, labels


def build_dataset(args, split: str):
    return ChangeCaptionDataset(
        args.data_root,
        split,
        dataset_name=args.dataset_name,
        captions_per_image=args.captions_per_image,
        min_word_freq=args.min_word_freq,
    )


def build_binary_pretrain_dataset(args, split: str):
    return BinaryPretrainDataset11Style(
        args.data_root,
        split,
        dataset_name=args.dataset_name,
        captions_per_image=args.captions_per_image,
        min_word_freq=args.min_word_freq,
        image_size=args.image_size,
    )


def make_loader(dataset, batch_size: int, shuffle: bool, num_workers: int, generator=None, worker_init_fn=None):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        generator=generator,
        worker_init_fn=worker_init_fn,
    )


def make_generator(seed: int):
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def make_worker_init_fn(seed: int):
    def seed_worker(worker_id: int) -> None:
        worker_seed = (int(seed) + int(worker_id)) % 2**32
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return seed_worker


def configure_binary_determinism(enabled: bool, seed: int) -> None:
    if not enabled:
        return
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("PYTHONHASHSEED", str(int(seed)))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True, warn_only=True)
    print(
        "[deterministic] binary pretraining enabled "
        f"seed={seed} CUBLAS_WORKSPACE_CONFIG={os.environ.get('CUBLAS_WORKSPACE_CONFIG')} "
        f"PYTHONHASHSEED={os.environ.get('PYTHONHASHSEED')} warn_only=True"
    )


def maybe_limit_caption_level(dataset, limit: int):
    if limit and limit > 0:
        return Subset(dataset, list(range(min(limit, len(dataset)))))
    return dataset


def maybe_limit_image_level(dataset, limit: int):
    if limit and limit > 0:
        return Subset(dataset, list(range(min(limit, len(dataset)))))
    return dataset


def optim_params(model: torch.nn.Module):
    params = []
    for name, param in model.named_parameters():
        if name.startswith("llm."):
            continue
        if "clip_model" in name and "lora_A" not in name and "lora_B" not in name:
            continue
        params.append(param)
    return params


def optimizer_param_groups(model: torch.nn.Module, args):
    binary_params = []
    other_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("llm."):
            continue
        if "clip_model" in name and "lora_A" not in name and "lora_B" not in name:
            continue
        if name.startswith("binary_branch."):
            binary_params.append(param)
        else:
            other_params.append(param)
    groups = []
    if other_params:
        groups.append({"params": other_params, "lr": args.lr})
    if binary_params:
        groups.append({"params": binary_params, "lr": args.binary_lr})
    return groups


def binary_optim_params(model: torch.nn.Module):
    params = []
    for name, param in model.named_parameters():
        if not name.startswith("binary_branch."):
            continue
        if "clip_model" in name and "lora_A" not in name and "lora_B" not in name:
            continue
        params.append(param)
    return params


def saveable_state_dict(model: torch.nn.Module):
    state = {}
    for name, tensor in model.state_dict().items():
        if name.startswith("llm."):
            continue
        if "clip_model" in name and "lora_A" not in name and "lora_B" not in name:
            continue
        state[name] = tensor.detach().cpu()
    return state


def saveable_binary_state_dict(binary_branch: torch.nn.Module):
    state = {}
    for name, tensor in binary_branch.state_dict().items():
        if name.startswith("encoder.clip_model") and "lora_A" not in name and "lora_B" not in name:
            continue
        state[name] = tensor.detach().cpu()
    return state


def count_params(model: torch.nn.Module):
    total = sum(p.numel() for p in model.parameters())
    optim_count = sum(p.numel() for p in optim_params(model))
    llm_trainable = sum(p.numel() for p in model.llm.parameters() if p.requires_grad)
    binary_trainable = sum(p.numel() for p in model.binary_branch.parameters() if p.requires_grad)
    return {
        "total": total,
        "optim": optim_count,
        "binary_trainable": binary_trainable,
        "llm_trainable": llm_trainable,
    }


def total_loss(out, changeflags: torch.Tensor, args):
    loss_binary = focal_loss(
        out["binary_logits"],
        changeflags,
        args.binary_gamma,
        args.binary_alpha_nochange,
        args.binary_alpha_change,
    )
    loss = args.lambda_caption * out["loss_caption"] + args.lambda_binary * loss_binary
    return loss, loss_binary


def progressive_alpha(epoch: int, args) -> float:
    if args.alpha_max <= 0:
        return 0.0
    warmup = max(1, int(args.progressive_warmup_epochs))
    progress = min(max((epoch - 1) / float(warmup), 0.0), 1.0)
    return float(args.alpha_max) * 0.5 * (1.0 - torch.cos(torch.tensor(progress * torch.pi)).item())


def focal_loss(logits: torch.Tensor, labels: torch.Tensor, gamma: float, alpha_nochange: float, alpha_change: float) -> torch.Tensor:
    ce = torch.nn.functional.cross_entropy(logits, labels.long(), reduction="none")
    pt = torch.exp(-ce)
    alpha = torch.where(
        labels.long().eq(1),
        torch.full_like(pt, float(alpha_change)),
        torch.full_like(pt, float(alpha_nochange)),
    )
    return (alpha * (1.0 - pt).pow(float(gamma)) * ce).mean()


def binary_pretrain_loss(logits: torch.Tensor, labels: torch.Tensor, args, epoch: int | None = None) -> tuple[torch.Tensor, str]:
    if epoch is not None and int(epoch) <= max(0, int(args.binary_ce_warmup_epochs)):
        return torch.nn.functional.cross_entropy(logits, labels.long()), "ce"
    return focal_loss(logits, labels, args.binary_gamma, args.binary_alpha_nochange, args.binary_alpha_change), "focal"


def binary_metrics_from_counts(tp: int, fp: int, tn: int, fn: int) -> dict:
    total = tp + fp + tn + fn
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    nochange_recall = tn / max(1, tn + fp)
    balanced_accuracy = 0.5 * (recall + nochange_recall)
    return {
        "accuracy": (tp + tn) / max(1, total),
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "nochange_recall": nochange_recall,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def binary_metrics_at_threshold(labels: torch.Tensor, probs: torch.Tensor, threshold: float) -> dict:
    labels = labels.long()
    pred = probs.ge(float(threshold)).long()
    tp = int(((pred == 1) & (labels == 1)).sum().item())
    fp = int(((pred == 1) & (labels == 0)).sum().item())
    tn = int(((pred == 0) & (labels == 0)).sum().item())
    fn = int(((pred == 0) & (labels == 1)).sum().item())
    metrics = binary_metrics_from_counts(tp=tp, fp=fp, tn=tn, fn=fn)
    metrics["threshold"] = float(threshold)
    return metrics


def binary_threshold_sweep(labels: list[int], probs: list[float]) -> list[dict]:
    if not labels:
        return []
    label_tensor = torch.tensor(labels, dtype=torch.long)
    prob_tensor = torch.tensor(probs, dtype=torch.float32)
    rows = []
    for idx in range(1001):
        threshold = idx / 1000.0
        rows.append(binary_metrics_at_threshold(label_tensor, prob_tensor, threshold))
    return rows


def best_recall_at_precision(rows: list[dict], min_precision: float = 0.9) -> dict:
    valid = [row for row in rows if row["precision"] >= min_precision]
    if not valid:
        return {"recall": 0.0, "precision": 0.0, "threshold": 1.0, "f1": 0.0}
    return max(valid, key=lambda row: (row["recall"], row["precision"], row["f1"]))


def binary_score(metrics: dict, metric_name: str) -> float:
    if metric_name == "val_loss":
        return -float(metrics.get("loss", float("inf")))
    if metric_name == "balanced_accuracy":
        return float(metrics.get("balanced_accuracy", 0.0))
    if metric_name == "f1":
        return float(metrics.get("f1", 0.0))
    if metric_name == "guarded_f1":
        precision = float(metrics.get("precision", 0.0))
        nochange_recall = float(metrics.get("nochange_recall", 0.0))
        f1 = float(metrics.get("f1", 0.0))
        if precision >= 0.92 and nochange_recall >= 0.92:
            return f1
        # Keep invalid epochs comparable with each other, but below any valid epoch.
        return f1 * 1e-3
    if metric_name == "change_recall":
        return float(metrics.get("recall", 0.0))
    if metric_name == "change_precision":
        return float(metrics.get("precision", 0.0))
    if metric_name == "recall_at_precision90":
        return float(metrics.get("recall_at_precision90", 0.0))
    raise ValueError(f"unknown binary metric: {metric_name}")


def update_binary_counts(logits: torch.Tensor, labels: torch.Tensor, counts: dict[str, int]) -> None:
    pred = logits.argmax(dim=-1)
    labels = labels.long()
    counts["tp"] += int(((pred == 1) & (labels == 1)).sum().item())
    counts["fp"] += int(((pred == 1) & (labels == 0)).sum().item())
    counts["tn"] += int(((pred == 0) & (labels == 0)).sum().item())
    counts["fn"] += int(((pred == 0) & (labels == 1)).sum().item())


def train_binary_one_epoch(model, loader, optimizer, device, epoch: int, args):
    model.set_stage(0)
    model.train()
    meter = AverageMeter()
    counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    start = time.time()
    params = [p for p in binary_optim_params(model) if p.requires_grad]
    for step, batch in enumerate(loader, start=1):
        img_a, img_b, changeflags, _ = batch
        img_a = img_a.to(device, non_blocking=True)
        img_b = img_b.to(device, non_blocking=True)
        changeflags = changeflags.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        out = model.forward_binary(img_a, img_b)
        loss, loss_type = binary_pretrain_loss(out["binary_logits"], changeflags, args, epoch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, args.binary_pretrain_grad_clip)
        optimizer.step()
        n = img_a.size(0)
        meter.update(float(loss.item()), n)
        update_binary_counts(out["binary_logits"].detach(), changeflags, counts)
        if step % args.log_interval == 0 or step == 1:
            metrics = binary_metrics_from_counts(**counts)
            print(
                f"binary epoch {epoch:03d} step {step:04d}/{len(loader)} "
                f"{loss_type}_loss {meter.avg:.4f} acc {metrics['accuracy']:.4f} "
                f"P {metrics['precision']:.4f} R {metrics['recall']:.4f} F1 {metrics['f1']:.4f} "
                f"time {time.time() - start:.1f}s"
            )
    metrics = binary_metrics_from_counts(**counts)
    metrics["loss"] = meter.avg
    metrics["loss_type"] = loss_type
    return metrics


@torch.no_grad()
def eval_binary(model, loader, device, args, tag: str = "VAL_BINARY", epoch: int | None = None):
    model.set_stage(0)
    model.eval()
    meter = AverageMeter()
    counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    all_labels: list[int] = []
    all_probs: list[float] = []
    for batch in loader:
        img_a, img_b, changeflags, _ = batch
        img_a = img_a.to(device, non_blocking=True)
        img_b = img_b.to(device, non_blocking=True)
        changeflags = changeflags.to(device, non_blocking=True)
        out = model.forward_binary(img_a, img_b)
        loss, loss_type = binary_pretrain_loss(out["binary_logits"], changeflags, args, epoch)
        meter.update(float(loss.item()), img_a.size(0))
        update_binary_counts(out["binary_logits"], changeflags, counts)
        all_labels.extend(changeflags.detach().cpu().long().tolist())
        all_probs.extend(out["binary_logits"].softmax(dim=-1)[:, 1].detach().cpu().float().tolist())
    metrics = binary_metrics_from_counts(**counts)
    threshold_sweep = binary_threshold_sweep(all_labels, all_probs)
    recall_p90 = best_recall_at_precision(threshold_sweep, min_precision=0.9)
    metrics["loss"] = meter.avg
    metrics["loss_type"] = loss_type
    metrics["recall_at_precision90"] = float(recall_p90["recall"])
    metrics["precision_at_recall_at_precision90"] = float(recall_p90["precision"])
    metrics["threshold_at_precision90"] = float(recall_p90["threshold"])
    metrics["threshold_sweep"] = threshold_sweep
    print(
        f"[{tag}] loss {metrics['loss']:.4f} acc {metrics['accuracy']:.4f} "
        f"P {metrics['precision']:.4f} R {metrics['recall']:.4f} F1 {metrics['f1']:.4f} "
        f"R@P90 {metrics['recall_at_precision90']:.4f}@thr{metrics['threshold_at_precision90']:.3f} "
        f"nochangeR {metrics['nochange_recall']:.4f} TP {metrics['tp']} FP {metrics['fp']} TN {metrics['tn']} FN {metrics['fn']}"
    )
    return metrics


def compact_binary_metrics(metrics: dict) -> dict:
    return {key: value for key, value in metrics.items() if key != "threshold_sweep"}


def save_binary_checkpoint(path: str, model, optimizer, epoch: int, metrics: dict, args) -> None:
    ensure_dir(os.path.dirname(path))
    torch.save(
        {
            "epoch": epoch,
            "model": saveable_binary_state_dict(model.binary_branch),
            "optimizer": optimizer.state_dict(),
            "metrics": compact_binary_metrics(metrics),
            "config": {
                "encoder_arch": args.encoder_arch,
                "encoder_checkpoint": args.checkpoint,
                "encoder_lora_rank": args.encoder_lora_rank,
                "encoder_lora_alpha": args.encoder_lora_alpha,
                "binary_semantic_dim": args.binary_semantic_dim,
                "binary_attn_heads": args.binary_attn_heads,
                "binary_loss": "ce_warmup_then_focal" if args.binary_ce_warmup_epochs > 0 else "focal",
                "binary_gamma": args.binary_gamma,
                "binary_alpha_change": args.binary_alpha_change,
                "binary_alpha_nochange": args.binary_alpha_nochange,
                "binary_ce_warmup_epochs": args.binary_ce_warmup_epochs,
                "binary_classifier_dropout": args.binary_classifier_dropout,
                "binary_pretrain_lr": args.binary_pretrain_lr,
                "binary_pretrain_weight_decay": args.binary_pretrain_weight_decay,
                "binary_pretrain_grad_clip": args.binary_pretrain_grad_clip,
                "binary_pretrain_select_metric": args.binary_pretrain_select_metric,
                "binary_pretrain_early_stop_metric": args.binary_pretrain_early_stop_metric,
                "binary_deterministic": bool(args.binary_deterministic),
                "dataset_name": args.dataset_name,
                "data_root": args.data_root,
            },
        },
        path,
    )


def selected_binary_checkpoint_path(binary_dir: str, metric_name: str) -> str:
    filename = {
        "val_loss": "best_by_val_loss.pth",
        "balanced_accuracy": "best_by_balanced_accuracy.pth",
        "f1": "best_by_f1.pth",
        "guarded_f1": "best_by_guarded_f1.pth",
        "change_recall": "best_by_change_recall.pth",
        "change_precision": "best_by_change_precision.pth",
        "recall_at_precision90": "best_by_recall_at_precision90.pth",
    }[metric_name]
    return os.path.join(binary_dir, filename)


def run_binary_pretrain(model, train_base, val_base, test_base, device, args, output_dir: str) -> str:
    if args.binary_deterministic:
        set_seed(args.seed)
        configure_binary_determinism(True, args.seed)
        print(f"[Binary Pretrain] reset RNG for deterministic binary stage seed={args.seed}")
    binary_dir = args.binary_pretrain_output_dir or output_dir
    ensure_dir(binary_dir)
    train_image_set = maybe_limit_image_level(build_binary_pretrain_dataset(args, "train"), args.limit_train)
    val_image_set = maybe_limit_image_level(build_binary_pretrain_dataset(args, "val"), args.limit_val)
    binary_generator = make_generator(args.seed) if args.binary_deterministic else None
    binary_worker_init = make_worker_init_fn(args.seed) if args.binary_deterministic else None
    train_loader = make_loader(
        train_image_set,
        args.binary_pretrain_batch_size,
        True,
        args.binary_pretrain_num_workers,
        generator=binary_generator,
        worker_init_fn=binary_worker_init,
    )
    val_loader = make_loader(
        val_image_set,
        args.binary_pretrain_eval_batch_size,
        False,
        args.binary_pretrain_num_workers,
        generator=make_generator(args.seed + 100_000) if args.binary_deterministic else None,
        worker_init_fn=binary_worker_init,
    )
    model.set_stage(0)
    optimizer = torch.optim.AdamW(
        binary_optim_params(model),
        lr=args.binary_pretrain_lr,
        weight_decay=args.binary_pretrain_weight_decay,
    )

    best_score = float("-inf")
    best_early_stop_score = float("-inf")
    epochs_since_improvement = 0
    history = []
    print(
        f"\n[Binary Pretrain] output_dir={binary_dir} epochs={args.binary_pretrain_epochs} "
        f"bs={args.binary_pretrain_batch_size} eval_bs={args.binary_pretrain_eval_batch_size} "
        f"lr={args.binary_pretrain_lr} wd={args.binary_pretrain_weight_decay} grad_clip={args.binary_pretrain_grad_clip} "
        f"ce_warmup_epochs={args.binary_ce_warmup_epochs} "
        f"select={args.binary_pretrain_select_metric} early_stop={args.binary_pretrain_early_stop_metric} "
        f"deterministic={args.binary_deterministic} "
        f"data_pipeline=baseline11_tensor_resize"
    )

    for epoch in range(1, args.binary_pretrain_epochs + 1):
        train_metrics = train_binary_one_epoch(model, train_loader, optimizer, device, epoch, args)
        val_metrics = eval_binary(model, val_loader, device, args, tag="VAL_BINARY", epoch=epoch)
        score = binary_score(val_metrics, "val_loss")
        if score > best_score + 1e-12:
            best_score = score
            save_binary_checkpoint(selected_binary_checkpoint_path(binary_dir, "val_loss"), model, optimizer, epoch, val_metrics, args)
            print(f"  * binary new best val_loss={val_metrics['loss']:.4f} at epoch={epoch}")

        early_score = binary_score(val_metrics, args.binary_pretrain_early_stop_metric)
        if early_score > best_early_stop_score + 1e-12:
            best_early_stop_score = early_score
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        row = {
            "epoch": epoch,
            "train": compact_binary_metrics(train_metrics),
            "val": compact_binary_metrics(val_metrics),
            "best_score": best_score,
            "epochs_since_improvement": epochs_since_improvement,
        }
        history.append(row)
        save_json(history, os.path.join(binary_dir, "train_log.json"))
        print(
            f"binary epoch {epoch:03d} train_loss {train_metrics['loss']:.4f} val_loss {val_metrics['loss']:.4f} "
            f"val_P {val_metrics['precision']:.4f} val_R {val_metrics['recall']:.4f} "
            f"val_F1 {val_metrics['f1']:.4f} val_R@P90 {val_metrics['recall_at_precision90']:.4f}"
        )
        if (
            args.binary_pretrain_early_stop_patience > 0
            and epoch >= args.binary_pretrain_min_epochs
            and epochs_since_improvement >= args.binary_pretrain_early_stop_patience
        ):
            print(
                f"[Binary Pretrain Early Stop] no {args.binary_pretrain_early_stop_metric} improvement "
                f"for {epochs_since_improvement} epochs; patience={args.binary_pretrain_early_stop_patience}."
            )
            break

    selected = selected_binary_checkpoint_path(binary_dir, args.binary_pretrain_select_metric)
    if not os.path.exists(selected):
        raise FileNotFoundError(f"Selected binary checkpoint was not saved: {selected}")
    model.binary_branch.load_binary_checkpoint(selected)
    print(f"[Binary Pretrain] selected checkpoint: {selected}")
    if args.binary_pretrain_test_eval:
        test_image_set = maybe_limit_image_level(build_binary_pretrain_dataset(args, args.final_eval_split), args.limit_test)
        test_loader = make_loader(
            test_image_set,
            args.binary_pretrain_eval_batch_size,
            False,
            args.binary_pretrain_num_workers,
            generator=make_generator(args.seed + 200_000) if args.binary_deterministic else None,
            worker_init_fn=binary_worker_init,
        )
        test_metrics = eval_binary(model, test_loader, device, args, tag="TEST_BINARY_SELECTED")
        compact = compact_binary_metrics(test_metrics)
        save_json(compact, os.path.join(binary_dir, "selected_test_metrics.json"))
        print(
            f"[Binary Pretrain TEST selected] "
            f"precision={test_metrics['precision']:.4f} "
            f"f1={test_metrics['f1']:.4f} "
            f"accuracy={test_metrics['accuracy']:.4f} "
            f"recall={test_metrics['recall']:.4f}"
        )
        model.binary_branch.load_binary_checkpoint(selected)
    return selected


def update_aux_meters(meters, out, n: int):
    meters["p_change"].update(float(out["p_change"].detach().mean().item()), n)
    meters["route_change"].update(float(out["route_change"].detach().mean().item()), n)


def train_one_epoch(model, loader, optimizer, device, epoch: int, idx2word, args):
    model.set_stage(2)
    model.train()
    alpha = progressive_alpha(epoch, args)
    meters = {key: AverageMeter() for key in ["loss", "binary", "caption", "residual", "p_change", "route_change", "alpha"]}
    start = time.time()
    params = [p for p in optim_params(model) if p.requires_grad]

    for step, batch in enumerate(loader, start=1):
        img_a, img_b, captions, _, changeflags = batch
        img_a = img_a.to(device, non_blocking=True)
        img_b = img_b.to(device, non_blocking=True)
        captions = captions.to(device, non_blocking=True)
        changeflags = changeflags.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        answers = [caption_ids_to_text(row.tolist(), idx2word) for row in captions]
        prompt_ids, answer_ids, qwen_labels = make_qwen_batch(model.tokenizer, answers, device, args)
        out = model(
            img_a,
            img_b,
            prompt_ids,
            answer_ids,
            qwen_labels,
            changeflags,
            route_threshold=args.route_threshold,
            evidence_alpha=alpha,
        )
        loss, loss_binary = total_loss(out, changeflags, args)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
        optimizer.step()

        n = img_a.size(0)
        meters["loss"].update(float(loss.item()), n)
        meters["binary"].update(float(loss_binary.item()), n)
        meters["caption"].update(float(out["loss_caption"].item()), n)
        meters["residual"].update(float(out["loss_residual"].item()), n)
        meters["alpha"].update(alpha, n)
        update_aux_meters(meters, out, n)
        if step % args.log_interval == 0 or step == 1:
            elapsed = time.time() - start
            print(
                f"epoch {epoch:03d} step {step:04d}/{len(loader)} "
                f"loss {meters['loss'].avg:.4f} cap {meters['caption'].avg:.4f} bin {meters['binary'].avg:.4f} "
                f"res {meters['residual'].avg:.5f} pchg {meters['p_change'].avg:.3f} "
                f"routeC {meters['route_change'].avg:.3f} alpha {meters['alpha'].avg:.3f} time {elapsed:.1f}s"
            )
    return {key: meter.avg for key, meter in meters.items()}


@torch.no_grad()
def validate_loss(model, loader, device, idx2word, args, epoch: int):
    model.set_stage(2)
    model.eval()
    alpha = progressive_alpha(epoch, args)
    meters = {key: AverageMeter() for key in ["loss", "binary", "caption", "residual", "p_change", "route_change", "alpha"]}
    sample = None
    for batch in loader:
        img_a, img_b, captions, _, changeflags, _, image_ids = batch
        img_a = img_a.to(device, non_blocking=True)
        img_b = img_b.to(device, non_blocking=True)
        captions = captions.to(device, non_blocking=True)
        changeflags = changeflags.to(device, non_blocking=True)
        answers = [caption_ids_to_text(row.tolist(), idx2word) for row in captions]
        prompt_ids, answer_ids, qwen_labels = make_qwen_batch(model.tokenizer, answers, device, args)
        out = model(
            img_a,
            img_b,
            prompt_ids,
            answer_ids,
            qwen_labels,
            changeflags,
            route_threshold=args.route_threshold,
            evidence_alpha=alpha,
        )
        loss, loss_binary = total_loss(out, changeflags, args)
        n = img_a.size(0)
        meters["loss"].update(float(loss.item()), n)
        meters["binary"].update(float(loss_binary.item()), n)
        meters["caption"].update(float(out["loss_caption"].item()), n)
        meters["residual"].update(float(out["loss_residual"].item()), n)
        meters["alpha"].update(alpha, n)
        update_aux_meters(meters, out, n)
        if sample is None:
            pred, aux = model.generate(
                img_a[:1],
                img_b[:1],
                slice_prompt_batch(prompt_ids, slice(0, 1)),
                max_new_tokens=args.max_new_tokens,
                threshold=args.route_threshold,
            )
            sample = {
                "image_id": image_ids[0],
                "prediction": pred[0].strip(),
                "reference": answers[0],
                "p_change": float(aux["p_change"][0].detach().cpu()),
                "route_change": float(aux["route_change"][0].detach().cpu()),
            }
    return {key: meter.avg for key, meter in meters.items()}, sample


def checkpoint_state(model, optimizer, epoch, train_losses, val_losses, best_val_sm, best_val_loss, args):
    return {
        "epoch": epoch,
        "train_loss": train_losses["loss"],
        "val_loss": val_losses["loss"],
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_sm": best_val_sm,
        "best_val_loss": best_val_loss,
        "model": saveable_state_dict(model),
        "optimizer": optimizer.state_dict(),
        "config": {
            "qwen_model": args.qwen_model,
            "qwen_prompt_style": args.qwen_prompt_style,
            "qwen_system_prompt": args.qwen_system_prompt,
            "max_prompt_tokens": args.max_prompt_tokens,
            "encoder_arch": args.encoder_arch,
            "encoder_checkpoint": args.checkpoint,
            "encoder_lora_rank": args.encoder_lora_rank,
            "encoder_lora_alpha": args.encoder_lora_alpha,
            "train_stage": args.train_stage,
            "binary_init_ckpt": args.binary_init_ckpt,
            "binary_pretrain_select_metric": args.binary_pretrain_select_metric,
            "binary_semantic_dim": args.binary_semantic_dim,
            "binary_attn_heads": args.binary_attn_heads,
            "binary_loss": "focal",
            "binary_gamma": args.binary_gamma,
            "binary_alpha_change": args.binary_alpha_change,
            "binary_alpha_nochange": args.binary_alpha_nochange,
            "num_visual_tokens": args.num_visual_tokens,
            "mode_dim": args.mode_dim,
            "controller_mode": args.controller_mode,
            "controller_condition_mode": args.controller_condition_mode,
            "routing_mode": args.routing_mode,
            "disable_binary_cross_attn": args.disable_binary_cross_attn,
            "controller_residual_scale": args.controller_residual_scale,
            "dataset_name": args.dataset_name,
            "data_root": args.data_root,
            "prompt": PROMPT_USER,
            "route_threshold": args.route_threshold,
            "limit_val_gen": args.limit_val_gen,
            "lambda_caption": args.lambda_caption,
            "lambda_binary": args.lambda_binary,
            "binary_lr": args.binary_lr,
            "alpha_max": args.alpha_max,
            "progressive_warmup_epochs": args.progressive_warmup_epochs,
        },
    }


def save_checkpoint(path, state):
    ensure_dir(os.path.dirname(path))
    torch.save(state, path)


def maybe_update_top_checkpoints(output_dir, top_checkpoints, score, epoch, state, top_k: int = 3):
    if score is None:
        return top_checkpoints
    if len(top_checkpoints) >= top_k:
        worst = min(top_checkpoints, key=lambda item: (item["score"], -item["epoch"]))
        if score <= worst["score"] + 1e-12:
            return top_checkpoints

    candidate_path = os.path.join(output_dir, f"BEST_candidate_epoch{epoch:03d}.pth")
    save_checkpoint(candidate_path, state)
    entries = top_checkpoints + [{"score": float(score), "epoch": int(epoch), "path": candidate_path}]
    entries.sort(key=lambda item: (-item["score"], item["epoch"]))
    selected = entries[:top_k]
    selected_paths = {item["path"] for item in selected}

    for item in entries[top_k:]:
        path = item["path"]
        if path not in selected_paths and os.path.basename(path).startswith("BEST_candidate_") and os.path.exists(path):
            os.remove(path)

    temp_paths = []
    for rank, item in enumerate(selected, start=1):
        temp_path = os.path.join(output_dir, f".BEST_top{rank}.tmp.pth")
        shutil.copy2(item["path"], temp_path)
        temp_paths.append(temp_path)

    for rank, temp_path in enumerate(temp_paths, start=1):
        target_path = os.path.join(output_dir, f"BEST_top{rank}.pth")
        os.replace(temp_path, target_path)
        selected[rank - 1]["path"] = target_path

    for rank in range(len(selected) + 1, top_k + 1):
        stale_path = os.path.join(output_dir, f"BEST_top{rank}.pth")
        if os.path.exists(stale_path):
            os.remove(stale_path)

    best_path = os.path.join(output_dir, "BEST.pth")
    if selected:
        shutil.copy2(selected[0]["path"], best_path)

    for candidate in Path(output_dir).glob("BEST_candidate_epoch*.pth"):
        if candidate.exists():
            candidate.unlink()

    metadata = [
        {"rank": rank, "epoch": item["epoch"], "score": item["score"], "path": item["path"]}
        for rank, item in enumerate(selected, start=1)
    ]
    save_json(metadata, os.path.join(output_dir, "best_checkpoints.json"))
    print(
        "  top val checkpoints: "
        + ", ".join(f"top{item['rank']}=epoch{item['epoch']} score={item['score']:.4f}" for item in metadata)
    )
    return selected


@torch.no_grad()
def run_generation_eval(model, dataset, idx2word, device, batch_size, num_workers, limit, max_new_tokens, route_threshold, tag: str, args):
    eval_dataset = build_eval_subset(dataset, limit)
    loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    result = eval_generation(
        model,
        loader,
        idx2word,
        device,
        max_new_tokens=max_new_tokens,
        route_threshold=route_threshold,
        prompt_style=args.qwen_prompt_style,
        system_prompt=args.qwen_system_prompt,
        max_prompt_tokens=args.max_prompt_tokens,
    )
    print_scores(f"{tag} ALL", result["metrics_all"], n=result["n_all"])
    if result["metrics_change"]:
        print_scores(f"{tag} CHANGE", result["metrics_change"], n=result["n_change"])
    if result["metrics_nochange"]:
        print_scores(f"{tag} NOCHANGE", result["metrics_nochange"], n=result["n_nochange"])
    print(f"[{tag} BINARY] {result['binary']}")
    print(f"[{tag} CONTROL] {result['control']}")
    return result


def main():
    args = parse_args()
    os.environ.setdefault("MKL_SERVICE_FORCE_INTEL", "1")
    os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
    maybe_use_transformers_path(args.transformers_path)
    set_seed(args.seed)
    if args.binary_deterministic and args.train_stage != "binary":
        print("[deterministic] --binary-deterministic ignored because --train-stage is not binary")
        args.binary_deterministic = False
    configure_binary_determinism(args.binary_deterministic, args.seed)
    local_files_only = not args.allow_remote_model
    if local_files_only and not Path(args.qwen_model).exists():
        raise FileNotFoundError(args.qwen_model)
    if args.train_stage == "caption" and not Path(args.binary_init_ckpt).exists():
        raise FileNotFoundError(args.binary_init_ckpt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = resolve_dtype(args.dtype, device)
    output_dir = args.output_dir
    ensure_dir(output_dir)

    train_base = build_dataset(args, "train")
    val_base = build_dataset(args, "val")
    test_base = build_dataset(args, args.final_eval_split)
    train_set = maybe_limit_caption_level(train_base, args.limit_train)
    val_loss_set = build_eval_subset(val_base, args.limit_val)
    word2idx, idx2word = load_vocab(train_base.wordmap_path)
    train_loader = make_loader(train_set, args.batch_size, True, args.num_workers)
    val_loss_loader = make_loader(val_loss_set, args.eval_batch_size, False, args.num_workers)

    model = ChangeAwareHiddenControllerQwen(
        qwen_model_path=args.qwen_model,
        vocab_size=len(word2idx),
        pad_idx=PAD_IDX,
        encoder_arch=args.encoder_arch,
        checkpoint_path=args.checkpoint,
        encoder_lora_rank=args.encoder_lora_rank,
        encoder_lora_alpha=args.encoder_lora_alpha,
        binary_init_ckpt=args.binary_init_ckpt if args.train_stage == "caption" else "",
        binary_semantic_dim=args.binary_semantic_dim,
        binary_attn_heads=args.binary_attn_heads,
        binary_classifier_dropout=args.binary_classifier_dropout,
        num_visual_tokens=args.num_visual_tokens,
        mode_dim=args.mode_dim,
        controller_mode=args.controller_mode,
        controller_condition_mode=args.controller_condition_mode,
        routing_mode=args.routing_mode,
        controller_residual_scale=args.controller_residual_scale,
        disable_binary_cross_attn=args.disable_binary_cross_attn,
        torch_dtype=dtype,
        local_files_only=local_files_only,
    ).to(device)
    if args.train_stage == "binary":
        selected_binary_ckpt = run_binary_pretrain(model, train_base, val_base, test_base, device, args, output_dir)
        print(f"[Binary Pretrain Only] done. selected_binary_ckpt={selected_binary_ckpt}")
        return
    selected_binary_ckpt = ""

    model.set_stage(2)
    params_stage2 = count_params(model)

    print(f"qwen={args.qwen_model}")
    print(f"qwen_prompt_style={args.qwen_prompt_style} max_prompt_tokens={args.max_prompt_tokens}")
    print(f"encoder={args.encoder_arch} checkpoint={args.checkpoint} lora_rank={args.encoder_lora_rank}")
    print(
        f"binary=semantic_only_focal_style dim={args.binary_semantic_dim} "
        f"train_stage={args.train_stage} init_ckpt={args.binary_init_ckpt} "
        f"selected_binary_ckpt={selected_binary_ckpt or ''} "
        f"loss=focal(gamma={args.binary_gamma}, alpha_change={args.binary_alpha_change}, alpha_nochange={args.binary_alpha_nochange})"
    )
    print(
        f"controller={args.controller_mode} condition={args.controller_condition_mode} "
        f"routing={args.routing_mode} binary_cross_attn={not args.disable_binary_cross_attn}"
    )
    print(f"optim lr={args.lr} binary_lr={args.binary_lr}")
    print(f"progressive_evidence_gradient alpha_max={args.alpha_max} warmup_epochs={args.progressive_warmup_epochs}")
    print(f"dataset={args.dataset_name} data_root={args.data_root}")
    print(f"device={device} dtype={dtype} train_caps={len(train_set)} val_images={len(val_loss_set)}")
    print(f"output_dir={output_dir}")
    print(
        f"params total={params_stage2['total']:,} optim={params_stage2['optim']:,} "
        f"binary_trainable={params_stage2['binary_trainable']:,} "
        f"llm_trainable={params_stage2['llm_trainable']:,}"
    )

    model.set_stage(2)
    optimizer = torch.optim.AdamW(optimizer_param_groups(model, args), weight_decay=args.weight_decay)

    best_val_sm = float("-inf")
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_since_improvement = 0
    top_checkpoints = []
    history = []
    val_metric_every = max(1, args.val_metric_every)
    val_gen_limit = args.limit_val_gen if args.limit_val_gen and args.limit_val_gen > 0 else args.limit_val
    early_stop_patience = max(0, args.early_stop_patience)

    for epoch in range(1, args.epochs + 1):
        train_losses = train_one_epoch(model, train_loader, optimizer, device, epoch, idx2word, args)
        val_losses, sample = validate_loss(model, val_loss_loader, device, idx2word, args, epoch)
        best_val_loss = min(best_val_loss, val_losses["loss"])
        should_eval_gen = (
            (not args.disable_val_gen)
            and ((epoch % val_metric_every == 0) or epoch == args.epochs)
        )
        val_sm = None
        checkpoint_score = None
        binary_eval = None
        control_eval = None
        is_best = False
        if should_eval_gen:
            val_result = run_generation_eval(
                model,
                val_base,
                idx2word,
                device,
                args.eval_batch_size,
                args.num_workers,
                val_gen_limit,
                args.max_new_tokens,
                args.route_threshold,
                tag="VAL",
                args=args,
            )
            val_sm = val_result["metrics_all"]["S_m"]
            checkpoint_score = val_sm
            binary_eval = val_result["binary"]
            control_eval = val_result["control"]
            is_best = val_sm > best_val_sm + 1e-12
            if is_best:
                best_val_sm = val_sm
                best_epoch = epoch
                epochs_since_improvement = 0
                print(f"  * new best val_Sm={best_val_sm:.4f} at epoch={epoch}")
            else:
                epochs_since_improvement += 1
                print(f"  epochs_since_improvement={epochs_since_improvement}/{early_stop_patience}")
        elif args.disable_val_gen:
            checkpoint_score = -val_losses["loss"]
            is_best = val_losses["loss"] <= best_val_loss + 1e-12
            if is_best:
                best_epoch = epoch
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1

        state = checkpoint_state(model, optimizer, epoch, train_losses, val_losses, best_val_sm, best_val_loss, args)
        save_checkpoint(os.path.join(output_dir, "last.pth"), state)
        top_checkpoints = maybe_update_top_checkpoints(output_dir, top_checkpoints, checkpoint_score, epoch, state)
        row = {
            "epoch": epoch,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_sm": val_sm,
            "best_val_sm": best_val_sm,
            "binary": binary_eval,
            "control": control_eval,
            "epochs_since_improvement": epochs_since_improvement,
            "sample": sample,
        }
        history.append(row)
        save_json(history, os.path.join(output_dir, "train_log.json"))
        print(
            f"epoch {epoch:03d} train {train_losses['loss']:.4f} val {val_losses['loss']:.4f} "
            f"val_Sm {val_sm:.4f}" if val_sm is not None else f"epoch {epoch:03d} train {train_losses['loss']:.4f} val {val_losses['loss']:.4f} val_gen skip"
        )
        if sample:
            print(f"sample pred: {sample['prediction']}")
            print(f"sample ref : {sample['reference']}")
            print(f"sample ctl : p_change={sample['p_change']:.3f} route_change={sample['route_change']:.3f}")
        if early_stop_patience > 0 and epochs_since_improvement >= early_stop_patience:
            print(f"\n[Early Stop] no improvement for {epochs_since_improvement} evaluated epochs; patience={early_stop_patience}.")
            break

    print(f"\ntraining done. best_epoch={best_epoch} best_val_Sm={best_val_sm:.4f} best_val_loss={best_val_loss:.4f}")
    best_ckpt = os.path.join(output_dir, "BEST.pth")
    if not args.disable_final_eval and os.path.exists(best_ckpt):
        print(f"\n[Final Eval] split={args.final_eval_split} ckpt={best_ckpt}")
        state = torch.load(best_ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"], strict=False)
        model.set_stage(2)
        model.to(device).eval()
        test_result = run_generation_eval(
            model,
            test_base,
            idx2word,
            device,
            args.eval_batch_size,
            args.num_workers,
            args.limit_test,
            args.max_new_tokens,
                args.route_threshold,
                tag=args.final_eval_split.upper(),
                args=args,
            )
        save_results(best_ckpt, args.final_eval_split, test_result)
        print(f"[Final Eval] saved to {os.path.join(os.path.dirname(best_ckpt), 'eval_results')}")


if __name__ == "__main__":
    main()
