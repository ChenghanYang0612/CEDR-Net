#!/usr/bin/env python3
"""Run binary pretraining and caption training as isolated stage_executor.py processes."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from config import cfg

SCRIPT_PATH = Path(__file__).resolve()
MYCODE_DIR = SCRIPT_PATH.parent
DEFAULT_PROJECT_ROOT = Path(cfg.paths.project_root)

QWEN_NAME_CHOICES = [
    "Qwen2-1.5B-Instruct",
    "Qwen2.5-1.5B-Instruct",
    "Qwen2.5-3B-Instruct",
    "Qwen3-0.6B",
    "Qwen3-1.7B",
    "Qwen3-4B",
]
QWEN_PROMPT_STYLE_CHOICES = ["plain", "chat", "auto"]
CONTROLLER_MODE_CHOICES = ["none", "nochange_only", "change_only", "shared", "dual"]
CONTROLLER_CONDITION_MODE_CHOICES = ["full", "route_only"]
ROUTING_MODE_CHOICES = ["hard", "confidence"]
FINAL_EVAL_SPLIT_CHOICES = ["val", "test"]
DTYPE_CHOICES = ["auto", "float16", "bfloat16", "float32"]
BINARY_SELECT_METRIC_CHOICES = ["val_loss"]
WORKFLOW_CHOICES = ["two_stage", "binary", "caption"]


def env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def env_flag(name: str, default: str = "0") -> bool:
    return env(name, default).strip().lower() in {"1", "true", "yes", "on"}


def split_ints(text: str) -> list[int]:
    parts = text.replace(",", " ").split()
    return [int(part) for part in parts]


def add_kv(cmd: list[str], key: str, value: Any) -> None:
    cmd.extend([key, str(value)])


def shell_join(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def resolve_project_path(project_root: Path, value: str) -> str:
    if not value:
        return value
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path)
    return str(project_root / path)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def append_log(log_path: Path, text: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(text)
        if not text.endswith("\n"):
            f.write("\n")


def init_run_log(
    log_path: Path,
    args: argparse.Namespace,
    config: dict[str, str],
    seeds: list[int],
    run_index: int,
    seed: int,
    run_root_name: str,
) -> None:
    payload = {
        "run_group": args.run_group,
        "run_root_name": run_root_name,
        "run_index": run_index,
        "seed": seed,
        "project_root": args.project_root,
        "output_root": args.output_root,
        "seeds": seeds,
        "qwen": config,
        "args": jsonable(vars(args)),
    }
    write_json(log_path.with_name("run_config.json"), payload)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        f.write("# Two-stage isolated training log\n\n")
        f.write("[run_config]\n")
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n\n")


def python_cmd(args: argparse.Namespace) -> list[str]:
    if args.force_conda_run or os.environ.get("CONDA_DEFAULT_ENV") != args.conda_env:
        return ["conda", "run", "--no-capture-output", "-n", args.conda_env, "python", "-u"]
    return [sys.executable, "-u"]


def resolve_qwen_model(project_root: Path, qwen_model: str, qwen_name: str) -> str:
    if qwen_name:
        name_path = Path(qwen_name).expanduser()
        if name_path.is_absolute():
            return str(name_path)
        if len(name_path.parts) > 1:
            return resolve_project_path(project_root, str(name_path))
        return str(project_root / "checkpoint" / "Qwen" / qwen_name)
    return resolve_project_path(project_root, qwen_model)


def require_choice(name: str, value: str, choices: list[str]) -> None:
    if value not in choices:
        raise ValueError(f"{name}={value!r} is not in {choices}")


def validate_args(args: argparse.Namespace) -> None:
    require_choice("workflow", args.workflow, WORKFLOW_CHOICES)
    if args.qwen_name and not args.qwen_model:
        require_choice("qwen_name", args.qwen_name, QWEN_NAME_CHOICES)
    require_choice("qwen_prompt_style", args.qwen_prompt_style, QWEN_PROMPT_STYLE_CHOICES)
    require_choice("controller_mode", args.controller_mode, CONTROLLER_MODE_CHOICES)
    require_choice("controller_condition_mode", args.controller_condition_mode, CONTROLLER_CONDITION_MODE_CHOICES)
    require_choice("routing_mode", args.routing_mode, ROUTING_MODE_CHOICES)
    require_choice("final_eval_split", args.final_eval_split, FINAL_EVAL_SPLIT_CHOICES)
    require_choice("dtype", args.dtype, DTYPE_CHOICES)
    require_choice("binary_pretrain_select_metric", args.binary_pretrain_select_metric, BINARY_SELECT_METRIC_CHOICES)
    require_choice("binary_pretrain_early_stop_metric", args.binary_pretrain_early_stop_metric, BINARY_SELECT_METRIC_CHOICES)


def qwen_settings(project_root: Path, args: argparse.Namespace) -> dict[str, str]:
    if args.qwen_model:
        qwen_model = resolve_qwen_model(project_root, args.qwen_model, "")
    else:
        qwen_model = resolve_qwen_model(project_root, cfg.paths.qwen_default, args.qwen_name)
    return {
        "tag": args.run_tag,
        "qwen_model": qwen_model,
        "prompt_style": args.qwen_prompt_style,
        "system_prompt": args.qwen_system_prompt,
    }


def common_train_args(
    args: argparse.Namespace,
    project_root: Path,
    run_name: str,
    output_dir: Path,
    seed: int,
    config: dict[str, str],
    train_stage: str,
    binary_ckpt: str,
) -> list[str]:
    cmd = [
        str(MYCODE_DIR / "stage_executor.py"),
        "--data-root",
        args.data_root,
        "--dataset-name",
        args.dataset_name,
        "--captions-per-image",
        str(args.captions_per_image),
        "--min-word-freq",
        str(args.min_word_freq),
        "--image-size",
        str(args.image_size),
        "--encoder-arch",
        args.encoder_arch,
        "--checkpoint",
        args.encoder_ckpt,
        "--qwen-model",
        config["qwen_model"],
        "--qwen-prompt-style",
        config["prompt_style"],
        "--qwen-system-prompt",
        config["system_prompt"],
        "--train-stage",
        train_stage,
        "--binary-init-ckpt",
        binary_ckpt,
        "--binary-pretrain-epochs",
        str(args.binary_pretrain_epochs),
        "--binary-pretrain-batch-size",
        str(args.binary_pretrain_batch_size),
        "--binary-pretrain-eval-batch-size",
        str(args.binary_pretrain_eval_batch_size),
        "--binary-pretrain-num-workers",
        str(args.binary_pretrain_num_workers),
        "--binary-pretrain-lr",
        args.binary_pretrain_lr,
        "--binary-pretrain-weight-decay",
        args.binary_pretrain_weight_decay,
        "--binary-pretrain-grad-clip",
        args.binary_pretrain_grad_clip,
        "--binary-pretrain-min-epochs",
        str(args.binary_pretrain_min_epochs),
        "--binary-pretrain-early-stop-patience",
        str(args.binary_pretrain_early_stop_patience),
        "--binary-pretrain-early-stop-metric",
        args.binary_pretrain_early_stop_metric,
        "--binary-pretrain-select-metric",
        args.binary_pretrain_select_metric,
        "--encoder-lora-rank",
        str(args.encoder_lora_rank),
        "--encoder-lora-alpha",
        args.encoder_lora_alpha,
        "--binary-semantic-dim",
        str(args.binary_semantic_dim),
        "--binary-attn-heads",
        str(args.binary_attn_heads),
        "--binary-classifier-dropout",
        args.binary_classifier_dropout,
        "--binary-gamma",
        args.binary_gamma,
        "--binary-alpha-change",
        args.binary_alpha_change,
        "--binary-alpha-nochange",
        args.binary_alpha_nochange,
        "--binary-ce-warmup-epochs",
        str(args.binary_ce_warmup_epochs),
        "--num-visual-tokens",
        str(args.num_visual_tokens),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--eval-batch-size",
        str(args.eval_batch_size),
        "--num-workers",
        str(args.num_workers),
        "--lr",
        args.lr,
        "--binary-lr",
        args.binary_lr,
        "--weight-decay",
        args.weight_decay,
        "--grad-clip",
        args.grad_clip,
        "--log-interval",
        str(args.log_interval),
        "--lambda-caption",
        args.lambda_caption,
        "--lambda-binary",
        args.lambda_binary,
        "--alpha-max",
        args.alpha_max,
        "--progressive-warmup-epochs",
        str(args.progressive_warmup_epochs),
        "--route-threshold",
        args.route_threshold,
        "--controller-mode",
        args.controller_mode,
        "--controller-condition-mode",
        args.controller_condition_mode,
        "--routing-mode",
        args.routing_mode,
        "--seed",
        str(seed),
        "--run-name",
        run_name,
        "--output-dir",
        str(output_dir),
        "--val-metric-every",
        str(args.val_metric_every),
        "--early-stop-patience",
        str(args.early_stop_patience),
        "--final-eval-split",
        args.final_eval_split,
        "--max-answer-tokens",
        str(args.max_answer_tokens),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--dtype",
        args.dtype,
        "--limit-train",
        str(args.limit_train),
        "--limit-val",
        str(args.limit_val),
        "--limit-val-gen",
        str(args.limit_val_gen),
        "--limit-test",
        str(args.limit_test),
    ]
    if args.transformers_path:
        add_kv(cmd, "--transformers-path", args.transformers_path)
    if args.disable_binary_cross_attn:
        cmd.append("--disable-binary-cross-attn")
    return cmd


def run_subprocess(
    cmd: list[str],
    args: argparse.Namespace,
    seed: int,
    deterministic_env: bool,
    log_path: Path,
    stage_label: str,
) -> None:
    env_vars = os.environ.copy()
    env_vars["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    if deterministic_env:
        env_vars["CUBLAS_WORKSPACE_CONFIG"] = env_vars.get("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        env_vars["PYTHONHASHSEED"] = str(seed)
        env_vars["NVIDIA_TF32_OVERRIDE"] = "0"
    header = f"\n[{stage_label}]\n[cmd] {shell_join(cmd)}\n"
    print(header.rstrip(), flush=True)
    if args.dry_run:
        return
    append_log(log_path, header)
    with log_path.open("a", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            cmd,
            cwd=args.project_root,
            env=env_vars,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
            log_file.flush()
        return_code = process.wait()
    append_log(log_path, f"[{stage_label}] exit_code={return_code}\n")
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd)


def selected_binary_ckpt(binary_run_dir: Path) -> Path:
    ckpt = binary_run_dir / "best_by_val_loss.pth"
    if not ckpt.exists():
        raise FileNotFoundError(f"Binary checkpoint not found: {ckpt}")
    return ckpt


def collect_caption_metrics(run_dir: Path, split: str) -> dict[str, Any]:
    metrics_path = run_dir / "eval_results" / f"BEST_{split}_metrics.json"
    if not metrics_path.exists():
        return {"status": "missing_metrics", "metrics_path": str(metrics_path)}
    metrics = read_json(metrics_path)
    return {
        "status": "success",
        "metrics_path": str(metrics_path),
        "test_sm": metrics.get("metrics_all", {}).get("S_m"),
        "test_cider": metrics.get("metrics_all", {}).get("CIDEr_D"),
        "binary_acc": metrics.get("binary", {}).get("accuracy"),
        "change_recall": metrics.get("binary", {}).get("change_recall"),
        "avg_route_change": metrics.get("control", {}).get("avg_route_change"),
    }


def collect_binary_metrics(run_dir: Path) -> dict[str, Any]:
    metrics_path = run_dir / "selected_test_metrics.json"
    if not metrics_path.exists():
        ckpt_path = run_dir / "best_by_val_loss.pth"
        return {
            "binary_status": "missing_test_metrics" if ckpt_path.exists() else "missing_binary_ckpt",
            "binary_metrics_path": str(metrics_path),
        }
    metrics = read_json(metrics_path)
    return {
        "binary_status": "success",
        "binary_metrics_path": str(metrics_path),
        "binary_acc": metrics.get("accuracy"),
        "change_recall": metrics.get("recall"),
        "binary_f1": metrics.get("f1"),
        "binary_precision": metrics.get("precision"),
        "nochange_recall": metrics.get("nochange_recall"),
        "tp": metrics.get("tp"),
        "fp": metrics.get("fp"),
        "tn": metrics.get("tn"),
        "fn": metrics.get("fn"),
    }


def append_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "index",
        "workflow",
        "run_root_name",
        "run_tag",
        "seed",
        "qwen_model",
        "qwen_prompt_style",
        "status",
        "binary_status",
        "binary_run_name",
        "binary_run_dir",
        "caption_run_name",
        "selected_binary_ckpt",
        "test_sm",
        "test_cider",
        "binary_acc",
        "binary_f1",
        "binary_precision",
        "change_recall",
        "nochange_recall",
        "tp",
        "fp",
        "tn",
        "fn",
        "avg_route_change",
        "caption_run_dir",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def parse_args() -> argparse.Namespace:
    project_root = Path(env("PROJECT_ROOT", str(DEFAULT_PROJECT_ROOT))).resolve()
    parser = argparse.ArgumentParser(description="Two-stage isolated training wrapper for Change_Caption_DU")
    parser.add_argument("--project-root", default=str(project_root))
    parser.add_argument("--workflow", default=env("WORKFLOW", cfg.two_stage.workflow))
    parser.add_argument("--run-group", default=env("RUN_GROUP", cfg.two_stage.run_group))
    parser.add_argument("--run-tag", default=env("RUN_TAG", cfg.train.run_tag))
    parser.add_argument("--seeds", default=env("SEEDS", " ".join(str(seed) for seed in cfg.two_stage.seeds)))
    parser.add_argument("--conda-env", default=env("CONDA_ENV", cfg.two_stage.conda_env))
    parser.add_argument("--force-conda-run", action="store_true")
    parser.add_argument("--gpu-id", default=env("GPU_ID", cfg.two_stage.gpu_id))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--qwen-model", default=env("QWEN_MODEL", ""))
    parser.add_argument("--qwen-name", default=env("QWEN_NAME", cfg.train.qwen_name))
    parser.add_argument("--qwen-prompt-style", default=env("QWEN_PROMPT_STYLE", cfg.train.qwen_prompt_style))
    parser.add_argument("--qwen-system-prompt", default=env("QWEN_SYSTEM_PROMPT", cfg.train.qwen_system_prompt))

    parser.add_argument("--output-root", default=env("OUTPUT_ROOT", cfg.paths.output_root))
    parser.add_argument("--data-root", default=env("DATA_ROOT", cfg.paths.data_root))
    parser.add_argument("--dataset-name", default=env("DATASET_NAME", cfg.data.dataset_name))
    parser.add_argument("--captions-per-image", type=int, default=int(env("CAPTIONS_PER_IMAGE", str(cfg.data.captions_per_image))))
    parser.add_argument("--min-word-freq", type=int, default=int(env("MIN_WORD_FREQ", str(cfg.data.min_word_freq))))
    parser.add_argument("--image-size", type=int, default=int(env("IMAGE_SIZE", str(cfg.data.image_size))))
    parser.add_argument("--encoder-arch", default=env("ENCODER_ARCH", cfg.model.clip_arch))
    parser.add_argument("--encoder-ckpt", default=env("ENCODER_CKPT", cfg.paths.remoteclip_b32))
    parser.add_argument("--binary-init-ckpt", default=env("BINARY_INIT_CKPT", cfg.paths.binary_init_ckpt))

    parser.add_argument("--binary-pretrain-epochs", type=int, default=int(env("BINARY_PRETRAIN_EPOCHS", str(cfg.binary_pretrain.epochs))))
    parser.add_argument("--binary-pretrain-batch-size", type=int, default=int(env("BINARY_PRETRAIN_BATCH_SIZE", str(cfg.binary_pretrain.batch_size))))
    parser.add_argument("--binary-pretrain-eval-batch-size", type=int, default=int(env("BINARY_PRETRAIN_EVAL_BATCH_SIZE", str(cfg.binary_pretrain.eval_batch_size))))
    parser.add_argument("--binary-pretrain-num-workers", type=int, default=int(env("BINARY_PRETRAIN_NUM_WORKERS", str(cfg.binary_pretrain.num_workers))))
    parser.add_argument("--binary-pretrain-lr", default=env("BINARY_PRETRAIN_LR", str(cfg.binary_pretrain.lr)))
    parser.add_argument("--binary-pretrain-weight-decay", default=env("BINARY_PRETRAIN_WEIGHT_DECAY", str(cfg.binary_pretrain.weight_decay)))
    parser.add_argument("--binary-pretrain-grad-clip", default=env("BINARY_PRETRAIN_GRAD_CLIP", str(cfg.binary_pretrain.grad_clip)))
    parser.add_argument("--binary-pretrain-min-epochs", type=int, default=int(env("BINARY_PRETRAIN_MIN_EPOCHS", str(cfg.binary_pretrain.min_epochs))))
    parser.add_argument("--binary-pretrain-early-stop-patience", type=int, default=int(env("BINARY_PRETRAIN_EARLY_STOP_PATIENCE", str(cfg.binary_pretrain.early_stop_patience))))
    parser.add_argument("--binary-pretrain-early-stop-metric", default=env("BINARY_PRETRAIN_EARLY_STOP_METRIC", cfg.binary_pretrain.early_stop_metric))
    parser.add_argument("--binary-pretrain-select-metric", default=env("BINARY_PRETRAIN_SELECT_METRIC", cfg.binary_pretrain.select_metric))
    parser.add_argument("--binary-pretrain-test-eval", type=int, default=int(env("BINARY_PRETRAIN_TEST_EVAL", "1" if cfg.binary_pretrain.test_eval else "0")))
    parser.add_argument("--binary-deterministic", action="store_true", default=env_flag("BINARY_DETERMINISTIC", "1" if cfg.binary_pretrain.deterministic else "0"))

    parser.add_argument("--encoder-lora-rank", type=int, default=int(env("ENCODER_LORA_RANK", "8")))
    parser.add_argument("--encoder-lora-alpha", default=env("ENCODER_LORA_ALPHA", "16.0"))
    parser.add_argument("--binary-semantic-dim", type=int, default=int(env("BINARY_SEMANTIC_DIM", str(cfg.model.binary_semantic_dim))))
    parser.add_argument("--binary-attn-heads", type=int, default=int(env("BINARY_ATTN_HEADS", str(cfg.model.binary_attn_heads))))
    parser.add_argument("--binary-classifier-dropout", default=env("BINARY_CLASSIFIER_DROPOUT", str(cfg.binary_pretrain.classifier_dropout)))
    parser.add_argument("--binary-gamma", default=env("BINARY_GAMMA", str(cfg.binary_pretrain.gamma)))
    parser.add_argument("--binary-alpha-change", default=env("BINARY_ALPHA_CHANGE", str(cfg.binary_pretrain.alpha_change)))
    parser.add_argument("--binary-alpha-nochange", default=env("BINARY_ALPHA_NOCHANGE", str(cfg.binary_pretrain.alpha_nochange)))
    parser.add_argument("--binary-ce-warmup-epochs", type=int, default=int(env("BINARY_CE_WARMUP_EPOCHS", str(cfg.binary_pretrain.ce_warmup_epochs))))
    parser.add_argument("--num-visual-tokens", type=int, default=int(env("NUM_VISUAL_TOKENS", "16")))

    parser.add_argument("--epochs", type=int, default=int(env("EPOCHS", str(cfg.train.epochs))))
    parser.add_argument("--batch-size", type=int, default=int(env("BATCH_SIZE", str(cfg.train.batch_size))))
    parser.add_argument("--eval-batch-size", type=int, default=int(env("EVAL_BATCH_SIZE", str(cfg.train.eval_batch_size))))
    parser.add_argument("--num-workers", type=int, default=int(env("NUM_WORKERS", str(cfg.train.num_workers))))
    parser.add_argument("--lr", default=env("LR", str(cfg.train.lr)))
    parser.add_argument("--binary-lr", default=env("BINARY_LR", str(cfg.train.binary_lr)))
    parser.add_argument("--weight-decay", default=env("WEIGHT_DECAY", str(cfg.train.weight_decay)))
    parser.add_argument("--grad-clip", default=env("GRAD_CLIP", str(cfg.train.grad_clip)))
    parser.add_argument("--log-interval", type=int, default=int(env("LOG_INTERVAL", str(cfg.train.log_interval))))
    parser.add_argument("--lambda-caption", default=env("LAMBDA_CAPTION", str(cfg.train.lambda_caption)))
    parser.add_argument("--lambda-binary", default=env("LAMBDA_BINARY", str(cfg.train.lambda_binary)))
    parser.add_argument("--alpha-max", default=env("ALPHA_MAX", str(cfg.train.alpha_max)))
    parser.add_argument("--progressive-warmup-epochs", type=int, default=int(env("PROGRESSIVE_WARMUP_EPOCHS", str(cfg.train.progressive_warmup_epochs))))
    parser.add_argument("--route-threshold", default=env("ROUTE_THRESHOLD", str(cfg.train.route_threshold)))
    parser.add_argument("--controller-mode", default=env("CONTROLLER_MODE", cfg.model.controller_mode))
    parser.add_argument("--controller-condition-mode", default=env("CONTROLLER_CONDITION_MODE", cfg.model.controller_condition_mode))
    parser.add_argument("--routing-mode", default=env("ROUTING_MODE", cfg.model.routing_mode))
    parser.add_argument("--disable-binary-cross-attn", action="store_true", default=env_flag("DISABLE_BINARY_CROSS_ATTN", "0"))

    parser.add_argument("--val-metric-every", type=int, default=int(env("VAL_METRIC_EVERY", str(cfg.train.val_metric_every))))
    parser.add_argument("--early-stop-patience", type=int, default=int(env("EARLY_STOP_PATIENCE", str(cfg.train.early_stop_patience))))
    parser.add_argument("--final-eval-split", default=env("FINAL_EVAL_SPLIT", cfg.train.final_eval_split))
    parser.add_argument("--max-answer-tokens", type=int, default=int(env("MAX_ANSWER_TOKENS", str(cfg.train.max_answer_tokens))))
    parser.add_argument("--max-new-tokens", type=int, default=int(env("MAX_NEW_TOKENS", str(cfg.train.max_new_tokens))))
    parser.add_argument("--dtype", default=env("DTYPE", cfg.train.dtype))
    parser.add_argument("--transformers-path", default=env("TRANSFORMERS_PATH", ""))
    parser.add_argument("--limit-train", type=int, default=int(env("LIMIT_TRAIN", str(cfg.train.limit_train))))
    parser.add_argument("--limit-val", type=int, default=int(env("LIMIT_VAL", str(cfg.train.limit_val))))
    parser.add_argument("--limit-val-gen", type=int, default=int(env("LIMIT_VAL_GEN", str(cfg.train.limit_val_gen))))
    parser.add_argument("--limit-test", type=int, default=int(env("LIMIT_TEST", str(cfg.train.limit_test))))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.project_root = str(Path(args.project_root).resolve())
    project_root = Path(args.project_root)
    args.output_root = resolve_project_path(project_root, args.output_root)
    args.data_root = resolve_project_path(project_root, args.data_root)
    args.encoder_ckpt = resolve_project_path(project_root, args.encoder_ckpt)
    args.binary_init_ckpt = resolve_project_path(project_root, args.binary_init_ckpt)
    output_root = Path(args.output_root)
    train_python = python_cmd(args)
    seeds = split_ints(args.seeds)
    if not seeds:
        seeds = [42]

    summary_rows: list[dict[str, Any]] = []
    config = qwen_settings(project_root, args)
    multi_run = len(seeds) > 1

    for index, seed in enumerate(seeds, start=1):
        run_tag = config["tag"]
        run_root_name = f"{args.run_group}_run{index}" if multi_run else args.run_group
        run_root_dir = output_root / run_root_name
        binary_run_name = f"{run_root_name}/binary"
        caption_run_name = run_root_name
        binary_run_dir = run_root_dir / "binary"
        caption_run_dir = run_root_dir
        global_log_path = run_root_dir / "run.log"
        if not args.dry_run:
            init_run_log(global_log_path, args, config, seeds, index, seed, run_root_name)

        print("=" * 80, flush=True)
        print(f"[two-stage] {index}/{len(seeds)} seed={seed}", flush=True)
        print(f"[two-stage] workflow={args.workflow}", flush=True)
        print(f"[two-stage] run_root_name={run_root_name}", flush=True)
        print(f"[two-stage] qwen_model={config['qwen_model']}", flush=True)
        print(f"[two-stage] qwen_prompt_style={config['prompt_style']}", flush=True)
        if args.workflow in {"two_stage", "binary"}:
            print(f"[two-stage] binary_run_name={binary_run_name}", flush=True)
        if args.workflow in {"two_stage", "caption"}:
            print(f"[two-stage] caption_run_name={caption_run_name}", flush=True)
            print(f"[two-stage] binary_init_ckpt={args.binary_init_ckpt}", flush=True)
        print("=" * 80, flush=True)

        binary_ckpt = Path(args.binary_init_ckpt)

        if args.workflow in {"two_stage", "binary"}:
            binary_args = common_train_args(
                args=args,
                project_root=project_root,
                run_name=binary_run_name,
                output_dir=binary_run_dir,
                seed=seed,
                config=config,
                train_stage="binary",
                binary_ckpt=args.binary_init_ckpt,
            )
            if args.binary_pretrain_test_eval:
                binary_args.append("--binary-pretrain-test-eval")
            if args.binary_deterministic:
                binary_args.append("--binary-deterministic")

            print("[two-stage] stage binary: binary pretrain process", flush=True)
            run_subprocess(
                train_python + binary_args,
                args,
                seed,
                deterministic_env=args.binary_deterministic,
                log_path=global_log_path,
                stage_label=f"run{index} binary",
            )

            if args.dry_run:
                binary_ckpt = binary_run_dir / "best_by_val_loss.pth"
            else:
                binary_ckpt = selected_binary_ckpt(binary_run_dir)

        if args.workflow in {"two_stage", "caption"}:
            caption_args = common_train_args(
                args=args,
                project_root=project_root,
                run_name=caption_run_name,
                output_dir=caption_run_dir,
                seed=seed,
                config=config,
                train_stage="caption",
                binary_ckpt=str(binary_ckpt),
            )

            print("[two-stage] stage caption: caption training process", flush=True)
            print(f"[two-stage] selected_binary_ckpt={binary_ckpt}", flush=True)
            run_subprocess(
                train_python + caption_args,
                args,
                seed,
                deterministic_env=False,
                log_path=global_log_path,
                stage_label=f"run{index} caption",
            )
        if args.dry_run:
            continue

        row = {
            "index": index,
            "workflow": args.workflow,
            "run_root_name": run_root_name,
            "run_tag": run_tag,
            "seed": seed,
            "qwen_model": config["qwen_model"],
            "qwen_prompt_style": config["prompt_style"],
            "binary_run_name": binary_run_name if args.workflow in {"two_stage", "binary"} else "",
            "binary_run_dir": str(binary_run_dir) if args.workflow in {"two_stage", "binary"} else "",
            "caption_run_name": caption_run_name,
            "caption_run_dir": str(caption_run_dir) if args.workflow in {"two_stage", "caption"} else "",
            "selected_binary_ckpt": str(binary_ckpt),
        }
        if args.workflow in {"two_stage", "binary"}:
            row.update(collect_binary_metrics(binary_run_dir))
        if args.workflow in {"two_stage", "caption"}:
            row.update(collect_caption_metrics(caption_run_dir, args.final_eval_split))
        else:
            row["status"] = "binary_complete"
        summary_rows.append(row)
        write_json(run_root_dir / "summary.json", [row])
        append_summary_csv(run_root_dir / "_run_logs" / "summary.csv", [row])
        if multi_run:
            write_json(output_root / f"{args.run_group}_summary.json", summary_rows)
            append_summary_csv(output_root / f"{args.run_group}_summary.csv", summary_rows)
            print(f"[two-stage] run summary saved under {run_root_dir}", flush=True)
            print(f"[two-stage] aggregate summary saved under {output_root}", flush=True)
        else:
            print(f"[two-stage] summary saved under {run_root_dir}", flush=True)


if __name__ == "__main__":
    main()
