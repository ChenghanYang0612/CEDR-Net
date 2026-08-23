from __future__ import annotations

import argparse
import importlib.machinery
import os
import subprocess
import sys
import threading
import types
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader, Subset

from config import cfg
from dataset import ChangeCaptionDataset
from model import ChangeAwareHiddenControllerQwen
from prompting import DEFAULT_PROMPT, make_prompt_batch
from utils import PAD_IDX, decode_caption, ensure_dir, load_vocab, save_json


PROMPT_USER = DEFAULT_PROMPT


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate baseline 18 joint binary-caption-controller with progressive binary evidence")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data-root", default=cfg.paths.data_root)
    parser.add_argument("--dataset-name", default=cfg.data.dataset_name)
    parser.add_argument("--captions-per-image", type=int, default=cfg.data.captions_per_image)
    parser.add_argument("--min-word-freq", type=int, default=cfg.data.min_word_freq)
    parser.add_argument("--encoder-arch", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--qwen-model", default=None)
    parser.add_argument("--qwen-prompt-style", default=None, choices=["plain", "chat", "auto"])
    parser.add_argument("--qwen-system-prompt", default=None)
    parser.add_argument("--max-prompt-tokens", type=int, default=None)
    parser.add_argument("--binary-init-ckpt", default=None)
    parser.add_argument("--encoder-lora-rank", type=int, default=None)
    parser.add_argument("--encoder-lora-alpha", type=float, default=None)
    parser.add_argument("--binary-semantic-dim", type=int, default=None)
    parser.add_argument("--binary-attn-heads", type=int, default=None)
    parser.add_argument("--num-visual-tokens", type=int, default=None)
    parser.add_argument("--mode-dim", type=int, default=None)
    parser.add_argument("--controller-mode", default=None, choices=["none", "nochange_only", "change_only", "shared", "dual"])
    parser.add_argument("--controller-condition-mode", default=None, choices=["full", "route_only"])
    parser.add_argument("--routing-mode", default=None, choices=["hard", "confidence"])
    parser.add_argument("--disable-binary-cross-attn", action="store_true")
    parser.add_argument("--controller-residual-scale", type=float, default=None)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=cfg.train.eval_batch_size)
    parser.add_argument("--num-workers", type=int, default=cfg.train.num_workers)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--route-threshold", type=float, default=0.5)
    parser.add_argument("--baseline-compare-limit", type=int, default=0, help="also save controller-off captions for first N samples")
    parser.add_argument("--output", default=None)
    parser.add_argument("--metrics-output", default=None)
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


class FixedMeteor:
    """Local METEOR wrapper with Java options before -jar.

    The installed pycocoevalcap wrapper in this environment passes -Xmx2G
    after -jar, which can make the METEOR process return command help instead
    of numeric scores.
    """

    def __init__(self):
        import pycocoevalcap.meteor.meteor as meteor_mod

        jar_path = os.path.join(os.path.dirname(os.path.abspath(meteor_mod.__file__)), "meteor-1.5.jar")
        self.meteor_cmd = ["java", "-Xmx2G", "-jar", jar_path, "-", "-", "-stdio", "-l", "en", "-norm"]
        self.meteor_p = subprocess.Popen(
            self.meteor_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.lock = threading.Lock()

    def method(self):
        return "METEOR"

    @staticmethod
    def _clean(text: str) -> str:
        return " ".join(str(text).replace("|||", " ").split())

    def _stat(self, hypothesis_str, reference_list):
        hypothesis_str = self._clean(hypothesis_str)
        references = [self._clean(ref) for ref in reference_list]
        score_line = " ||| ".join(("SCORE", " ||| ".join(references), hypothesis_str))
        self.meteor_p.stdin.write(f"{score_line}\n".encode())
        self.meteor_p.stdin.flush()
        return self.meteor_p.stdout.readline().decode().strip()

    def compute_score(self, gts, res):
        assert gts.keys() == res.keys()
        img_ids = gts.keys()
        scores = []
        eval_line = "EVAL"
        self.lock.acquire()
        try:
            for i in img_ids:
                assert len(res[i]) == 1
                stat = self._stat(res[i][0], gts[i])
                eval_line += f" ||| {stat}"
            self.meteor_p.stdin.write(f"{eval_line}\n".encode())
            self.meteor_p.stdin.flush()
            for _ in range(0, len(img_ids)):
                scores.append(float(self.meteor_p.stdout.readline().strip()))
            score = float(self.meteor_p.stdout.readline().strip())
        finally:
            self.lock.release()
        return score, scores

    def __del__(self):
        try:
            self.lock.acquire()
            if self.meteor_p.stdin:
                self.meteor_p.stdin.close()
            self.meteor_p.kill()
            self.meteor_p.wait()
        except Exception:
            pass
        finally:
            try:
                self.lock.release()
            except Exception:
                pass


def compute_scores(gts: Dict[str, List[str]], hyps: Dict[str, List[str]]) -> Dict[str, float]:
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.rouge.rouge import Rouge

    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (FixedMeteor(), ["METEOR"]),
        (Rouge(), ["ROUGE_L"]),
        (Cider(), ["CIDEr_D"]),
    ]
    scores: Dict[str, float] = {}
    for scorer, keys in scorers:
        try:
            score, _ = scorer.compute_score(gts, hyps)
        finally:
            if hasattr(scorer, "__del__"):
                try:
                    scorer.__del__()
                except Exception:
                    pass
        if isinstance(score, list):
            for key, value in zip(keys, score):
                scores[key] = round(float(value), 4)
        else:
            scores[keys[0]] = round(float(score), 4)
    scores["S_m"] = round(
        (scores.get("Bleu_4", 0.0) + scores.get("METEOR", 0.0) + scores.get("ROUGE_L", 0.0) + scores.get("CIDEr_D", 0.0)) / 4.0,
        4,
    )
    return scores


def print_scores(tag: str, scores: Dict[str, float], n: int = 0) -> None:
    sep = "-" * 64
    print(sep)
    print(f"  [{tag}]  (n={n})")
    print(
        f"  BLEU-1/2/3/4 : {scores.get('Bleu_1', 0.0):.4f}  {scores.get('Bleu_2', 0.0):.4f}  "
        f"{scores.get('Bleu_3', 0.0):.4f}  {scores.get('Bleu_4', 0.0):.4f}"
    )
    print(f"  METEOR       : {scores.get('METEOR', 0.0):.4f}")
    print(f"  ROUGE-L      : {scores.get('ROUGE_L', 0.0):.4f}")
    print(f"  CIDEr-D      : {scores.get('CIDEr_D', 0.0) * 100:.2f}  (raw={scores.get('CIDEr_D', 0.0):.4f})")
    print(f"  S_m          : {scores.get('S_m', 0.0):.4f}")
    print(sep)


def build_dataset(data_root: str, split: str, dataset_name: str, captions_per_image: int, min_word_freq: int):
    return ChangeCaptionDataset(data_root, split, dataset_name=dataset_name, captions_per_image=captions_per_image, min_word_freq=min_word_freq)


def build_eval_subset(dataset: ChangeCaptionDataset, limit: int = 0):
    indices = list(range(0, len(dataset), dataset.cpi))
    if limit and limit > 0:
        indices = indices[:limit]
    return Subset(dataset, indices)


def resolve_eval_prompt_options(args, state) -> tuple[str, str, int]:
    ckpt_config = state.get("config", {})
    prompt_style = args.qwen_prompt_style or ckpt_config.get("qwen_prompt_style", cfg.train.qwen_prompt_style)
    system_prompt = (
        args.qwen_system_prompt
        if args.qwen_system_prompt is not None
        else ckpt_config.get("qwen_system_prompt", cfg.train.qwen_system_prompt)
    )
    max_prompt_tokens = args.max_prompt_tokens or ckpt_config.get("max_prompt_tokens", 128)
    return prompt_style, system_prompt, int(max_prompt_tokens)


def binary_metrics(truth: list[int], pred: list[int]) -> Dict[str, float | int]:
    tp = sum(1 for y, p in zip(truth, pred) if y == 1 and p == 1)
    tn = sum(1 for y, p in zip(truth, pred) if y == 0 and p == 0)
    fp = sum(1 for y, p in zip(truth, pred) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(truth, pred) if y == 1 and p == 0)
    total = max(1, len(truth))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "accuracy": round((tp + tn) / total, 4),
        "change_precision": round(precision, 4),
        "change_recall": round(recall, 4),
        "change_f1": round(f1, 4),
        "nochange_accuracy": round(tn / max(1, tn + fp), 4),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def mean_or_zero(values: list[float]) -> float:
    return round(sum(values) / max(1, len(values)), 4)


def load_model(args, device: torch.device, vocab_size: int):
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    ckpt_config = state.get("config", {})
    qwen_model = args.qwen_model or ckpt_config.get("qwen_model", cfg.paths.qwen_default)
    local_files_only = not args.allow_remote_model
    if local_files_only and not Path(qwen_model).exists():
        raise FileNotFoundError(qwen_model)
    model = ChangeAwareHiddenControllerQwen(
        qwen_model_path=qwen_model,
        vocab_size=vocab_size,
        pad_idx=PAD_IDX,
        encoder_arch=args.encoder_arch or ckpt_config.get("encoder_arch", cfg.model.clip_arch),
        checkpoint_path=args.checkpoint or ckpt_config.get("encoder_checkpoint", cfg.paths.remoteclip_b32),
        encoder_lora_rank=args.encoder_lora_rank or ckpt_config.get("encoder_lora_rank", 8),
        encoder_lora_alpha=args.encoder_lora_alpha or ckpt_config.get("encoder_lora_alpha", 16.0),
        binary_init_ckpt=args.binary_init_ckpt or ckpt_config.get("binary_init_ckpt", cfg.paths.binary_init_ckpt),
        binary_semantic_dim=args.binary_semantic_dim or ckpt_config.get("binary_semantic_dim", cfg.model.binary_semantic_dim),
        binary_attn_heads=args.binary_attn_heads or ckpt_config.get("binary_attn_heads", cfg.model.binary_attn_heads),
        num_visual_tokens=args.num_visual_tokens or ckpt_config.get("num_visual_tokens", 16),
        mode_dim=args.mode_dim or ckpt_config.get("mode_dim", cfg.model.mode_dim),
        controller_mode=args.controller_mode or ckpt_config.get("controller_mode", cfg.model.controller_mode),
        controller_condition_mode=args.controller_condition_mode or ckpt_config.get("controller_condition_mode", cfg.model.controller_condition_mode),
        routing_mode=args.routing_mode or ckpt_config.get("routing_mode", cfg.model.routing_mode),
        controller_residual_scale=args.controller_residual_scale or ckpt_config.get("controller_residual_scale", cfg.model.controller_residual_scale),
        disable_binary_cross_attn=args.disable_binary_cross_attn or ckpt_config.get("disable_binary_cross_attn", False),
        torch_dtype=resolve_dtype(args.dtype, device),
        local_files_only=local_files_only,
    )
    missing, unexpected = model.load_state_dict(state["model"], strict=False)
    unexpected = [key for key in unexpected if not key.startswith("llm.")]
    missing_trainable = [
        key
        for key in missing
        if not key.startswith("llm.")
        and "clip_model" not in key
    ]
    if unexpected:
        print(f"[eval] unexpected checkpoint keys: {unexpected[:20]}")
    if missing_trainable:
        print(f"[eval] missing non-LLM/head keys: {missing_trainable[:20]}")
    model.set_stage(2)
    model.to(device).eval()
    print(f"[eval] epoch={state.get('epoch', '?')} best_val_Sm={state.get('best_val_sm', '?')}")
    return model, state


@torch.no_grad()
def generate_with_controller_mode(model, mode: str, img_a, img_b, prompt_ids, idx2word, max_new_tokens: int, route_threshold: float, num_beams: int = 1):
    old_mode = model.controller.mode
    try:
        model.controller.mode = mode
        return model.generate(img_a, img_b, prompt_ids, idx2word, max_new_tokens=max_new_tokens, threshold=route_threshold, num_beams=num_beams)
    finally:
        model.controller.mode = old_mode


@torch.no_grad()
def evaluate(
    model,
    loader,
    idx2word,
    device,
    max_new_tokens: int = 48,
    num_beams: int = 1,
    route_threshold: float = 0.5,
    baseline_compare_limit: int = 0,
    prompt_style: str = "plain",
    system_prompt: str = "",
    max_prompt_tokens: int = 128,
):
    model.eval()
    gts_all, hyps_all = {}, {}
    gts_change, hyps_change = {}, {}
    gts_nochange, hyps_nochange = {}, {}
    sample_results = []
    truth_flags, pred_flags = [], []
    route_change_values, route_nochange_values, lengths = [], [], []
    sample_idx = 0

    for batch_idx, batch in enumerate(loader, start=1):
        img_a, img_b, _, _, changeflags, all_captions, image_ids = batch
        img_a = img_a.to(device)
        img_b = img_b.to(device)
        prompt_ids = make_prompt_batch(
            model.tokenizer,
            img_a.size(0),
            device,
            prompt_style=prompt_style,
            instruction=PROMPT_USER,
            system_prompt=system_prompt,
            max_prompt_tokens=max_prompt_tokens,
        )
        generated, aux = model.generate(
            img_a,
            img_b,
            prompt_ids,
            idx2word,
            max_new_tokens=max_new_tokens,
            threshold=route_threshold,
            num_beams=num_beams,
        )
        baseline_generated = None
        if baseline_compare_limit > 0 and sample_idx < baseline_compare_limit:
            baseline_generated, _ = generate_with_controller_mode(
                model,
                "none",
                img_a,
                img_b,
                prompt_ids,
                idx2word,
                max_new_tokens=max_new_tokens,
                route_threshold=route_threshold,
                num_beams=num_beams,
            )
        probs = aux["p_change"].detach().cpu().tolist()
        route_change = aux["route_change"].detach().cpu().tolist()
        route_nochange = aux["route_nochange"].detach().cpu().tolist()
        gen_lengths = aux["caption_lengths"]

        for i in range(img_a.size(0)):
            hyp = generated[i].strip()
            refs = [decode_caption(row.tolist(), idx2word) for row in all_captions[i]]
            key = str(sample_idx)
            gts_all[key] = refs
            hyps_all[key] = [hyp]
            changeflag = int(changeflags[i])
            pred_flag = int(probs[i] >= route_threshold)
            truth_flags.append(changeflag)
            pred_flags.append(pred_flag)
            route_change_values.append(float(route_change[i]))
            route_nochange_values.append(float(route_nochange[i]))
            lengths.append(float(gen_lengths[i]))
            if changeflag == 0:
                gts_nochange[key] = refs
                hyps_nochange[key] = [hyp]
            else:
                gts_change[key] = refs
                hyps_change[key] = [hyp]
            row = {
                "image_id": image_ids[i],
                "changeflag": changeflag,
                "binary_p_change": probs[i],
                "binary_pred": pred_flag,
                "route_change": route_change[i],
                "route_nochange": route_nochange[i],
                "generated_length": gen_lengths[i],
                "hyp": hyp,
                "refs": refs,
            }
            if baseline_generated is not None and sample_idx < baseline_compare_limit:
                row["baseline_no_controller_hyp"] = baseline_generated[i].strip()
            sample_results.append(row)
            sample_idx += 1
        if batch_idx == 1 or batch_idx % 50 == 0 or batch_idx == len(loader):
            print(
                f"[eval] generated {sample_idx}/{len(loader.dataset)} images "
                f"avg_len={mean_or_zero(lengths):.2f} "
                f"avg_route_change={mean_or_zero(route_change_values):.3f}"
            )

    return {
        "metrics_all": compute_scores(gts_all, hyps_all),
        "metrics_change": compute_scores(gts_change, hyps_change) if gts_change else {},
        "metrics_nochange": compute_scores(gts_nochange, hyps_nochange) if gts_nochange else {},
        "binary": binary_metrics(truth_flags, pred_flags),
        "control": {
            "avg_route_change": mean_or_zero(route_change_values),
            "avg_route_nochange": mean_or_zero(route_nochange_values),
            "avg_generated_length": mean_or_zero(lengths),
            "num_beams": int(num_beams),
        },
        "n_all": len(gts_all),
        "n_change": len(gts_change),
        "n_nochange": len(gts_nochange),
        "sample_results": sample_results,
        "_gts": gts_all,
        "_hyps": hyps_all,
    }


def save_results(ckpt_path: str, split: str, result: Dict, metrics_output: str | None = None, captions_output: str | None = None):
    eval_dir = os.path.join(os.path.dirname(ckpt_path), "eval_results")
    ensure_dir(eval_dir)
    stem = Path(ckpt_path).stem
    metrics = {
        "split": split,
        "metrics_all": result["metrics_all"],
        "metrics_change": result["metrics_change"],
        "metrics_nochange": result["metrics_nochange"],
        "binary": result["binary"],
        "control": result["control"],
        "n_all": result["n_all"],
        "n_change": result["n_change"],
        "n_nochange": result["n_nochange"],
    }
    detail = {"split": split, "gts": result["_gts"], "hyps": result["_hyps"], "sample_results": result["sample_results"]}
    metrics_path = metrics_output or os.path.join(eval_dir, f"{stem}_{split}_metrics.json")
    captions_path = captions_output or os.path.join(eval_dir, f"{stem}_{split}_captions.json")
    save_json(metrics, metrics_path)
    save_json(detail, captions_path)
    return metrics_path, captions_path


def main():
    args = parse_args()
    os.environ.setdefault("MKL_SERVICE_FORCE_INTEL", "1")
    os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
    maybe_use_transformers_path(args.transformers_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = build_dataset(args.data_root, "train", args.dataset_name, args.captions_per_image, args.min_word_freq)
    word2idx, idx2word = load_vocab(train_dataset.wordmap_path)
    base_dataset = build_dataset(args.data_root, args.split, args.dataset_name, args.captions_per_image, args.min_word_freq)
    eval_dataset = build_eval_subset(base_dataset, args.limit)
    loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    model, state = load_model(args, device, len(word2idx))
    prompt_style, system_prompt, max_prompt_tokens = resolve_eval_prompt_options(args, state)
    print(f"[eval] qwen_prompt_style={prompt_style} max_prompt_tokens={max_prompt_tokens}")
    result = evaluate(
        model,
        loader,
        idx2word,
        device,
        args.max_new_tokens,
        args.num_beams,
        args.route_threshold,
        args.baseline_compare_limit,
        prompt_style=prompt_style,
        system_prompt=system_prompt,
        max_prompt_tokens=max_prompt_tokens,
    )
    print_scores("ALL", result["metrics_all"], n=result["n_all"])
    if result["metrics_change"]:
        print_scores("CHANGE", result["metrics_change"], n=result["n_change"])
    if result["metrics_nochange"]:
        print_scores("NOCHANGE", result["metrics_nochange"], n=result["n_nochange"])
    print(f"[BINARY] {result['binary']}")
    print(f"[CONTROL] {result['control']}")
    metrics_path, captions_path = save_results(args.ckpt, args.split, result, args.metrics_output, args.output)
    print(f"saved metrics to {metrics_path}")
    print(f"saved captions to {captions_path}")


if __name__ == "__main__":
    main()
