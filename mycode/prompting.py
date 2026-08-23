from __future__ import annotations

from typing import Any

import torch


DEFAULT_PROMPT = (
    "Compare image 1 and image 2. If there is a change, briefly describe the main change in one sentence. "
    "If there is no change, state that no change has occurred."
)

IMAGE1_PLACEHOLDER = "<image1>"
IMAGE2_PLACEHOLDER = "<image2>"


def resolve_prompt_style(tokenizer: Any, prompt_style: str) -> str:
    style = (prompt_style or "plain").lower()
    if style not in {"plain", "chat", "auto"}:
        raise ValueError(f"qwen_prompt_style must be plain, chat or auto, got {prompt_style}")
    if style != "auto":
        return style
    model_name = str(getattr(tokenizer, "name_or_path", "")).lower()
    has_chat_template = bool(getattr(tokenizer, "chat_template", None))
    return "chat" if has_chat_template and "instruct" in model_name else "plain"


def _tokenize_no_special(tokenizer: Any, texts: list[str], max_length: int) -> torch.Tensor:
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
        return_tensors="pt",
    )
    return enc["input_ids"]


def _chat_segments(tokenizer: Any, instruction: str, system_prompt: str) -> list[str]:
    content = f"Image 1: {IMAGE1_PLACEHOLDER}\nImage 2: {IMAGE2_PLACEHOLDER}\n{instruction}"
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": content})
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if IMAGE1_PLACEHOLDER not in formatted or IMAGE2_PLACEHOLDER not in formatted:
        raise ValueError("chat template removed visual placeholders; cannot insert visual embeddings")
    before_image1, rest = formatted.split(IMAGE1_PLACEHOLDER, 1)
    between_images, after_image2 = rest.split(IMAGE2_PLACEHOLDER, 1)
    return [before_image1, between_images, after_image2]


def make_prompt_batch(
    tokenizer: Any,
    batch_size: int,
    device: torch.device,
    prompt_style: str = "plain",
    instruction: str = DEFAULT_PROMPT,
    system_prompt: str = "",
    max_prompt_tokens: int = 128,
):
    style = resolve_prompt_style(tokenizer, prompt_style)
    if style == "plain":
        ids = _tokenize_no_special(tokenizer, [instruction for _ in range(batch_size)], max_prompt_tokens)
        return ids.to(device)

    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("qwen_prompt_style=chat requires tokenizer.chat_template")
    segments = _chat_segments(tokenizer, instruction, system_prompt)
    segment_ids = [
        _tokenize_no_special(tokenizer, [segment for _ in range(batch_size)], max_prompt_tokens).to(device)
        for segment in segments
    ]
    return {"style": "chat", "segments": segment_ids}


def make_answer_batch(tokenizer: Any, answers: list[str], device: torch.device, max_answer_tokens: int):
    eos_text = tokenizer.eos_token or ""
    answer_ids = _tokenize_no_special(tokenizer, [answer + eos_text for answer in answers], max_answer_tokens).to(device)
    labels = answer_ids.clone()
    labels[answer_ids.eq(tokenizer.pad_token_id)] = -100
    return answer_ids, labels


def slice_prompt_batch(prompt_batch, item):
    if torch.is_tensor(prompt_batch):
        return prompt_batch[item]
    if isinstance(prompt_batch, dict):
        return {
            "style": prompt_batch.get("style", "chat"),
            "segments": [segment[item] for segment in prompt_batch["segments"]],
        }
    raise TypeError(f"Unsupported prompt batch type: {type(prompt_batch)!r}")
