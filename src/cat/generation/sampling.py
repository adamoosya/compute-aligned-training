"""Chunked decoder-only sampling. No rewards, retries, or answer-based filtering."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from typing import Any, Iterator

import torch

from cat.data.math import MathExample
from cat.evaluation.records import CandidateRecord
from .config import SamplingConfig, positive_int


def _generation_config(config: SamplingConfig, count: int, tokenizer: Any):
    try:
        from transformers import GenerationConfig
    except ImportError as error:
        raise ImportError("Generation requires python -m pip install -e '.[hf]'") from error
    # Fresh settings avoid inheriting beam search, forced tokens or penalties
    # from a model's stored generation_config.json.
    return GenerationConfig(
        do_sample=True, num_beams=1, num_return_sequences=count,
        max_new_tokens=config.max_new_tokens, temperature=config.temperature,
        top_p=config.top_p, top_k=config.top_k, use_cache=True,
        bos_token_id=getattr(tokenizer, "bos_token_id", None),
        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
        return_dict_in_generate=False, output_scores=False, output_logits=False,
        disable_compile=True,
    )


def chunk_seed(seed: int, example_id: str, start: int) -> int:
    """Stable across problem ordering; does not use Python's salted hash()."""
    positive_int(seed, "seed", 0)
    positive_int(start, "start", 0)
    if not isinstance(example_id, str) or not example_id.strip():
        raise ValueError("example_id must be nonempty")
    data = json.dumps([seed, example_id, start], separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(data).digest()[:8], "big") % (2**63)


def encode_prompt(tokenizer: Any, problem: str, config: SamplingConfig) -> tuple[int, ...]:
    """The reference answer and solution are not arguments to prompt encoding."""
    if not isinstance(problem, str) or not problem.strip():
        raise ValueError("problem must be nonempty")
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("Tokenizer has no chat template")
    text = config.prompt_template.format(problem=problem)
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=True, add_generation_prompt=True,
    )
    if (not isinstance(ids, (list, tuple)) or not ids
            or any(type(token) is not int or token < 0 for token in ids)):
        raise ValueError("chat template must produce a nonempty list of nonnegative token IDs")
    if len(ids) > config.max_prompt_tokens:
        raise ValueError(f"Prompt has {len(ids)} tokens, exceeding max_prompt_tokens={config.max_prompt_tokens}; not truncated")
    return tuple(ids)


def _device_and_context(model: Any, ids: tuple[int, ...], config: SamplingConfig) -> torch.device:
    if getattr(model.config, "is_encoder_decoder", False):
        raise ValueError("Only decoder-only generation is supported")
    device = model.get_input_embeddings().weight.device
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Generation currently supports CPU or a single CUDA device")
    if any(p.device != device for p in model.parameters()):
        raise ValueError("Model must be on one device; offloading/distributed generation is not supported")
    maximum = getattr(model.config, "max_position_embeddings", None)
    if type(maximum) is int and len(ids) + config.max_new_tokens > maximum:
        raise ValueError("Prompt plus max_new_tokens exceeds the model context limit")
    size = model.get_input_embeddings().weight.shape[0]
    if max(ids) >= size:
        raise ValueError("Prompt token exceeds the model vocabulary")
    return device


@contextmanager
def _sampling_state(model: Any, device: torch.device, seed: int) -> Iterator[None]:
    # Restore individual submodule modes, including mixed train/eval states.
    modes = [(module, module.training) for module in model.modules()]
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    try:
        model.eval()
        with torch.random.fork_rng(devices=devices, device_type="cuda"):
            # torch.manual_seed also touches other device types; seed only the
            # CPU and selected CUDA generators whose states are being preserved.
            torch.random.default_generator.manual_seed(seed)
            if devices:
                with torch.cuda.device(devices[0]):
                    torch.cuda.manual_seed(seed)
            with torch.inference_mode():
                yield
    finally:
        for module, mode in modes:
            module.training = mode


def sample_example(
    model: Any, tokenizer: Any, example: MathExample, config: SamplingConfig,
    *, prompt_ids: tuple[int, ...] | None = None,
) -> tuple[CandidateRecord, dict[str, Any]]:
    """Generate exactly num_candidates; token slicing removes the input prompt.

    Chunk size is part of reproducibility: changing it can change sampled tokens.
    Errors abort rather than silently omitting an example or resampling a failure.
    """
    if not isinstance(config, SamplingConfig) or not isinstance(example, MathExample):
        raise TypeError("Expected SamplingConfig and MathExample")
    eos, pad = getattr(tokenizer, "eos_token_id", None), getattr(tokenizer, "pad_token_id", None)
    for name, value in (("EOS", eos), ("PAD", pad)):
        if type(value) is not int or value < 0:
            raise ValueError(f"Tokenizer requires one nonnegative {name} token ID")
    if prompt_ids is None:
        prompt_ids = encode_prompt(tokenizer, example.problem, config)
    elif (not isinstance(prompt_ids, tuple) or not prompt_ids
          or any(type(i) is not int or i < 0 for i in prompt_ids)):
        raise ValueError("prompt_ids must be a nonempty tuple of nonnegative token IDs")
    if len(prompt_ids) > config.max_prompt_tokens:
        raise ValueError("Prompt exceeds max_prompt_tokens")
    device = _device_and_context(model, prompt_ids, config)
    vocab_size = model.get_input_embeddings().weight.shape[0]
    if max(eos, pad) >= vocab_size:
        raise ValueError("EOS/PAD token exceeds the model vocabulary")
    inputs = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    mask = torch.ones_like(inputs)
    prompt_length = len(prompt_ids)
    attempts = []
    completions = []
    chunks = []
    for start in range(0, config.num_candidates, config.chunk_size):
        count = min(config.chunk_size, config.num_candidates - start)
        seed = chunk_seed(config.seed, example.example_id, start)
        settings = _generation_config(config, count, tokenizer)
        with _sampling_state(model, device, seed):
            sequences = model.generate(
                input_ids=inputs, attention_mask=mask, generation_config=settings,
                use_model_defaults=False, synced_gpus=False,
            )
        if (not isinstance(sequences, torch.Tensor) or sequences.ndim != 2
                or sequences.dtype != torch.long or sequences.shape[0] != count):
            raise ValueError("generate() returned an unexpected shape or token dtype")
        length = sequences.shape[1] - prompt_length
        if not 1 <= length <= config.max_new_tokens:
            raise ValueError("generate() returned an invalid completion length")
        if not torch.equal(sequences[:, :prompt_length], inputs.expand(count, -1)):
            raise ValueError("Generated sequences do not retain the exact prompt token prefix")
        chunks.append({"start": start, "count": count, "seed": seed})
        for row in sequences[:, prompt_length:].detach().cpu().tolist():
            if any(token < 0 or token >= vocab_size for token in row):
                raise ValueError("Generated token exceeds the model vocabulary")
            if eos in row:
                stop = row.index(eos) + 1
                if any(token != pad for token in row[stop:]):
                    raise ValueError("Non-padding tokens follow generated EOS")
                token_ids = row[:stop]
                reason = "eos"
            else:
                if len(row) != config.max_new_tokens:
                    raise ValueError("Generation stopped before EOS or the requested length limit")
                token_ids = row
                reason = "length"
            text = tokenizer.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            if not isinstance(text, str):
                raise TypeError("Tokenizer.decode must return a string")
            completions.append(text)  # Includes empty/unparseable answers.
            attempts.append({"index": len(attempts), "token_ids": token_ids,
                             "num_tokens": len(token_ids), "finish_reason": reason})
    record = CandidateRecord.from_example(example, completions)
    details = {"schema_version": 1, "example_id": example.example_id,
               "prompt_token_ids": list(prompt_ids), "prompt_length": prompt_length,
               "attempts": attempts, "chunks": chunks}
    return record, details
