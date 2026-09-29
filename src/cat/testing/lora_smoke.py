"""Offline integration test using a tiny Mistral architecture, not pretrained 7B weights."""
from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile

# Set before importing the optional Hub libraries. The smoke test never uses a repo ID.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import torch
from torch.utils.data import DataLoader

from cat.data import CausalCollator, encode_math_selection, select_math_examples
from cat.models.hf import HFModelConfig, LoRASettings, load_lora_model, save_lora_adapter
from cat.training import SFTObjective, evaluate_sft_loss, train_sft_epoch


_CHAT = """{{ bos_token }}{% for message in messages %}{% if message['role'] == 'user' %} [USER] {{ message['content'] }} [ASSISTANT]{% elif message['role'] == 'assistant' %} {{ message['content'] }} {{ eos_token }}{% endif %}{% endfor %}"""
_RECORDS = [
    {"problem": "1 + 1 =", "answer": "2", "solution": "The sum is 2 .", "level": 1},
    {"problem": "2 + 1 =", "answer": "3", "solution": "The sum is 3 .", "level": 1},
    {"problem": "2 + 2 =", "answer": "4", "solution": "The sum is 4 .", "level": 1},
    {"problem": "3 + 2 =", "answer": "5", "solution": "The sum is 5 .", "level": 1},
]


def _local_base(path: Path):
    try:
        from transformers import MistralConfig, MistralForCausalLM, PreTrainedTokenizerFast
        from tokenizers import Tokenizer, models, pre_tokenizers
        import peft
    except ImportError as error:
        raise ImportError("Install the real backend before this smoke test: python -m pip install -e '.[hf]'") from error
    words = ["[UNK]", "[BOS]", "[EOS]", "[PAD]", "[USER]", "[ASSISTANT]",
             "Problem:", "1", "2", "3", "4", "5", "+", "=", "The", "sum", "is", "."]
    engine = Tokenizer(models.WordLevel({word: i for i, word in enumerate(words)}, unk_token="[UNK]"))
    engine.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=engine, unk_token="[UNK]", bos_token="[BOS]",
        eos_token="[EOS]", pad_token="[PAD]",
        additional_special_tokens=["[USER]", "[ASSISTANT]"],
    )
    tokenizer.chat_template = _CHAT
    torch.manual_seed(123)
    config = MistralConfig(
        vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=128, sliding_window=None, attention_dropout=0.0,
        bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id, use_cache=False,
    )
    config._attn_implementation = "eager"
    model = MistralForCausalLM(config)
    path.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(str(path), safe_serialization=True)
    tokenizer.save_pretrained(str(path))


def _params(model, trainable):
    return {name: p.detach().cpu().clone() for name, p in model.named_parameters()
            if bool(p.requires_grad) == trainable}


def _assert_same(left, right):
    if left.keys() != right.keys():
        raise AssertionError("Parameter keys changed")
    for name in left:
        torch.testing.assert_close(left[name], right[name], rtol=0, atol=0)


def run_lora_smoke(output_parent: str | Path = "outputs") -> Path:
    torch.set_num_threads(1)
    parent = Path(output_parent)
    parent.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="lora-smoke-", dir=parent)).resolve()
    base = root / "tiny-base"
    _local_base(base)
    model_config = HFModelConfig(name=str(base), device="cpu", precision="fp32", attention="eager")
    lora = LoRASettings(r=4, alpha=8, target_modules=("q_proj", "k_proj", "v_proj", "o_proj"))
    selection = select_math_examples(_RECORDS, split="smoke_train", shuffle=False)
    model, tokenizer = load_lora_model(model_config, lora)
    encoded = encode_math_selection(selection, tokenizer, target="answer", max_length=128)
    loader = DataLoader(encoded, batch_size=2, collate_fn=CausalCollator(tokenizer.pad_token_id))
    warmup_objective = SFTObjective("ce", reduction="sequence_mean")
    frozen = _params(model, False)
    initial = _params(model, True)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01, weight_decay=0)
    for _ in range(5):
        train_sft_epoch(model, loader, optimizer, warmup_objective, accumulation_steps=2, max_grad_norm=1)
    _assert_same(frozen, _params(model, False))
    if not any(not torch.equal(p, _params(model, True)[name]) for name, p in initial.items()):
        raise AssertionError("Warmup did not update the adapters")
    warmup = save_lora_adapter(model, tokenizer, root / "warmup", model_config=model_config,
                               lora=lora, provenance={"kind": "smoke", "stage": "warmup"})
    shared_state = _params(model, True)
    del model, optimizer
    results = []
    for strategy, n, k in [("ce", 1, None), ("passn", 4, None), ("majority", 8, 3)]:
        model, tokenizer = load_lora_model(model_config, lora, initial_adapter=warmup)
        _assert_same(shared_state, _params(model, True))
        frozen = _params(model, False)
        objective = SFTObjective(strategy, n, k, "sequence_mean")
        before = evaluate_sft_loss(model, loader, objective).loss
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.005, weight_decay=0)
        for _ in range(4):
            train_sft_epoch(model, loader, optimizer, objective, accumulation_steps=2, max_grad_norm=1)
        after = evaluate_sft_loss(model, loader, objective).loss
        _assert_same(frozen, _params(model, False))
        trained = _params(model, True)
        if not any(not torch.equal(shared_state[name], trained[name]) for name in trained):
            raise AssertionError(f"{strategy}: adapters did not update")
        batch = next(iter(loader))
        model.eval()
        with torch.no_grad():
            expected_logits = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask']).logits
        saved = save_lora_adapter(model, tokenizer, root / strategy, model_config=model_config,
                                   lora=lora, provenance={"kind": "smoke", "strategy": strategy})
        loaded, loaded_tokenizer = load_lora_model(model_config, lora, initial_adapter=saved)
        loaded.eval()
        with torch.no_grad():
            actual_logits = loaded(input_ids=batch['input_ids'], attention_mask=batch['attention_mask']).logits
        torch.testing.assert_close(actual_logits, expected_logits, rtol=1e-5, atol=1e-6)
        if not torch.isfinite(actual_logits).all():
            raise AssertionError("Reloaded logits are not finite")
        results.append({"strategy": strategy, "loss_before": before, "loss_after": after,
                        "adapter_updates": True, "base_frozen": True, "save_reload": True})
        print(f"{strategy:8s} loss {before:.6f} -> {after:.6f}; adapters updated; base frozen; save/reload OK")
        del model, loaded, optimizer
    (root / "summary.json").write_text(json.dumps({"kind": "offline_lora_smoke", "results": results}, indent=2) + '\n')
    print("LoRA smoke test passed (tiny random Mistral; CPU; no model or dataset downloads).")
    print(f"Local smoke outputs: {root}")
    return root


if __name__ == "__main__":
    run_lora_smoke()
