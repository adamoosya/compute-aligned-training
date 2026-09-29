"""A byte tokenizer and tiny GRU causal model; no files or network required."""
from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn


class ByteChatTokenizer:
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2
    vocab_size = 259

    def apply_chat_template(self, conversation, *, tokenize=True, add_generation_prompt=False):
        if not tokenize:
            raise ValueError("The offline byte fixture only supports tokenize=True")
        if len(conversation) not in (1, 2) or conversation[0]["role"] != "user":
            raise ValueError("Expected one user and optionally one assistant")
        def encode(text):
            return [byte + 3 for byte in text.encode("utf-8")]
        ids = [self.bos_token_id] + encode("User: " + conversation[0]["content"] + "\nAssistant: ")
        if len(conversation) == 1:
            if not add_generation_prompt:
                raise ValueError("Prompt encoding requires add_generation_prompt=True")
            return ids
        if conversation[1]["role"] != "assistant" or add_generation_prompt:
            raise ValueError("Full training chat needs an assistant and no generation prompt")
        return ids + encode(conversation[1]["content"]) + [self.eos_token_id]


class TinyCausalLM(nn.Module):
    def __init__(self, vocab_size=259, hidden_size=16):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.recurrent = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        hidden, _ = self.recurrent(self.embedding(input_ids))
        return SimpleNamespace(logits=self.head(hidden))


def tiny_math_records():
    """Handwritten fixture examples; not a sample from the MATH benchmark."""
    return [
        {"problem": "What is 1 + 1?", "answer": "2", "solution": "Adding gives 2.", "level": "Level 1"},
        {"problem": "What is 2 + 3?", "answer": "5", "solution": "Adding gives 5.", "level": "Level 1"},
        {"problem": "What is 5 - 2?", "answer": "3", "solution": "Subtracting gives 3.", "level": "Level 2"},
        {"problem": "What is 6 / 2?", "answer": "3", "solution": "Dividing gives 3.", "level": "Level 2"},
        {"problem": "What is 2 * 5?", "answer": "10", "solution": "Multiplying gives 10.", "level": "Level 3"},
    ]
