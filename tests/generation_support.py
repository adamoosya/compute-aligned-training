"""Small deterministic test doubles. They are not language-model performance fixtures."""
from types import SimpleNamespace

import torch
from torch import nn


class FakeGenerationConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    def to_dict(self):
        return dict(self.__dict__)


class FakeTokenizer:
    chat_template = "unit_test_template"
    eos_token_id = 2
    pad_token_id = 0
    bos_token_id = 1
    eos_token = "[EOS]"
    padding_side = "right"
    def __init__(self):
        self.messages = []
        self.decoded = []
        self.ids = [1, 4, 5]
    def __len__(self):
        return 12
    def apply_chat_template(self, messages, **kwargs):
        self.messages.append(messages)
        assert kwargs == {"tokenize": True, "add_generation_prompt": True}
        return self.ids
    def decode(self, tokens, **kwargs):
        assert kwargs == {"skip_special_tokens": True, "clean_up_tokenization_spaces": False}
        self.decoded.append(list(tokens))
        return " ".join(str(i) for i in tokens if i not in {0, 1, 2, 4, 5})


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(12, 3)
        self.drop = nn.Dropout(0.1)
        self.config = SimpleNamespace(is_encoder_decoder=False, max_position_embeddings=128, use_cache=False)
        self.calls = []
        self.behavior = "normal"
    def get_input_embeddings(self):
        return self.emb
    def generate(self, input_ids, attention_mask, generation_config, **kwargs):
        assert not self.training and not torch.is_grad_enabled()
        assert kwargs == {"use_model_defaults": False, "synced_gpus": False}
        assert torch.equal(attention_mask, torch.ones_like(input_ids))
        self.calls.append(generation_config.to_dict())
        if self.behavior == "fail":
            torch.rand(3)
            raise RuntimeError("simulated generation error")
        if self.behavior == "fail_second" and len(self.calls) > 1:
            raise RuntimeError("simulated generation error")
        count = generation_config.num_return_sequences
        length = generation_config.max_new_tokens
        out = torch.randint(6, 12, (count, length))
        if self.behavior == "empty":
            out[:] = generation_config.pad_token_id
            out[:, 0] = generation_config.eos_token_id
        elif self.behavior == "eos":
            out[:, -1] = generation_config.eos_token_id
        elif self.behavior == "bad_after_eos":
            out[:, 0] = generation_config.eos_token_id
        elif self.behavior == "short":
            out = out[:, :-1]
        elif self.behavior == "long":
            out = torch.cat([out, out[:, -1:]], dim=1)
        elif self.behavior == "token_range":
            out[:, -1] = 999
        result = torch.cat([input_ids.expand(count, -1), out], dim=1)
        if self.behavior == "prefix":
            result[:, 0] = 9
        elif self.behavior == "count":
            result = result[:-1]
        elif self.behavior == "float":
            result = result.float()
        return result
