from dataclasses import replace

import pytest
import torch

from cat.data import (CausalCollator, EncodedMathDataset, encode_math_example,
                      encode_math_selection, select_math_examples)
from cat.testing.tiny import ByteChatTokenizer, tiny_math_records


@pytest.fixture
def examples():
    return select_math_examples(tiny_math_records(), split='fixture', shuffle=False)


def test_prompt_masking_and_first_completion_boundary(examples):
    tokenizer = ByteChatTokenizer()
    encoded = encode_math_example(examples.examples[0], tokenizer, target='answer')
    assert encoded.labels[:encoded.prompt_tokens] == (-100,) * encoded.prompt_tokens
    assert encoded.labels[encoded.prompt_tokens] == ord('2') + 3
    assert encoded.labels[-1] == tokenizer.eos_token_id
    assert encoded.target_tokens == 2
    assert not encoded.truncated


def test_full_solution_is_used_without_adding_new_text(examples):
    encoded = encode_math_example(examples.examples[0], ByteChatTokenizer(), target='solution')
    content = bytes(x - 3 for x in encoded.labels[encoded.prompt_tokens:-1]).decode()
    assert content == 'Adding gives 2.'


def test_padding_keeps_real_eos_when_pad_equals_eos(examples):
    tokenizer = ByteChatTokenizer()
    first = encode_math_example(examples.examples[0], tokenizer, target='answer')
    last = encode_math_example(examples.examples[-1], tokenizer, target='answer')
    batch = CausalCollator(tokenizer.eos_token_id)([first, last])
    assert batch['labels'][0, len(first.input_ids) - 1] == tokenizer.eos_token_id
    assert (batch['labels'][0, len(first.input_ids):] == -100).all()
    assert (batch['attention_mask'][0, len(first.input_ids):] == 0).all()
    assert batch['attention_mask'][0, len(first.input_ids) - 1] == 1


def test_overflow_is_explicit_and_reported(examples):
    tokenizer = ByteChatTokenizer()
    full = encode_math_example(examples.examples[0], tokenizer, target='solution')
    with pytest.raises(ValueError, match='exceed max_length'):
        encode_math_example(examples.examples[0], tokenizer, target='solution', max_length=len(full.input_ids)-2)
    truncated = encode_math_example(examples.examples[0], tokenizer, target='solution',
                                    max_length=len(full.input_ids)-2, overflow='truncate_target')
    assert truncated.truncated
    assert truncated.input_ids == full.input_ids[:-2]
    assert truncated.input_ids[-1] != tokenizer.eos_token_id
    assert EncodedMathDataset([truncated]).manifest()['truncated_count'] == 1
    with pytest.raises(ValueError, match='prompt fills'):
        encode_math_example(examples.examples[0], tokenizer, target='answer', max_length=full.prompt_tokens,
                            overflow='truncate_target')


def test_prefix_mismatch_rejected(examples):
    class BadTokenizer(ByteChatTokenizer):
        def apply_chat_template(self, conversation, **kwargs):
            ids = super().apply_chat_template(conversation, **kwargs)
            if len(conversation) == 2:
                ids[0] = 33
            return ids
    with pytest.raises(ValueError, match='changes the prompt prefix'):
        encode_math_example(examples.examples[0], BadTokenizer(), target='answer')


@pytest.mark.parametrize('kwargs', [
    {'max_length': 1}, {'max_length': True}, {'overflow': 'silent'},
    {'prompt_template': 'missing problem'}, {'target': 'wrong'},
])
def test_encoding_validation(examples, kwargs):
    with pytest.raises(ValueError):
        encode_math_example(examples.examples[0], ByteChatTokenizer(), **{'target': 'answer', **kwargs})


def test_dataset_manifest_and_no_mutation(examples):
    dataset = encode_math_selection(examples, ByteChatTokenizer(), target='answer')
    first = dataset[0]
    before = first.input_ids
    batch = CausalCollator(0)([dataset[0], dataset[-1]])
    batch['input_ids'][0, 0] = 0
    assert first.input_ids == before
    assert dataset.manifest()['count'] == 5
    assert dataset.manifest()['target_tokens'] == 11


def test_invalid_encoded_records_and_empty_batches(examples):
    encoded = encode_math_example(examples.examples[0], ByteChatTokenizer(), target='answer')
    with pytest.raises(ValueError, match='prompt labels'):
        replace(encoded, labels=encoded.input_ids).validate()
    with pytest.raises(ValueError, match='nonzero length'):
        replace(encoded, labels=(-100,)).validate()
    with pytest.raises(ValueError, match='original_tokens'):
        replace(encoded, original_tokens=1).validate()
    with pytest.raises(ValueError):
        EncodedMathDataset([])
    with pytest.raises(ValueError):
        CausalCollator(0)([])
    with pytest.raises(ValueError):
        CausalCollator(-1)
