import json
import random
import sys
from types import ModuleType

import pytest

from cat.data import assert_disjoint, load_math_hf, load_math_jsonl, parse_level, select_math_examples
from cat.testing.tiny import tiny_math_records


@pytest.mark.parametrize('value,expected', [('Level 1', 1), (' level 3 ', 3), ('5', 5), (2, 2), (None, None)])
def test_level_forms(value, expected):
    assert parse_level(value) == expected


@pytest.mark.parametrize('value', [True, 0, 6, 1.5, 'unknown', 'Level 6'])
def test_bad_level(value):
    with pytest.raises(ValueError):
        parse_level(value)


def test_filter_select_and_manifest():
    result = select_math_examples(tiny_math_records(), split='train', levels=[2, 3], max_examples=2,
                                  seed=7, source='fixture', revision='abc')
    manifest = result.manifest()
    assert manifest['input_count'] == 5
    assert manifest['eligible_count'] == 3
    assert manifest['selected_count'] == 2
    assert {x.level for x in result.examples} <= {2, 3}
    assert manifest['revision'] == 'abc'
    assert len(manifest['selection_sha256']) == 64
    assert 'problem' not in json.dumps(manifest['selected_records']).replace('problem_sha256', '')
    assert json.loads(json.dumps(manifest)) == manifest


def test_selection_is_repeatable_without_changing_global_random_state():
    state = random.getstate()
    first = select_math_examples(tiny_math_records(), split='train', seed=19)
    assert random.getstate() == state
    second = select_math_examples(tiny_math_records(), split='train', seed=19)
    third = select_math_examples(tiny_math_records(), split='train', seed=20)
    assert first == second
    assert first.examples != third.examples


def test_unshuffled_source_indices_survive_filtering():
    selection = select_math_examples(tiny_math_records(), split='test', levels=[2], shuffle=False)
    assert [e.source_index for e in selection.examples] == [2, 3]
    assert all(e.source_split == 'test' for e in selection.examples)


@pytest.mark.parametrize('kwargs', [
    {'max_examples': 6}, {'max_examples': 0}, {'max_examples': -1},
    {'seed': True}, {'levels': []}, {'levels': [True]}, {'levels': [6]},
    {'split': ''}, {'shuffle': 'yes'},
])
def test_selection_rejects_bad_requests(kwargs):
    options = {'split': 'train', **kwargs}
    with pytest.raises((ValueError, TypeError)):
        select_math_examples(tiny_math_records(), **options)


@pytest.mark.parametrize('rows', [[], [{'answer': '2'}], [{'problem': 'Question'}], ['not a mapping']])
def test_invalid_records(rows):
    with pytest.raises(ValueError):
        select_math_examples(rows, split='train')


def test_missing_level_errors_only_when_filtering():
    rows = [{'problem': 'q', 'answer': 'a'}]
    assert select_math_examples(rows, split='train').examples[0].level is None
    with pytest.raises(ValueError, match='level is missing'):
        select_math_examples(rows, split='train', levels=[1, 2, 3])


def test_target_modes_no_silent_fallback():
    example = select_math_examples(tiny_math_records(), split='train', shuffle=False).examples[0]
    assert example.target_text('answer') == '2'
    assert example.target_text('solution') == 'Adding gives 2.'
    missing = select_math_examples([{'problem': 'q', 'answer': 'a'}], split='train').examples[0]
    with pytest.raises(ValueError, match='solution target'):
        missing.target_text('solution')
    with pytest.raises(ValueError, match='target must'):
        example.target_text('unknown')


def test_jsonl_loading_and_errors(tmp_path):
    path = tmp_path/'fixture.jsonl'
    path.write_text('\n'.join(json.dumps(row) for row in tiny_math_records()) + '\n\n')
    selection = load_math_jsonl(path, split='train', shuffle=False)
    assert len(selection.examples) == 5
    assert selection.source == 'jsonl:fixture.jsonl'
    assert len(selection.source_fingerprint) == 64
    path.write_text('{}\nnot JSON\n')
    with pytest.raises(ValueError, match='fixture.jsonl:2: invalid JSON'):
        load_math_jsonl(path, split='train')


def test_overlap_detection_including_duplicate_within_split():
    rows = tiny_math_records()
    train = select_math_examples(rows[:2], split='train', shuffle=False)
    test = select_math_examples(rows[2:], split='test', shuffle=False)
    assert_disjoint(train, test)
    duplicate = select_math_examples([dict(rows[0], problem=' What  is 1 + 1? ')], split='test')
    with pytest.raises(ValueError, match='Duplicate problem'):
        assert_disjoint(train, duplicate)
    repeated = select_math_examples([rows[0], rows[0]], split='train')
    with pytest.raises(ValueError, match='Duplicate problem'):
        assert_disjoint(repeated)


def test_hf_loader_passes_split_revision_and_never_falls_back(monkeypatch):
    calls = []
    class FakeDataset(list):
        _fingerprint = 'fixture-fingerprint'
    def load_dataset(name, **kwargs):
        calls.append((name, kwargs))
        if kwargs['split'] == 'unavailable':
            raise ValueError('missing split')
        return FakeDataset(tiny_math_records())
    module = ModuleType('datasets')
    module.load_dataset = load_dataset
    monkeypatch.setitem(sys.modules, 'datasets', module)
    selection = load_math_hf(dataset_name='fixture/data', revision='a'*40, split='test',
                             cache_dir='/tmp/cache-fixture', max_examples=2)
    assert selection.source_fingerprint == 'fixture-fingerprint'
    assert calls == [('fixture/data', {'revision': 'a'*40, 'split': 'test', 'cache_dir': '/tmp/cache-fixture'})]
    with pytest.raises(ValueError, match='missing split'):
        load_math_hf(dataset_name='fixture/data', revision='a'*40, split='unavailable')
    assert len(calls) == 2


def test_hf_loader_requires_dependency_and_revision(monkeypatch):
    monkeypatch.setitem(sys.modules, 'datasets', None)
    with pytest.raises(ImportError, match=r'\[data\]'):
        load_math_hf(dataset_name='fixture/data', revision='a'*40, split='train')
    with pytest.raises(ValueError, match='revision'):
        load_math_hf(dataset_name='fixture/data', revision='', split='train')
