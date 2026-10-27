"""Independent heldout views and immutable native fixture integration checks."""
import hashlib
import copy
import importlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from test_multi_pair_ablation import multi_camera_fixture, ablation


def heldout():
    return importlib.import_module('integrations.demo_pipeline.heldout_multiview')


def payloads(root):
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob('*') if path.is_file()}


def make_prior(tmp_path):
    source, binary, frozen, truth = multi_camera_fixture(tmp_path)
    old = tmp_path / 'multi_pair'
    ablation().experiment(frozen, old)
    return source, old, truth


def test_split_is_deterministic_and_disjoint():
    expected = dict(status='SPLIT_READY', pair=[3, 4], additional_views=[1, 2, 5, 6],
                    train_extra=[1, 5], heldout=[2, 6])
    assert heldout().split_views([6, 3, 2, 4, 5, 1], [3, 4]) == expected
    assert heldout().split_views([1, 2, 3, 4, 5, 6], [3, 4]) == expected
    assert set(expected['pair']).isdisjoint(expected['train_extra'])
    assert set(expected['heldout']).isdisjoint(expected['pair'] + expected['train_extra'])


@pytest.mark.parametrize('views', [[1, 2], [1, 2, 3]])
def test_insufficient_extra_views_are_explicit(views):
    result = heldout().split_views(views, [1, 2])
    assert result['status'] == 'INSUFFICIENT_EXTRA_VIEWS'
    assert result['additional_views'] == views[2:]


@pytest.mark.parametrize('views,pair', [
    ([1, 1, 2, 3], [1, 2]), ([True, 2, 3, 4], [True, 2]),
    ([1., 2, 3, 4], [1, 2]), (['1', 2, 3, 4], [1, 2]),
    ([1, 2, 3, 4], [1, 1]), ([1, 2, 3, 4], [1, 7]),
    ([1, 2, 3, 4], [1]), ([1, 2, 3, 4], [1, 2, 3]),
])
def test_invalid_split_identities_are_rejected(views, pair):
    with pytest.raises((ValueError, TypeError)):
        heldout().split_views(views, pair)


def test_fixture_experiment_preserves_inputs_and_point_identities(tmp_path):
    source, old, _ = make_prior(tmp_path)
    before_source, before_prior = payloads(source), payloads(old)
    output = tmp_path / 'heldout'
    result = heldout().experiment(old, output)
    assert payloads(source) == before_source
    assert payloads(old) == before_prior
    assert result['schema'] == 'heldout-multiview-result-v1'
    assert result['status'] == 'EXPERIMENT_COMPLETE_NOT_GEOMETRY_ACCEPTED'
    assert result['counts']['positive']['requested'] == 72
    assert result['counts']['negative_control']['requested'] == 72
    assert json.loads((output / 'RESULT.json').read_text(encoding='utf-8')) == result
    samples = json.loads((output / 'SAMPLES.json').read_text(encoding='utf-8'))['samples']
    assert len(samples) == 144
    for method in ('A', 'J'):
        good = [s['methods'][method]['heldout']['rms'] for s in samples
                if s['case_kind'] == 'positive'
                and s['methods'][method]['heldout']['status'] == 'EVALUATED']
        wrong = [s['methods'][method]['heldout']['rms'] for s in samples
                 if s['case_kind'] == 'negative_control'
                 and s['methods'][method]['heldout'] is not None
                 and s['methods'][method]['heldout']['status'] == 'EVALUATED']
        assert len(good) == 72
        assert len(wrong) > 0
        assert max(good) < 0.1
        assert np.median(wrong) > max(1., 100 * max(good))
    for kind in ('positive', 'negative_control'):
        rows = [s for s in samples if s['case_kind'] == kind]
        assert sorted((s['pair_index'], s['point_id']) for s in rows) == [
            (pair, pid) for pair in range(3) for pid in range(1, 25)]
    for sample in samples:
        split = sample['split']
        assert split['status'] == 'SPLIT_READY'
        assert len(split['train_extra']) == len(split['heldout']) == 2
        assert set(split['heldout']).isdisjoint(split['pair'] + split['train_extra'])
        assert sample['right_source_track_id'] == (sample['point_id']
            if sample['case_kind'] == 'positive' else
            ((sample['point_id'] - 1) // 6) * 6 + sample['point_id'] % 6 + 1)
        for bucket in ('pair', 'train_extra', 'heldout'):
            assert len(sample['observations'][bucket]) == len(split[bucket])
            assert all('source_track_id' in obs for obs in sample['observations'][bucket])
        assert {'A', 'J'}.issubset(sample['methods'])
    # No accepted geometry or native model/mesh is produced by this diagnostic.
    assert not any(p.suffix.lower() in {'.obj', '.stl', '.off', '.blend', '.step', '.stp'}
                   or p.name in {'images.bin', 'cameras.bin', 'points3D.bin'}
                   for p in output.rglob('*') if p.is_file())
    assert not list(output.rglob('*.ply'))
    positive = next(s for s in samples if s['case_kind'] == 'positive')
    original = copy.deepcopy(positive)
    first = heldout().evaluate_sample(positive)
    poisoned = copy.deepcopy(positive)
    for obs in poisoned['observations']['heldout']:
        obs['raw_pixels'] = [obs['raw_pixels'][0] + 60., obs['raw_pixels'][1] - 35.]
    second = heldout().evaluate_sample(poisoned)
    assert positive == original
    assert first['methods']['J']['status'] == second['methods']['J']['status'] == 'FITTED'
    assert np.array_equal(first['methods']['J']['xyz'], second['methods']['J']['xyz'])
    assert first['methods']['J']['training'] == second['methods']['J']['training']
    assert second['methods']['J']['heldout']['rms'] > 50.
    before_output = payloads(output)
    with pytest.raises(ValueError):
        heldout().experiment(old, output)
    assert payloads(output) == before_output


def test_prior_raw_identity_tamper_rejected_before_output(tmp_path):
    _, old, _ = make_prior(tmp_path)
    target = old / 'groups/p0_g0000_negative_control.json'
    row = json.loads(target.read_text(encoding='utf-8'))
    row['right_source_track_ids'][0] = row['left_source_track_ids'][0]
    target.write_text(json.dumps(row), encoding='utf-8')
    output = tmp_path / 'rejected'
    with pytest.raises(ValueError):
        heldout().experiment(old, output)
    assert not output.exists()


@pytest.mark.parametrize('relative', ['images/1.png', 'models/0/binary/cameras.bin'])
def test_native_source_drift_rejected_before_output(tmp_path, relative):
    source, old, _ = make_prior(tmp_path)
    target = source / relative
    target.write_bytes(target.read_bytes() + b'bound source drift')
    output = tmp_path / 'rejected'
    with pytest.raises(ValueError):
        heldout().experiment(old, output)
    assert not output.exists()
