"""TEST_DATA: object-neutral entry points; no native reconstruction or GPU use."""
import json
import sys
from types import SimpleNamespace

import pytest
from PIL import Image

from integrations.video_masks import region_comparison as comparison
from integrations.video_masks import region_experiment as experiment
from integrations.video_masks import region_report as report


@pytest.mark.parametrize('count', [2, 7, 64])
def test_prepare_uses_actual_endpoint_count_and_keeps_native_budget(tmp_path, monkeypatch, count):
    from integrations.video_masks import reconstruction_input
    source = tmp_path / 'source'
    masks = source / 'final_masks'
    masks.mkdir(parents=True)
    decode = tmp_path / 'decode'
    decode.mkdir()
    sfm = tmp_path / 'sfm'
    sfm.mkdir()
    selected, rows, decoded, mapping = [], [], [], []
    for i in range(count):
        image = source / f'{i}.png'
        image.write_bytes(f'TEST_DATA image {i}'.encode())
        mask = masks / f'{i}.png'
        mask.write_bytes(f'TEST_DATA mask {i}'.encode())
        digest = experiment.sha(image)
        selected.append(dict(decode_index=i, input_index=i, frame_id=f'frame{i}',
            source_image=str(image), source_image_sha256=digest, width=16, height=16))
        rows.append(dict(decode_index=i, placeholder=False, path=str(mask), sha256=experiment.sha(mask)))
        decoded.append(dict(decode_index=i, sha256=digest))
        mapping.append(dict(input_index=i, canonical_image_sha256=digest))
    experiment.save(source / 'MANIFEST.json', dict(rows=rows))
    experiment.save(source / 'START_BINDINGS.json', dict(source_run=str(decode)))
    experiment.save(decode / 'DECODE.json', dict(frames=decoded))
    experiment.save(sfm / 'run.json', {})
    (sfm / 'database.db').write_bytes(b'TEST_DATA database')
    (sfm / 'frames_map.jsonl').write_text('\n'.join(json.dumps(row) for row in mapping), encoding='utf8')
    monkeypatch.setattr(reconstruction_input, '_collect',
                        lambda _: (source, source, {}, {}, selected, {}))
    plan = experiment.prepare(source, sfm, source, tmp_path / 'output')
    assert plan['endpoint_count'] == count
    assert plan['same_factors'][0] == f'{count} original canonical keyframes'
    assert plan['schema'] == 'glove-region-plan-v1'
    assert 'object-agnostic' in plan['object_scope']
    assert plan['budgets'] == dict(tracking_seconds=600, geometry_seconds=600, job_commit_bytes=4 << 30)


@pytest.mark.parametrize('count', [1, 65])
def test_endpoint_bounds_are_unchanged(tmp_path, monkeypatch, count):
    from integrations.video_masks import reconstruction_input
    monkeypatch.setattr(reconstruction_input, '_collect',
                        lambda _: (tmp_path, tmp_path, {}, {}, [{}] * count, {}))
    with pytest.raises(ValueError, match='2..64'):
        experiment.prepare(tmp_path, tmp_path, tmp_path, tmp_path / 'output')
    assert not (tmp_path / 'output').exists()


def _report_fixture(tmp_path, indices=(8, 33, 56)):
    (tmp_path / 'tracking').mkdir()
    image = tmp_path / 'image.png'
    mask = tmp_path / 'mask.png'
    Image.new('RGB', (16, 16), (60, 90, 120)).save(image)
    Image.new('L', (16, 16), 255).save(mask)
    experiment.save(tmp_path / 'PLAN.json', {})
    experiment.save(tmp_path / 'tracking' / 'REGION_TRACKS.json',
        dict(schema='glove-region-tracks-v1', frames=[dict(input_index=i, decode_index=i,
            image_path=str(image), mask_path=str(mask), width=16, height=16,
            observations=[]) for i in indices]))


def test_report_does_not_inherit_unbound_historical_body_parts(tmp_path):
    _report_fixture(tmp_path)
    old = dict(visual_back_group=[8, 33, 56],
        globalmetrics=dict(sift_plus_regions=dict(labelled_back_registered=3)),
        models=[dict(model_id=0, previously_labelled_back_views=[8, 33, 56])])
    experiment.save(tmp_path / 'COMPARISON.json', old)
    before = (tmp_path / 'COMPARISON.json').read_bytes()
    result = report.render(tmp_path)
    assert (tmp_path / 'COMPARISON.json').read_bytes() == before
    assert result['visual_back_group'] is None
    assert result['globalmetrics']['sift_plus_regions']['labelled_back_registered'] is None
    assert result['models'][0]['previously_labelled_back_views'] is None
    assert result['region_annotation_scope']['status'] == 'NOT_EVALUATED'
    assert all(row['native_associated_observations'] is None for row in result['frames'])
    page = (tmp_path / 'index.html').read_text(encoding='utf8')
    assert '手套' not in page and '背面' not in page
    assert 'id="frame8" open' not in page
    assert '<h1>目标：' in page
    assert result['schema'] == 'glove-region-report-v1'


def test_report_pair_sampling_is_uniform_without_special_frame_priority(tmp_path):
    _report_fixture(tmp_path, range(40))
    result = report.render(tmp_path)
    adjacent = [[i, i + 1] for i in range(39)]
    expected = [adjacent[i * 38 // 23] for i in range(24)]
    assert [pair['input_indices'] for pair in result['pairs']] == expected


def test_report_ambiguous_geometry_refuses_then_honors_explicit_selection(tmp_path):
    _report_fixture(tmp_path)
    for name in ('geometry', 'geometry02'):
        (tmp_path / name).mkdir()
        experiment.save(tmp_path / name / 'RESULT.json', dict(models=[]))
    with pytest.raises(ValueError, match='ambiguous'):
        report.render(tmp_path)
    assert not (tmp_path / 'OUTPUT_REPORT.json').exists()
    assert report.render(tmp_path, source_geometry='geometry')['source_geometry'] == 'geometry'


def test_report_uses_comparison_bound_geometry_selection(tmp_path):
    _report_fixture(tmp_path)
    for name in ('geometry', 'geometry02'):
        (tmp_path / name).mkdir()
        experiment.save(tmp_path / name / 'RESULT.json', dict(models=[]))
    selected = tmp_path / 'geometry' / 'RESULT.json'
    experiment.save(tmp_path / 'COMPARISON.json', dict(source_geometry='geometry',
        source_bindings={str(selected): experiment.sha(selected)}))
    assert report.render(tmp_path)['source_geometry'] == 'geometry'


@pytest.mark.parametrize('binding', ['missing', 'drifted'])
def test_report_refuses_unbound_or_drifted_comparison_selection(tmp_path, binding):
    _report_fixture(tmp_path)
    (tmp_path / 'geometry').mkdir()
    selected = tmp_path / 'geometry' / 'RESULT.json'
    experiment.save(selected, dict(models=[]))
    refs = {} if binding == 'missing' else {str(selected): '0' * 64}
    experiment.save(tmp_path / 'COMPARISON.json', dict(source_geometry='geometry', source_bindings=refs))
    with pytest.raises(ValueError, match='binding missing or drifted'):
        report.render(tmp_path)


def test_comparison_never_guesses_object_part_from_original_frame_indices(tmp_path, monkeypatch):
    from integrations.demo_pipeline import camera_domain_gate
    baseline = tmp_path / 'baseline'
    baseline.mkdir()
    geometry = tmp_path / 'geometry'
    geometry.mkdir()
    control = tmp_path / 'sift_control'
    control.mkdir()
    (tmp_path / 'tracking').mkdir()
    model = dict(binary_path=str(tmp_path / 'TEST_DATA_model'),
                 ply_path=str(tmp_path / 'missing.ply'), model_id=0)
    experiment.save(tmp_path / 'PLAN.json', dict(baseline_sfm=str(baseline)))
    experiment.save(baseline / 'run.json', dict(models=[model]))
    (baseline / 'frames_map.jsonl').write_text(json.dumps(dict(database_image_id=1, input_index=8)), encoding='utf8')
    experiment.save(tmp_path / 'tracking' / 'REGION_TRACKS.json', dict(frames=[dict(input_index=8)]))
    experiment.save(geometry / 'RESULT.json', dict(models=[model], pairs=[], unique_registered_input_indices=[8]))
    experiment.save(geometry / 'PLAN.json', dict(frames=[dict(image_id=1, row_by_track={})]))
    experiment.save(control / 'RESULT.json', dict(models=[model]))
    rec = SimpleNamespace(reg_image_ids=lambda: [1], points3D={}, num_points3D=lambda: 0)
    monkeypatch.setitem(sys.modules, 'pycolmap', SimpleNamespace(Reconstruction=lambda _: rec))
    monkeypatch.setattr(camera_domain_gate, 'audit_camera_domain',
                        lambda _: dict(status='CAMERA_DOMAIN_REJECTED'))
    result = comparison.analyse(tmp_path, geometry_name='geometry')
    assert result['visual_back_group'] is None
    assert result['region_annotation_scope']['status'] == 'NOT_EVALUATED'
    assert result['source_geometry'] == 'geometry'
    assert all(m['previously_labelled_back_views'] is None for m in result['models'])
    for key in ('historical_sift', 'reverified_sift', 'sift_plus_regions'):
        assert result['globalmetrics'][key]['unique_registered_views'] == 1
        assert result['globalmetrics'][key]['labelled_back_registered'] is None
    assert all('glove' not in note.lower() for note in result['notes'])
