"""Population presentation must preserve missing and rejected evidence."""
import json
from pathlib import Path

import numpy as np
from PIL import Image

from integrations.demo_pipeline.intermediate_population_report import (
    CONTINUING, NEW, _accepted, render_population,
)


def _fixture(tmp_path, count=240):
    frames = {}
    for index in range(3):
        path = tmp_path / f'raw{index}.png'
        Image.new('RGB', (200, 150), 'black').save(path)
        frames[index] = dict(path=str(path), width=200, height=150,
                             mask=np.ones((150, 200), dtype=np.uint8) if index != 1 else None,
                             original_mask_status='OK' if index != 1 else 'UNKNOWN_NOT_SEGMENTED')
    points = [dict(group_id=f'p{i}', source_raw_xy=[5.5+(i%20)*9, 5.5+(i//20)*10],
                   birth_kind='NEW_SEED' if i%2 == 0 else 'CONTINUING_OBSERVATION') for i in range(count)]
    plan = dict(source_input_index=0, target_input_index=1, source_decode_index=0,
                target_decode_index=2, decode_path=[0, 1, 2], points=points)
    return frames, dict(status='COMPARED', plan=dict(edges=[plan]), edges=[])


def test_hundreds_preserved_and_missing_never_passed(tmp_path):
    decoded, result = _fixture(tmp_path)
    output = tmp_path/'report'
    paths = render_population(output, result, decoded)
    evidence = json.loads((output/'population_evidence.json').read_text(encoding='utf-8'))['edges'][0]
    assert len(evidence['points']) == 240
    assert evidence['counts']['new_seed'] == evidence['counts']['continuing'] == 120
    assert evidence['counts']['measured'] == evidence['counts']['dense_passed'] == 0
    assert all(p['outcome'] == 'NOT_MEASURED' and not p['accepted_dense'] for p in evidence['points'])
    assert {'source_population_000.png', 'endpoint_comparison_000.png', 'lifecycle_000.png'} <= {p.name for p in paths}
    with Image.open(output/'source_population_000.png') as picture:
        colors = set(picture.getdata())
    assert tuple(bytes.fromhex(NEW[1:])) in colors
    assert tuple(bytes.fromhex(CONTINUING[1:])) in colors


def test_only_completed_fb_legs_draw_lifecycle():
    good = dict(forward=dict(to_decode_index=1, observed_raw_xy=[4, 5]),
                backward=dict(observed_raw_xy=[2, 3],status=[[1]],error=[[0.1]]), fb_error_px=.2)
    good['forward'].update(status=[[1]],error=[[0.1]])
    bad = dict(forward=dict(to_decode_index=2, observed_raw_xy=[6, 7]), backward={}, fb_error_px=.1)
    assert _accepted(dict(steps=[good, bad, good])) == [dict(decode_index=1, raw_xy=[4, 5])]
    assert _accepted(dict(steps=[dict(good, fb_error_px=1.01)])) == []


def test_json_script_escape_and_counts_from_actual_rows(tmp_path):
    decoded, result = _fixture(tmp_path, 1)
    gid = '</script><script>alert(1)</script>'
    result['plan']['edges'][0]['points'][0]['group_id'] = gid
    result['edges'] = [dict(summary=dict(dense_passed=999), rows=[dict(group_id=gid, outcome='BOTH_FAIL', direct=dict(status='REJECTED'), dense=dict(status='REJECTED', reasons=['failure']))])]
    output=tmp_path/'report';render_population(output,result,decoded)
    page=(output/'index.html').read_text(encoding='utf-8')
    assert gid not in page
    evidence=json.loads((output/'population_evidence.json').read_text(encoding='utf-8'))['edges'][0]
    assert evidence['points'][0]['group_id'] == gid
    assert evidence['counts']['dense_passed'] == 0
    assert evidence['points'][0]['measurement']['dense']['reasons'] == ['failure']


def test_empty_population_report(tmp_path):
    paths=render_population(tmp_path,dict(status='EMPTY',plan=dict(edges=[]),edges=[]),{})
    assert all(Path(p).exists() for p in paths)
    assert json.loads((tmp_path/'population_evidence.json').read_text(encoding='utf-8')) == dict(status='EMPTY',edges=[])


def test_kernel_summary_displayed_without_image_grid(tmp_path):
    decoded,result=_fixture(tmp_path,1)
    result['edges']=[dict(rows=[],summary=dict(coverage=dict(source_all=dict(occupied=3,total=12,fraction=.25)),by_birth_kind=dict(NEW_SEED=dict(evaluated=1)),dense_per_leg_survival=[dict(from_decode_index=0,to_decode_index=1,survived=0)],survival_histogram=[dict(verified_legs=0,count=1)]))]
    render_population(tmp_path/'report',result,decoded)
    page=(tmp_path/'report'/'index.html').read_text(encoding='utf-8')
    assert 'source_all' in page and '有效对象网格' in page and '逐段通过' in page
    assert 'source_image_grid_8x8' not in page


def test_failed_lk_status_never_lifecycle_dot():
    step=dict(forward=dict(to_decode_index=1,observed_raw_xy=[4,5],status=[[0]],error=[[.1]]),backward=dict(observed_raw_xy=[2,3],status=[[1]],error=[[.1]]),fb_error_px=.2)
    assert _accepted(dict(steps=[step])) == []
