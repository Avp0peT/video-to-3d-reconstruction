from copy import deepcopy
from pathlib import Path

import pytest
from PIL import Image

from integrations.demo_pipeline.neighborhood_report import render_neighborhood


def fixture(tmp_path):
    decoded = {}
    for index in (0, 1):
        path = tmp_path/f'actual frame {index}.png'
        Image.new('RGB', (100, 80), (20+index, 30, 40)).save(path)
        decoded[index] = dict(path=str(path), width=100, height=80)
    polygon = [[10.5, 10.5], [42.5, 10.5], [42.5, 42.5], [10.5, 42.5]]
    record = dict(region_id='region0', source_input_index=27, source_decode_index=0,
                  target_decode_index=1, strict_status='REJECTED', strict_reasons=['cycle failed'],
                  cycle_error_px=3., eligibility_reason=None,
                  measurement=dict(source_polygon_raw4x2=polygon, source_center_raw_xy=[26.5, 26.5],
                                   target_polygon_raw4x2=[[x+2, y] for x, y in polygon], target_center_raw_xy=[28.5, 26.5]),
                  neighborhood=dict(status='NEIGHBOR_CONSISTENT_CANDIDATE', reasons=[],
                                    matrix_global_raw3x3=[[1., 0., 5.], [0., 1., 0.], [0., 0., 1.]]),
                  continuation=dict(status='NOT_RUN', legs=[], counts={}))
    return dict(profile={}, counts={}, records=[record]), decoded


def test_actual_local_images_and_distinct_prediction_preserve_raw_pixel_centers(tmp_path):
    result, decoded = fixture(tmp_path)
    before = deepcopy((result, decoded))
    files = render_neighborhood(tmp_path/'report', result, decoded)
    assert {p.name for p in files} == {'index.html', 'evidence_000.svg'}
    svg = (tmp_path/'report/evidence_000.svg').read_text(encoding='utf-8')
    assert '../actual%20frame%200.png' in svg and '../actual%20frame%201.png' in svg
    assert '12.5,10.5 44.5,10.5' in svg  # Actual +2, not prediction +5.
    assert '15.5,10.5 47.5,10.5' in svg
    assert '10.5,10.5 42.5,10.5' in svg
    assert 'stroke-dasharray="8 5"' in svg
    assert '邻域预测，仅用于比较，不是新测量' in svg
    assert (result, decoded) == before


def test_unknown_measurement_and_missing_image_do_not_create_positions(tmp_path):
    result, decoded = fixture(tmp_path)
    result['records'][0]['neighborhood'] = None
    decoded[1]['path'] = str(tmp_path/'missing.png')
    result['records'].append(dict(region_id='pending', strict_status='PENDING', measurement=None))
    render_neighborhood(tmp_path/'report', result, decoded)
    svg = (tmp_path/'report/evidence_000.svg').read_text(encoding='utf-8')
    assert 'data-kind="neighborhood-prediction"' not in svg
    assert 'stroke="#ffad45"' not in svg
    assert '实际图像或尺寸缺失：不画图像和区域' in svg
    page = (tmp_path/'report/index.html').read_text(encoding='utf-8')
    assert 'pending' in page and 'PENDING' in page and 'UNKNOWN' in page
    assert len(list((tmp_path/'report').glob('*.svg'))) == 1


def test_all_rows_remain_and_candidate_evidence_is_prioritized_deterministically(tmp_path):
    result, decoded = fixture(tmp_path)
    base = result['records'][0]
    result['records'] = []
    for index in range(11):
        row = deepcopy(base)
        row['region_id'] = f'unique-region-{index:02d}'
        row['neighborhood']['status'] = 'NEIGHBOR_CONSISTENT_CANDIDATE' if index in (9, 10) else 'UNKNOWN'
        result['records'].append(row)
    render_neighborhood(tmp_path/'report', result, decoded)
    page = (tmp_path/'report/index.html').read_text(encoding='utf-8')
    assert page.count('<tr>') == 12
    for row in result['records']:
        assert row['region_id'] in page
    assert len(list((tmp_path/'report').glob('*.svg'))) == 8
    assert 'unique-region-09' in (tmp_path/'report/evidence_000.svg').read_text(encoding='utf-8')
    assert 'unique-region-10' in (tmp_path/'report/evidence_001.svg').read_text(encoding='utf-8')
    assert 'unique-region-00' in (tmp_path/'report/evidence_002.svg').read_text(encoding='utf-8')


def test_html_and_svg_escape_raw_records_and_never_overwrite(tmp_path):
    result, decoded = fixture(tmp_path)
    attack = '</pre><script>bad()</script>'
    result['records'][0]['region_id'] = attack
    output = tmp_path/'report'
    files = render_neighborhood(output, result, decoded)
    for path in files:
        text = path.read_text(encoding='utf-8')
        assert attack not in text and '&lt;script&gt;' in text
    before = {p: p.read_bytes() for p in files}
    with pytest.raises(FileExistsError):
        render_neighborhood(output, result, decoded)
    assert {p: p.read_bytes() for p in files} == before


def test_budget_stops_before_writing_and_optional_keys_are_supported(tmp_path):
    result, decoded = fixture(tmp_path)
    def stop():
        raise TimeoutError('report deadline')
    with pytest.raises(TimeoutError, match='report deadline'):
        render_neighborhood(tmp_path/'stopped', result, decoded, budget=stop)
    assert not (tmp_path/'stopped').exists()
    files = render_neighborhood(tmp_path/'empty', dict(records=[{}]), {})
    assert [p.name for p in files] == ['index.html']
    assert 'UNKNOWN' in files[0].read_text(encoding='utf-8')


def test_existing_svg_prevents_any_artifact_write(tmp_path):
    result, decoded = fixture(tmp_path)
    output = tmp_path/'report'
    output.mkdir()
    (output/'evidence_000.svg').write_text('sealed', encoding='utf-8')
    with pytest.raises(FileExistsError):
        render_neighborhood(output, result, decoded)
    assert not (output/'index.html').exists()
    assert (output/'evidence_000.svg').read_text(encoding='utf-8') == 'sealed'
