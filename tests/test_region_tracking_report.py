from copy import deepcopy

import pytest
from PIL import Image

from integrations.demo_pipeline import region_tracking_report as report


def _fixture(tmp_path):
    decoded = {}
    for index in range(3):
        path = tmp_path/f'actual_{index}.png'
        Image.new('RGB', (160, 120), (17+index, 29, 41)).save(path)
        decoded[index] = dict(path=str(path), width=160, height=120,
                              time_seconds=index/30, original_mask_status='OK' if index == 0 else 'UNKNOWN_NOT_SEGMENTED')
    points = [dict(point_id='center', grid_ordinal=40, offset_xy=[0, 0], source_raw_xy=[80.5, 60.5],
                   is_center=True, status='TRACKLET_2D', verified_legs=2, reason='PATH_COMPLETE',
                   observations=[dict(decode_index=i, raw_xy=[80.5+i, 60.5+2*i]) for i in range(3)],
                   steps=[dict(fb_error_px=.25, forward=dict(error=2.), backward=dict(error=3.),
                               ncc=dict(status='EVALUATED', value=.95))]),
              dict(point_id='failed', grid_ordinal=41, offset_xy=[4, 0], source_raw_xy=[84.5, 60.5],
                   is_center=False, status='REJECTED', verified_legs=0, reason='LK_STATUS_FAILED',
                   observations=[dict(decode_index=0, raw_xy=[84.5, 60.5])], steps=[]),
              dict(point_id='outside-mask', grid_ordinal=0, offset_xy=[-16, -16], source_raw_xy=[64.5, 44.5],
                   is_center=False, status='EXCLUDED_SOURCE_MASK', verified_legs=0,
                   reason='SOURCE_MASK_NOT_FOREGROUND', observations=[], steps=[])]
    region = dict(region_id='region0', tracklet_id='track0', source_input_index=4,
                  source_decode_index=0, decode_path=[0, 1, 2], center_raw_xy=[80.5, 60.5],
                  source_bbox_raw_xy=[64.5, 44.5, 96.5, 76.5], points=points,
                  center_baseline=dict(total=1, excluded=0, evaluated=1, tracklets_2d=1, tracking_rejected=0),
                  expanded_grid=dict(total=2, excluded=1, evaluated=1, tracklets_2d=0, tracking_rejected=1),
                  affine_diagnostics=[dict(from_decode_index=0, to_decode_index=1, status='BLOCKED',
                      reason='INSUFFICIENT_SUPPORT', train_ids=[], holdout_ids=[], matrix=None)])
    return dict(status='REGION_MEASUREMENTS_COMPLETE', regions=[region], counts={}, flow_calls=4,
                geometry_accepted=False, new_model_points=0), decoded


def test_actual_region_panels_marks_statistics_and_input_preservation(tmp_path):
    result, decoded = _fixture(tmp_path)
    before = deepcopy(result)
    output = tmp_path/'report'
    files = report.render_regions(output, result, decoded)
    assert {p.name for p in files} == {'region_000.png', 'index.html'}
    with Image.open(output/'region_000.png') as image:
        colors = set(image.getdata())
        assert (17, 29, 41) in colors  # Actual source pixels survive the panel.
        assert (53, 216, 121) in colors
        assert (255, 91, 98) in colors
        assert (155, 164, 175) in colors
        assert image.width == 1020
    page = (output/'index.html').read_text(encoding='utf-8')
    assert 'UNKNOWN_NOT_SEGMENTED' in page and 'LK_STATUS_FAILED' in page
    assert '源域排除不计作跟踪失败' in page
    assert '新增网格2D连续点 0/1' in page and '中心2D连续点 1/1' in page
    assert '0.25' in page and '0.95' in page and 'INSUFFICIENT_SUPPORT' in page
    assert result == before


def test_missing_observations_never_use_center_displacement(tmp_path):
    result, decoded = _fixture(tmp_path)
    points = result['regions'][0]['points']
    assert report._observations(points[1]) == {0: [84.5, 60.5]}
    assert report._observations(points[2]) == {}
    report.render_regions(tmp_path/'report', result, decoded)
    page = (tmp_path/'report/index.html').read_text(encoding='utf-8')
    assert '未复制中心位移' in page and '未补出缺失点' in page
    # A later stop preserves a genuinely measured two-leg prefix as green.
    points[0]['reason'] = 'LATER_NCC_FAILED'
    assert report._color(points[0]) == report.GREEN


def test_html_escaping_and_no_actual_frame_placeholder(tmp_path):
    result, _ = _fixture(tmp_path)
    attack = '</td><script>attack()</script>'
    result['regions'][0]['region_id'] = attack
    result['regions'][0]['points'][0]['reason'] = attack
    files = report.render_regions(tmp_path/'report', result, {})
    assert [p.name for p in files] == ['index.html']
    page = files[0].read_text(encoding='utf-8')
    assert attack not in page and '&lt;script&gt;' in page
    assert '实际帧不可用' in page


def test_empty_result_is_valid_without_placeholder(tmp_path):
    files = report.render_regions(tmp_path/'report', dict(regions=[]), {})
    assert [p.name for p in files] == ['index.html']
    assert '没有区域记录' in files[0].read_text(encoding='utf-8')


def test_budget_exception_propagates_before_writing_files(tmp_path):
    result, decoded = _fixture(tmp_path)
    def exhausted():
        raise TimeoutError('region deadline')
    with pytest.raises(TimeoutError, match='region deadline'):
        report.render_regions(tmp_path/'report', result, decoded, budget=exhausted)
    assert not list((tmp_path/'report').iterdir())


def test_output_files_never_overwrite_existing_evidence(tmp_path):
    output = tmp_path/'report'
    output.mkdir()
    (output/'index.html').write_text('sealed', encoding='utf-8')
    with pytest.raises(FileExistsError):
        report.render_regions(output, {}, {})
    assert (output/'index.html').read_text(encoding='utf-8') == 'sealed'


def test_actual_frame_dimensions_are_checked(tmp_path):
    result, decoded = _fixture(tmp_path)
    decoded[0]['width'] += 1
    with pytest.raises(ValueError, match='REGION_REPORT_DECODED_IMAGE_SIZE_MISMATCH'):
        report.render_regions(tmp_path/'report', result, decoded)


def test_recursive_layers_parentage_and_correlated_samples_are_explicit(tmp_path):
    result, decoded = _fixture(tmp_path)
    root = result['regions'][0]
    root.update(depth=0, parent_region_id=None, root_tracklet_id='track0')
    child = deepcopy(root)
    child.update(region_id='child<&>', depth=1, parent_region_id='region0')
    result['regions'].append(child)
    report.render_regions(tmp_path/'report', result, decoded)
    page = (tmp_path/'report/index.html').read_text(encoding='utf-8')
    assert '逐层2D摘要' in page and '父区域 region0' in page
    assert 'child&lt;&amp;&gt;' in page and 'depth 1' in page
    assert '相关性' in page and '可能重复采样' in page and '不同图像坐标系' in page
