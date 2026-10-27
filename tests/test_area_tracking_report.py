from copy import deepcopy
import builtins

import pytest
from PIL import Image

from integrations.demo_pipeline import area_tracking_report as report


def fixture(tmp_path):
    template = [[20+3*x+2*y for x in range(33)] for y in range(33)]
    decoded = {}
    for index in (0, 1):
        image = Image.new('RGB', (200, 160), (17, 29, 41))
        for y in range(33):
            for x in range(33):
                value = template[y][x]+10*index
                image.putpixel((64+x+2*index, 44+y+index), (value,)*3)
        path = tmp_path/f'actual_{index}.png'
        image.save(path)
        decoded[index] = dict(path=str(path), width=200, height=160, time_seconds=index/30,
                              original_mask_status='OK' if index == 0 else 'UNKNOWN_NOT_SEGMENTED')
    all_valid = [[255]*33 for _ in range(33)]
    train = [[255 if y < 16 else 0 for _ in range(33)] for y in range(33)]
    held = [[255 if y >= 20 else 0 for _ in range(33)] for y in range(33)]
    pixels = dict(template_uint8=template, template_mask_uint8=deepcopy(all_valid),
        train_mask_uint8=train, holdout_mask_uint8=held,
        warped_search_gray=[[float(v+10) for v in row] for row in template],
        valid_mask_uint8=deepcopy(all_valid),search_foreground_valid_mask_uint8=deepcopy(all_valid),
        residual_after_train_photometric=[[0.]*33 for _ in range(33)])
    metrics = dict(count=400,valid_count=400,coverage=1.,ncc=1.,rmse=0.,normalized_rmse=0.)
    alignment = dict(status='ACCEPTED',pixel_evidence=pixels,train=metrics,holdout=deepcopy(metrics),
        photometric=dict(scale=1.,offset=-10.,fit_scope='TRAIN_ONLY'),
        native_calls=dict(matchTemplate=1,findTransformECCWithMask=1))
    forward = dict(alignment=alignment,template_roi=dict(left=64,top=44,width=33,height=33),
        search_roi=dict(left=40,top=20,width=97,height=97),matrix_global_raw3x3=[[1.,0.,2.],[0.,1.,1.],[0.,0.,1.]],
        polygon_raw4x2=[[66.5,45.5],[98.5,45.5],[98.5,77.5],[66.5,77.5]],
        template_mask_status='OK',search_mask_status='UNKNOWN_NOT_SEGMENTED')
    region = dict(region_id='area0',depth=0,source_input_index=4,source_decode_index=0,decode_path=[0,1],
        source_center_raw_xy=[80.5,60.5],template_roi=dict(left=64,top=44,width=33,height=33),
        source_polygon_raw4x2=[[64.5,44.5],[96.5,44.5],[96.5,76.5],[64.5,76.5]],
        region_legs=[dict(target_decode_index=1,status='ACCEPTED',reasons=[],forward=forward,
                          backward=None,cycle_points=[],cycle_error_px=.125)],
        verified_targets=1,status='INCOMPLETE',accepted_polygons=[],stop_reason='ROUTE_END')
    return dict(status='AREA_TRACKING_COMPLETE',regions=[region],
        point_baseline=[dict(region_id='area0',center_baseline=dict(tracklets_2d=1),expanded_grid=dict(tracklets_2d=40),counts={})]), decoded


def test_actual_crops_joint_quad_warp_residual_and_fixed_pixel_split(tmp_path):
    result, decoded = fixture(tmp_path)
    before = deepcopy((result, decoded))
    output = tmp_path/'report'
    files = report.render_area_tracking(output, result, decoded)
    assert {path.name for path in files} == {'area_000.png','index.html','motion.html','motion.json'}
    with Image.open(output/'area_000.png') as image:
        colors = set(image.getdata())
        assert image.size == (1440,670)
        assert (255,173,69) in colors  # One whole-region quadrilateral.
        assert (75,213,255) in colors and (255,211,90) in colors
        assert (0,90,255) in colors  # Saved zero photometric residual.
        assert (30,30,30) in colors  # Actual template and saved warp pixels.
    page = (output/'index.html').read_text(encoding='utf-8')
    for text in ('整体6参数','光度联合配准','UNKNOWN_NOT_SEGMENTED','TRAIN_ONLY','0.125','点与整体区域口径不同'):
        assert text in page
    assert '不把变换网格算作新对应' in page
    assert '不称完整物体恢复' in page
    assert (result, decoded) == before


def test_motion_view_uses_actual_next_position_and_does_not_extend_failed_track(tmp_path):
    result, decoded = fixture(tmp_path)
    region = result['regions'][0]
    region['decode_path'] = [0, 1, 2, 3]
    for index in (2, 3):
        decoded[index] = dict(decoded[1], time_seconds=index/30)
    first = region['region_legs'][0]
    first['forward']['search_roi'] = dict(left=32, top=12, width=97, height=97)
    failed = deepcopy(first)
    failed.update(target_decode_index=2, status='REJECTED', reasons=['independent reverse failed'])
    failed['forward']['search_roi'] = dict(left=34, top=13, width=97, height=97)
    failed['forward']['polygon_raw4x2'] = [[p[0]+9, p[1]+8] for p in first['forward']['polygon_raw4x2']]
    region['region_legs'].append(failed)
    before = deepcopy(result)
    data = report.motion_records(result, decoded, tmp_path)
    source, accepted, rejected, not_run = data['regions'][0]['frames']
    assert source['status'] == 'SOURCE_REFERENCE' and source['search_roi'] is None
    assert accepted['search_requested_raw_xy'] == [80.5,60.5]
    assert accepted['accepted_center_raw_xy'] == [82.5,61.5]
    assert rejected['search_requested_raw_xy'] == [82.5,61.5]
    assert rejected['search_anchor_decode_index'] == 1
    assert accepted['search_roi_matches_last_accepted_center'] is True
    assert rejected['search_roi_matches_last_accepted_center'] is True
    assert rejected['fitted_center_raw_xy'] == [91.5,69.5]
    assert rejected['accepted_center_raw_xy'] is None and not rejected['accepted']
    assert not_run['status'] == 'NOT_RUN' and not_run['fitted_polygon'] is None
    assert not_run['search_roi'] is None and not_run['search_requested_raw_xy'] is None
    assert result == before


def test_motion_report_does_not_silently_draw_invalid_or_missing_accepted_positions(tmp_path):
    result, decoded = fixture(tmp_path)
    leg = result['regions'][0]['region_legs'][0]
    leg['forward']['polygon_raw4x2'] = None
    with pytest.raises(ValueError,match='ACCEPTED_WITHOUT_POSITION'):
        report.render_area_motion(tmp_path/'missing',result,decoded)
    leg['forward']['polygon_raw4x2'] = [[float('nan'),0]]*4
    with pytest.raises(ValueError,match='INVALID_TARGET_POLYGON'):
        report.render_area_motion(tmp_path/'nan',result,decoded)
    assert not list((tmp_path/'missing').iterdir())
    assert not list((tmp_path/'nan').iterdir())


def test_motion_inline_json_is_safe_and_search_mismatch_is_not_silently_corrected(tmp_path):
    import json
    result, decoded = fixture(tmp_path)
    result['regions'][0]['region_id'] = '</script><script>bad()</script>'
    output=tmp_path/'motion'
    report.render_area_motion(output,result,decoded)
    page=(output/'motion.html').read_text(encoding='utf-8')
    assert '</script><script>bad()' not in page
    data=json.loads((output/'motion.json').read_text(encoding='utf-8'))
    assert data['regions'][0]['frames'][1]['search_roi_matches_last_accepted_center'] is False
    assert data['regions'][0]['frames'][1]['search_roi']['left'] == 40
    with pytest.raises(FileExistsError):
        report.render_area_motion(output,result,decoded)


def test_rejected_alignment_without_warp_does_not_create_fake_pixel_evidence(tmp_path):
    result, decoded = fixture(tmp_path)
    leg = result['regions'][0]['region_legs'][0]
    leg.update(status='REJECTED',reasons=['NATIVE_NONCONVERGENCE'])
    leg['forward']['polygon_raw4x2'] = None
    leg['forward']['alignment']['status'] = 'REJECTED'
    for key in ('warped_search_gray','valid_mask_uint8','search_foreground_valid_mask_uint8','residual_after_train_photometric'):
        leg['forward']['alignment']['pixel_evidence'][key] = None
    report.render_area_tracking(tmp_path/'report',result,decoded)
    with Image.open(tmp_path/'report/area_000.png') as image:
        assert (0,90,255) not in set(image.getdata())
    page = (tmp_path/'report/index.html').read_text(encoding='utf-8')
    assert 'NATIVE_NONCONVERGENCE' in page and 'REJECTED' in page
    assert leg['forward']['alignment']['pixel_evidence']['warped_search_gray'] is None


def test_early_template_refusal_can_show_actual_source_without_fake_roi(tmp_path):
    result, decoded = fixture(tmp_path)
    region = result['regions'][0]
    region.update(template_roi=None,source_polygon_raw4x2=None,region_legs=[],
                  verified_targets=0,stop_reason='INSUFFICIENT_REAL_CROP_BOUNDS')
    report.render_area_tracking(tmp_path/'report',result,decoded)
    assert 'INSUFFICIENT_REAL_CROP_BOUNDS' in (tmp_path/'report/index.html').read_text(encoding='utf-8')


def test_html_escaping_missing_actual_source_and_empty_lazy_pil(tmp_path,monkeypatch):
    result, _ = fixture(tmp_path)
    attack = '</td><script>bad()</script>'
    result['regions'][0]['region_id'] = attack
    result['regions'][0]['stop_reason'] = attack
    files = report.render_area_tracking(tmp_path/'missing',result,{})
    assert [path.name for path in files] == ['index.html']
    page = files[0].read_text(encoding='utf-8')
    assert attack not in page and '&lt;script&gt;' in page and '实际源图不可用' in page
    original = builtins.__import__
    def no_pil(name,*args,**kwargs):
        if name == 'PIL' or name.startswith('PIL.'):
            raise AssertionError('PIL must remain lazy for an empty report')
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',no_pil)
    assert [p.name for p in report.render_area_tracking(tmp_path/'empty',{}, {})] == ['index.html']


def test_no_existing_artifacts_are_overwritten(tmp_path):
    output = tmp_path/'report'
    output.mkdir()
    (output/'index.html').write_text('sealed',encoding='utf-8')
    with pytest.raises(FileExistsError):
        report.render_area_tracking(output,{}, {})
    assert (output/'index.html').read_text(encoding='utf-8') == 'sealed'


@pytest.mark.parametrize('index',[0,1])
def test_actual_source_and_target_dimensions_checked(tmp_path,index):
    result, decoded = fixture(tmp_path)
    decoded[index]['width'] += 1
    with pytest.raises(ValueError,match='AREA_REPORT_SOURCE_DIMENSIONS_MISMATCH'):
        report.render_area_tracking(tmp_path/'report',result,decoded)


def test_budget_and_pixel_shape_fail_closed(tmp_path):
    result, decoded = fixture(tmp_path)
    def stop():
        raise TimeoutError('area deadline')
    with pytest.raises(TimeoutError,match='area deadline'):
        report.render_area_tracking(tmp_path/'stopped',result,decoded,budget=stop)
    assert not list((tmp_path/'stopped').iterdir())
    result['regions'][0]['region_legs'][0]['forward']['alignment']['pixel_evidence']['warped_search_gray'] = [[0.]]
    with pytest.raises(ValueError,match='AREA_REPORT_PIXEL_MATRIX_SHAPE'):
        report.render_area_tracking(tmp_path/'invalid',result,decoded)
