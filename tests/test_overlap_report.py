"""Real-source report fixtures; no native tracking or model execution."""
import base64
from copy import deepcopy
import xml.etree.ElementTree as ET

import pytest
from PIL import Image

from integrations.demo_pipeline.overlap_report import render_overlap


def fixture(tmp_path):
    source = tmp_path / 'source.png'
    Image.new('RGB', (96, 64), (79, 111, 141)).save(source)
    rois = [dict(left=x, top=8, width=33, height=33) for x in (8, 24)]
    items = [dict(candidate_id='c'+str(i), roi=roi, eligible=True,
                  selected=True, region_id='r'+str(i)) for i, roi in enumerate(rois)]
    items += [dict(candidate_id='<unselected>', roi=dict(left=40, top=24,width=33,height=33),
                   eligible=True, selected=False, reason='<budget>'),
              dict(candidate_id='boundary', roi=dict(left=80,top=48,width=33,height=33),
                   eligible=False,selected=False,reason='ROI_OUTSIDE_IMAGE')]
    edge = dict(edge_id='e01',region_a='r0',region_b='r1',source_input_index=0,
                source_decode_index=0,shared_foreground_pixels=17*33)
    plan = dict(sources=[dict(source_input_index=0,source_decode_index=0,decode_path=[0,1,2])],
                candidates=[dict(source_input_index=0,source_decode_index=0,items=items)],
                regions=[dict(region_id='r0'),dict(region_id='r1')],edges=[edge],
                frame_domains={'0':dict(width=96,height=64,original_mask_status='OK',
                                        mask_rle=[[[0,96]] for _ in range(64)])})
    result = dict(regions=[dict(region_id='r'+str(i),source_input_index=0,template_roi=roi,
                                region_legs=[{}]) for i,roi in enumerate(rois)],
                  overlap=dict(regions=[dict(region_id='r0',local_tracklet=True,network_tracklet=True),
                                        dict(region_id='r1',local_tracklet=True,network_tracklet=False)],
                               edges=[dict(edge,targets=[dict(target_decode_index=1,status='CONSISTENT',
                                                             max_error_px=.5,reasons=['<safe>'])])],
                               coverage=[dict(source_input_index=0,source_decode_index=0,
                                              foreground_pixels=6144,planned_union_pixels=1617,
                                              local_two_target_union_pixels=1617,
                                              overlap_two_target_union_pixels=1089)]),
                  batches=[dict(ordinal=0,regions=2,region_ids=['r0','r1'],report='batch_000/index.html')],
                  counts={},native_calls={},measurement_wall_seconds=1.2)
    decoded = {0:dict(path=str(source),width=96,height=64)}
    return plan,result,decoded,source


def test_actual_source_union_layers_links_and_escaping(tmp_path):
    plan,result,decoded,source = fixture(tmp_path)
    before = deepcopy((plan,result,decoded))
    output = tmp_path/'report'
    files = render_overlap(output,plan,result,decoded)
    assert files == [output/'overlap_source_000.svg',output/'index.html']
    svg = files[0].read_text(encoding='utf-8')
    root = ET.fromstring(svg)
    image = root.find('{http://www.w3.org/2000/svg}image')
    assert base64.b64decode(image.attrib['href'].split(',',1)[1]) == source.read_bytes()
    for layer in ('mask','candidates','selected','local','network','conflicts'):
        assert 'layer_'+layer in svg
    page = files[1].read_text(encoding='utf-8')
    assert '<td>1617</td>' in page  # union, not 2*1089
    assert '<td>1089</td>' in page
    assert 'href="batch_000/index.html">r0</a>' in page
    assert '&lt;unselected&gt;' in page and '&lt;safe&gt;' in page
    assert '<unselected>' not in page
    assert 'ROI_OUTSIDE_IMAGE' in page
    assert '总耗时于RESULT.json收尾写入' in page and 'None s' not in page
    assert '不是3D覆盖' in page and '不产生新3D点' in page
    assert (plan,result,decoded) == before


def test_missing_edge_assessment_is_explicitly_untested(tmp_path):
    plan,result,decoded,_ = fixture(tmp_path)
    result['overlap']['edges'] = []
    render_overlap(tmp_path/'report',plan,result,decoded)
    page = (tmp_path/'report/index.html').read_text(encoding='utf-8')
    assert page.count('class="edge_UNTESTED"') == 2
    assert 'class="edge_CONSISTENT"' not in page
    assert 'UNTESTED不等于一致' in page


@pytest.mark.parametrize('bad',[1618,True])
def test_incorrect_declared_union_is_rejected(tmp_path,bad):
    plan,result,decoded,_ = fixture(tmp_path)
    result['overlap']['coverage'][0]['planned_union_pixels'] = bad
    with pytest.raises(ValueError,match='DECLARED_COVERAGE_MISMATCH'):
        render_overlap(tmp_path/'report',plan,result,decoded)
    assert not list((tmp_path/'report').iterdir())


@pytest.mark.parametrize('where',['decoded','domain'])
def test_source_dimensions_must_match_actual_image(tmp_path,where):
    plan,result,decoded,_ = fixture(tmp_path)
    (decoded[0] if where == 'decoded' else plan['frame_domains']['0'])['width'] = 95
    with pytest.raises(ValueError,match='DIMENSIONS_MISMATCH'):
        render_overlap(tmp_path/'report',plan,result,decoded)


def test_batch_path_cannot_escape_output(tmp_path):
    plan,result,decoded,_ = fixture(tmp_path)
    result['batches'][0]['report'] = '../outside/index.html'
    with pytest.raises(ValueError,match='BATCH_LINK_OUTSIDE_OUTPUT'):
        render_overlap(tmp_path/'report',plan,result,decoded)
    assert not list((tmp_path/'report').iterdir())


def test_budget_is_checked_before_any_write(tmp_path):
    def stop():
        raise RuntimeError('budget exhausted')
    with pytest.raises(RuntimeError,match='budget exhausted'):
        render_overlap(tmp_path/'report',{}, {}, {},budget=stop)
    assert not (tmp_path/'report').exists()


def test_empty_report_and_existing_output_preserved(tmp_path):
    output = tmp_path/'report'
    assert render_overlap(output,{}, {}, {}) == [output/'index.html']
    original = (output/'index.html').read_bytes()
    with pytest.raises(FileExistsError):
        render_overlap(output,{}, {}, {})
    assert (output/'index.html').read_bytes() == original


def test_dynamic_profile_components_connections_and_prefix_comparison(tmp_path):
    plan,result,decoded,_ = fixture(tmp_path)
    plan['profile'] = dict(max_regions=256,max_sources=2)
    plan['planned_connected_components'] = [dict(source_input_index=0,source_decode_index=0,
        component_count=2,components=[['r0'],['r1']],isolated_region_ids=['r0','r1'])]
    plan['selection_counts'] = dict(selected_regions=2,remaining_region_slots=254)
    plan['sources'][0].update(selected_component_count=2,candidate_component_count=1,
        unreachable_eligible_candidates=0,budget_unselected_candidates=7,
        spatial_strata=[dict(stratum=0,row=0,column=0,eligible_candidates=9,selected_count=2,
                            eligible_union_pixels=2000,planned_union_pixels=1617,planned_fraction=.8085)])
    result['same_budget64'] = dict(counts={'regions':2},coverage=deepcopy(result['overlap']['coverage']),
        planned_connected_components=deepcopy(plan['planned_connected_components']),
        selection_counts={'selected_regions':2},note='<prefix evidence>')
    render_overlap(tmp_path/'report',plan,result,decoded)
    page = (tmp_path/'report/index.html').read_text(encoding='utf-8')
    svg = (tmp_path/'report/overlap_source_000.svg').read_text(encoding='utf-8')
    assert '区域预算最多256' in page
    assert '未连通' in page and "[&#x27;r0&#x27;], [&#x27;r1&#x27;]" in page
    assert '64前缀与全量区域' in page and '&lt;prefix evidence&gt;' in page
    assert '不是第二次运行，也不是速度收益证据' in page
    assert 'show_connections' in page and 'layer_connections' in svg
    assert '<line ' in svg and 'e01</title>' in svg
    assert '不表示目标帧已经验证连通' in page
    assert '空间选择' in page and 'budget_unselected_candidates' in page
    assert '<td>0.8085</td>' in page
    assert '格面积不能加总' in page


def test_old_plan_uses_default_budget_and_unknown_connectivity(tmp_path):
    render_overlap(tmp_path/'report',{}, {}, {})
    page = (tmp_path/'report/index.html').read_text(encoding='utf-8')
    assert '区域预算最多64' in page
    assert '计划连通性：未知' in page
    assert '64前缀与全量区域' not in page
