import pytest
from integrations.video_masks.region_improvement_assessment import reverse_ids,summarize,differences,render

def component(mid,views,points,valid=True):
    return dict(model_id=mid,registered_indices=views,points=points,camera_domain_status='CAMERA_DOMAIN_VALID' if valid else 'CAMERA_DOMAIN_REJECTED')

def test_largest_valid_component_and_union_not_sum():
    old=summarize([component(0,[0,1,2],100),component(1,[2,3],50),component(2,[4,5,6,7],200,False)])
    assert old['unique_registered_views']==8
    assert old['valid_unique_registered_views']==4
    assert old['largest_valid_component_views']==3
    assert old['largest_valid_component_points']==100
    new=summarize([component(0,[0,1,2,3,8],120)])
    delta=differences(old,new)
    assert delta['valid_registered_added']==[8] and delta['valid_registered_lost']==[]
    assert delta['registered_lost']==[4,5,6,7]
    assert delta['largest_valid_component_views_delta']==2

def test_reverse_provenance_explicit():
    tracks=dict(reverse_track_ids=[8],frames=[dict(observations=[dict(track_id=8),dict(track_id=1)])])
    assert reverse_ids(tracks)=={8}
    tracks['reverse_track_ids']=[9]
    with pytest.raises(ValueError):reverse_ids(tracks)
    with pytest.raises(ValueError):reverse_ids(dict(frames=[],tracks=[]))

def test_html_marks_shape_boundary_and_escapes(tmp_path):
    m=component(0,[1,2],5);m.update(label='new',viewer='viewers/a.html')
    s=summarize([m]);doc=dict(summary=dict(old=s,new=s),differences=differences(s,s),models=[m])
    output=tmp_path/'index.html';render(doc,output);text=output.read_text(encoding='utf-8')
    assert 'viewers/a.html' in text and '未验收' in text and '点数不可相加' in text
    assert '<h1>目标稀疏重建对照</h1>' in text and '手套' not in text


def test_source_geometry_from_bound_plan(tmp_path):
    import json
    from integrations.video_masks.region_improvement_assessment import source_geometry
    assert source_geometry(tmp_path)=='geometry02'
    (tmp_path/'PLAN.json').write_text(json.dumps(dict(source_geometry='geometry_custom')),encoding='utf-8')
    assert source_geometry(tmp_path)=='geometry_custom'
    (tmp_path/'PLAN.json').write_text(json.dumps(dict(source_geometry='../bad')),encoding='utf-8')
    with pytest.raises(ValueError):source_geometry(tmp_path)


@pytest.mark.parametrize('absolute', [False, True])
def test_source_geometry_accepts_nested_contained_directory(tmp_path, absolute):
    import json
    from integrations.video_masks.region_improvement_assessment import source_geometry
    source=tmp_path/'source';geometry=source/'nested'/'geometry'
    geometry.mkdir(parents=True)
    (geometry/'RESULT.json').write_text('{}',encoding='utf-8')
    new=tmp_path/'new';new.mkdir()
    selection=str(geometry) if absolute else 'nested/geometry'
    (new/'PLAN.json').write_text(json.dumps(dict(source_run=str(source),source_geometry=selection)),encoding='utf-8')
    assert source_geometry(new)=='nested/geometry'


@pytest.mark.parametrize('selection', ['../outside', 'nested/../geometry', '.', 'geometry/RESULT.json'])
def test_source_geometry_rejects_parent_segments_and_nondirectory(tmp_path, selection):
    import json
    from integrations.video_masks.region_improvement_assessment import source_geometry
    source=tmp_path/'source';geometry=source/'geometry';geometry.mkdir(parents=True)
    (geometry/'RESULT.json').write_text('{}',encoding='utf-8')
    outside=tmp_path/'outside';outside.mkdir()
    (outside/'RESULT.json').write_text('{}',encoding='utf-8')
    new=tmp_path/'new';new.mkdir()
    (new/'PLAN.json').write_text(json.dumps(dict(source_run=str(source),source_geometry=selection)),encoding='utf-8')
    with pytest.raises(ValueError):source_geometry(new)


def test_source_geometry_rejects_absolute_directory_outside_source(tmp_path):
    import json
    from integrations.video_masks.region_improvement_assessment import source_geometry
    source=tmp_path/'source';source.mkdir()
    outside=tmp_path/'outside';outside.mkdir()
    (outside/'RESULT.json').write_text('{}',encoding='utf-8')
    new=tmp_path/'new';new.mkdir()
    (new/'PLAN.json').write_text(json.dumps(dict(source_run=str(source),source_geometry=str(outside))),encoding='utf-8')
    with pytest.raises(ValueError,match='inside source'):source_geometry(new)


def test_split_ratio_uses_largest_valid_component():
    large=component(0,[0,1,2],100);large.update(split_region_ids=[7,8],triangulated_region_ids=10)
    invalid=component(1,[0,1,2,3],500,False);invalid.update(split_region_ids=[1],triangulated_region_ids=1)
    summary=summarize([large,invalid])
    assert summary['largest_valid_component_split_ids']==2
    assert summary['largest_valid_component_triangulated_region_ids']==10
    assert summary['largest_valid_component_split_ratio']==.2
