import json
import pytest
from integrations.video_masks import region_bridge_assessment as assessment
from integrations.video_masks.region_bridge_assessment import remap_models,summarize,mapper_scope


def test_scope_missing_frame_cannot_evaluate():
    r=mapper_scope([{'name':'a'},{'name':'b'}],{'actual_options':{'mapping':{'image_names':['a']}}})
    assert r['status']=='INPUT_SCOPE_MISMATCH' and r['excluded_names']==['b']
    assert not r['evaluatable_bridge_effect']


def test_scope_all_frames_and_empty_all_semantics():
    frames=[{'name':'a'},{'name':'b'}]
    for names in (['b','a'],[]):
        assert mapper_scope(frames,{'actual_options':{'mapping':{'image_names':names}}})['status']=='INPUT_SCOPE_MATCH'


def test_scope_duplicate_names_rejected():
    assert mapper_scope([{'name':'a'}],{'actual_options':{'mapping':{'image_names':['a','a']}}})['status']=='INPUT_SCOPE_MISMATCH'
    assert mapper_scope([{'name':'a'},{'name':'a'}],{'actual_options':{'mapping':{'image_names':[]}}})['status']=='INPUT_SCOPE_MISMATCH'


def test_decode_identity_instead_of_cross_run_indices():
    old=[dict(input_index=0,decode_index=0),dict(input_index=1,decode_index=17)]
    new=[dict(input_index=0,decode_index=0),dict(input_index=8,decode_index=8),dict(input_index=17,decode_index=17)]
    model=dict(model_id=0,registered_indices=[0,17],points=3,camera_domain_status='CAMERA_DOMAIN_VALID')
    r=remap_models([model],new,{0,17})[0]
    assert r['registered_decode_indices']==[0,17] and r['added_registered_count']==0
    assert r['original_endpoint_registered_decode_indices']==r['original64_registered_decode_indices']==[0,17]
    assert r['original_endpoint_registered_count']==r['original64_registered_count']==2
    model['registered_indices']=[0,8,17]
    s=summarize(remap_models([model],new,{0,17}),{0,17})
    assert s['largest_valid_component_original64_views']==2
    assert s['largest_valid_component_original_endpoint_views']==2
    assert s['original_endpoint_registered_count']==s['original64_registered_count']==2
    assert s['valid_original_endpoint_registered_count']==s['valid_original64_registered_count']==2
    assert s['largest_valid_component_added_views']==1


def test_rejected_component_not_largest_valid():
    m=[dict(model_id=1,registered_indices=[1,2],points=100,camera_domain_status='REJECTED'),dict(model_id=2,registered_indices=[1],points=2,camera_domain_status='CAMERA_DOMAIN_VALID')]
    s=summarize(remap_models(m,[dict(input_index=1,decode_index=10),dict(input_index=2,decode_index=20)],{10}),{10})
    assert s['largest_valid_component_id']==2 and s['valid_added_registered_count']==0


def test_missing_explicit_decode_mapping_rejected():
    with pytest.raises(ValueError):remap_models([dict(registered_indices=[8])],[dict(input_index=0,decode_index=0)],{0})


@pytest.mark.parametrize('endpoint_count', [2,5])
def test_assessment_generic_fields_and_actual_report_denominator(tmp_path,monkeypatch,endpoint_count):
    """TEST_DATA: report-only inspection is stubbed; no native model is opened."""
    old=tmp_path/'old';new=tmp_path/'new'
    old_frames=[dict(input_index=i,decode_index=i*10,name=f'frame_{i}.png') for i in range(endpoint_count)]
    new_frames=[dict(f,input_index=f['decode_index']) for f in old_frames]
    def write(path,value):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value),encoding='utf-8')
    for run,frames in ((old,old_frames),(new,new_frames)):
        write(run/'geometry/PLAN.json',dict(frames=frames))
        write(run/'geometry/RESULT.json',dict(actual_options=dict(mapping=dict(image_names=[]))))
    write(new/'input/PLAN.json',dict(timeline=new_frames,added_decode_indices=[]))
    write(new/'tracking/REGION_TRACKS.json',{})
    def fake_inspect(run,geometry,label,ids,output):
        indices=[0] if label=='old' else [f['input_index'] for f in new_frames]
        return [dict(model_id=0,label=label,registered_indices=indices,points=3,
                     camera_domain_status='CAMERA_DOMAIN_VALID',viewer=f'{label}.html')],{}
    monkeypatch.setattr(assessment,'inspect',fake_inspect)
    result=assessment.assess(old,new)
    saved=json.loads((new/'assessment/BRIDGE_ASSESSMENT.json').read_text(encoding='utf-8'))
    assert saved==result and result['original_endpoint_count']==endpoint_count
    assert result['status']=='MEASURED_NOT_GEOMETRY_ACCEPTED'
    assert result['geometry_accepted'] is False and result['quality_status']=='NOT_EVALUATED'
    for section in [*result['models'],*result['summary'].values(),result['differences']]:
        for key,value in section.items():
            if 'original64' in key:
                assert section[key.replace('original64','original_endpoint')]==value
    assert result['differences']['original_endpoint_registered_added']==list(range(10,endpoint_count*10,10))
    page=(new/'assessment/index.html').read_text(encoding='utf-8')
    assert f'原始端点注册（共{endpoint_count}）' in page
    assert f'<td>1/{endpoint_count}</td>' in page
    assert f'<td>{endpoint_count}/{endpoint_count}</td>' in page
    assert 'original_endpoint_registered_added' in page
    assert '原64' not in page and 'original64' not in page and '手套' not in page
