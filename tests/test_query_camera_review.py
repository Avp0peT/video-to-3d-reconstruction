import json
from copy import deepcopy
from pathlib import Path
import pytest
from PIL import Image
from integrations.demo_pipeline.query_camera_review import review_query_cameras
from integrations.demo_pipeline.temporal_review import fingerprint


def test_empty_automatic_pipeline_without_native_backend(tmp_path):
    result=review_query_cameras(dict(records=[]),dict(records=[]),[],[],{}, {},tmp_path/'new',None)
    assert result['status']=='QUERY_CAMERA_EXTENSION_COMPLETE'
    assert result['conditional_vertices']==0 and result['new_cameras']==0
    assert (tmp_path/'new/index.html').is_file() and not (tmp_path/'new/viewer.html').exists()
    assert all(fingerprint(r['path'])==r for r in result['source_bindings'])


def test_image_domain_and_candidate_membership_pass_to_real_math(tmp_path,monkeypatch):
    from integrations.demo_pipeline import query_camera_extension as ext
    # A real undersized math cohort is BLOCKED before any native call.
    src=tmp_path/'source.png';Image.new('RGB',(20,30),'pink').save(src)
    row=dict(model_key='m',query_decode_index=10,cohort='STRICT',points=[dict(point_id='p',xyz=[0.,0.,3.],raw_xy=[10.,15.])],original_mask_status='UNKNOWN_NOT_SEGMENTED')
    before=deepcopy(row);seen=[]
    def extension(ledger,frames,masks,cameras,result,*,budget):
        seen.append(deepcopy(result));budget()
        return dict(status='QUERY_CAMERA_EXTENSION_COMPLETE',classifications=[],counts=dict(groups=0))
    monkeypatch.setattr(ext,'assess_query_extensions',extension)
    result=review_query_cameras(dict(records=[]),dict(records=[row]),[],[],{},
        {10:dict(width=20,height=30,path=str(src))},tmp_path/'new',None)
    assert row==before and seen[0]['records'][0]['points']==row['points']
    assert seen[0]['records'][0]['assessment']['status']=='BLOCKED'
    assert result['camera_counts']==dict(BLOCKED=2)
    assert (tmp_path/'new/camera_000.png').is_file()
    assert result['conditional_vertices']==0


def test_record_bound_and_duplicate_refused(tmp_path):
    with pytest.raises(ValueError,match='QUERY_RECORD_BOUND'):
        review_query_cameras({},dict(records=[{}]*33),[],[],{}, {},tmp_path/'new',None)
    failure=json.loads((tmp_path/'new/FAILURE.json').read_text())
    assert failure['status']=='FAILED' and not (tmp_path/'new/RESULT.json').exists()


def test_shared_deadline_is_not_converted_to_success(tmp_path):
    def budget():raise TimeoutError('same budget')
    with pytest.raises(TimeoutError,match='same budget'):
        review_query_cameras(dict(records=[]),dict(records=[]),[],[],{}, {},tmp_path/'new',None,budget=budget)
    assert json.loads((tmp_path/'new/FAILURE.json').read_text())['status']=='RESOURCE_LIMIT'
