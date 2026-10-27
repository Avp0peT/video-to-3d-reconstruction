import json
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
from integrations.video_masks import auto_boundary as auto


def rows(n=100):
    return [dict(decode_index=i,added_fraction_of_base=(1 if i in (12,37,62,87) else .01)) for i in range(n)]


def test_selection_spreads_risk_and_neighbours_without_assuming_complete_coverage():
    r=auto.select_indices(rows())
    assert r==auto.select_indices(list(reversed(rows())))
    assert len(r['indices'])==16 and r['unselected_count']==84
    assert 0 in r['indices'] and 99 in r['indices']
    assert all(i in r['indices'] for c in (12,37,62,87) for i in (c-1,c,c+1))
    assert r['unselected_status']=='NOT_EVALUATED'


@pytest.mark.parametrize('requested',[[True],[0,0],[100],[],list(range(17))])
def test_invalid_requested_selection_rejected(requested):
    with pytest.raises(ValueError):auto.select_indices(rows(),requested=requested)


@pytest.mark.parametrize('value',[True,float('nan'),float('inf'),-1])
def test_invalid_risk_rejected(value):
    r=rows();r[2]['added_fraction_of_base']=value
    with pytest.raises(ValueError):auto.select_indices(r)


def test_small_input_all_selected_and_metadata_correct():
    r=auto.select_indices(rows(2))
    assert r['indices']==[0,1] and r['unselected_count']==0


def test_decoder_uses_ordinal_and_same_canonical_recipe(tmp_path):
    a=auto.decode_argv('ffmpeg','video with space.mp4',tmp_path,[0,5,20],576,1024)
    f=a[a.index('-filter_complex')+1]
    assert 'eq(n\\,0)+eq(n\\,5)+eq(n\\,20)' in f
    assert 'scale=576:1024:flags=lanczos,format=yuvj444p' in f
    assert '-ss' not in a and '-r' not in a and a.count('passthrough')==2
    assert a[a.index('-i')+1]=='video with space.mp4'


def decoded_fixture(tmp_path):
    (tmp_path/'native').mkdir();(tmp_path/'canonical').mkdir()
    original=tmp_path/'original.jpg';Image.new('RGB',(32,64),(10,50,80)).save(original)
    (tmp_path/'canonical/000000.jpg').write_bytes(original.read_bytes())
    Image.new('RGB',(40,80),(10,50,80)).save(tmp_path/'native/000000.png')
    return [dict(decode_index=23,image=str(original))]


def test_decoded_association_requires_bytes_and_isotropic_geometry(tmp_path):
    frames=decoded_fixture(tmp_path);refs=[]
    result=auto.verify_decoded(tmp_path,frames,32,64,refs)
    assert result[0]['decode_index']==23 and result[0]['canonical_rederivation_byte_equal']
    assert result[0]['scale']==1.25 and result[0]['new_segmentation'] is False and len(refs)==2
    Image.new('RGB',(41,80)).save(tmp_path/'native/000000.png')
    with pytest.raises(ValueError,match='coordinate'):auto.verify_decoded(tmp_path,frames,32,64,[])


def test_different_reencoding_cannot_pass_native_identity(tmp_path):
    frames=decoded_fixture(tmp_path)
    Image.new('RGB',(32,64),(10,50,80)).save(tmp_path/'canonical/000000.jpg',quality=90)
    with pytest.raises(ValueError,match='REDERIVATION'):auto.verify_decoded(tmp_path,frames,32,64,[])


def test_attachment_keeps_source_result_frozen_and_runs_one_bounded_entry(tmp_path,monkeypatch):
    run=tmp_path/'fusion';run.mkdir();source=dict(status='FULL_TIMELINE_FUSED_EXPORTED',mode='union')
    original=json.dumps(source).encode();(run/'RESULT.json').write_bytes(original);calls=[]
    def invoke(root,run_arg,stage,argv,**kw):
        calls.append((run_arg,stage,argv,kw));out=Path(argv[argv.index('--output')+1]);out.mkdir()
        (out/'RESULT.json').write_text(json.dumps(dict(status=auto.READY)))
        (out/'index.html').write_text('report')
        return dict(status='completed',root_exit_code=0,reclaim_confirmed=True)
    monkeypatch.setattr(auto,'_invoke',invoke)
    r=auto.attach_auto_review(run,source)
    assert len(calls)==1 and calls[0][3]['seconds']==420
    assert calls[0][3]['additional_roots']==[run.with_name('fusion__boundary_review')]
    assert r['automatic_boundary_review']['status']==auto.READY
    assert (run/'RESULT.json').read_bytes()==original
    sidecar=(run/'AUTOMATIC_BOUNDARY.json').read_bytes()
    conflict=auto.attach_auto_review(run,source)
    assert conflict['automatic_boundary_review']['status']=='FAILED' and len(calls)==1
    assert (run/'AUTOMATIC_BOUNDARY.json').read_bytes()==sidecar


@pytest.mark.parametrize('source',[dict(status='PREPARED'),dict(status='FAILED'),
    dict(status='FULL_TIMELINE_EXPORTED'),dict(status='FULL_TIMELINE_FUSED_EXPORTED',mode='supported-holes'),
    dict(status='FULL_TIMELINE_FUSED_EXPORTED',mode='union',input_drift=['changed'])])
def test_inapplicable_source_does_not_launch(tmp_path,monkeypatch,source):
    def forbidden(*a,**k):raise AssertionError('must not launch')
    monkeypatch.setattr(auto,'_invoke',forbidden)
    r=auto.attach_auto_review(tmp_path,source)
    assert r['automatic_boundary_review']['status']=='NOT_APPLICABLE'


def test_supervisor_failure_is_not_mask_success(tmp_path,monkeypatch):
    monkeypatch.setattr(auto,'_invoke',lambda *a,**kw:dict(status='failed',root_exit_code=None,reclaim_confirmed=False))
    r=auto.attach_auto_review(tmp_path,dict(status='FULL_TIMELINE_EXPORTED',fusion=dict(mode='union')))
    assert r['status']=='FULL_TIMELINE_EXPORTED'
    assert r['automatic_boundary_review']['status']=='FAILED' and r['automatic_boundary_review']['required']
