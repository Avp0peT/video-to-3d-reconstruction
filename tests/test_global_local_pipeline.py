"""Synthetic contract counterexamples; not a real geometry acceptance."""
import json
import numpy as np
import pytest
from integrations.mvs_surface import global_local_pipeline as p
from scripts import run_region_mask_review as cli


def test_only_missing_pixels_added_existing_values_unchanged():
    whole=np.array([[1.,0.],[3.,0.]],np.float32)
    candidate=np.array([[99.,2.],[.1,0.]],np.float32)
    accepted=candidate>0; roi=np.ones((2,2),bool)
    result,added=p.add_missing_only(whole,candidate,accepted,roi,roi)
    np.testing.assert_array_equal(result,[[1,2],[3,0]])
    np.testing.assert_array_equal(whole,[[1,0],[3,0]])
    assert added.sum()==1 and not np.shares_memory(result,whole)


@pytest.mark.parametrize('field',['roi','allowed','depth','finite'])
def test_invalid_candidate_cannot_fill_hole(field):
    whole=np.zeros((2,2),np.float32); proposal=np.ones_like(whole)
    accepted=np.ones((2,2),bool);roi=accepted.copy();allowed=accepted.copy()
    if field=='roi':roi[0,0]=False
    if field=='allowed':allowed[0,0]=False
    if field=='depth':proposal[0,0]=0
    if field=='finite':proposal[0,0]=np.nan
    with pytest.raises(ValueError):p.add_missing_only(whole,proposal,accepted,roi,allowed)


def test_roi_uses_image_identity_and_mask_intersection():
    allowed=np.ones((4,5),bool);allowed[2,2]=False
    spec={'decode_index':840,'cells':[{'bounds':[1,1,4,3]}]}
    roi=p._roi(spec,840,allowed)
    assert roi.sum()==5 and not roi[2,2]
    with pytest.raises(ValueError):p._roi(spec,841,allowed)
    with pytest.raises(ValueError):p._roi(dict(spec,decode_index=True),1,allowed)


def test_historical_source_plan_hash_rejects_current_modified_plan(tmp_path):
    path=tmp_path/'plan.json';path.write_text('{}')
    prep={'source_plan':str(path),'refs':[p.binding(path)]}
    assert p.historical_plan_binding(prep)['sha256']==prep['refs'][0]['sha256']
    path.write_text('{"changed":true}')
    with pytest.raises(ValueError,match='historical source plan changed'):p.historical_plan_binding(prep)


@pytest.mark.parametrize('text',['{"a":1,"a":2}','{"a":NaN}','{"a":[1e999]}'])
def test_json_rejects_duplicate_or_nonfinite(tmp_path,text):
    path=tmp_path/'bad.json';path.write_text(text)
    with pytest.raises(ValueError):p.load(path)


@pytest.mark.parametrize('n',[0,2])
def test_cloud_writer_exact_readback_including_empty(tmp_path,n):
    xyz=np.array([[1.25,2.,3.],[4.,5.,6.]])[:n]
    rgb=np.array([[0,1,255],[128,77,22]],np.uint8)[:n]
    result=p._write_cloud(tmp_path/'new.ply',xyz,rgb)
    assert result['vertices']==n and result['exact_readback']


def test_auto_entry_routes_explicit_local_mode(monkeypatch,capsys):
    seen=[]
    def fake(*args):
        seen.append(args)
        return {'status':'NOT_ACCEPTED','metrics':{},'wall_seconds':0.}
    monkeypatch.setattr(p,'run',fake)
    assert cli.main(['--global-local-experiment','exp','--reference-image-id','80','--roi-spec','roi','--output','out'])==0
    assert seen==[('exp',80,'roi','out')]
    assert json.loads(capsys.readouterr().out)['status']=='NOT_ACCEPTED'


@pytest.mark.parametrize('extra',[
    ['--motion-manifest','m'],['--prepare-only'],['--source-plan','p'],['--raw-motion-only']])
def test_auto_entry_rejects_mixed_modes(extra):
    with pytest.raises(SystemExit) as e:
        cli.main(['--global-local-experiment','e','--reference-image-id','80','--roi-spec','r','--output','o',*extra])
    assert e.value.code==2


def test_regional_method_selected_explicitly(monkeypatch,capsys):
    seen=[]
    def fake(*args,**kwargs):
        seen.append((args,kwargs));return {'status':'INFERRED_NOT_ACCEPTED','metrics':{},'wall_seconds':0.}
    monkeypatch.setattr(p,'run',fake)
    assert cli.main(['--global-local-experiment','e','--reference-image-id','80','--roi-spec','r','--output','o','--local-method','regional-plane-v2'])==0
    assert seen==[(('e',80,'r','o'),{'method':'regional-plane-v2'})]
    assert json.loads(capsys.readouterr().out)['status']=='INFERRED_NOT_ACCEPTED'


def test_local_method_without_source_rejects():
    with pytest.raises(SystemExit) as e:cli.main(['--motion-manifest','m','--output','o','--local-method','regional-plane-v2'])
    assert e.value.code==2


def test_silhouette_routes_to_separate_inference_not_depth_combiner(monkeypatch,capsys):
    from integrations.mvs_surface import silhouette_pipeline as silhouette
    seen=[]
    def fake(*args,**kwargs):
        seen.append((args,kwargs));return {'status':'INFERRED_NOT_ACCEPTED','metrics':{},'wall_seconds':0.}
    monkeypatch.setattr(silhouette,'run',fake)
    monkeypatch.setattr(p,'run',lambda *a,**k:pytest.fail('must not combine silhouette into observed depths'))
    assert cli.main(['--global-local-experiment','e','--reference-image-id','80','--roi-spec','r','--output','o','--local-method','silhouette-interval-v1'])==0
    assert seen==[(('e',80,'r','o'),{})]
    assert json.loads(capsys.readouterr().out)['status']=='INFERRED_NOT_ACCEPTED'


def test_multiview_routes_to_explicit_progressive_consumer(monkeypatch,capsys):
    from integrations.mvs_surface import silhouette_multiview_pipeline as multiview
    from integrations.mvs_surface import silhouette_pipeline as old
    seen=[]
    def fake(*args,**kwargs):
        seen.append((args,kwargs));return {'status':'MULTIVIEW_NOT_ACCEPTED','metrics':{},'wall_seconds':0.}
    monkeypatch.setattr(multiview,'run',fake)
    monkeypatch.setattr(old,'run',lambda *a,**k:pytest.fail('must select the explicit new source policy'))
    monkeypatch.setattr(p,'run',lambda *a,**k:pytest.fail('must not adopt silhouette depths'))
    assert cli.main(['--global-local-experiment','e','--reference-image-id','80','--roi-spec','r','--output','o','--local-method','silhouette-multiview-v2'])==0
    assert seen==[(('e',80,'r','o'),{})]
    assert json.loads(capsys.readouterr().out)['status']=='MULTIVIEW_NOT_ACCEPTED'


def test_alignment_review_routes_without_reconstructing(monkeypatch,capsys):
    from integrations.mvs_surface import boundary_alignment_pipeline as alignment
    seen=[]
    def fake(*args):
        seen.append(args);return {'status':'DIAGNOSTIC_NOT_ADOPTED','wall_seconds':0.}
    monkeypatch.setattr(alignment,'run',fake)
    monkeypatch.setattr(p,'run',lambda *a,**k:pytest.fail('review must not reconstruct'))
    assert cli.main(['--alignment-review','prior','--output','out'])==0
    assert seen==[('prior','out')]
    assert json.loads(capsys.readouterr().out)['status']=='DIAGNOSTIC_NOT_ADOPTED'


@pytest.mark.parametrize('extra',[['--global-local-experiment','x'],['--motion-manifest','x'],['--reference-image-id','0'],['--prepare-only']])
def test_alignment_review_rejects_other_modes(extra):
    with pytest.raises(SystemExit) as e:cli.main(['--alignment-review','prior','--output','out',*extra])
    assert e.value.code==2


def test_model_entry_routes_to_complete_pipeline(monkeypatch,capsys):
    from integrations.mvs_surface import global_local_model_pipeline as model
    seen=[]
    def fake(*args,**kwargs):
        seen.append((args,kwargs));return dict(status='MODEL_GENERATED_NOT_GEOMETRY_ACCEPTED',new_meshes=2,new_triangulated_points=0,wall_seconds=0.,blender=dict(status='SUCCESS'))
    monkeypatch.setattr(model,'run',fake)
    assert cli.main(['--model-from-multiview','prior','--output','out','--blender'])==0
    assert seen==[(('prior','out'),{'blender':True})]
    assert json.loads(capsys.readouterr().out)['new_meshes']==2


@pytest.mark.parametrize('extra',[['--alignment-review','p'],['--motion-manifest','m'],['--reference-image-id','1'],['--prepare-only'],['--local-method','silhouette-interval-v1']])
def test_model_entry_rejects_mixed_modes(extra):
    with pytest.raises(SystemExit) as e:cli.main(['--model-from-multiview','prior','--output','out',*extra])
    assert e.value.code==2


def test_blender_is_not_implicitly_enabled_for_diagnostics():
    with pytest.raises(SystemExit) as e:cli.main(['--alignment-review','prior','--output','out','--blender'])
    assert e.value.code==2
