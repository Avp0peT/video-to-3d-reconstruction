"""Synthetic integration: actual mesh IO/rendering; fake depth evidence only."""
from pathlib import Path
import numpy as np
import pytest
from integrations.mvs_surface import strict_completion_comparison as c


def package(tmp_path,monkeypatch,empty=False):
    v=np.array([[1.,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]])
    f=np.array([[0,2,4],[2,1,4],[1,3,4],[3,0,4],[2,0,5],[1,2,5],[3,1,5],[0,3,5]])
    source=tmp_path/'inferred';source.mkdir();final=tmp_path/'fusion'/'final';final.mkdir(parents=True)
    def mesh(path,scale):return c.write_mesh(path,v*scale,f,np.tile([50,90,130],(6,1)).astype(np.uint8),provenance_note='Synthetic test fixture only.')
    observed=mesh(final/'combined.ply',.7);robust=mesh(source/'robust.ply',1);strict=mesh(source/'strict.ply',.8)
    c.save(final/'RESULT.json',dict(status='ORIGINAL_PLUS_AUDITED_FUSION_NOT_GEOMETRY_ACCEPTED',artifacts={'combined':observed}))
    params=lambda q:dict(quantile=q,smoothing_sigma=0,world_aabb=[[-3,-3,-3],[3,3,3]],spacing=[.1,.1,.1])
    c.save(source/'RESULT.json',dict(status='PARTIAL_INFERRED_ENVELOPE_NOT_GEOMETRY_ACCEPTED',artifacts=dict(robust=robust,strict=strict),variants=dict(robust={'status':'INFERRED','parameters':params(.1)},strict={'status':'INFERRED','parameters':params(0)})))
    c.save(source/'PLAN.json',dict(full_source='synthetic',view_ids=[3,6,9],fusion_run=str(final.parent/'raw')))
    base=tmp_path/'baseline';base.mkdir();c.save(base/'RESULT.json',dict(artifacts=dict(complete_envelope=robust,observed_depth=observed)))
    views=[dict(view_id=i,R=np.eye(3),image=np.zeros((32,32,3),np.uint8),mask=np.ones((32,32),bool),validity=np.ones((32,32),bool)) for i in [3,6,9]]
    monkeypatch.setattr(c,'load_full_inputs',lambda spec,check:dict(bindings=[],views=views))
    def audit(vv,ff,views,check):
        keep=np.zeros(len(vv),bool) if empty else np.array([True,True,True,False,False,False]);faces=np.zeros(len(ff),bool)
        return dict(keep_vertices=keep,keep_faces=faces,counts={'strict_supported_vertices':int(keep.sum()),'strict_supported_faces':0},policy={'synthetic':True})
    monkeypatch.setattr(c,'audit_native_mesh',audit)
    monkeypatch.setattr(c,'project_footprint',lambda vv,ff,view,check:np.full((32,32),float(abs(vv).max())>.75,bool))
    return source,base,tmp_path/'out'


@pytest.mark.parametrize('empty',[False,True])
def test_export_independent_layers_and_empty_strict_points(tmp_path,monkeypatch,empty):
    source,base,out=package(tmp_path,monkeypatch,empty)
    before={str(p):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    r=c.run(source,out,baseline_delivery=base,seconds=60)
    assert r['status']=='STRICT_COMPARISON_DELIVERED' and r['strict_closed']
    assert not r['automatic_adoption'] and not r['geometry_accepted']
    assert r['baseline']['geometry_identical']==dict(complete_envelope=True,observed_depth=True)
    assert [p['fg'] for p in r['projection_totals']]==[0,3072,3072]
    assert r['point_audit']['strict_supported_vertices']==(0 if empty else 3)
    assert not (out/'strict_depth_surface.ply').exists()
    assert (out/'index.html').exists() and (out/'comparison.png').exists()
    assert len(r['per_view'])==3 and all(p['source_bindings'] for p in r['per_view'])
    for name in c.NAMES:
        ref=r['artifacts'][name];assert c.binding(ref['path'])['sha256']==ref['sha256']
    assert all(Path(p).read_bytes()==blob for p,blob in before.items())


def test_drift_rejected_before_output(tmp_path,monkeypatch):
    source,base,out=package(tmp_path,monkeypatch)
    with (source/'strict.ply').open('ab') as f:f.write(b'changed')
    with pytest.raises(ValueError,match='source drift'):c.run(source,out,baseline_delivery=base)
    assert not out.exists()


def test_current_baseline_mismatch_rejected_before_output(tmp_path,monkeypatch):
    source,base,out=package(tmp_path,monkeypatch)
    r=c.load(base/'RESULT.json');r['artifacts']['complete_envelope']=c.load(source/'RESULT.json')['artifacts']['strict'];c.save(base/'RESULT.json',r)
    with pytest.raises(ValueError,match='baseline geometry'):c.run(source,out,baseline_delivery=base)
    assert not out.exists()


def test_absent_strict_envelope_is_diagnostic_not_fabricated_mesh(tmp_path,monkeypatch):
    source,base,out=package(tmp_path,monkeypatch)
    r=c.load(source/'RESULT.json');r['variants']['strict']={'status':'NO_ENVELOPE'};del r['artifacts']['strict'];c.save(source/'RESULT.json',r)
    result=c.run(source,out,baseline_delivery=base,blender=True,seconds=60)
    assert not result['strict_closed'] and result['status']=='STRICT_COMPARISON_DELIVERED'
    assert result['blender']==dict(status='NOT_APPLICABLE',reason='strict_envelope_absent')
    assert result['projection_totals'][2]['fg']==0 and result['projection_totals'][2]['missing']==3072
    assert not (out/'after_strict.ply').exists() and not (out/'after_strict.obj').exists()


def test_baseline_output_overlap_rejected(tmp_path,monkeypatch):
    source,base,out=package(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='disjoint from baseline'):c.run(source,base/'nested',baseline_delivery=base)
    assert not (base/'nested').exists()
