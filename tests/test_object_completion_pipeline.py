import json
from pathlib import Path
import numpy as np
import pytest
from integrations.mvs_surface import object_completion_pipeline as p

def test_options_are_explicit_and_disjoint(tmp_path):
    source=tmp_path/'raw';source.mkdir();spec=tmp_path/'inputs.json';spec.write_text('{}')
    assert p.validate_options(spec,source,tmp_path/'new',False,720)[0]==spec
    for dest,budget in ((source/'new',720),(tmp_path,720),(tmp_path/'new',True),(tmp_path/'new',1201)):
        with pytest.raises(ValueError):p.validate_options(spec,source,dest,False,budget)

def test_layered_pipeline_preserves_exact_closed_mesh(tmp_path,monkeypatch):
    source=tmp_path/'raw';source.mkdir();spec=tmp_path/'inputs.json';p.save(spec,{})
    v=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    f=np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]],np.int64)
    for name in ('FUSION','COVERAGE'):np.savez(source/(name+'.npz'),coverage_view_1=np.zeros((8,8),np.uint8))
    p.save(source/'FUSION.json',{'parameters':{'world_aabb':[[-2,-2,-2],[2,2,2]]}})
    p.save(source/'PLAN.json',{'full_source':str(spec),'view_ids':[1]})
    refs={k:{'arrays':p.binding(source/(name+'.npz'))} for k,name in (('fusion','FUSION'),('coverage','COVERAGE'))};refs['fusion']['metadata']=p.binding(source/'FUSION.json')
    p.save(source/'RESULT.json',dict(status='FUSED_SURFACE_GENERATED_NOT_GEOMETRY_ACCEPTED',artifacts=refs))
    calls=[]
    monkeypatch.setattr(p,'load_full_inputs',lambda *a:{'views':[{'view_id':1}],'bindings':[],'local_vertices':v,'local_faces':f})
    def envelope(views,bounds,**kw):
        calls.append(kw);kw['check']()
        return dict(vertices=v.copy(),faces=f.copy(),colors=np.full((4,3),128,np.uint8),counts={'vertices':4,'faces':4},status='INFERRED_ENVELOPE_NOT_GEOMETRY_ACCEPTED',boundary_known_negative=True,parameters={'q':kw['quantile']})
    monkeypatch.setattr(p,'infer_silhouette_completion',envelope)
    monkeypatch.setattr(p,'assess_points',lambda *a,**kw:dict(classification=np.array([1,0,2,1],np.uint8),counts={'supported':2,'unverified':1,'conflict':1}))
    monkeypatch.setattr(p,'project_footprint',lambda *a:np.zeros((8,8),bool))
    monkeypatch.setattr(p,'write_coverage_panel',lambda *a:dict(counts={'view_id':1}))
    monkeypatch.setattr(p,'export_html',lambda spec,dest,**kw:dict(status='TEST_EXPORT_ONLY'))
    out=tmp_path/'new';result=p.run(spec,source,out,seconds=20)
    assert [c['quantile'] for c in calls]==[0.,.1]
    assert all(c['smoothing_sigma']==0 for c in calls)
    assert result['closed_estimated_outer_envelope'] and not result['complete_observed_object'] and not result['geometry_accepted']
    assert result['layer_face_identity_exact']
    assert sum(x['faces'] for x in result['layer_records'])==4
    files=[out/(x['name']+'_IDENTITY.npz') for x in result['layer_records']]
    ids=np.concatenate([np.load(path)['original_face_ids'] for path in files]);assert sorted(ids.tolist())==[0,1,2,3]
    assert 'not directly measured' in (out/'robust_envelope.obj').read_text()
    assert 'No smoothing or hole filling' not in (out/'robust_envelope.obj').read_text()
    assert result['variants']['robust']['topology']['counts']['boundary_edges']==0
    assert (out/'index.html').exists()

def test_cli_inference_flag_required(tmp_path):
    import scripts.complete_object_surface as cli
    with pytest.raises(SystemExit) as e:cli.main(['--full-source','x','--fusion-run','y','--output',str(tmp_path/'new')])
    assert e.value.code==2
    assert not (tmp_path/'new').exists()

def test_note_rejects_multiline_before_write(tmp_path):
    from integrations.mvs_surface.inferred_mesh_io import write_mesh
    v=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]]);f=np.array([[0,1,2]]);c=np.zeros((3,3),np.uint8)
    with pytest.raises(ValueError):write_mesh(tmp_path/'a.ply',v,f,c,provenance_note='a\nb')
    assert not (tmp_path/'a.ply').exists()
