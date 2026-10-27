from pathlib import Path
import numpy as np
from integrations.mvs_surface import completion_delivery as p

def test_deliver_closed_estimate_without_relabelling_observations(tmp_path,monkeypatch):
    v=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    f=np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]],np.int64);c=np.ones((4,3),np.uint8)
    inferred=tmp_path/'inferred';inferred.mkdir();raw=tmp_path/'fusion'/'raw';raw.mkdir(parents=True);final=raw.parent/'final';final.mkdir()
    ref=p.write_mesh(inferred/'robust_envelope.ply',v,f,c,provenance_note='Inferred test only.')
    spec=tmp_path/'spec.json';p.save(spec,{})
    p.save(inferred/'PLAN.json',dict(full_source=str(spec),fusion_run=str(raw)))
    p.save(inferred/'RESULT.json',dict(status='PARTIAL_INFERRED_ENVELOPE_NOT_GEOMETRY_ACCEPTED',artifacts={'robust':ref},variants={'robust':{'parameters':{'world_aabb':[[-2,-2,-2],[2,2,2]],'spacing':[.1,.1,.1]}}},layer_records=[{'name':'inferred_completion','faces':4}]))
    np.savez(raw/'COVERAGE.npz',coverage_view_1=np.zeros((8,8),np.uint8));p.save(raw/'RESULT.json',{'artifacts':{'coverage':{'arrays':p.binding(raw/'COVERAGE.npz')}}})
    observed=p.write_mesh(final/'source.ply',v,f,c,provenance_note='Observed test fixture.');p.save(final/'RESULT.json',{'artifacts':{'combined':observed}})
    before=p.binding(observed['path'])
    monkeypatch.setattr(p,'load_full_inputs',lambda *a:{'views':[{'view_id':1}],'bindings':[]})
    monkeypatch.setattr(p,'export_html',lambda *a,**k:None)
    monkeypatch.setattr(p,'project_footprint',lambda *a:np.zeros((8,8),bool))
    counts=dict(view_id=1,valid_foreground_pixels=12,foreground_native_depth=4,old_foreground_covered=4,new_foreground_covered=10,old_background_projection=1,new_background_projection=2)
    monkeypatch.setattr(p,'write_coverage_panel',lambda *a:dict(counts=counts))
    out=tmp_path/'delivery';r=p.finalize_completion(inferred,out,seconds=20)
    assert r['closed_estimated_outer_model'] and not r['complete_measured_object'] and not r['geometry_accepted']
    assert r['observed_depth_geometry_unchanged'] and p.binding(observed['path'])==before
    assert r['topology']['counts']['boundary_edges']==0 and r['selection']['retained_faces']==4
    assert r['estimated_surface_has_directly_measured_vertices'] is False
    assert r['projection_totals']['estimated_foreground_coverage']==10/12
    spec=p.load(out/'MODEL_SPEC.json');assert [s['title'] for s in spec['scenes']]==['Inferred complete envelope','Observed-depth reconstruction']
    assert p.load(out/'RESULT.json')['status']=='CLOSED_ESTIMATED_OUTER_MODEL_DELIVERED'
