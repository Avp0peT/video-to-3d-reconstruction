"""End-to-end file/provenance adapter checks on explicit synthetic inputs."""
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from integrations.mvs_surface import global_local_model_pipeline as p


def fixture(tmp_path,monkeypatch):
    inp=tmp_path/'inputs';inp.mkdir();prior=tmp_path/'prior';prior.mkdir();frames=[];refs=[];maps={}
    for i in range(4):
        im=inp/f'i{i}.png';mask=inp/f'm{i}.png';valid=inp/f'v{i}.png'
        Image.fromarray(np.full((32,32,3),100,np.uint8)).save(im)
        for file in (mask,valid):Image.fromarray(np.full((32,32),255,np.uint8)).save(file)
        d=SimpleNamespace(view_ids=[i],K=np.array([[30.,0,16],[0,30,16],[0,0,1.]]),R=np.eye(3),C=np.array([i*.1,0,0]),depth=np.full((32,32),2.,np.float32))
        f=dict(image=dict(global_id=i,width=32,height=32,K=d.K.tolist(),R=d.R.tolist(),C=d.C.tolist()),image_file=p.binding(im),mask=p.binding(mask),validity=p.binding(valid));frames.append(f)
        refs.extend([f['image_file'],f['mask'],f['validity']])
        if i!=2:
            path=inp/f'd{i}.dmap';path.write_bytes(b'SYNTHETIC'+bytes([i]));maps[str(path)]=d;refs.append(p.binding(path))
    historical=inp/'source_plan.json';p.save(historical,dict(frames=frames));refs.append(p.binding(historical))
    prepared=inp/'PREPARED.json';p.save(prepared,dict(views=[dict(global_id=i,decode_index=i*5) for i in range(4)]));refs.append(p.binding(prepared))
    cloud=inp/'__mvs_dense.ply';p._write_cloud(cloud,np.array([[x,y,z] for x in (-.2,.2) for y in (-.2,.2) for z in (1.5,2.5)]),np.full((8,3),100,np.uint8));refs.append(p.binding(cloud))
    p.save(prior/'PLAN.json',dict(reference_id=1,source_ids=[0,2,3],withheld_source_id=3,source_plan_historical_binding=p.binding(historical),input_bindings=refs))
    p.save(prior/'RESULT.json',dict(status='MULTIVIEW_SILHOUETTE_ESTIMATES_COMPLETE_NOT_GEOMETRY_ACCEPTED'))
    roi=np.zeros((32,32),bool);roi[8:24,8:24]=True;np.savez_compressed(prior/'SILHOUETTE_INTERVALS.npz',roi=roi)
    monkeypatch.setattr(p,'read_dmap',lambda path:maps[str(path)])
    monkeypatch.setattr(p,'authenticate_depth',lambda d,c:np.testing.assert_array_equal(d.K,np.array(c['K'])))
    protected={f:f.read_bytes() for folder in (inp,prior) for f in folder.iterdir() if f.is_file()}
    return prior,tmp_path/'out',protected


def fake_kernels(monkeypatch,*,empty=False):
    seen=[]
    def track(views,refid,roi,check):
        seen.append(('tracks',[v['view_id'] for v in views]));check()
        return dict(points=np.empty((0,3)),rgb=np.empty((0,3),np.uint8),reference_xy=np.empty((0,2)),accepted=np.empty(0,bool),observation_mask=np.empty((0,4),bool),observations=np.empty((0,4,2)),view_ids=[0,1,2,3],counters=dict(selected=0,accepted=0),geometry_accepted=False)
    verts=np.array([[-.1,-.1,2],[.1,-.1,2],[-.1,.1,2]])
    faces=np.empty((0,3),np.int32) if empty else np.array([[0,1,2]],np.int32)
    def surface(views,check):
        seen.append(('surface',[v['view_id'] for v in views]));check()
        assert not views[2]['depth_available'] and (views[2]['depth']==0).all()
        return dict(worldvertices=verts.copy(),colors=np.full((3,3),90,np.uint8),faces=faces.copy(),counts=dict(faces=len(faces)),geometry_accepted=False)
    def hull(views,refid,roi,bounds,check):
        seen.append(('hull',[v['view_id'] for v in views]));check()
        assert len(views)==3 and 3 not in [v['view_id'] for v in views]
        return dict(vertices=verts+np.array([0,0,.1]),colors=np.tile(np.array([255,150,30],np.uint8),(3,1)),faces=faces.copy(),counts=dict(faces=len(faces)),geometry_accepted=False)
    monkeypatch.setattr(p,'recover_local_geometry',track);monkeypatch.setattr(p,'build_depth_surface',surface);monkeypatch.setattr(p,'build_local_visual_hull',hull)
    return seen


def test_layers_complete_bound_readback_no_fusion(tmp_path,monkeypatch):
    prior,out,protected=fixture(tmp_path,monkeypatch);seen=fake_kernels(monkeypatch)
    result=p.run(prior,out)
    assert result['status']=='MODEL_GENERATED_NOT_GEOMETRY_ACCEPTED'
    assert result['new_meshes']==2 and not result['geometry_fused'] and result['blender']['status']=='NOT_RUN'
    assert seen==[('tracks',[1,0,2,3]),('surface',[1,0,2,3]),('hull',[1,0,2])]
    assert p.load(out/'PLAN.json')['temporal_order']==[0,1,2,3]
    mesh=p.read_ply(out/'layered_preview.ply',max_bytes=16<<20,max_vertices=200000,max_faces=400000)
    assert len(mesh['vertices'])==6 and len(mesh['faces'])==2
    assert mesh['faces']==[[0,1,2],[3,4,5]]
    for key in ('observed_surface','inferred_shell','layered_preview'):assert result['artifacts'][key]['exact_readback']
    assert (out/'layers_viewer.html').exists() and (out/'model_viewer.html').exists()
    for file,data in protected.items():assert file.read_bytes()==data


def test_unknown_masks_do_not_turn_empty_result_into_model(tmp_path,monkeypatch):
    prior,out,_=fixture(tmp_path,monkeypatch);fake_kernels(monkeypatch,empty=True)
    result=p.run(prior,out,blender=True)
    assert result['new_meshes']==0 and result['status']=='NO_EVIDENCE_SUPPORTED_MODEL'
    assert result['blender']['status']=='NOT_RUN' and not (out/'blender').exists()


def test_changed_bound_input_stops_before_kernels(tmp_path,monkeypatch):
    prior,out,_=fixture(tmp_path,monkeypatch);seen=fake_kernels(monkeypatch)
    (prior.parent/'inputs/i1.png').write_bytes(b'DRIFT')
    with pytest.raises(ValueError,match='binding drift'):p.run(prior,out)
    assert not seen and p.load(out/'RESULT.json')['status']=='FAILED_MODEL_PIPELINE'


def test_blender_export_receives_separate_layers(tmp_path,monkeypatch):
    from integrations.blender_export import export as b
    prior,out,_=fixture(tmp_path,monkeypatch);fake_kernels(monkeypatch);seen=[]
    def export(spec,output,profile):
        seen.append(p.load(spec));assert profile=='surface400k';return dict(status='SUCCESS',scenes=2)
    monkeypatch.setattr(b,'export_blend',export)
    result=p.run(prior,out,blender=True)
    assert result['blender']['status']=='SUCCESS' and len(seen[0]['scenes'])==2
    assert 'INFERRED' in seen[0]['scenes'][1]['note']


def test_mesh_io_rejects_invalid_indices_before_writing(tmp_path):
    path=tmp_path/'bad.ply'
    with pytest.raises(ValueError):p.write_mesh(path,np.zeros((3,3)),np.array([[0,1,3]]),np.zeros((3,3),np.uint8))
    assert not path.exists()


def test_inconsistent_temporal_identity_rejected(tmp_path,monkeypatch):
    prior,out,_=fixture(tmp_path,monkeypatch);seen=fake_kernels(monkeypatch)
    plan=p.load(prior/'PLAN.json');file=prior.parent/'inputs/PREPARED.json';value=p.load(file);value['views'][1]['decode_index']=0;p.save(file,value)
    plan['input_bindings']=[p.binding(file) if Path(b['path'])==file else b for b in plan['input_bindings']];p.save(prior/'PLAN.json',plan)
    with pytest.raises(ValueError,match='temporal decode'):p.run(prior,out)
    assert not seen


def test_hull_diagnostic_does_not_claim_vertices_cover_triangle():
    mask=np.ones((9,9),bool);mask[3,3]=False
    view=dict(view_id=4,C=np.zeros(3),R=np.eye(3),K=np.eye(3),mask=mask,validity=np.ones_like(mask))
    vertices=np.array([[1.,1,1],[7,1,1],[1,7,1]])
    r=p.hull_view_diagnostic(vertices,np.array([[0,1,2]]),view)
    assert r['samples']['vertices']['foreground']==3
    assert r['samples']['triangle_centers']['background']==1
    assert not r['adopted'] and 'not full-triangle' in r['semantics']


def test_aggregate_export_budget_requires_split_not_truncation():
    r=p.combined_layer_budget(dict(observed_surface=dict(vertices=150000,faces=250000),inferred_shell=dict(vertices=90000,faces=200000)))
    assert r['vertices']==240000 and r['faces']==450000 and not r['combined_allowed']
    assert p.combined_layer_budget(dict(observed_surface=dict(vertices=1000,faces=100)))['combined_allowed']
