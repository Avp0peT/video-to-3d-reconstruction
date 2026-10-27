"""Synthetic adapter contracts; native evidence and fusion kernels remain real."""
from copy import deepcopy
import numpy as np
import pytest
from integrations.mvs_surface import mutual_model_pipeline as p


def fixture(tmp_path,monkeypatch,unknown_midpoint=False):
    source=tmp_path/'source';source.mkdir()
    gate=source/'sealed.bin';gate.write_bytes(b'sealed original native inputs')
    shell=source/'inferred_shell.ply';shell.write_bytes(b'NEVER INGEST: inferred geometry not native depth')
    views=[]
    for i,c in enumerate((-.06,-.02,.02,.06),1):
        valid=np.ones((64,64),bool)
        if unknown_midpoint and i!=1:valid[29:34,24:40]=False
        views.append(dict(view_id=i,image=np.full((64,64,3),110,np.uint8),mask=np.ones((64,64),bool),validity=valid,depth=np.full((64,64),2.,np.float32),K=np.array([[100.,0,32],[0,100.,32],[0,0,1]]),R=np.eye(3),C=np.array([c,0,0]),depth_available=True))
    vertices=np.array([[-.1,-.1,2.],[.1,-.1,2.],[0,.1,2.]])
    global_points=np.vstack((vertices[:1],[[.3,0,2.],[0,0,2.2],[0,0,1.8]]))
    data=dict(global_points=global_points,global_colors=np.tile(np.array([15,35,55],np.uint8),(4,1)),local_vertices=vertices,local_colors=np.tile(np.array([50,80,120],np.uint8),(3,1)),local_faces=np.array([[0,1,2]],np.int64),local_vertex_source=np.ones(3,np.int64),face_source=np.ones(1,np.int64),views=views,reference_id=1,bindings=[p.binding(gate),p.binding(shell)],historical_plan_binding=p.binding(gate),inferred_shell_vertices=np.array([[999.,999.,999.]]))
    before=deepcopy(data)
    monkeypatch.setattr(p,'load_model_inputs',lambda source,check:data)
    def export(spec,out,profile):
        out.write_text('<html>synthetic viewer only</html>')
        return dict(path=str(out))
    monkeypatch.setattr(p,'export_html',export)
    return source,tmp_path/'output',data,before,gate,shell


def test_native_evidence_outputs_readback_origin_maps_and_shell_never_ingested(tmp_path,monkeypatch):
    source,out,data,before,gate,shell=fixture(tmp_path,monkeypatch)
    protected={path:path.read_bytes() for path in (gate,shell)}
    result=p.run(source,out)
    assert result['status']=='SUPPORTED_MODEL_GENERATED_NOT_GEOMETRY_ACCEPTED'
    assert not result['geometry_accepted'] and not result['inferred_shell_adopted']
    assert result['global_counts']['supported']==2
    assert result['global_counts']['conflict']==1
    assert result['global_counts']['unverified']==1
    assert result['face_counts']['retained_faces']==1
    assert result['fusion_counts']['local_origins_exact_duplicate_global']==1
    with np.load(out/'FUSION_PROVENANCE.npz',allow_pickle=False) as z:
        assert z['global_to_output'][2:].tolist()==[-1,-1]
        assert z['global_to_output'][0]==z['local_to_output'][0]
        assert z['origin_offsets'][-1]==5
        assert np.isin(z['origin_kind'],[0,1]).all()
        assert z['faces'].shape==(1,3)
        assert not (z['vertices']==999).any()
        mesh=p.read_ply(out/'supported_model.ply',max_bytes=1<<20,max_vertices=20,max_faces=20)
        np.testing.assert_array_equal(np.asarray(mesh['vertices']),z['vertices'])
        np.testing.assert_array_equal(np.asarray(mesh['faces']),z['faces'])
    for key in ('global_points','global_colors','local_vertices','local_colors','local_faces','local_vertex_source','face_source'):
        np.testing.assert_array_equal(data[key],before[key])
    for path,raw in protected.items():assert path.read_bytes()==raw
    assert (out/'supported_model.obj').is_file()
    assert (out/'overlays/global_4.png').is_file()
    assert (out/'index.html').is_file()
    face=p.load(out/'FACE_EVIDENCE.json')
    assert 'not exhaustive' in face['semantics']
    assert result['blender']['status']=='NOT_RUN'


def test_unknown_face_midpoints_reject_bridge_despite_supported_vertices(tmp_path,monkeypatch):
    source,out,data,_,_,_=fixture(tmp_path,monkeypatch,unknown_midpoint=True)
    result=p.run(source,out)
    assert result['local_counts']['supported']==3
    assert result['face_counts']['vertex_eligible']==1
    assert result['face_counts']['retained_faces']==0
    assert result['new_meshes']==0
    assert not (out/'supported_model.ply').exists()
    with np.load(out/'FACE_EVIDENCE.npz',allow_pickle=False) as z:
        assert not z['supported'].any()
        assert (z['sample_state']==4).any()
    with np.load(out/'FUSION_PROVENANCE.npz',allow_pickle=False) as z:
        assert z['faces'].shape==(0,3)
    assert 'not complete per-pixel' in ' '.join(result['limitations'])


def test_injected_bound_drift_prevents_success_status(tmp_path,monkeypatch):
    source,out,_,_,gate,_=fixture(tmp_path,monkeypatch)
    def corrupt(spec,target,profile):
        gate.write_bytes(b'changed after binding')
        target.write_text('<html>synthetic</html>')
        return dict(path=str(target))
    monkeypatch.setattr(p,'export_html',corrupt)
    with pytest.raises(ValueError,match='bound input/code changed'):
        p.run(source,out)
    result=p.load(out/'RESULT.json')
    assert result['status']=='FAILED_SUPPORT_MODEL'
    assert not result['geometry_accepted']


def test_output_must_be_fresh_and_disjoint(tmp_path,monkeypatch):
    source,out,*_=fixture(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='fresh disjoint'):
        p.run(source,source/'nested')
    out.mkdir()
    with pytest.raises(ValueError,match='fresh disjoint'):
        p.run(source,out)


def test_incomplete_production_source_rejected_without_native_reads(tmp_path):
    source=tmp_path/'incomplete';source.mkdir()
    p.save(source/'RESULT.json',dict(status='RUNNING'))
    p.save(source/'PLAN.json',{})
    with pytest.raises((ValueError,FileNotFoundError)):
        p.load_model_inputs(source,lambda:None)


def test_command_routes_once_and_requires_blender_success(monkeypatch,capsys):
    from scripts import run_support_gated_model as cli
    calls=[]
    def fake(*args,**kwargs):
        calls.append((args,kwargs))
        return dict(status='SUPPORTED_MODEL_GENERATED_NOT_GEOMETRY_ACCEPTED',global_counts={},local_counts={},fusion_counts={},wall_seconds=0.,blender=dict(status='FAILED'))
    monkeypatch.setattr(cli,'run',fake)
    assert cli.main(['--source-model','source','--output','out','--blender'])==1
    assert len(calls)==1 and calls[0][1]['blender'] is True
    assert 'SUPPORTED_MODEL' in capsys.readouterr().out



def _with_unsupported_local_points(data):
    # The original face stays unchanged. Additional contradictory/occluded local
    # candidates are isolated vertices, never permission to invent new triangles.
    data['local_vertices']=np.vstack((data['local_vertices'],[[0.,0.,1.8],[0.,0.,2.2]]))
    data['local_colors']=np.vstack((data['local_colors'],[[200,40,40],[180,140,30]])).astype(np.uint8)
    data['local_vertex_source']=np.ones(5,np.int64)


def test_all_points_adopts_local_conflict_unknown_without_relabeling_evidence(tmp_path,monkeypatch):
    source,out,data,_,_,_=fixture(tmp_path,monkeypatch)
    _with_unsupported_local_points(data)
    result=p.run(source,out,local_policy='all-points')
    assert not result['geometry_accepted']
    assert result['global_counts']['supported']==2
    assert result['local_counts']['supported']==3
    assert result['local_counts']['conflict']==1
    assert result['local_counts']['unverified']==1
    with np.load(out/'LOCAL_EVIDENCE.npz',allow_pickle=False) as local:
        assert local['classification'].tolist()==[1,1,1,2,0]
        assert local['supported'].tolist()==[True,True,True,False,False]
    with np.load(out/'FUSION_PROVENANCE.npz',allow_pickle=False) as fusion:
        # Global candidates at z1.8/z2.2 remain excluded as GLOBAL origins,
        # while the explicitly adopted local counterparts retain LOCAL origins.
        assert fusion['global_to_output'][2:].tolist()==[-1,-1]
        assert (fusion['local_to_output']>=0).all()
        assert fusion['local_evidence_supported'].tolist()==[True,True,True,False,False]
        assert fusion['faces'].shape==(1,3)
        mesh=p.read_ply(out/'local_all_model.ply',max_bytes=1<<20,max_vertices=20,max_faces=20)
        np.testing.assert_array_equal(np.asarray(mesh['vertices']),fusion['vertices'])
        assert len(fusion['vertices'])==6
    assert (out/'local_all_combined_points.ply').is_file()
    assert (out/'local_evidence_audit.ply').is_file()
    assert not (out/'supported_model.ply').exists()
    assert not (out/'supported_combined_points.ply').exists()


def test_all_points_does_not_relax_unknown_face_sample_rejection(tmp_path,monkeypatch):
    source,out,data,_,_,_=fixture(tmp_path,monkeypatch,unknown_midpoint=True)
    _with_unsupported_local_points(data)
    result=p.run(source,out,local_policy='all-points')
    assert result['face_counts']['vertex_eligible']==1
    assert result['face_counts']['retained_faces']==0
    assert result['new_meshes']==0
    with np.load(out/'FUSION_PROVENANCE.npz',allow_pickle=False) as fusion:
        assert (fusion['local_to_output']>=0).all()
        assert fusion['faces'].shape==(0,3)
        assert fusion['local_evidence_supported'].tolist()==[True,True,True,False,False]
    assert (out/'local_all_combined_points.ply').is_file()
    assert not (out/'local_all_model.ply').exists()


def test_cli_explicit_all_points_policy_routes_once(monkeypatch,capsys):
    from scripts import run_support_gated_model as cli
    calls=[]
    def fake(*args,**kwargs):
        calls.append((args,kwargs))
        return dict(status='SUPPORTED_MODEL_GENERATED_NOT_GEOMETRY_ACCEPTED',global_counts={},local_counts={},fusion_counts={},wall_seconds=0.,blender=dict(status='NOT_RUN'))
    monkeypatch.setattr(cli,'run',fake)
    cli.main(['--source-model','source','--output','out','--local-policy','all-points'])
    assert len(calls)==1
    assert calls[0][1]['local_policy']=='all-points'
    capsys.readouterr()



def test_all_original_points_and_faces_preserve_coordinates_and_evidence(tmp_path,monkeypatch):
    source,out,data,_,_,_=fixture(tmp_path,monkeypatch,unknown_midpoint=True)
    _with_unsupported_local_points(data)
    result=p.run(source,out,local_policy='all-points',global_policy='all-points',face_policy='original')
    assert result['status']=='ALL_ORIGINAL_COMPARISON_NOT_GEOMETRY_ACCEPTED'
    assert not result['geometry_accepted']
    assert not result['inferred_shell_adopted']
    # Evidence audit still rejects the unknown interior samples, while the new
    # explicit original-face admission keeps this unverified original face.
    assert result['face_counts']['retained_faces']==0
    with np.load(out/'GLOBAL_EVIDENCE.npz',allow_pickle=False) as g:
        assert g['classification'].tolist()==[1,1,0,2]
    with np.load(out/'LOCAL_EVIDENCE.npz',allow_pickle=False) as l:
        assert not l['supported'][3:].any()
    with np.load(out/'FACE_EVIDENCE.npz',allow_pickle=False) as f:
        assert not f['supported'].any()
        assert (f['sample_state']==4).any()
    with np.load(out/'FUSION_PROVENANCE.npz',allow_pickle=False) as fusion:
        assert (fusion['global_to_output']>=0).all()
        assert (fusion['local_to_output']>=0).all()
        np.testing.assert_array_equal(fusion['vertices'][fusion['global_to_output']],data['global_points'])
        np.testing.assert_array_equal(fusion['vertices'][fusion['local_to_output']],data['local_vertices'])
        assert fusion['faces'].shape==(1,3)
        np.testing.assert_array_equal(fusion['vertices'][fusion['faces']],data['local_vertices'][data['local_faces']])
        assert fusion['face_original_ids'].tolist()==[0]
        assert fusion['origin_offsets'][-1]==9
        assert not (fusion['vertices']==999).any()
        model=p.read_ply(out/'all_original_model.ply',max_bytes=1<<20,max_vertices=20,max_faces=20)
        np.testing.assert_array_equal(np.asarray(model['vertices']),fusion['vertices'])
        np.testing.assert_array_equal(np.asarray(model['faces']),fusion['faces'])
    assert (out/'all_original_points.ply').is_file()
    assert (out/'all_original_points_viewer.html').is_file()
    assert not (out/'local_all_model.ply').exists()
    assert not (out/'supported_model.ply').exists()


def test_original_faces_require_all_local_points_before_loading(tmp_path,monkeypatch):
    source,out,*_=fixture(tmp_path,monkeypatch)
    called=[]
    def forbidden(*args,**kwargs):
        called.append(True)
        raise AssertionError('invalid policy combination reached input loading')
    monkeypatch.setattr(p,'load_model_inputs',forbidden)
    with pytest.raises(ValueError):
        p.run(source,out,global_policy='all-points',face_policy='original')
    assert not called


def test_cli_all_original_routes_all_three_explicit_policies_once(monkeypatch,capsys):
    from scripts import run_support_gated_model as cli
    calls=[]
    def fake(*args,**kwargs):
        calls.append((args,kwargs))
        return dict(status='ALL_ORIGINAL_COMPARISON_NOT_GEOMETRY_ACCEPTED',global_counts={},local_counts={},fusion_counts={},wall_seconds=0.,blender=dict(status='NOT_RUN'))
    monkeypatch.setattr(cli,'run',fake)
    assert cli.main(['--source-model','source','--output','out','--global-policy','all-points','--local-policy','all-points','--face-policy','original'])==0
    assert len(calls)==1
    for key,value in [('global_policy','all-points'),('local_policy','all-points'),('face_policy','original')]:
        assert calls[0][1][key]==value
    capsys.readouterr()
