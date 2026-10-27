"""Finalizer contracts with synthetic evidence and real lightweight exporters."""
import copy
import json
from pathlib import Path
import numpy as np
import pytest
from integrations.mvs_surface import depth_fusion_finalize as product
from integrations.viewer.ply_data import read_ply


def evidence():
    return dict(keep=np.array([True,False,True,True,True,True]),full_foreground_view_count=np.array([2,2,1,2,2,2],np.uint8)),np.array([3,3,3,1,3,3],np.uint16),np.array([1,1,1,1,0,2],np.uint8)


def test_four_admission_gates_and_no_evidence_mutation():
    gate,bits,labels=evidence();before=copy.deepcopy((gate,bits,labels))
    r=product.select_additions(gate,bits,labels)
    assert r['accepted'].tolist()==[True,False,False,False,False,False]
    assert r['counts']==dict(candidates=6,mask_gate_pass=5,two_complete_foreground_views=5,two_common_near_views=5,centroid_supported=4,accepted=1)
    for key in gate:np.testing.assert_array_equal(gate[key],before[0][key])
    np.testing.assert_array_equal(bits,before[1]);np.testing.assert_array_equal(labels,before[2])


def test_64_native_source_bits_and_67_mask_votes_are_not_truncated():
    gate=dict(keep=np.array([True,True]),full_foreground_view_count=np.array([67,128],np.uint8))
    bits=np.array([(1<<63)|(1<<62),(1<<64)-1],np.uint64)
    result=product.select_additions(gate,bits,np.array([1,1],np.uint8))
    assert result['accepted'].tolist()==[True,True]
    assert result['common_near_view_count'].tolist()==[2,64]


@pytest.mark.parametrize('field,value',[
    ('keep',np.array([1])),('keep',np.array([[True]])),('keep',np.array(True)),
    ('fg',np.array([-1])),('fg',np.array([129])),('fg',np.array([2.])),
    ('bits',np.array([-3])),('bits',np.array([2**64],dtype=object)),('bits',np.array([3.])),
    ('labels',np.array([3])),('labels',np.array(['SUPPORTED'])),('labels',np.array([True])),
    ('labels',np.array([1,1]))])
def test_shapes_types_negative_sources_and_unknown_vocabulary_rejected(field,value):
    gate=dict(keep=np.array([True]),full_foreground_view_count=np.array([2],np.uint8));bits=np.array([3],np.uint16);labels=np.array([1],np.uint8)
    if field=='keep':gate['keep']=value
    elif field=='fg':gate['full_foreground_view_count']=value
    elif field=='bits':bits=value
    else:labels=value
    with pytest.raises(ValueError):product.select_additions(gate,bits,labels)


def setup(tmp_path,monkeypatch):
    model=tmp_path/'model';model.mkdir();source=tmp_path/'fusion';source.mkdir();out=tmp_path/'final'
    tracked=model/'bound.dat';tracked.write_bytes(b'UNCHANGED SOURCE')
    views=[]
    for iid in (1,2):
        views.append(dict(view_id=iid,image=np.full((16,16,3),110,np.uint8),mask=np.ones((16,16),bool),validity=np.ones((16,16),bool),
            depth=np.full((16,16),2.,np.float32),depth_available=True,K=np.array([[10.,0,8],[0,10.,8],[0,0,1]]),R=np.eye(3),C=np.zeros(3)))
    old_v=np.array([[-.6,-.6,2.],[-.2,-.6,2.],[-.6,-.2,2.]])
    old_f=np.array([[0,1,2]],np.int64);old_c=np.full((3,3),[120,90,60],np.uint8)
    new_v=np.array([[.2,-.6,2.],[.6,-.6,2.],[.6,-.2,2.],[.2,.2,2.],[.6,.2,2.],[.6,.6,2.]])
    new_f=np.array([[0,1,2],[3,4,5]],np.int64)
    files={}
    for name,arrays in [('fusion',dict(vertices=new_v,faces=new_f,colors=np.full((6,3),[70,175,225],np.uint8),face_common_near_bits=np.array([3,1],np.uint16))),
                        ('review',dict(centroid_classification=np.array([1,1],np.uint8))),
                        ('coverage',dict(coverage_view_1=np.full((16,16),3,np.uint8),coverage_view_2=np.full((16,16),3,np.uint8)))]:
        path=source/(name+'.npz');np.savez_compressed(path,**arrays);files[name]=product.binding(path)
    product.save(source/'PLAN.json',dict(source=str(model),view_ids=[1,2]))
    product.save(source/'RESULT.json',dict(status='FUSED_SURFACE_GENERATED_NOT_GEOMETRY_ACCEPTED',source_bindings=[product.binding(tracked)],artifacts=dict(fusion=dict(arrays=files['fusion']),new_review=dict(arrays=files['review']),coverage=dict(arrays=files['coverage']))))
    data=dict(views=views,bindings=[product.binding(tracked)],local_vertices=old_v,local_faces=old_f,local_colors=old_c)
    monkeypatch.setattr(product,'load_model_inputs',lambda path,check:data)
    return source,out,data,new_v,new_f,tracked


def mesh(path):return read_ply(path,max_bytes=1<<20,max_vertices=100,max_faces=100)


def test_real_tiny_gate_export_preserves_old_prefix_and_offsets_additions(tmp_path,monkeypatch):
    source,out,data,new_v,new_f,tracked=setup(tmp_path,monkeypatch)
    before=tracked.read_bytes();r=product.finalize(source,out)
    assert r['status']=='ORIGINAL_PLUS_AUDITED_FUSION_NOT_GEOMETRY_ACCEPTED'
    assert r['blender']['status']=='NOT_RUN' and not r['geometry_accepted']
    combined=mesh(out/'combined_surface.ply')
    vertices=np.asarray(combined['vertices']);faces=np.asarray(combined['faces'])
    np.testing.assert_array_equal(vertices[:3],data['local_vertices'])
    np.testing.assert_array_equal(faces[:1],data['local_faces'])
    np.testing.assert_array_equal(vertices[3:],new_v[:3])
    np.testing.assert_array_equal(faces[1:],new_f[:1]+3)
    assert r['admission']['accepted']==1 and r['projection_regression_views']==[]
    for panel in r['per_view']:
        counts=panel['counts']
        assert counts['new_foreground_covered']>=counts['old_foreground_covered']
        assert counts['new_background_projection']<=counts['old_background_projection']
    with np.load(out/'PROVENANCE.npz',allow_pickle=False) as arrays:
        np.testing.assert_array_equal(arrays['accepted_raw_face_ids'],[0])
        np.testing.assert_array_equal(arrays['accepted_raw_vertex_ids'],[0,1,2])
    provenance=json.loads((out/'PROVENANCE.json').read_text(encoding='utf-8'))
    assert provenance['old_vertices_exact'] and provenance['old_faces_exact']
    assert (out/'viewer/viewer.html').is_file() and (out/'index.html').is_file()
    assert tracked.read_bytes()==before


def test_bound_source_mutation_fails_final_acceptance(tmp_path,monkeypatch):
    source,out,_,_,_,tracked=setup(tmp_path,monkeypatch)
    original=product.review_mesh
    def mutate(*args,**kwargs):
        tracked.write_bytes(b'CHANGED DURING REVIEW');return original(*args,**kwargs)
    monkeypatch.setattr(product,'review_mesh',mutate)
    with pytest.raises(ValueError,match='input changed'):product.finalize(source,out)
    assert (out/'FAILURE.json').is_file() and not (out/'RESULT.json').exists()


def test_initial_changed_raw_artifact_rejected(tmp_path,monkeypatch):
    source,out,_,_,_,_=setup(tmp_path,monkeypatch)
    with (source/'fusion.npz').open('ab') as stream:stream.write(b'TAMPERED')
    with pytest.raises(ValueError,match='changed fusion artifact'):product.finalize(source,out)
    assert not out.exists()


def test_projection_regression_is_terminal_failure(tmp_path,monkeypatch):
    source,out,_,_,_,_=setup(tmp_path,monkeypatch);original=product.write_coverage_panel
    def regression(*args):
        panel=original(*args);panel['counts']['new_background_projection']=panel['counts']['old_background_projection']+1;return panel
    monkeypatch.setattr(product,'write_coverage_panel',regression)
    with pytest.raises(ValueError,match='projection regression'):product.finalize(source,out)
    assert json.loads((out/'FAILURE.json').read_text())['status']=='INCOMPLETE'
    assert not (out/'RESULT.json').exists()


def test_fresh_outputs_and_ancestor_overlap_rejected(tmp_path,monkeypatch):
    source,out,_,_,_,_=setup(tmp_path,monkeypatch);out.mkdir()
    for target in (out,source/'nested',tmp_path):
        with pytest.raises(ValueError,match='fresh disjoint'):product.finalize(source,target)


def cli(monkeypatch,ticks):
    import importlib.util
    import os
    import sys
    from types import SimpleNamespace
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        monkeypatch.setenv(key,os.environ.get(key,'2'))
    monkeypatch.setattr(sys,'dont_write_bytecode',sys.dont_write_bytecode)
    monkeypatch.setattr(sys,'path',sys.path.copy())
    path=Path(__file__).resolve().parents[1]/'scripts/run_depth_fusion_model.py'
    spec=importlib.util.spec_from_file_location('depth_fusion_cli_contract',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    clock=iter(ticks);monkeypatch.setattr(module,'time',SimpleNamespace(monotonic=lambda:next(clock)))
    return module


def test_cli_default_raw_then_final_only_one_blender_and_shared_deadline(tmp_path,monkeypatch):
    module=cli(monkeypatch,[100.,123.2,124.]);calls=[];out=tmp_path/'delivery'
    def raw(source,output,**kwargs):
        calls.append(('raw',Path(output),kwargs));Path(output).mkdir(parents=True)
        return dict(status='FUSED_SURFACE_GENERATED_NOT_GEOMETRY_ACCEPTED',blender={'status':'NOT_RUN'})
    def final(source,output,**kwargs):
        calls.append(('final',Path(source),Path(output),kwargs));Path(output).mkdir()
        return dict(status='ORIGINAL_PLUS_AUDITED_FUSION_NOT_GEOMETRY_ACCEPTED',blender={'status':'SUCCESS'})
    monkeypatch.setattr(module,'run',raw);monkeypatch.setattr(product,'finalize',final)
    assert module.main(['--source-model','synthetic','--output',str(out),'--blender','--seconds','100'])==0
    assert calls[0][0:2]==('raw',out/'raw')
    assert calls[0][2]['blender'] is False and calls[0][2]['seconds']==100
    assert calls[1][0:3]==('final',out/'raw',out/'final')
    assert calls[1][3]['blender'] is True and calls[1][3]['seconds']==76
    assert 'final/index.html' in (out/'index.html').read_text(encoding='utf-8')


def test_cli_raw_only_uses_requested_blender_without_finalization(tmp_path,monkeypatch):
    module=cli(monkeypatch,[100.,104.]);calls=[];out=tmp_path/'raw'
    def raw(source,output,**kwargs):
        calls.append((Path(output),kwargs));Path(output).mkdir()
        return dict(status='FUSED_SURFACE_GENERATED_NOT_GEOMETRY_ACCEPTED',blender={'status':'SUCCESS'})
    monkeypatch.setattr(module,'run',raw)
    monkeypatch.setattr(product,'finalize',lambda *a,**k:pytest.fail('raw-only must not finalize'))
    assert module.main(['--source-model','synthetic','--output',str(out),'--raw-only','--blender'])==0
    assert calls[0][0]==out and calls[0][1]['blender'] is True


def test_cli_exhausted_shared_deadline_does_not_restart_budget(tmp_path,monkeypatch):
    module=cli(monkeypatch,[100.,201.]);out=tmp_path/'delivery'
    def raw(source,output,**kwargs):
        Path(output).mkdir(parents=True);return dict(status='RAW',blender={'status':'NOT_RUN'})
    monkeypatch.setattr(module,'run',raw)
    monkeypatch.setattr(product,'finalize',lambda *a,**k:pytest.fail('must not reset exhausted deadline'))
    with pytest.raises(TimeoutError,match='shared automatic pipeline deadline'):
        module.main(['--source-model','synthetic','--output',str(out),'--seconds','100'])
    assert (out/'raw').is_dir() and not (out/'final').exists()
