"""Bound/domain and adapter contract tests; native algorithms are mocked."""
import copy
import json
from pathlib import Path
import numpy as np
import pytest
from integrations.mvs_surface import depth_fusion_pipeline as pipeline


def views():
    result=[]
    for i in range(2):
        depth=np.full((5,5),2.,np.float32) if i==0 else np.zeros((5,5),np.float32)
        if i:depth[1,1]=3
        result.append(dict(view_id=i,image=np.full((5,5,3),100,np.uint8),mask=np.ones((5,5),bool),validity=np.ones((5,5),bool),depth=depth,depth_available=True,K=np.eye(3),R=np.eye(3),C=np.zeros(3)))
    return result


def test_native_bounds_every_foreground_pixel_including_image_edge():
    v=views();before=copy.deepcopy(v)
    bounds,meta=pipeline.native_bounds(v,16)
    np.testing.assert_allclose(meta['raw_bounds'],[[0,0,2],[8,8,3]])
    assert meta['padding']==2
    np.testing.assert_allclose(bounds,[[-2,-2,0],[10,10,5]])
    assert [x['native_foreground_pixels'] for x in meta['per_view']]==[25,1]
    # An extreme observed pixel, far outside any old mesh, must change the box.
    v[0]['depth'][4,4]=20
    _,extreme=pipeline.native_bounds(v,16)
    np.testing.assert_allclose(extreme['raw_bounds'][1],[80,80,20])
    for old,new in zip(before,views()):
        for key in old:
            if isinstance(old[key],np.ndarray):np.testing.assert_array_equal(old[key],new[key])


def test_large_source_regular_sampling_declared_and_not_prefix_truncated():
    v=views();v[0]['mask']=np.ones((100,100),bool);v[0]['validity']=v[0]['mask'].copy();v[0]['depth']=np.ones((100,100),np.float32)
    stride,count=pipeline.coverage_stride(v,max_points=100)
    assert stride>3 and count<=100
    actual=sum(np.count_nonzero(x['mask'][::stride,::stride]&x['validity'][::stride,::stride]&(x['depth'][::stride,::stride]>0)&x['depth_available']) for x in v)
    assert count==actual
    with pytest.raises(ValueError):pipeline.coverage_stride(v,True)


@pytest.mark.parametrize('excluded_by',['mask','validity'])
def test_bounds_respect_foreground_validity_without_quantile_cropping(excluded_by):
    v=views();v[0]['depth'][4,4]=20;v[0][excluded_by][4,4]=False
    _,meta=pipeline.native_bounds(v,16)
    assert meta['raw_bounds'][1][2]==3
    assert meta['per_view'][0]['native_foreground_pixels']==24


def test_missing_native_depth_never_invents_bounds():
    v=views()
    for item in v:item['depth_available']=False
    with pytest.raises(ValueError,match='bounds unavailable'):pipeline.native_bounds(v,16)
    v=views();v[1]['depth_available']=False
    with pytest.raises(ValueError,match='bounds unavailable'):pipeline.native_bounds(v,16)


@pytest.mark.parametrize('axis',[True,7,193,16.])
def test_native_bounds_axis_validation(axis):
    with pytest.raises(ValueError):pipeline.native_bounds(views(),axis)


def test_native_bounds_nonfinite_projection_and_cancellation():
    v=views();v[0]['C'][0]=np.inf
    with pytest.raises(ValueError,match='nonfinite native point'):pipeline.native_bounds(v,16)
    def stop():raise RuntimeError('cancelled')
    with pytest.raises(RuntimeError,match='cancelled'):pipeline.native_bounds(views(),16,stop)


def review():
    return dict(topology=dict(vertices=3,faces=1,connected_components=1,boundary_edges=3,nonmanifold_edges=0,degenerate_faces=0,components=[{'not':'summary'}]),
        centroid_classification_legend={'0':'UNVERIFIED','1':'SUPPORTED','2':'CONFLICT'},centroid_classification=np.array([0,1,1,2],np.uint8),
        face_known_background_votes=np.array([0,1,2]),face_foreground_votes=np.array([0,2,3]),sampling='Mock four samples; not exhaustive',geometry_accepted=False)


def test_review_summary_string_json_keys_and_distinct_vote_mappings():
    result=pipeline.review_summary(review())
    assert result['centroid_labels']=={'UNVERIFIED':1,'SUPPORTED':2,'CONFLICT':1}
    assert 'components' not in result['topology']
    assert result['faces_with_any_sampled_background']==2
    assert result['faces_with_at_least_two_foreground_views']==2
    assert result['geometry_accepted'] is False


def test_run_rejects_existing_or_overlapping_output_before_loading(tmp_path,monkeypatch):
    source=tmp_path/'source';source.mkdir();existing=tmp_path/'existing';existing.mkdir()
    monkeypatch.setattr(pipeline,'load_model_inputs',lambda *a:pytest.fail('must reject before loading'))
    for out in (existing,source/'nested',tmp_path):
        with pytest.raises(ValueError,match='fresh disjoint'):pipeline.run(source,out)


@pytest.mark.parametrize('options',[{'max_axis':True},{'min_views':1},{'truncation_voxels':np.nan},{'truncation_voxels':0},{'seconds':False},{'blender':1}])
def test_run_invalid_parameters_rejected(tmp_path,options):
    source=tmp_path/'source';source.mkdir()
    with pytest.raises(ValueError):pipeline.run(source,tmp_path/'out',**options)


def fixture(tmp_path,monkeypatch,empty=False):
    source=tmp_path/'source';source.mkdir();out=tmp_path/'new'
    v=views();vertices=np.array([[0.,0.,2.],[1.,0.,2.],[0.,1.,2.]])
    faces=np.array([[0,1,2]],np.int64)
    monkeypatch.setattr(pipeline,'load_model_inputs',lambda path,check:dict(views=v,bindings=[],reference_id=0,local_vertices=vertices,local_faces=faces,global_points=vertices))
    summary=dict(total_valid_mask_foreground_pixels=50,total_native_depth_foreground_pixels=26,total_foreground_without_native_depth=24)
    monkeypatch.setattr(pipeline,'classify_depth_coverage',lambda *a,**k:dict(status='COMPLETE',coverage_view_0=np.zeros((5,5),np.uint8),coverage_view_1=np.zeros((5,5),np.uint8),summary=summary))
    def fused(*a,**kwargs):
        assert kwargs['max_axis']==16 and kwargs['min_views']==2 and kwargs['truncation_voxels']==3
        return dict(vertices=vertices,faces=faces if not empty else np.empty((0,3),np.int64),colors=np.full((3,3),100,np.uint8),face_common_near_bits=np.array([3],np.uint16),counts=dict(vertices=3,faces=0 if empty else 1))
    monkeypatch.setattr(pipeline,'fuse_depth_views',fused)
    monkeypatch.setattr(pipeline,'review_mesh',lambda *a:review())
    # Use real tiny projection/panel exports so path contracts are not mocked away.
    return source,out


def test_mocked_run_exports_terminal_result_exact_arrays_and_working_html(tmp_path,monkeypatch):
    source,out=fixture(tmp_path,monkeypatch)
    # Real lightweight PLY, NPZ and HTML exporters exercise their actual signatures.
    result=pipeline.run(source,out,max_axis=16)
    assert result['status']=='FUSED_SURFACE_GENERATED_NOT_GEOMETRY_ACCEPTED'
    assert result['blender']['status']=='NOT_RUN' and result['geometry_accepted'] is False
    assert (out/'fused_surface.ply').is_file() and (out/'fused_surface.obj').is_file()
    assert (out/'RESULT.json').is_file() and (out/'index.html').is_file()
    assert (out/'viewer/viewer.html').is_file()
    from PIL import Image
    with Image.open(out/'coverage/view_0.png') as image:assert image.format=='PNG'
    with np.load(out/'FUSION.npz',allow_pickle=False) as arrays:
        np.testing.assert_array_equal(arrays['faces'],[[0,1,2]])
    saved=json.loads((out/'COMPARISON.json').read_text(encoding='utf-8'))
    assert saved['common_near_views_per_cell']=={'zero':0,'one':0,'two_or_more':1}
    assert len(saved['per_view'])==2


def test_empty_fusion_records_failure_no_fake_hull(tmp_path,monkeypatch):
    source,out=fixture(tmp_path,monkeypatch,empty=True)
    with pytest.raises(ValueError,match='no supported TSDF'):pipeline.run(source,out,max_axis=16)
    failed=json.loads((out/'FAILURE.json').read_text(encoding='utf-8'))
    assert failed['status']=='INCOMPLETE'
    assert not (out/'fused_surface.ply').exists() and not (out/'RESULT.json').exists()
