import copy
import numpy as np
import pytest
from integrations.mvs_surface.fused_surface_mask_gate import gate_surface


def scene():
    # Identity K: native UV equals XY/Z; all three vertex samples are foreground.
    vertices=np.array([[2.,2.,2.],[14.,2.,2.],[14.,14.,2.]])
    faces=np.array([[0,1,2]],np.int64)
    view=dict(view_id=1,image=np.full((10,10,3),100,np.uint8),mask=np.ones((10,10),bool),validity=np.ones((10,10),bool),depth=np.full((10,10),2.,np.float32),K=np.eye(3),R=np.eye(3),C=np.zeros(3),depth_available=True)
    return vertices,faces,[view]


def test_tiny_gap_inside_triangle_rejected_not_only_vertices_or_centroid():
    v,f,views=scene();views[0]['mask'][2,5]=False
    r=gate_surface(v,f,views)
    assert not r['keep'][0] and r['reason'][0]==1
    assert r['known_background_view_count'][0]==1
    assert r['background_view_bits'][0]==1


def test_bbox_background_outside_footprint_does_not_reject():
    v,f,views=scene();views[0]['mask'][6,1]=False
    r=gate_surface(v,f,views)
    assert r['keep'][0] and r['full_foreground_view_count'][0]==1
    assert not r['bbox_fast_accept'][0,0] and r['sat_touched_pixel_count'][0,0]>0


def test_unknown_interior_not_background_or_support():
    v,f,views=scene();views[0]['validity'][2,5]=False;views[0]['mask'][2,5]=False
    r=gate_surface(v,f,views)
    assert r['keep'][0] and r['view_state'][0,0]==4
    assert r['full_foreground_view_count'][0]==0


def test_inside_occlusion_is_separate_diagnostic_and_no_mutation():
    v,f,views=scene();views[0]['depth'][:]=1;before=copy.deepcopy((v,f,views))
    r=gate_surface(v,f,views)
    assert r['keep'][0] and r['full_foreground_view_count'][0]==1
    assert r['occluded'][0,0] and r['occlusion_evaluated'][0,0]
    assert r['bbox_fast_accept'][0,0]
    np.testing.assert_array_equal(v,before[0]);np.testing.assert_array_equal(f,before[1])
    for key in views[0]:
        if isinstance(views[0][key],np.ndarray):np.testing.assert_array_equal(views[0][key],before[2][0][key])


def test_offimage_and_camera_plane_not_known_background():
    v,f,views=scene();views[0]['C'][0]=100
    r=gate_surface(v,f,views)
    assert r['keep'][0] and r['view_state'][0,0]==2 and r['off_image'][0,0]
    v,f,views=scene();v[0,2]=-1
    r=gate_surface(v,f,views)
    assert r['keep'][0] and r['view_state'][0,0]==5


def test_partial_offimage_can_still_have_known_background_veto():
    v,f,views=scene();v[0,0]=-2;views[0]['mask'][2,3]=False
    r=gate_surface(v,f,views)
    assert not r['keep'][0] and r['off_image'][0,0]


def test_missing_depth_has_mask_evidence_but_no_occlusion_evaluation():
    v,f,views=scene();views[0]['depth_available']=False;views[0]['depth'][:]=0
    r=gate_surface(v,f,views)
    assert r['keep'][0] and r['view_state'][0,0]==0
    assert not r['occlusion_evaluated'].any()


def test_duplicate_invalid_indices_nonfinite_validity_and_cancel():
    v,f,views=scene()
    with pytest.raises(ValueError):gate_surface(v,np.array([[0,1,3]]),views)
    bad=v.copy();bad[0,0]=np.nan
    with pytest.raises(ValueError):gate_surface(bad,f,views)
    with pytest.raises(ValueError):gate_surface(v,f,views+copy.deepcopy(views))
    views[0]['validity']=np.ones((10,10),np.uint8)
    with pytest.raises(ValueError):gate_surface(v,f,views)
    def stop():raise RuntimeError('cancel')
    with pytest.raises(RuntimeError,match='cancel'):gate_surface(v,f,scene()[2],check=stop)


@pytest.mark.parametrize('count',[16,64,67,128])
def test_background_bitset_words_preserve_columns_beyond16_and64(count):
    v,f,base=scene();selected=[]
    expected=[index for index in (0,15,16,63,64,66,127) if index<count]
    for index in range(count):
        view=copy.deepcopy(base[0]);view['view_id']=1000+index
        if index in expected:view['mask'][2,5]=False
        selected.append(view)
    r=gate_surface(v,f,selected);bits=r['background_view_bits']
    assert bits.dtype==np.uint64
    assert bits.shape==((1,) if count<=64 else (1,2))
    words=bits.reshape(1,-1)[0]
    decoded=[index for index in range(count) if int(words[index//64])&(1<<(index%64))]
    assert decoded==expected
    assert int(r['known_background_view_count'][0])==len(expected)
    assert int(r['full_foreground_view_count'][0])==count-len(expected)
    assert not r['keep'][0]
    schema=r['background_view_bits_schema']
    assert schema['words']==(count+63)//64 and schema['view_ids']==[1000+i for i in range(count)]


def test_128_foreground_gate_votes_and_129_rejection():
    v,f,base=scene();selected=[]
    for index in range(128):
        item=copy.deepcopy(base[0]);item['view_id']=index;selected.append(item)
    r=gate_surface(v,f,selected)
    assert r['keep'][0] and int(r['full_foreground_view_count'][0])==128
    assert r['background_view_bits'].shape==(1,2) and not r['background_view_bits'].any()
    with pytest.raises(ValueError,match='128'):gate_surface(v,f,selected+[copy.deepcopy(base[0])])
