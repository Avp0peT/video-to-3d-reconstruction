import numpy as np
import pytest
from kozlov_lab.stereo.appearance_evidence import compare_patches, descriptor_evidence


def images():
    a=np.random.default_rng(42).integers(0,80,(80,80),dtype=np.uint8)
    return a, np.full(a.shape,255,np.uint8)


@pytest.mark.parametrize('radius',[8,16])
def test_identity_linear_brightness_swap_and_immutable(radius):
    a,m=images();target=(a*2+50).astype(np.uint8);before=[x.copy() for x in (a,m,target)]
    r=compare_patches(a,m,[25.25,25.5],target,m,[25.25,25.5],[55.25,55.5],radius)
    assert r['status']=='EVALUATED' and r['original_zncc']==pytest.approx(1)
    assert r['support']=='original'
    swapped=compare_patches(a,m,[25.25,25.5],target,m,[55.25,55.5],[25.25,25.5],radius)
    assert swapped['support']=='alternative'
    assert all(np.array_equal(x,y) for x,y in zip((a,m,target),before))


def test_mask_boundary_checks_positive_weight_neighbors():
    a,m=images();m[33,25]=0
    r=compare_patches(a,m,[25.5,24.5],a,m,[50,50],[55,55],8)
    assert r['status']=='NOT_EVALUATED_MASK_BOUNDARY'
    assert r['patch_statistics']['reference']['foreground_fraction']<1
    assert len(r['patch_statistics'])==3


def test_integer_edge_zero_weight_does_not_require_external_neighbor():
    a,m=images()
    r=compare_patches(a,m,[71,71],a,m,[71,71],[30,30],8)
    assert r['status']=='EVALUATED'


def test_bounds_constant_and_tie():
    a,m=images()
    assert compare_patches(a,m,[7.9,20],a,m,[20,20],[30,30],8)['status']=='NOT_EVALUATED'
    constant=np.full(a.shape,60,np.uint8)
    assert compare_patches(constant,m,[20,20],a,m,[20,20],[30,30],8)['status']=='NOT_EVALUATED'
    assert compare_patches(a,m,[20,20],a,m,[20,20],[20,20],8)['support']=='tie'


@pytest.mark.parametrize('bad',['boolxy','nan','masklabels','boolimage','radius'])
def test_invalid_inputs(bad):
    a,m=images();xy=[20.,20.];radius=8
    if bad=='boolxy':xy[0]=True
    elif bad=='nan':xy[0]=float('nan')
    elif bad=='masklabels':m[0,0]=1
    elif bad=='boolimage':a=a.astype(bool)
    elif bad=='radius':radius=9
    with pytest.raises(ValueError):compare_patches(a,m,xy,a,m,[20,20],[40,40],radius)


def test_descriptor_identity_difference_swap_zero_and_immutable():
    a=np.zeros(128,np.uint8);a[0]=200;b=np.zeros(128,np.uint8);b[1]=200;before=a.copy()
    r=descriptor_evidence(a,a,b)
    assert r['original_distance']==0 and r['alternative_distance']==pytest.approx(np.sqrt(2))
    assert r['support']=='original' and descriptor_evidence(a,b,a)['support']=='alternative'
    assert descriptor_evidence(a,a,a)['support']=='tie'
    assert descriptor_evidence(a,a,np.zeros(128,np.uint8))['status']=='NOT_EVALUATED'
    assert np.array_equal(a,before)
    with pytest.raises(ValueError):descriptor_evidence(a,a,a.astype(bool))
