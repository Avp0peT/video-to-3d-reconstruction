import numpy as np
import pytest
from integrations.video_masks.face_edge_gate import EdgePolicy,evaluate_edge_view


def evaluate(points,mask=None,valid=None,depth=None,policy=EdgePolicy(),check=lambda:None):
    if mask is None:
        mask=np.zeros((100,100),bool); mask[10:90,10:90]=True
    if valid is None: valid=np.ones_like(mask)
    return evaluate_edge_view(np.array(points,float),np.array([[0,1,2]]),K=np.eye(3),R=np.eye(3),C=np.zeros(3),
        mask=mask,validity=valid,depth=depth,view_id='test',policy=policy,check=check)


def test_small_edge_face_eligible_no_interior_veto():
    row=evaluate([[9,45,1],[11,45,1],[10,47,1]],depth=np.ones((100,100)))
    assert row['edge_eligible'].tolist()==[True]
    assert not row['core_protected'][0]
    assert row['enclosing_contour_bound_pixels'][0]<=row['edge_band_pixels']==6


def test_small_deep_interior_is_protected_not_edge():
    row=evaluate([[40,40,1],[44,40,1],[40,44,1]],depth=np.ones((100,100)))
    assert row['core_protected'].tolist()==[True]
    assert row['protected_sample_count'].tolist()==[4]
    assert not row['edge_eligible'][0]


def test_large_crossing_triangle_cannot_be_deleted_as_edge():
    row=evaluate([[8,10,1],[90,10,1],[10,90,1]],depth=np.ones((100,100)))
    assert not row['edge_eligible'][0]
    assert row['core_protected'][0]  # centroid anchors otherwise unprotected corners


@pytest.mark.parametrize('depth',[None,np.zeros((100,100)),np.full((100,100),np.nan),np.full((100,100),.5)])
def test_missing_invalid_or_occluding_depth_is_not_positive_support(depth):
    row=evaluate([[40,40,1],[44,40,1],[40,44,1]],depth=depth)
    assert not row['core_protected'][0]
    assert not row['edge_eligible'][0]


def test_unknown_hole_inside_disk_blocks_edge():
    mask=np.zeros((100,100),bool);mask[10:90,10:90]=True
    valid=np.ones_like(mask);valid[44,9]=False
    row=evaluate([[9,45,1],[11,45,1],[10,47,1]],mask,valid)
    assert not row['edge_eligible'][0]


def test_image_border_not_an_object_contour():
    mask=np.ones((100,100),bool)
    row=evaluate([[1,40,1],[3,40,1],[1,43,1]],mask,depth=np.ones((100,100)))
    assert not row['edge_eligible'][0]
    assert row['core_protected'][0]


def test_empty_mask_and_out_of_frame_do_not_supply_eligibility():
    assert not evaluate([[1,1,1],[2,1,1],[1,2,1]],np.zeros((100,100),bool))['edge_eligible'][0]
    assert not evaluate([[-1,1,1],[2,1,1],[1,2,1]])['edge_eligible'][0]


def test_same_image_coordinate_far_background_not_edge():
    assert not evaluate([[1,1,1],[3,1,1],[1,3,1]])['edge_eligible'][0]


def test_interior_support_not_based_on_parent_depth_or_relative_rgb():
    points=[[80,80,2],[84,80,2],[80,84,2]]  # projects to 40..42
    assert not evaluate(points,depth=np.ones((100,100)))['core_protected'][0]
    assert evaluate(points,depth=np.full((100,100),2.))['core_protected'][0]


def test_deadline_input_immutability_and_policy_validation():
    mask=np.zeros((100,100),bool);mask[10:90,10:90]=True
    copy=mask.copy(); evaluate([[9,45,1],[11,45,1],[10,47,1]],mask)
    assert np.array_equal(mask,copy)
    def stop():raise TimeoutError('test stop')
    with pytest.raises(TimeoutError):evaluate([[9,45,1],[11,45,1],[10,47,1]],check=stop)
    for args in ({'band_extent_fraction':True},{'maximum_band_px':float('nan')},{'core_band_multiplier':1},
                 {'minimum_band_px':40,'maximum_band_px':32}):
        with pytest.raises(ValueError): EdgePolicy(**args)
