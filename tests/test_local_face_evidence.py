import numpy as np
import pytest

from integrations.video_masks.local_face_evidence import (
    LocalPolicy, prepare_probe_membership, evaluate_local_view)


def scene():
    mask = np.zeros((120,120),bool)
    mask[:,10:30] = True; mask[:,80:110] = True
    valid = np.ones_like(mask)
    gap = np.zeros_like(mask); gap[10:110,34:76] = True
    return mask,valid,gap


def evaluate(vertices, faces=None, *, original=None, parents=None, eligible=None,
             mask=None, valid=None, gap=None, depth=None, membership=None, policy=LocalPolicy()):
    m,va,g = scene(); m=m if mask is None else mask; va=va if valid is None else valid; g=g if gap is None else gap
    v = np.asarray(vertices,float)
    f = np.array([[0,1,2]]) if faces is None else np.asarray(faces)
    ov,of = (v,f) if original is None else original
    ids = np.arange(len(f)) if parents is None else np.asarray(parents)
    eligible = np.ones(len(f),bool) if eligible is None else np.asarray(eligible)
    return evaluate_local_view(v,f,K=np.eye(3),R=np.eye(3),C=np.zeros(3),mask=m,validity=va,
        gap_mask=g,depth=depth,view_id='test',original_vertices=ov,original_faces=of,
        parent_ids=ids,eligible=eligible,policy=policy,probe_membership=membership)


def test_whole_child_gap_negative_without_depth_and_real_witnesses():
    row = evaluate([[48.2,40.2,1],[57.8,40.2,1],[48.2,49.8,1]])
    assert row['negative'].tolist() == [True]
    assert row['status'].tolist() == [2]
    assert row['actual_pixel_count'].tolist() == [4]
    assert row['pixel_count_is_lower_bound'].tolist() == [True]
    assert row['certified_gap_fraction'].tolist() == [1.]
    assert len(np.unique(row['pixel_witness_xy'][0],axis=0)) == 4
    assert not row['core_protected'][0]


def test_center_in_gap_does_not_certify_large_crossing_triangle():
    row = evaluate([[20,30,1],[90,30,1],[55,80,1]])
    assert not row['whole_gap_disk'][0]
    assert not row['negative'][0]


def test_actual_centres_not_projected_area_proxy():
    # Area exceeds four pixels, but this narrow strip contains no integer row.
    row = evaluate([[45.1,50.1,1],[65.1,50.1,1],[45.1,50.6,1]])
    assert row['projected_area_pixels'][0] > 4
    assert row['whole_gap_disk'][0] and row['deep_background_disk'][0]
    assert row['actual_pixel_count'][0] == 0 and not row['negative'][0]
    assert row['reason_names'][int(row['reason'][0])] == 'INSUFFICIENT_ACTUAL_PIXEL_CENTERS'


def split_fixture():
    original = np.array([[20.,20.,1.],[80.,20.,1.],[20.,80.,1.]])
    # Split through the original centroid along its line to the first vertex.
    v = np.vstack((original,(original[1]+original[2])/2))
    f = np.array([[0,1,3],[0,3,2]])
    return v,f,original,np.array([[0,1,2]])


def test_original_centroid_on_shared_child_edge_protects_both_incident_faces():
    v,f,ov,of = split_fixture()
    membership = prepare_probe_membership(v,f,ov,of,np.zeros(2,int))
    assert membership[:,3].tolist() == [True,True]
    mask = np.zeros((120,120),bool); mask[5:110,5:110] = True
    depth = np.zeros(mask.shape); depth[40,40] = 1
    row = evaluate(v,f,original=(ov,of),parents=[0,0],mask=mask,gap=np.zeros_like(mask),depth=depth,membership=membership)
    assert row['original_probe_protected'].tolist() == [True,True]
    assert row['core_protected'].tolist() == [True,True]
    assert not row['local_core_protected'].any()


def test_original_vertex_guard_is_local_not_whole_parent_veto():
    ov = np.array([[20.,20.,1.],[80.,20.,1.],[20.,80.,1.]])
    v = np.vstack((ov,(ov[0]+ov[1])/2,(ov[1]+ov[2])/2,(ov[2]+ov[0])/2))
    f = np.array([[0,3,5],[3,1,4],[5,4,2],[3,4,5]])
    membership = prepare_probe_membership(v,f,ov,np.array([[0,1,2]]),np.zeros(4,int))
    assert membership[:,0].tolist() == [True,False,False,False]
    mask = np.zeros((120,120),bool); mask[5:110,5:110] = True
    depth = np.zeros(mask.shape); depth[20,20] = 1
    row = evaluate(v,f,original=(ov,np.array([[0,1,2]])),parents=[0]*4,
                   mask=mask,gap=np.zeros_like(mask),depth=depth,membership=membership)
    assert row['original_probe_protected'].tolist() == [True,False,False,False]


@pytest.mark.parametrize('depth_value',[0.,np.nan,.5,2.])
def test_unknown_or_mismatched_depth_does_not_protect(depth_value):
    mask = np.zeros((120,120),bool); mask[5:110,5:110] = True
    row = evaluate([[40,40,1],[45,40,1],[40,45,1]],mask=mask,gap=np.zeros_like(mask),depth=np.full(mask.shape,depth_value))
    assert not row['core_protected'].any()


def test_closure_face_has_core_but_cannot_have_negative_vote():
    mask = np.zeros((120,120),bool); mask[5:110,5:110] = True
    row = evaluate([[40,40,1],[45,40,1],[40,45,1]],eligible=[False],
                   mask=mask,gap=np.zeros_like(mask),depth=np.ones(mask.shape))
    assert row['core_protected'][0] and not row['negative'][0]
    assert row['reason_names'][int(row['reason'][0])] == 'INELIGIBLE_PARENT'


def test_unknown_hole_in_circle_blocks_even_with_valid_vertices():
    mask,valid,gap = scene(); valid[44,54] = False; gap[44,54] = False
    row = evaluate([[48.2,40.2,1],[57.8,40.2,1],[48.2,49.8,1]],mask=mask,valid=valid,gap=gap)
    assert not row['negative'][0] and not row['known_domain_disk'][0]


@pytest.mark.parametrize('vertices',[
    [[-1,40,1],[57,40,1],[48,49,1]],
    [[48,40,-1],[57,40,1],[48,49,1]],
    [[48,40,0],[57,40,1],[48,49,1]],
])
def test_camera_crossings_and_offscreen_are_unknown(vertices):
    row = evaluate(vertices)
    assert row['status'].tolist() == [0]
    assert not row['negative'][0]


def test_minimum_background_clearance_is_independent_of_incoming_gap():
    mask,valid,gap = scene()
    gap[:,30:80] = True
    row = evaluate([[30.1,40.1,1],[33.9,40.1,1],[30.1,43.9,1]],mask=mask,valid=valid,gap=gap)
    assert not row['deep_background_disk'][0] and not row['negative'][0]


def test_budget_and_finite_policy_fail_closed():
    row = evaluate([[48.2,40.2,1],[57.8,40.2,1],[48.2,49.8,1]],policy=LocalPolicy(max_raster_pixels_per_face=2))
    assert not row['negative'][0] and not row['pixel_count_evaluated'][0]
    assert row['reason_names'][int(row['reason'][0])] == 'PIXEL_BUDGET'
    for kwargs in ({'min_background_pixels':3},{'min_views':1},{'boundary_px':float('nan')},{'min_views':True}):
        with pytest.raises(ValueError): LocalPolicy(**kwargs)


def test_probe_parent_identity_plane_and_immutability():
    v,f,ov,of = split_fixture()
    copies = [a.copy() for a in (v,f,ov,of)]
    result = prepare_probe_membership(v,f,ov,of,np.zeros(2,int))
    assert result.shape == (2,4)
    assert all(np.array_equal(a,b) for a,b in zip((v,f,ov,of),copies))
    wrong = v.copy(); wrong[3,2] += .1
    with pytest.raises(ValueError): prepare_probe_membership(wrong,f,ov,of,np.zeros(2,int))
    with pytest.raises(ValueError): prepare_probe_membership(v,f,ov,of,np.ones(2,int))
    def stop(): raise TimeoutError('synthetic deadline')
    with pytest.raises(TimeoutError): prepare_probe_membership(v,f,ov,of,np.zeros(2,int),check=stop)


def test_full_evaluation_does_not_mutate_source_arrays():
    v = np.array([[48.2,40.2,1],[57.8,40.2,1],[48.2,49.8,1]])
    m,valid,gap = scene(); depth = np.where(m,1.,0.)
    before = [a.copy() for a in (v,m,valid,gap,depth)]
    evaluate(v,mask=m,valid=valid,gap=gap,depth=depth)
    assert all(np.array_equal(a,b) for a,b in zip((v,m,valid,gap,depth),before))
