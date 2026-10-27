"""Independent pinhole truth exercises projective fitting, not camera acceptance."""
from copy import deepcopy
import json
import numpy as np
import pytest
from integrations.demo_pipeline.tracklet_camera_math import assess_projection


def truth(count=20):
    # Projection is computed directly from a fixed camera, independently of the
    # estimator's normalization, DLT design, and SVD implementation.
    rng=np.random.default_rng(415)
    xyz=rng.uniform(-2,2,size=(count,3)); xyz[:,2]+=7
    P=np.array([[800.,35.,320.,120.],[-20.,790.,240.,-80.],[.01,-.02,1.,.3]])
    records=[]
    for i, X in enumerate(xyz):
        q=P @ np.append(X,1.)
        records.append(dict(point_id=f'p{i:03d}',xyz=X.tolist(),raw_xy=(q[:2]/q[2]).tolist()))
    return records,P


def test_independent_projection_truth_is_recovered_and_json_finite():
    points,P=truth(); original=deepcopy(points)
    result=assess_projection(points)
    fitted=np.array(result['all_fit']['P'])
    P=P/np.linalg.norm(P)
    if np.sum(fitted*P)<0: P=-P
    np.testing.assert_allclose(fitted,P,rtol=1e-8,atol=1e-10)
    assert result['all_fit']['reprojection']['max_px'] < 1e-8
    assert result['status']=='PROJECTIVE_FIT_DIAGNOSTIC' and result['numeric_fits']==3
    assert result['geometry_accepted'] is False and result['camera_pose_established'] is False
    assert points==original
    json.dumps(result,allow_nan=False)


@pytest.mark.parametrize('count',[6,7,8])
def test_minimal_fits_are_never_called_validated(count):
    result=assess_projection(truth(count)[0])
    assert result['status']=='MINIMAL_FIT_NOT_VALIDATED'
    assert result['holdout_ids']==[] and result['heldout_fit'] is None
    assert result['negative_control'] is None and result['numeric_fits']==1


def test_holdout_values_cannot_change_training_fit():
    points,_=truth(12); before=assess_projection(points)
    assert before['holdout_ids']==['p003','p007','p011']
    altered=deepcopy(points)
    for i in (3,7,11): altered[i]['raw_xy'][0]+=100
    after=assess_projection(altered)
    np.testing.assert_array_equal(before['heldout_fit']['P'],after['heldout_fit']['P'])
    assert after['heldout_fit']['holdout_reprojection']['median_px'] > 90
    assert before['heldout_fit']['holdout_reprojection']['max_px'] < 1e-8
    assert not np.allclose(before['all_fit']['P'],after['all_fit']['P'])


def test_nine_points_get_two_frozen_holdouts():
    result=assess_projection(truth(9)[0])
    assert len(result['holdout_ids'])==2 and len(result['train_ids'])==7
    assert result['heldout_fit'] is not None


def test_noise_rank_twelve_does_not_get_rejected_as_degenerate():
    points,_=truth()
    for i,p in enumerate(points):
        p['raw_xy'][0]+=.15*np.sin(i)
        p['raw_xy'][1]+=.15*np.cos(i*1.7)
    result=assess_projection(points)
    assert result['status']=='PROJECTIVE_FIT_DIAGNOSTIC'
    assert result['all_fit']['design_singular_values'][-1] > 1e-8
    assert result['all_fit']['reprojection']['max_px'] > 0


def test_planar_world_and_collinear_image_support_refused():
    points,_=truth()
    for p in points:p['xyz'][2]=3.
    assert assess_projection(points)['status']=='BLOCKED_DEGENERATE'
    points,_=truth()
    for i,p in enumerate(points):p['raw_xy']=[float(i),2.*i]
    assert assess_projection(points)['status']=='BLOCKED_DEGENERATE'


@pytest.mark.parametrize('mutation',['duplicate','bool_xyz','bool_pixel','nan','string'])
def test_invalid_geometry_rejected(mutation):
    points,_=truth()
    if mutation=='duplicate':points[1]['point_id']=points[0]['point_id']
    if mutation=='bool_xyz':points[0]['xyz'][0]=True
    if mutation=='bool_pixel':points[0]['raw_xy'][1]=False
    if mutation=='nan':points[0]['xyz'][0]=float('nan')
    if mutation=='string':points[0]['raw_xy'][0]='1'
    with pytest.raises(ValueError):assess_projection(points)


def test_shifted_wrong_correspondences_degrade_frozen_holdout():
    result=assess_projection(truth()[0])
    assert result['negative_control']['status']=='EVALUATED'
    correct=result['heldout_fit']['holdout_reprojection']['median_px']
    wrong=result['negative_control']['holdout_reprojection']['median_px']
    assert wrong > correct+1.


def test_budget_exception_propagates_during_svd_and_does_not_become_degeneracy():
    points,_=truth(); calls=[]
    class Stop(RuntimeError):pass
    def budget():
        calls.append(1)
        # Validation N, fit begin, normalization pre-SVD, normalization post-SVD.
        if len(calls)==len(points)+3:raise Stop('shared budget')
    with pytest.raises(Stop,match='shared budget'):assess_projection(points,budget=budget)


def test_empty_and_point_bound():
    assert assess_projection([])['status']=='BLOCKED_INSUFFICIENT_POINTS'
    with pytest.raises(ValueError,match='BOUND'):assess_projection([{}]*769)
