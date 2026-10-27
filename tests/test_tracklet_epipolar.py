"""Synthetic perspective correspondences, with independent held-out points."""
from copy import deepcopy
import json
import numpy as np
import pytest
from integrations.demo_pipeline.tracklet_epipolar import assess_epipolar


def fixture(n=40,planar=False):
    rng=np.random.default_rng(42)
    world=rng.uniform([-2,-1,4],[2,1,9],size=(n,3))
    if planar:world[:,2]=6
    a=.13;c,s=np.cos(a),np.sin(a);rotation=np.array([[c,0,s],[0,1,0],[-s,0,c]])
    views=[]
    for r,t in [(np.eye(3),np.zeros(3)),(rotation,np.array([.7,.15,.1])),(rotation,np.array([1.,.2,.1]))]:
        cam=world@r.T+t;views.append(cam[:,:2]/cam[:,2,None]*500+np.array([320.,240.]))
    records=[]
    for i in range(n):
        pid=f'track:{i}:segment:0:source:0'
        records.append(dict(tracklet_id=pid+':edge:0',point_id=pid,track_id=i,identity_segment=0,source_input_index=0,edge_ordinal=0,
            status='CANDIDATE_2D_TRACKLET',verified_legs=2,
            source_point=dict(point_id=pid,source_input_index=0,source_decode_index=0,source_raw_xy=views[0][i].tolist()),
            observations=[dict(decode_index=j,raw_xy=v[i].tolist()) for j,v in enumerate(views)]))
    return dict(records=records)


def test_perspective_heldout_f_negative_control_and_immutable():
    data=fixture();before=deepcopy(data);result=assess_epipolar(data)
    assert data==before and len(result['pairs'])==2
    for p in result['pairs']:
        assert p['status']=='UNCALIBRATED_EPIPOLAR_DIAGNOSTIC'
        assert p['F']['holdout']['max']<1e-7
        assert p['negative_control']['F']['median']>.1
        assert not p['negative_control']['refit']
        assert len(p['train_ids'])==30 and len(p['holdout_ids'])==10
        assert not set(p['train_ids'])&set(p['holdout_ids'])
        assert p['holdout_ids']==sorted(q['point_id'] for q in data['records'])[3::4]
    assert not result['geometry_accepted'] and not result['poses_verified']
    json.dumps(result,allow_nan=False)


def test_planar_homography_generalization():
    result=assess_epipolar(fixture(planar=True))
    for p in result['pairs']:
        assert p['H']['status']=='EVALUATED'
        assert p['H']['holdout']['max']<1e-7
        assert p['negative_control']['H']['median']>1


def test_small_cohort_blocked_no_fit():
    result=assess_epipolar(fixture(11))
    assert all(p['reasons']==['BLOCKED_INSUFFICIENT_HOLDOUT'] and p['F'] is None for p in result['pairs'])


def test_collinear_explicitly_blocked():
    data=fixture(16)
    for i,r in enumerate(data['records']):
        for j,o in enumerate(r['observations']):o['raw_xy']=[float(i+j),float(2*(i+j))]
        r['source_point']['source_raw_xy']=r['observations'][0]['raw_xy'][:]
    result=assess_epipolar(data)
    assert all(p['status']=='BLOCKED' and p['image_support']['source']['centered_rank']==1 for p in result['pairs'])


@pytest.mark.parametrize('change',[lambda d:d['records'].append(deepcopy(d['records'][0])),
    lambda d:d['records'][0].update(point_id='forged'),
    lambda d:d['records'][0]['observations'][1].update(raw_xy=[float('nan'),2]),
    lambda d:d['records'][0]['observations'][1].update(decode_index=8),
    lambda d:d['records'][0]['source_point'].update(source_raw_xy=[1.,2.])])
def test_bad_identity_nonfinite_and_path_fail_closed(change):
    data=fixture();change(data)
    with pytest.raises(ValueError):assess_epipolar(data)


def test_truncated_prefix_does_not_invent_later_observations():
    data=fixture(16)
    for r in data['records'][:8]:
        r['status']='EXCLUDED';r['verified_legs']=1;r['observations']=r['observations'][:2]
    result=assess_epipolar(data)
    assert [p['counts']['points'] for p in result['pairs']]==[8,8]
    assert all(p['target_decode_index']<=2 for p in result['pairs'])


def test_descending_prefix_valid():
    data=fixture(16)
    for r in data['records']:
        for o in r['observations']:o['decode_index']=10-o['decode_index']
        r['source_point']['source_decode_index']=10
    assert [(p['source_decode_index'],p['target_decode_index']) for p in assess_epipolar(data)['pairs']]==[(9,8),(10,9)]


def test_budget_exception_propagates():
    def budget():raise TimeoutError('shared clock')
    with pytest.raises(TimeoutError,match='shared clock'):assess_epipolar(fixture(),budget=budget)


def test_value_error_budget_after_numeric_fit_is_not_swallowed(monkeypatch):
    from kozlov_lab.stereo import viewpoint_consistency as math_kernel
    fitted=[]; original=math_kernel.fit_fundamental
    def fit(value):
        result=original(value); fitted.append(True); return result
    monkeypatch.setattr(math_kernel,'fit_fundamental',fit)
    def budget():
        if fitted:raise ValueError('budget must propagate')
    with pytest.raises(ValueError,match='budget must propagate'):
        assess_epipolar(fixture(),budget=budget)


def test_pair_and_point_bounds():
    data=fixture(12);records=[]
    for e in range(18):
        for original in data['records']:
            r=deepcopy(original);r['edge_ordinal']=e;r['tracklet_id']=r['point_id']+f':edge:{e}';records.append(r)
    result=assess_epipolar(dict(records=records))
    assert len(result['pairs'])==16 and len(result['omitted_pairs'])==20
    with pytest.raises(ValueError,match='PAIR_POINT_BOUND'):assess_epipolar(fixture(769))


def test_holdout_coordinates_never_enter_fitting():
    data=fixture();baseline=assess_epipolar(data)
    held=set(baseline['pairs'][0]['holdout_ids'])
    for r in data['records']:
        if r['point_id'] in held:
            r['observations'][1]['raw_xy'][1]+=20
    changed=assess_epipolar(data)
    assert changed['pairs'][0]['F']['model']==baseline['pairs'][0]['F']['model']
    assert changed['pairs'][0]['H']['model']==baseline['pairs'][0]['H']['model']
    assert changed['pairs'][0]['F']['holdout']['rms']>baseline['pairs'][0]['F']['holdout']['rms']+1
