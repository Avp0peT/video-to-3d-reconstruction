"""Synthetic 3D points and independent forward projection; no video/models."""
from copy import deepcopy
import json
import numpy as np
import pytest
from integrations.demo_pipeline.query_camera_math import assess_query_camera


def data(k=0.,n=40):
    rng=np.random.default_rng(812)
    X=rng.uniform([-2.,-1.5,4.],[2.,1.5,9.],(n,3))
    a=.17;c,s=np.cos(a),np.sin(a);R=np.array([[c,0,s],[0,1,0],[-s,0,c]])
    camera=X@R.T+np.array([.2,-.1,.4]);q=camera[:,:2]/camera[:,2,None]
    xy=600*q*(1+k*np.sum(q*q,axis=1))[:,None]+[400.,300.]
    return [dict(point_id=f'p{i:03}',xyz=x.tolist(),raw_xy=y.tolist()) for i,(x,y) in enumerate(zip(X,xy))]


def cvmodule():return pytest.importorskip('cv2')


@pytest.mark.parametrize('k',[0.,.1,-.1])
def test_independent_synthetic_camera_radial(k):
    cv=cvmodule();points=data(k);before=deepcopy(points);result=assess_query_camera(points,800,600,cv)
    assert points==before
    radial=result['models'][1]
    assert radial['status']=='CONDITIONAL_CAMERA_CANDIDATE',radial['reasons']
    assert abs(radial['f']-600)<1 and abs(radial['k']-k)<.01
    assert radial['train']['max_px']<.01 and radial['holdout']['max_px']<.01
    assert radial['jacobian']['rank']==8
    assert radial['radial_domain']['full_image_coverage']['covered']
    assert all(r['depth']>0 for r in radial['train']['rows']+radial['holdout']['rows'])
    assert radial['negative_control']['status']!='CONDITIONAL_CAMERA_CANDIDATE'
    assert not result['camera_pose_established'] and not result['native_injected']
    json.dumps(result,allow_nan=False)


def test_holdout_pollution_cannot_change_training_or_initialization():
    cv=cvmodule();points=data(.1);baseline=assess_query_camera(points,800,600,cv)
    for i in range(3,len(points),4):points[i]['raw_xy'][0]+=30
    changed=assess_query_camera(points,800,600,cv)
    for before,after in zip(baseline['models'],changed['models']):
        assert before['initialization']==after['initialization']
        assert before['K']==after['K'] and before['R']==after['R'] and before['t']==after['t']
        assert after['holdout']['max_px']>20
        assert 'HOLDOUT_REPROJECTION_GATE' in after['reasons']


def test_permuted_correspondence_not_admitted():
    points=data();targets=[p['raw_xy'] for p in points]
    for i,p in enumerate(points):p['raw_xy']=targets[(i+1)%len(points)]
    result=assess_query_camera(points,800,600,cvmodule())
    assert all(m['status']!='CONDITIONAL_CAMERA_CANDIDATE' for m in result['models'])


def test_planar_support_explicit_block():
    points=data()
    for p in points:p['xyz'][2]=6.
    result=assess_query_camera(points,800,600,cvmodule())
    assert all(m['status']=='BLOCKED' and 'DEGENERATE_NONPLANAR_TRAINING_SUPPORT' in m['reasons'] for m in result['models'])


def test_less_than_twelve_blocked():
    result=assess_query_camera(data(n=11),800,600,cvmodule())
    assert result['status']=='BLOCKED' and len(result['models'])==2


@pytest.mark.parametrize('change',[lambda p:p[0]['xyz'].__setitem__(0,True),lambda p:p[0]['raw_xy'].__setitem__(0,float('nan')),lambda p:p[0]['raw_xy'].__setitem__(0,-1),lambda p:p.append(deepcopy(p[0]))])
def test_bad_inputs_raise(change):
    points=data();change(points)
    with pytest.raises(ValueError):assess_query_camera(points,800,600,cvmodule())


def test_boolean_dimension_raise():
    with pytest.raises(ValueError):assess_query_camera(data(),True,600,cvmodule())


def test_shared_budget_propagates():
    def budget():raise TimeoutError('deadline')
    with pytest.raises(TimeoutError,match='deadline'):assess_query_camera(data(),800,600,cvmodule(),budget=budget)


class Proxy:
    def __init__(self,cv,kind):self.cv=cv;self.kind=kind;self.calls=[]
    def __getattr__(self,key):return getattr(self.cv,key)
    def calibrateCamera(self,*args,**kwargs):
        self.calls.append((deepcopy(args),deepcopy(kwargs)))
        rms,K,d,rv,tv=self.cv.calibrateCamera(*args,**kwargs)
        if self.kind=='focal':K[0,0]=K[1,1]=10000.
        if self.kind=='fold':K[0,0]=K[1,1]=100.;d.flat[0]=-.9
        return rms,K,d,rv,tv


@pytest.mark.parametrize('kind,reason',[('focal','FOCAL_LENGTH_BOUND'),('fold','RADIAL_IMAGE_DOMAIN_NOT_INVERTIBLE')])
def test_parameter_and_entire_image_fold_gates(kind,reason):
    proxy=Proxy(cvmodule(),kind);result=assess_query_camera(data(),800,600,proxy)
    radial=result['models'][1]
    assert radial['status']=='REJECTED' and reason in radial['reasons']
    assert proxy.calls[0][1]['criteria']==(3,60,1e-9)
    for args,kwargs in proxy.calls:
        assert len(args[0][0])==30 and args[0][0].dtype==np.float32
        assert kwargs['flags']&proxy.CALIB_USE_INTRINSIC_GUESS
        assert kwargs['flags']&proxy.CALIB_FIX_PRINCIPAL_POINT
        assert kwargs['flags']&proxy.CALIB_FIX_ASPECT_RATIO


def test_budget_after_native_is_not_swallowed():
    cv=cvmodule();calls=0
    class NativeHook:
        def __getattr__(self,key):return getattr(cv,key)
        def decomposeProjectionMatrix(self,*args):
            nonlocal calls
            calls=1
            return cv.decomposeProjectionMatrix(*args)
    def budget():
        if calls:raise TimeoutError('after-native')
    with pytest.raises(TimeoutError,match='after-native'):assess_query_camera(data(),800,600,NativeHook(),budget=budget)


@pytest.mark.parametrize('cv',[None,object()])
@pytest.mark.parametrize('n',[0,11])
def test_insufficient_points_does_not_require_cv_constants(cv,n):
    result=assess_query_camera(data(n=n),800,600,cv)
    assert result['status']=='BLOCKED'
    assert all(m['reasons']==['INSUFFICIENT_FROZEN_TRAIN_HOLDOUT'] for m in result['models'])


def test_empty_input_still_honors_budget_before_return():
    def budget():raise TimeoutError('empty deadline')
    with pytest.raises(TimeoutError,match='empty deadline'):
        assess_query_camera([],800,600,None,budget=budget)


def test_native_control_failure_is_not_evidence_for_candidate():
    cv=cvmodule()
    class ControlFailure:
        def __init__(self):self.calibrations=0
        def __getattr__(self,key):return getattr(cv,key)
        def calibrateCamera(self,*args,**kwargs):
            self.calibrations+=1
            if self.calibrations%2==0:raise cv.error('synthetic wrong-control optimizer failure')
            return cv.calibrateCamera(*args,**kwargs)
    result=assess_query_camera(data(),800,600,ControlFailure())
    for model in result['models']:
        assert model['status']!='CONDITIONAL_CAMERA_CANDIDATE'
        assert 'NEGATIVE_CONTROL_NOT_EVALUATED' in model['reasons']
        assert any('OPENCV:' in reason for reason in model['negative_control']['reasons'])
