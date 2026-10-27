import numpy as np
import pytest
from kozlov_lab.stereo import viewpoint_consistency as module


def parallel():
    xyz=np.random.default_rng(31).uniform([-2,-1,4],[2,1,9],(32,3))
    x,y,z=xyz.T
    return np.column_stack((700*x/z+320,700*y/z+240,700*(x-1)/z+320,700*y/z+240))


def test_parallel_correct_and_wrong_frozen_models(monkeypatch):
    raw=parallel(); before=raw.copy(); train=list(range(20)); hold=list(range(20,32)); calls=[]
    for name in ('fit_affine_flat','fit_fundamental'):
        original=getattr(module,name)
        def wrapped(rows, original=original, name=name):
            calls.append(name); assert np.array_equal(rows,raw[train]); return original(rows)
        monkeypatch.setattr(module,name,wrapped)
    result=module.analyze(raw,train,hold)
    assert calls==['fit_affine_flat','fit_fundamental']
    assert np.array_equal(raw,before)
    for name in ('affine_flat','fundamental'):
        model=result[name]; assert model['status']=='EVALUATED'; assert model['correct']['max']<1e-8
        for control in model['negative_controls'].values():
            assert control['rms']>1 and control['increase_count']>=10
            assert sum(control[k] for k in ('increase_count','decrease_count','equal_count'))==12


def test_perspective_fundamental_holdout():
    xyz=np.random.default_rng(4).uniform([-2,-1,5],[2,1,10],(40,3))
    theta=.24;c,s=np.cos(theta),np.sin(theta);R=np.array([[c,0,s],[0,1,0],[-s,0,c]])
    second=xyz@R.T+np.array([-1,.2,.1]); a=800*xyz[:,:2]/xyz[:,2:]+[320,240]; b=800*second[:,:2]/second[:,2:]+[320,240]
    result=module.analyze(np.column_stack((a,b)),list(range(25)),list(range(25,40)))
    assert result['fundamental']['correct']['max']<1e-8
    assert result['fundamental']['negative_controls']['cyclic_one']['rms']>1


def test_near_degenerate_rejected():
    t=np.arange(16,dtype=float);raw=np.column_stack((t,t*2,t+1,t*2+1e-12*t*t))
    result=module.analyze(raw,list(range(10)),list(range(10,16)))
    assert all(result[k]['status']=='NOT_EVALUATED' for k in ('affine_flat','fundamental'))


def test_genuine_parallel_projection_not_only_rectified_perspective():
    xyz=np.random.default_rng(16).uniform(-3,3,(32,3))
    first=xyz@np.array([[1.,.2],[0.,1.],[.3,0.]])+[100,200]
    second=xyz@np.array([[.2,1.],[1.,.3],[1.,-.2]])+[300,400]
    result=module.analyze(np.column_stack((first,second)),tuple(range(20)),tuple(range(20,32)))
    score=result['affine_flat']
    assert score['status']=='EVALUATED' and score['correct']['max']<1e-10
    assert score['negative_controls']['cyclic_one']['rms']>.1


def test_holdout_changes_cannot_change_fitted_parameters():
    raw=parallel();changed=raw.copy();changed[20:,2:]+=np.array([9.,-7.])
    a=module.analyze(raw,list(range(20)),list(range(20,32)))
    b=module.analyze(changed,list(range(20)),list(range(20,32)))
    for kind in ('affine_flat','fundamental'):
        assert a[kind]['model']==b[kind]['model']
        assert a[kind]['correct']['rms']<b[kind]['correct']['rms']


@pytest.mark.parametrize('bad', ['overlap','missing','boolindex','boolpixel','nan','short'])
def test_invalid_input_or_split(bad):
    raw=parallel().tolist();train=list(range(20));hold=list(range(20,32))
    if bad=='overlap':hold[0]=0
    elif bad=='missing':hold.pop()
    elif bad=='boolindex':train[0]=False
    elif bad=='boolpixel':raw[0][0]=True
    elif bad=='nan':raw[0][0]=float('nan')
    elif bad=='short':raw=raw[:11]
    with pytest.raises(ValueError):module.analyze(raw,train,hold)
