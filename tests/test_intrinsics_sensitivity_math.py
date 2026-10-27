import json
import numpy as np
import pytest
from integrations.demo_pipeline.intrinsics_sensitivity_math import analyse_pair, invert_radial


def scene(planar=False, rotation=False):
    rng=np.random.default_rng(51)
    P=rng.uniform([-2,-1,4],[2,1,8],(48,3))
    if planar: P[:,2]=6
    angle=.09
    R=np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]])
    Q=P@R.T + (np.zeros(3) if rotation else np.array([.5,.1,.05]))
    return P[:,:2]/P[:,2,None]*800+[500,400],Q[:,:2]/Q[:,2,None]*800+[500,400],np.column_stack((np.arange(48),np.arange(48)+100))


def run(a,b,ids,**kw):
    return analyse_pair(a,b,ids,[500,400],[1000,800],800,**kw)


def test_nonplanar_and_json():
    a,b,ids=scene(); orig=[x.copy() for x in (a,b,ids)]
    result=run(a,b,ids,config=6)
    assert result['status']=='SENSITIVITY_EVALUATED'
    assert result['principal_profiles']==[]
    zero=result['distortion_scenarios'][1]
    for fold in zero['folds']:
        assert fold['models']['F']['rms']<1e-8
        assert fold['models']['F']['rank2_enforced_at_fit']
        assert not (set(map(tuple,fold['training_feature_pairs'])) & set(map(tuple,fold['heldout_feature_pairs'])))
    assert not result['calibration_accepted'] and not result['new_ba_authorized']
    json.dumps(result,allow_nan=False)
    for x,y in zip((a,b,ids),orig): np.testing.assert_array_equal(x,y)


def test_planar_and_rotation():
    a,b,ids=scene(planar=True)
    result=run(a,b,ids)['distortion_scenarios'][1]
    assert result['homography_compatible']
    assert all(f['models']['H']['rms']<1e-8 for f in result['folds'])
    a,b,ids=scene(rotation=True)
    folds=run(a,b,ids)['distortion_scenarios'][1]['folds']
    assert all(f['models']['R']['maximum']<2e-6 for f in folds)


@pytest.mark.parametrize('k',[-.1,0,.1])
def test_radial(k):
    ideal=np.array([[500.,400.],[600,470],[330,290]])
    u=(ideal-[500,400])/800
    distorted=[500,400]+800*u*(1+k*(u*u).sum(1))[:,None]
    np.testing.assert_allclose(invert_radial(distorted,[500,400],800,k),ideal,atol=1e-10)
    if k==0: np.testing.assert_array_equal(invert_radial(ideal,[500,400],800,k),ideal)


def test_negative_domain():
    with pytest.raises(ValueError,match='monotonic'):
        invert_radial([[10000,400]],[500,400],800,-.1)


def test_rejections_and_insufficient():
    a,b,ids=scene()
    assert run(a.astype(bool),b,ids)['status']=='SENSITIVITY_REFUSED'
    bad=ids.copy(); bad[1,0]=bad[0,0]
    assert run(a,b,bad)['status']=='SENSITIVITY_REFUSED'
    bad=ids.astype(float); bad[0,0]=.5
    assert run(a,b,bad)['status']=='SENSITIVITY_REFUSED'
    assert run(a,b,ids,config=True)['status']=='SENSITIVITY_REFUSED'
    small=run(a[:23],b[:23],ids[:23])
    for scenario in small['distortion_scenarios']:
        assert all(f['status']=='INSUFFICIENT' and not f['models'] for f in scenario['folds'])


def test_training_independence_and_sort():
    a,b,ids=scene()
    base=run(a,b,ids)
    changed=b.copy(); changed[1::2]+=3
    other=run(a,changed,ids)
    np.testing.assert_array_equal(base['distortion_scenarios'][1]['folds'][0]['models']['F']['matrix'],
                                  other['distortion_scenarios'][1]['folds'][0]['models']['F']['matrix'])
    perm=np.random.default_rng(1).permutation(len(a))
    assert run(a[perm],b[perm],ids[perm])==base


def test_stored_principal_profiles():
    a,b,ids=scene()
    F=np.array(run(a,b,ids)['distortion_scenarios'][1]['folds'][0]['models']['F']['matrix'])
    result=run(a,b,ids,stored_f=F)
    assert len(result['principal_profiles'])==5
    assert all(len(row['profile']['curve'])==101 for row in result['principal_profiles'])
    json.dumps(result,allow_nan=False)


def test_bounded_inputs_and_integer_ids():
    a,b,ids=scene()
    for size in ([1000.5,800],[True,800],[1000,16385]):
        assert analyse_pair(a,b,ids,[500,400],size,800)['status']=='SENSITIVITY_REFUSED'
    assert run(a,b,ids.astype(float))['status']=='SENSITIVITY_REFUSED'
    for p in ([-1,400],[1000,400]):
        assert analyse_pair(a,b,ids,p,[1000,800],800)['status']=='SENSITIVITY_REFUSED'
    outside=a.copy(); outside[0,0]=1000
    assert run(outside,b,ids)['status']=='SENSITIVITY_REFUSED'
    assert analyse_pair(a,b,ids,[500,400],[1000,800],1e9)['status']=='SENSITIVITY_REFUSED'
    many=np.tile(a,(86,1)); many_ids=np.column_stack((np.arange(len(many)),np.arange(len(many))))
    assert run(many,many,many_ids)['status']=='SENSITIVITY_REFUSED'


def test_unknown_h_compatibility():
    a,b,ids=scene()
    for scenario in run(a[:23],b[:23],ids[:23])['distortion_scenarios']:
        assert scenario['homography_compatible'] is None
        assert scenario['homography_compatibility_reason']=='INSUFFICIENT_FOLD'
    collapsed=np.full_like(a,[500,400])
    result=run(collapsed,collapsed,ids)['distortion_scenarios'][1]
    assert result['homography_compatible'] is None
    assert result['homography_compatibility_reason']=='H_MODEL_FAILED'


def test_radial_direct_boolean_and_turnpoint():
    for f,k in ((True,.1),(800,False)):
        with pytest.raises(ValueError): invert_radial([[500,400]],[500,400],f,k)
    limit=np.sqrt(1/.3)*(1-.1*(1/.3))
    with pytest.raises(ValueError,match='monotonic'):
        invert_radial([[limit,0]],[0,0],1,-.1)


def test_stats_extreme_is_finite():
    from integrations.demo_pipeline.intrinsics_sensitivity_math import _stats
    result=_stats([1e300,1e300],2)
    assert result['mean']==pytest.approx(1e300)
    assert result['rms']==pytest.approx(1e300)
    json.dumps(result,allow_nan=False)


def test_h_invalid_holdout_is_unknown(monkeypatch):
    import integrations.demo_pipeline.intrinsics_sensitivity_math as math
    original=math._fit
    def fit(a,b,model):
        return (np.diag([1.,1.,1e-15]),8) if model=='H' else original(a,b,model)
    monkeypatch.setattr(math,'_fit',fit)
    a,b,ids=scene()
    result=run(a,b,ids)['distortion_scenarios'][1]
    assert result['homography_compatible'] is None
    assert result['homography_compatibility_reason']=='H_INVALID_HOLDOUT'
    assert result['folds'][0]['models']['H']['failed_or_undefined_count']==24


def test_nonfinite_derived_output_refused(monkeypatch):
    import integrations.demo_pipeline.intrinsics_sensitivity_math as math
    original=math._fold
    def fold(*args):
        result=original(*args)
        result['test_injected_nonfinite']=float('inf')
        return result
    monkeypatch.setattr(math,'_fold',fold)
    a,b,ids=scene()
    result=run(a,b,ids)
    assert result['status']=='SENSITIVITY_REFUSED'
    assert result['reason']=='NONFINITE_DERIVED_OUTPUT'
    json.dumps(result,allow_nan=False)
