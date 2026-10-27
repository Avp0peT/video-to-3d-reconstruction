from fractions import Fraction as F
from copy import deepcopy
from itertools import combinations,permutations
import json
import numpy as np
import pytest
from kozlov_lab.stereo import approximate_parallel as a


def parallel_rows(n,seed=101):
    rng=np.random.default_rng(seed);body=rng.normal(size=(n,3))*50
    x,y,z=body.T
    return np.column_stack((x,y,x+z,y+2*z))


def fitted():
    r=a.fit_affine_flat(parallel_rows(64))
    assert r['status']=='ENGINEERING_APPROXIMATION_FITTED'
    return r


def det(rows):
    n=len(rows)
    return sum((-1)**sum(p[i]>p[j] for i in range(n) for j in range(i+1,n)) * np.prod([rows[i][p[i]] for i in range(n)]) for p in permutations(range(n)))


def test_known_rank3_training_and_separate_frozen_validation(monkeypatch):
    train=parallel_rows(80);r=a.fit_affine_flat(train)
    assert r['scope']=='ENGINEERING_APPROXIMATION' and r['training_rms_pixels']<1e-10
    assert r['rank_support_ratio']>1e-6 and r['flatness_ratio_smallest_over_third']<1e-12
    frozen=deepcopy(r['model'])
    def forbidden(*args,**kwargs):raise AssertionError('validation refitted SVD')
    monkeypatch.setattr(np.linalg,'svd',forbidden)
    held=a.evaluate_affine_flat(frozen,parallel_rows(40,102))
    assert held['status']=='ENGINEERING_APPROXIMATION_VALIDATED' and held['rms_pixels']<1e-10
    assert held['refitted'] is False and held['quality_threshold_decided'] is False
    assert frozen==r['model'] and held['physical_model_accepted'] is False


def test_heldout_wrong_correspondence_is_visible_without_refit():
    model=fitted()['model'];rows=parallel_rows(40,103)
    correct=a.evaluate_affine_flat(model,rows)
    wrong=rows.copy();wrong[:,2:]=np.roll(wrong[:,2:],1,axis=0)
    result=a.evaluate_affine_flat(model,wrong)
    assert correct['max_abs_pixels']<1e-10 and result['rms_pixels']>1
    assert result['refitted'] is False and len(result['signed_residuals_pixels'])==40
    assert result['original_exact_model_decided'] is False


def test_generic_perspective_observations_are_not_silently_exact_or_accepted():
    def rows(n,seed):
        rng=np.random.default_rng(seed);x,y=rng.uniform(-1,1,(2,n));z=rng.uniform(3,8,n)
        theta=0.4;xp=np.cos(theta)*x+np.sin(theta)*z-1;zp=-np.sin(theta)*x+np.cos(theta)*z
        return np.column_stack((400*x/z,400*y/z,400*xp/zp,400*y/zp))
    r=a.fit_affine_flat(rows(80,105));held=a.evaluate_affine_flat(r['model'],rows(50,106))
    assert r['flatness_ratio_smallest_over_third']>1e-4
    assert held['rms_pixels']>0.1 and held['quality_threshold_decided'] is False
    assert held['physical_model_accepted'] is False and r['original_exact_model_decided'] is False


def test_exact_rational_surrogate_rank3_and_full_modification_ledger():
    from kozlov_lab.stereo.official.given_parallel_correspondence import assess_given_parallel_input
    model=fitted()['model'];rows=parallel_rows(6,107);rows[0,2]+=0.02
    labels=[91,-4,8,500,31,72];r=a.build_exact_surrogate(model,rows,labels)
    assert r['status']=='ENGINEERING_APPROXIMATION_SURROGATE'
    assert r['plane']['grid_bits']==18 and r['affine_rank_upper_bound']==3
    assert r['exact_rank_three_proved'] is False and r['max_change_pixels']>0
    plane=r['plane'];dep=plane['dependent_component'];free=plane['free_components']
    combined=[]
    for item,original in zip(r['ledger'],rows):
        q=item['rational_normalized'];combined.append(q+(F(1),))
        assert all(type(v) is F and abs(v.numerator)<=10**12 and v.denominator<=10**12 for v in q)
        assert q[dep]==plane['offset']+sum(s*q[i] for s,i in zip(plane['slopes'],free))
        assert item['original_pixels']==tuple(original)
        assert np.allclose(np.array(item['projection_change_pixels'])+item['quantization_change_pixels'],item['total_change_pixels'],rtol=0,atol=1e-12)
        assert np.allclose([(float(v)-original[i]/model['scale'])*model['scale'] for i,v in enumerate(q)],item['total_change_pixels'])
    assert all(det(rows)==0 for rows in combinations(combined,5))
    gate=assess_given_parallel_input(r['source_xy'],r['target_xy'],r['mapping'])
    assert gate['status']=='CONSTRUCTED'
    assert r['mapping']=={i:i for i in labels}
    assert r['scope']=='ENGINEERING_APPROXIMATION' and r['original_exact_model_decided'] is False


def test_full4096_fit_and_validation_are_not_sampled():
    rows=parallel_rows(4096);r=a.fit_affine_flat(rows)
    assert r['status']=='ENGINEERING_APPROXIMATION_FITTED' and r['model']['training_count']==4096
    assert len(r['training_observations'])==len(r['training_residuals_pixels'])==4096
    held=a.evaluate_affine_flat(r['model'],rows)
    assert len(held['observations'])==len(held['signed_residuals_pixels'])==4096


@pytest.mark.parametrize('kind',['nan','bool','string','few','shape','too_many'])
def test_fit_rejects_invalid_complete_inputs(kind):
    rows=parallel_rows(8).tolist()
    if kind=='nan':rows[0][0]=float('nan')
    elif kind=='bool':rows[0][0]=True
    elif kind=='string':rows[0][0]='1'
    elif kind=='few':rows=rows[:4]
    elif kind=='shape':rows[0]=[1,2,3]
    else:rows=parallel_rows(4097)
    assert a.fit_affine_flat(rows)['status']=='INVALID_INPUT'


def test_degenerate_support_is_explicit_weak_geometry():
    rows=np.column_stack((np.arange(8),np.arange(8),2*np.arange(8),3*np.arange(8)))
    r=a.fit_affine_flat(rows)
    assert r['status']=='WEAK_GEOMETRY' and r['model'] is None


@pytest.mark.parametrize('kwargs',[{'max_work':0},{'seconds':0},{'max_work':1}])
def test_exhaustion_all_stages_is_unknown(kwargs):
    model=fitted()['model'];rows=parallel_rows(6)
    assert a.fit_affine_flat(rows,**kwargs)['status']=='UNKNOWN'
    assert a.evaluate_affine_flat(model,rows,**kwargs)['status']=='UNKNOWN'
    r=a.build_exact_surrogate(model,rows,list(range(6)),**kwargs)
    assert r['status']=='UNKNOWN' and r['source_xy'] is None


@pytest.mark.parametrize('kwargs',[{'scale':True},{'scale':0},{'min_support_ratio':False},{'max_work':True},{'seconds':float('nan')}])
def test_invalid_frozen_parameters(kwargs):
    assert a.fit_affine_flat(parallel_rows(6),**kwargs)['status']=='INVALID_INPUT'


def test_grid_cannot_retune_and_algT_bound_failure_keeps_ledger():
    model=fitted()['model'];bad=deepcopy(model);bad['grid_bits']=20
    assert a.build_exact_surrogate(bad,parallel_rows(6),list(range(6)))['status']=='INVALID_INPUT'
    train=parallel_rows(64);train[:,3]=train[:,1]+(train[:,2]-train[:,0])*2/7
    bounded_model=a.fit_affine_flat(train)['model']
    rows=parallel_rows(6)*1e3;rows[:,3]=rows[:,1]+(rows[:,2]-rows[:,0])*2/7
    r=a.build_exact_surrogate(bounded_model,rows,list(range(6)))
    assert r['status']=='UNSUPPORTED_RATIONAL_BOUND' and len(r['ledger'])==6
    assert r['source_xy'] is None and r['plane']['grid_bits']==18


def test_validation_and_surrogate_bool_nan_labels_rejected():
    model=fitted()['model'];rows=parallel_rows(6).tolist();rows[0][3]=False
    assert a.evaluate_affine_flat(model,rows)['status']=='INVALID_INPUT'
    rows=parallel_rows(6)
    assert a.build_exact_surrogate(model,rows,[False,1,2,3,4,5])['status']=='INVALID_INPUT'
    assert a.build_exact_surrogate(model,rows,[0,0,2,3,4,5])['status']=='INVALID_INPUT'
    rows[0,0]=np.inf
    assert a.evaluate_affine_flat(model,rows)['status']=='INVALID_INPUT'



def test_frozen_model_plain_json_roundtrip():
    model=json.loads(json.dumps(fitted()['model'],allow_nan=False))
    assert a.evaluate_affine_flat(model,parallel_rows(6))['status']=='ENGINEERING_APPROXIMATION_VALIDATED'
    assert a.build_exact_surrogate(model,parallel_rows(6),list(range(6)))['status']=='ENGINEERING_APPROXIMATION_SURROGATE'
