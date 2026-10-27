from itertools import combinations
import json
import numpy as np
import pytest
from kozlov_lab.stereo.area_code import compare_area_codes


SOURCE=np.array([[21.,65.],[105.,16.],[147.,25.],[83.,190.],[135.,116.],[101.,114.]])
TARGET=SOURCE.copy();TARGET[5]=[100.94424093687824,113.99242374043777]


def oracle(a,b,scope):
    def area(p,ids):
        rows=p[list(ids)]
        return abs(sum(rows[i,0]*rows[(i+1)%3,1]-rows[(i+1)%3,0]*rows[i,1] for i in range(3)))/2
    triangles=list(combinations(range(len(a)),3));differences=[]
    for den in triangles:
        for num in triangles:
            if num==den or (scope=='four_point' and len(set(den)|set(num))!=4):continue
            assert min(area(a,den),area(b,den))>=32
            differences.append(abs(area(a,num)/area(a,den)-area(b,num)/area(b,den)))
    return len(differences),max(differences)


@pytest.mark.parametrize('scope,count',[('full',380),('four_point',180)])
def test_independent_unsigned_shoelace_oracle(scope,count):
    a=SOURCE.copy();b=TARGET.copy();r=compare_area_codes(a,b,scope=scope)
    n,maximum=oracle(a,b,scope)
    assert r['total_slots']==r['checked']==n==count
    assert r['max_difference']==pytest.approx(maximum,abs=1e-13)
    w=r['worst_slot'];assert w['difference']==r['max_difference']
    assert abs(w['source_ratio']-w['target_ratio'])==r['max_difference']
    assert np.array_equal(a,SOURCE) and np.array_equal(b,TARGET)
    json.dumps(r,allow_nan=False)


def test_fixed_short_passes_but_full_same_epsilon_fails():
    short=compare_area_codes(SOURCE,TARGET,scope='four_point')
    full=compare_area_codes(SOURCE,TARGET)
    assert short['status']=='PASSED' and short['max_difference']==pytest.approx(.0631039722130815)
    assert full['status']=='CODE_MISMATCH' and full['max_difference']==pytest.approx(.08497454671624105)
    assert short['epsilon']==full['epsilon']==.08


def test_reflection_ordinary_area_is_nonnegative():
    r=compare_area_codes(SOURCE,SOURCE*np.array([-1,1]))
    assert r['status']=='PASSED' and r['max_difference']==0
    assert r['worst_slot']['source_ratio']>=0 and r['worst_slot']['target_ratio']>=0


def test_all_collinear_not_pass_and_every_slot_accounted():
    a=np.array([[i*20.,0.] for i in range(6)])
    r=compare_area_codes(a,a)
    assert r['status']=='NO_DEFINED_STABLE_SLOTS' and r['total_slots']==380
    assert r['undefined']==380 and r['checked']==0 and r['max_difference'] is None


def test_single_zero_and_unstable_denominators_no_early_exit():
    a=np.array([[0.,0.],[100.,0.],[0.,100.],[50.,0.]])
    b=a.copy();b[-1,1]=1
    r=compare_area_codes(a,b)
    assert r['status']=='ZERO_DENOMINATOR_MISMATCH' and r['zero_denominator_mismatch_slots']==3
    assert r['total_slots']==12 and r['checked']==9
    a[-1]=[.1,.1];r=compare_area_codes(a,a)
    assert r['status']=='UNSTABLE_DENOMINATOR'
    assert r['checked']+r['undefined']+r['unsupported_slots']==r['total_slots']==12
    assert r['unstable_denominator_slots']==6


@pytest.mark.parametrize('bad',[float('nan'),float('inf'),-float('inf'),True,1000001.])
def test_invalid_coordinate_values_rejected(bad):
    a=SOURCE.tolist();a[0][0]=bad
    with pytest.raises(ValueError):compare_area_codes(a,SOURCE)


@pytest.mark.parametrize('kwargs',[dict(scope='short'),dict(epsilon=True),dict(epsilon=float('nan')),
    dict(min_denominator_area=-1),dict(min_denominator_area=float('inf'))])
def test_bad_scope_and_thresholds(kwargs):
    with pytest.raises(ValueError):compare_area_codes(SOURCE,SOURCE,**kwargs)


def test_shape_count_and_non_numeric_rejected():
    for bad in (SOURCE[:3],np.zeros((13,2)),np.zeros((6,3)),[['x','y']]*6,np.ones((6,2),dtype=bool)):
        with pytest.raises(ValueError):compare_area_codes(bad,bad)
    with pytest.raises(ValueError):compare_area_codes(SOURCE,SOURCE[:5])
