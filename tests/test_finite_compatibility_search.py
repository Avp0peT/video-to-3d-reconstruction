import copy
from unittest.mock import patch
from kozlov_lab.kozlov2025.finite_compatibility_search import (
    search_finite_compatibility, assess_finite_compatibility_certificate)

ENV = {0:(0,0),1:(1,0),2:(3,0),3:(0,10),4:(1,10),5:(2,10)}
GROUP = ((0,1,2),(3,4,5))


def test_automatic_nonzero_collinear_tight_and_independent_verifier():
    r = search_finite_compatibility(ENV, GROUP)
    assert r['status'] == 'EXACT_GRANULARITY_CERTIFIED', r
    assert r['granularity'] == 3
    fits = r['certificate']['compatibility_certificate']['member_certificates']
    assert any(f['schema']=='collinear-fits-v1' and f['alignment_certificate']['error_squared']>0 for f in fits)
    with patch('kozlov_lab.kozlov2025.finite_compatibility_search.search_finite_compatibility', side_effect=AssertionError), patch('kozlov_lab.kozlov2025.exact_center.solve_exact_center',side_effect=AssertionError):
        assert assess_finite_compatibility_certificate(ENV,GROUP,r['certificate'])['status']=='VERIFIED'


def test_full_2d_affine_triangle_uses_zero_error():
    env={0:(0,0),1:(2,0),2:(0,2),3:(10,10),4:(14,10),5:(12,16)}
    r=search_finite_compatibility(env,GROUP)
    assert r['status']=='EXACT_GRANULARITY_CERTIFIED',r
    assert all(f['schema']=='zero-error-fits-v1' for f in r['certificate']['compatibility_certificate']['member_certificates'])


def test_partial_budget_is_unknown_with_no_maximum():
    for kw in ({'max_work':0},{'seconds':0},{'max_work':50}):
        r=search_finite_compatibility(ENV,GROUP,**kw)
        assert r['status']=='UNKNOWN'
        assert r['granularity'] is None and r['certificate'] is None
        # Existing Budget records the operation that crosses the cap before
        # raising; retain that evidence instead of clamping the counter.
        assert r['work_used'] >= 0
        assert 'budget' in r['reason'].lower() or 'time' in r['reason'].lower()
        assert r['general_compatibility_decided'] is False


def test_changed_complete_input_group_bool_and_maximum_forgery():
    c=search_finite_compatibility(ENV,GROUP)['certificate']
    for field in ('lower_bound','upper_bound'):
        bad=copy.deepcopy(c);bad[field]=True
        assert assess_finite_compatibility_certificate(ENV,GROUP,bad)['status']=='INVALID'
    assert assess_finite_compatibility_certificate({**ENV,6:(99,99)},GROUP,c)['status']=='INVALID'
    assert assess_finite_compatibility_certificate(ENV,tuple(reversed(GROUP)),c)['status']=='INVALID'
    bad=copy.deepcopy(c);bad['lower_bound']=2
    assert assess_finite_compatibility_certificate(ENV,GROUP,bad)['status']=='INVALID'
    bad=copy.deepcopy(c);bad['granularity_certificate']=None
    assert assess_finite_compatibility_certificate(ENV,GROUP,bad)['status']=='INVALID'


def test_m1_unknown_strict_input_no_false():
    r=search_finite_compatibility({0:(0,0),1:(1,1)},((0,),(1,)))
    assert r['status']=='UNKNOWN' and r['granularity'] is None
    for kw in ({'max_work':True},{'seconds':True}):
        assert search_finite_compatibility(ENV,GROUP,**kw)['status']=='UNSUPPORTED_INPUT'
    assert search_finite_compatibility(ENV,((0,1),(1,2)))['status']=='UNSUPPORTED_INPUT'


def test_lower_witness_not_promoted_and_all_source_partitions_visited():
    from kozlov_lab.kozlov2025.zero_error_fit_search import search_zero_error_fits as zero
    from kozlov_lab.kozlov2025.collinear_fits import collinear_fits as collinear
    def restricted(fn):
        def run(a,ap,b,bp,**kw):
            if len(ap)==3:
                return dict(status='NO_SUPPORTED_WITNESS',work_used=0)
            return fn(a,ap,b,bp,**kw)
        return run
    with patch('kozlov_lab.kozlov2025.finite_compatibility_search.search_zero_error_fits',side_effect=restricted(zero)), patch('kozlov_lab.kozlov2025.finite_compatibility_search.collinear_fits',side_effect=restricted(collinear)):
        r=search_finite_compatibility(ENV,GROUP)
    assert r['status']=='LOWER_BOUND',r
    assert r['lower_bound']==2 and r['granularity'] is None
    assert assess_finite_compatibility_certificate(ENV,GROUP,r['certificate'])['status']=='VERIFIED'
    bad=copy.deepcopy(r['certificate']);bad['complete']=True
    assert assess_finite_compatibility_certificate(ENV,GROUP,bad)['status']=='INVALID'
    # No supported witness forces all source partitions, including the
    # overlapping-hull collinear partition ((0,2),(1,)), to be attempted.
    no=dict(status='NO_SUPPORTED_WITNESS',work_used=0)
    with patch('kozlov_lab.kozlov2025.finite_compatibility_search.search_zero_error_fits',return_value=no), patch('kozlov_lab.kozlov2025.finite_compatibility_search.collinear_fits',return_value=no):
        exhausted=search_finite_compatibility(ENV,GROUP)
    assert exhausted['status']=='UNKNOWN'
    assert any(row['kind']=='source' and row['parts']==((0,2),(1,)) for row in exhausted['visited_candidates'])
    assert len({(row['kind'],row['id']) for row in exhausted['visited_candidates']})==len(exhausted['visited_candidates'])
