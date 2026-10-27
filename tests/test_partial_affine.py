"""Engineering partial affine candidates; no real-video or exact-theory claim."""
from copy import deepcopy
from itertools import combinations
import json
import numpy as np
import pytest
from kozlov_lab.stereo.partial_affine import search_partial_affine, _area_code_check


def fixture(count=8,noise=0.):
    xy=np.array([[20,20],[100,20],[30,100],[95,110],[60,55],[150,60],[120,155],[170,130]],dtype=float)[:count]
    a=np.array([[1.05,.07],[-.03,.95]]);t=np.array([7.,-4.])
    right=xy@a.T+t+np.random.default_rng(4).normal(0,noise,xy.shape)
    s=[dict(id='s'+str(i),xy=p.tolist()) for i,p in enumerate(xy)]
    d=[dict(id='t'+str(i),xy=p.tolist()) for i,p in enumerate(right)]
    e=[dict(source_id='s'+str(i),target_id='t'+str(i),descriptor_distance=float(i)) for i in range(count)]
    return s,d,e


def test_partial_missing_added_points_real_coordinates_unchanged():
    s,t,e=fixture(6);s.append(dict(id='source_only',xy=[180.,10.]));t.append(dict(id='target_only',xy=[5.,170.]))
    before=deepcopy((s,t,e));r=search_partial_affine(s,t,e)
    assert r['candidate_count']==1
    c=r['candidates'][0]
    assert c['mapping_count']==6 and c['extra_point_count']==3
    assert c['unmatched_source_ids']==['source_only'] and c['unmatched_target_ids']==['target_only']
    assert c['area_code']['comparisons']==380 and c['supporting_hypotheses']>1
    assert (s,t,e)==before
    assert not r['identity_verified'] and not r['native_injected'] and not r['geometry_acceptance']
    assert not r['absence_claim']
    json.dumps(r,allow_nan=False)


def test_noise_extra_points_and_complete_code():
    s,t,e=fixture(8,.00025);r=search_partial_affine(s,t,e)
    assert r['candidate_count']>=1
    c=max(r['candidates'],key=lambda c:c['mapping_count'])
    assert c['mapping_count']==6 and c['area_code']['comparisons']==380
    assert all(len(c['unmatched_source_ids'])==2 for c in r['candidates'])
    assert c['area_code']['max_difference']<=.08


def test_three_anchors_never_self_certify():
    r=search_partial_affine(*fixture(3))
    assert not r['candidates'] and r['rejection_counts']['INSUFFICIENT_EXTRA_POINTS']==1


def test_collinear_and_duplicate_coordinates():
    s,t,e=fixture(6)
    for i in range(6):s[i]['xy']=[float(i*20),0.];t[i]['xy']=[float(i*20+3),5.]
    r=search_partial_affine(s,t,e)
    assert not r['candidates'] and r['rejection_counts']['ANCHOR_AREA_REJECTED']==20
    s[1]['xy']=s[0]['xy'][:]
    with pytest.raises(ValueError,match='duplicate coordinates'):search_partial_affine(s,t,e)


def test_close_repeated_pattern_is_ambiguous_not_greedy():
    s,t,e=fixture(6)
    for i,p in enumerate(t[:]):
        t.append(dict(id='near'+str(i),xy=[p['xy'][0]+1,p['xy'][1]]))
        e.append(dict(source_id='s'+str(i),target_id='near'+str(i),descriptor_distance=.1))
    r=search_partial_affine(s,t,e)
    assert not r['candidates'] and r['rejection_counts'].get('AMBIGUOUS_ANCHOR_MAPPING',0)>0
    assert any(row.get('ambiguous_edge_indices') for row in r['hypothesis_ledger'])


def test_distinct_repeated_pattern_candidates_retained_ambiguous():
    s,t,e=fixture(6)
    for i,p in enumerate(t[:]):
        t.append(dict(id='copy'+str(i),xy=[p['xy'][0]+35,p['xy'][1]+3]))
        e.append(dict(source_id='s'+str(i),target_id='copy'+str(i),descriptor_distance=.1))
    r=search_partial_affine(s,t,e)
    assert len(r['candidates'])>=2 and all(c['ambiguous'] for c in r['candidates'])


def test_wrong_pairing_and_non_affine_rejected():
    s,t,e=fixture(8)
    for i,edge in enumerate(e):edge['target_id']='t'+str((i+1)%8)
    assert not search_partial_affine(s,t,e)['candidates']
    s,t,e=fixture(8)
    for p in t:
        x,y=p['xy'];p['xy']=[x+.008*y*y,y+.008*x*x]
    assert not search_partial_affine(s,t,e)['candidates']


def test_sampled_budget_is_reproducible_partial_not_absence():
    s,t,e=fixture(8)
    r=search_partial_affine(s,t,e,max_hypotheses=5)
    assert r['status']=='PARTIAL_SEARCH_NOT_EXHAUSTIVE'
    assert r['sampled_combination_indices']==[0,13,27,41,55]
    assert r['sampled_combinations']==[list(list(combinations(range(8),3))[i]) for i in [0,13,27,41,55]]
    assert not r['proposal_universe_exhausted'] and not r['absence_claim']


def test_over_twelve_has_explicit_globally_sampled_six_point_subsets():
    xy=np.random.default_rng(5).uniform(20,100,(13,2))
    s=[dict(id='s'+str(i),xy=p.tolist()) for i,p in enumerate(xy)]
    t=[dict(id='t'+str(i),xy=(p+[3,4]).tolist()) for i,p in enumerate(xy)]
    e=[dict(source_id='s'+str(i),target_id='t'+str(i),descriptor_distance=1.) for i in range(13)]
    r=search_partial_affine(s,t,e)
    assert r['candidates'] and r['total_subsets']>512 and r['checked_subsets']==512
    assert r['status']=='PARTIAL_SEARCH_NOT_EXHAUSTIVE'
    assert len(r['subset_ledger'])==512
    domains=[h for h in r['hypothesis_ledger'] if h['status']=='GEOMETRIC_SUBSET_DOMAIN']
    assert all(h['total_subsets']==120 for h in domains)
    expected=[i*(r['total_subsets']-1)//511 for i in range(512)]
    assert r['sampled_subset_indices']==expected
    assert len({row['hypothesis_combination_index'] for row in r['subset_ledger']})>1
    for c in r['candidates']:
        assert c['mapping_count']==6 and len(c['unmatched_source_ids'])==7
        assert len(c['unselected_geometric_edge_indices'])==7
    for row in r['subset_ledger']:
        h=next(h for h in domains if h['combination_index']==row['hypothesis_combination_index'])
        extras=[i for i in h['unique_edge_indices'] if i not in h['anchor_edge_indices']]
        assert row['extra_edge_indices']==[extras[i] for i in list(combinations(range(len(extras)),3))[row['local_subset_index']]]


def test_full_code_against_independent_shoelace_oracle():
    s,t,e=fixture(8,.1);left=np.array([p['xy'] for p in s]);right=np.array([p['xy'] for p in t])
    def area(points,ids):
        p=points[list(ids)]
        return abs(float(sum(p[i,0]*p[(i+1)%3,1]-p[(i+1)%3,0]*p[i,1] for i in range(3))/2))
    maximum=0.;count=0
    triples=list(combinations(range(8),3))
    for pivot in triples:
        assert min(area(left,pivot),area(right,pivot))>=32
        for ids in triples:
            if ids==pivot:continue
            maximum=max(maximum,abs(area(left,ids)/area(left,pivot)-area(right,ids)/area(right,pivot)));count+=1
    result=_area_code_check(left,right)
    assert result['comparisons']==count==56*55
    assert result['max_difference']==pytest.approx(maximum,abs=1e-12)
    assert _area_code_check(left,right*np.array([-1,1]))['max_difference']==pytest.approx(maximum,abs=1e-12)


def test_all_denominators_detect_hidden_small_denominator_mismatch():
    source=np.array([[0,0],[100,0],[0,100],[1,1]],dtype=float)
    target=source.copy();target[3]=[1.5,1]
    result=_area_code_check(source,target)
    assert result['status']=='CODE_MISMATCH' and result['max_difference']>33
    assert result['total_slots']==result['checked_slots']==12 and result['undefined_slots']==0


def test_zero_and_unstable_denominator_statuses():
    source=np.array([[0,0],[100,0],[0,100],[50,0]],dtype=float)
    same=_area_code_check(source,source)
    assert same['status']=='PASSED' and same['total_slots']==12 and same['checked_slots']==9 and same['undefined_slots']==3
    target=source.copy();target[3,1]=1
    assert _area_code_check(source,target)['status']=='ZERO_DENOMINATOR_MISMATCH'
    source[3]=[.1,.1]
    assert _area_code_check(source,source)['status']=='UNSTABLE_DENOMINATOR'


@pytest.mark.parametrize('bad',[True,0,513,1.5])
def test_bad_budget(bad):
    with pytest.raises(ValueError):search_partial_affine(*fixture(6),max_hypotheses=bad)


def test_bad_inputs_and_empty_scope():
    s,t,e=fixture(6)
    with pytest.raises(ValueError):search_partial_affine(s,t,e+e)
    s[0]['xy'][0]=float('nan')
    with pytest.raises(ValueError):search_partial_affine(s,t,e)
    r=search_partial_affine([],[],[])
    assert r['candidate_count']==0 and r['total_anchor_combinations']==0 and not r['absence_claim']
