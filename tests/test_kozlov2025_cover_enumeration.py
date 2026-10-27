"""Independent analytic small-set oracles, budget and certificate failure gates."""
import pytest
from fractions import Fraction as F
from kozlov_lab.kozlov2025.cover_enumeration import enumerate_independent_covers_2d as search


@pytest.mark.parametrize('points,total,valid', [
    ({0:(0,0)},1,1), ({0:(0,0),1:(1,0)},2,2),
    ({0:(0,0),1:(1,0),2:(0,1)},5,5),
    ({i:(i,0) for i in range(4)},15,8),
    ({0:(0,0),1:(1,0),2:(1,1),3:(0,1)},15,14),
])
def test_complete_analytic_truth(points,total,valid):
    r=search(points)
    assert r['status']=='COMPLETE' and r['complete'] is True
    assert r['visited_candidates']==r['judged_candidates']==r['expected_candidates']==total
    assert len(r['independent_covers'])==valid
    assert r['rejected_candidates']==total-valid and not r['uncertain_candidates']
    keys={tuple(tuple(p) for p in c['parts']) for c in r['independent_covers']}
    assert len(keys)==valid


def test_collinear_interval_oracle_independent_of_convex_hull_algorithm():
    points={i:(i,0) for i in range(4)};r=search(points)
    # Every partition of a strictly ordered line into disjoint convex intervals
    # is precisely determined by its three cut/no-cut decisions.
    oracle=set()
    for bits in range(8):
        groups=[[0]]
        for i in range(1,4):
            if bits & (1<<(i-1)):groups.append([])
            groups[-1].append(i)
        oracle.add(tuple(tuple(g) for g in groups))
    assert {tuple(tuple(p) for p in c['parts']) for c in r['independent_covers']}==oracle


def test_affine_and_label_permutation_preserve_cover_family():
    points={0:(0,0),1:(1,0),2:(1,1),3:(0,1)}
    labels={0:17,1:-8,2:5,3:2}
    moved={labels[i]:(2*x+F(y,3)+F(1,7),-y+F(2,9)) for i,(x,y) in points.items()}
    canonical=lambda groups:frozenset(frozenset(p) for p in groups)
    expected={canonical([[labels[i] for i in p] for p in c['parts']]) for c in search(points)['independent_covers']}
    assert {canonical(c['parts']) for c in search(moved)['independent_covers']}==expected


@pytest.mark.parametrize('kwargs', [{'seconds':0},{'max_candidates':0},{'max_candidates':14},{'total_max_work':0}])
def test_interruption_never_complete_or_empty_means_nonexistence(kwargs):
    r=search({i:(i,0) for i in range(4)},**kwargs)
    assert r['status']=='PARTIAL_UNKNOWN' and r['complete'] is False
    assert r['judged_candidates']<15


def test_singleton_zero_work_is_complete_but_zero_time_is_unknown():
    assert search({0:(0,0)},total_max_work=0)['complete'] is True
    assert search({0:(0,0)},seconds=0)['complete'] is False


@pytest.mark.parametrize('points', [None,{}, {True:(0,0)},{0:(True,0)},{0:(float('nan'),0)}, {0:(0,0),1:(0,0)}])
def test_bad_set_rejected(points):
    assert search(points)['status']=='UNSUPPORTED_INPUT'


@pytest.mark.parametrize('kwargs', [{'seconds':True},{'seconds':float('inf')},{'seconds':10**10000},{'max_candidates':True},{'total_max_work':-1}])
def test_budget_parameters_rejected(kwargs):
    assert search({0:(0,0)},**kwargs)['status']=='UNSUPPORTED_INPUT'


def test_large_complete_set_rejected_without_sampling():
    r=search({i:(i,0) for i in range(8)})
    assert r['status']=='UNSUPPORTED_SIZE' and r['visited_candidates']==0


def test_shared_work_decreases_and_independent_failure_stops(monkeypatch):
    import kozlov_lab.kozlov2025.cover_enumeration as m
    solver=m.verify_independent_cover_2d;seen=[];times=[]
    def wrapped(*args,**kwargs):
        seen.append(kwargs['max_work']);times.append(kwargs['seconds']);return solver(*args,**kwargs)
    monkeypatch.setattr(m,'verify_independent_cover_2d',wrapped)
    r=m.enumerate_independent_covers_2d({i:(i,0) for i in range(4)})
    assert r['complete'] and seen==sorted(seen,reverse=True) and seen[-1]<seen[0]
    assert times==sorted(times,reverse=True) and times[-1]<times[0]
    monkeypatch.setattr(m,'verify_cover_certificate',lambda *args:False)
    r=m.enumerate_independent_covers_2d({0:(0,0),1:(1,0)})
    assert not r['complete'] and r['judged_candidates']==0 and len(r['uncertain_candidates'])==1
    assert r['unresolved_candidates']==2 and r['unvisited_candidates']==1


def test_corrupt_geometry_and_unknown_solver_are_not_rejections(monkeypatch):
    import kozlov_lab.kozlov2025.cover_enumeration as m
    monkeypatch.setattr(m,'verify_independent_cover_2d',lambda *a,**kw:{'status':'UNKNOWN_BUDGET','work_used':0})
    r=m.enumerate_independent_covers_2d({0:(0,0)})
    assert not r['complete'] and r['rejected_candidates']==0 and r['uncertain_candidates']


def test_expired_during_verification_does_not_publish_complete(monkeypatch):
    import kozlov_lab.kozlov2025.cover_enumeration as m
    clock=[0.];monkeypatch.setattr(m.time,'monotonic',lambda:clock[0])
    real=m.verify_cover_certificate
    def verify(*args):
        value=real(*args);clock[0]=31.;return value
    monkeypatch.setattr(m,'verify_cover_certificate',verify)
    r=m.enumerate_independent_covers_2d({0:(0,0)})
    assert not r['complete'] and r['judged_candidates']==0


def test_seven_singleton_line_complete_count():
    r=search({i:(i,0) for i in range(7)})
    assert r['complete'] and r['judged_candidates']==877
    assert len(r['independent_covers'])==64  # six independent cut positions
