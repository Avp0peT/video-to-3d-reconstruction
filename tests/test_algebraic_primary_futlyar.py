from fractions import Fraction as F
from copy import deepcopy
import pytest
from kozlov_lab.kozlov2025 import algebraic_primary_futlyar as p
from kozlov_lab.kozlov2025.real_algebraic import Root,AN
from kozlov_lab.kozlov2025.independent_cover import Budget
from kozlov_lab.kozlov2025.exact_center import encode_exact,decode_exact
import json


@pytest.fixture(scope='module')
def cubic():
    points={0:(0,0),1:(1,0),2:(0,1),3:(2,F(3,2)),4:(100,100)}
    parts=[[0,1,2,3],[4]]
    answer=p.assess_algebraic_primary_futlyar(points,parts)
    assert answer['status']=='TRUE',answer
    return points,parts,answer


def test_genuine_cubic_complete_primary(cubic):
    points,parts,answer=cubic
    z=answer['certificate']
    assert z['blocks'][0]['result']['profile']=='contact4'
    assert len(z['number_context']['algebraic_root']['poly'])==4
    assert p.verify_algebraic_primary_certificate(points,z)
    r=Root.from_data(z['number_context']['algebraic_root'],Budget())
    c=[AN.from_data(x,r) for x in z['centers'][0]]
    assert any(len(x.numerator)>1 for x in c)
    # Independent radius identity for precisely two centers, no float decoding.
    distance=(c[0]-100)**2+(c[1]-100)**2
    assert AN.from_data(z['radius_squared'],r)==distance
    assert len(z['point_checks'])==len(points)
    assert all(x['uniquely_nearest'] and x['strictly_inside'] for x in z['point_checks'])
    restored=decode_exact(json.loads(json.dumps(encode_exact(z))))
    assert p.verify_algebraic_primary_certificate(points,restored)


def test_rational_primary_and_strict_boundary():
    answer=p.assess_algebraic_primary_futlyar({0:(0,0),1:(3,4)},[[0],[1]])
    assert answer['status']=='TRUE'
    z=answer['certificate'];r=Root.from_data(z['number_context']['algebraic_root'],Budget())
    assert AN.from_data(z['radius_squared'],r)==25 # full distance, not half
    boundary={0:(-1,0),1:(1,0),2:(0,1)}
    assert p.assess_algebraic_primary_futlyar(boundary,[[0,1],[2]])['status']=='UNSUPPORTED'
    wrong={0:(-1,0),1:(1,0),2:(1,1)}
    assert p.assess_algebraic_primary_futlyar(wrong,[[0,1],[2]])['status']=='FALSE'


def test_algebraic_verifier_independent_and_tamper(cubic,monkeypatch):
    points,_,answer=cubic;z=answer['certificate']
    def forbidden(*a,**k):raise AssertionError('constructor called')
    monkeypatch.setattr(p,'solve_exact_center',forbidden)
    monkeypatch.setattr(p,'verify_independent_cover_2d',forbidden)
    assert p.verify_algebraic_primary_certificate(points,z)
    bad=deepcopy(z);bad['radius_squared']['numerator']=[[1,1]]
    assert not p.verify_algebraic_primary_certificate(points,bad)
    bad=deepcopy(z);bad['point_checks'][0]['owner']=False
    assert not p.verify_algebraic_primary_certificate(points,bad)
    bad=deepcopy(z);bad['centers'][0][0]['numerator'][0][0]=False
    assert not p.verify_algebraic_primary_certificate(points,bad)
    bad=deepcopy(z);bad['number_context']['algebraic_root']['poly'][0][0]+=1
    assert not p.verify_algebraic_primary_certificate(points,bad)
    assert not p.verify_algebraic_primary_certificate({**points,5:(200,200)},z)


def test_budget_and_domain_are_not_false(cubic):
    points,parts,answer=cubic
    assert p.assess_algebraic_primary_futlyar(points,parts,max_work=1)['status']=='UNKNOWN'
    assert p.assess_algebraic_primary_certificate(points,answer['certificate'],max_work=1)['status']=='UNKNOWN'
    assert p.assess_algebraic_primary_futlyar({0:(0,0)},[[0]])['status']=='UNSUPPORTED'
    assert p.assess_algebraic_primary_futlyar({0:(True,0),1:(2,0)},[[0],[1]])['status']=='INVALID_INPUT'
    assert p.assess_algebraic_primary_futlyar(points,[[0,1,2,3]])['status']=='INVALID_INPUT'


def test_three_centers_include_two_rational_domains():
    points={0:(0,0),1:(1,0),2:(0,1),3:(2,F(3,2)),4:(100,100),5:(200,0)}
    answer=p.assess_algebraic_primary_futlyar(points,[[0,1,2,3],[4],[5]])
    assert answer['status']=='TRUE',answer
    assert p.verify_algebraic_primary_certificate(points,answer['certificate'])
