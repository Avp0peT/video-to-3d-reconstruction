from copy import deepcopy
from fractions import Fraction as F
import pytest
from kozlov_lab.kozlov2025 import primary_futlyar as p
from kozlov_lab.kozlov2025.independent_cover import Budget


def test_radius_is_full_minimum_center_distance():
    points={0:(0,0),1:(3,4)}
    answer=p.assess_primary_futlyar(points,[[0],[1]])
    assert answer['status']=='TRUE'
    assert answer['radius_squared']==25
    assert p.verify_primary_futlyar_certificate(points,answer['certificate'])


def test_segments_and_ellipse_sizes_exact():
    points={0:(-1,0),1:(1,0),2:(9,0),3:(11,0)}
    a=p.assess_primary_futlyar(points,[[0,1],[2,3]])
    assert a['status']=='TRUE'
    assert a['radius_squared']==100
    tri={0:(0,0),1:(1,0),2:(0,1),3:(10,10)}
    a=p.assess_primary_futlyar(tri,[[0,1,2],[3]])
    assert a['status']=='TRUE'
    assert a['certificate']['size_descriptors'][0]['kind']=='ellipse'


def test_half_size_screen_alone_is_insufficient():
    points={0:(-1,0),1:(1,0),2:(1,1)}
    a=p.assess_primary_futlyar(points,[[0,1],[2]])
    assert a['status']=='FALSE'
    assert all(a['certificate']['radius_screens'])
    assert not a['certificate']['point_checks'][1]['uniquely_nearest']


def test_radius_screen_rejection_and_primary_boundary():
    points={0:(-2,0),1:(2,0),2:(0,1)}
    assert p.assess_primary_futlyar(points,[[0,1],[2]])['status']=='FALSE'
    boundary={0:(-1,0),1:(1,0),2:(0,1)}
    a=p.assess_primary_futlyar(boundary,[[0,1],[2]])
    assert a['status']=='UNSUPPORTED'
    assert all(a['certificate']['radius_screens'])


def test_m1_and_coincident_centers_unresolved():
    assert p.assess_primary_futlyar({0:(0,0)},[[0]])['status']=='UNSUPPORTED'
    with pytest.raises(p.UnsupportedDomain):
        p._geometry({0:(0,0),1:(1,0)},[(0,),(1,)],[(F(0),F(0))]*2,
                    [dict(kind='squared',squared=0)]*2,Budget())


def test_invalid_and_nonindependent_are_distinct():
    assert p.assess_primary_futlyar({0:(0,0),1:(1,0)},[[0]])['status']=='INVALID_INPUT'
    assert p.assess_primary_futlyar({0:(True,0)},[[0]])['status']=='INVALID_INPUT'
    assert p.assess_primary_futlyar({0:(0.0,0)},[[0]])['status']=='INVALID_INPUT'
    assert p.assess_primary_futlyar({0:(0,0),1:(2,0),2:(1,0)},[[0,1],[2]])['status']=='FALSE'


def test_budget_exhaustion_is_unknown():
    assert p.assess_primary_futlyar({0:(0,0),1:(1,0)},[[0],[1]],max_work=1)['status']=='UNKNOWN'
    assert p.assess_primary_futlyar({0:(0,0),1:(1,0)},[[0],[1]],seconds=0)['status']=='UNKNOWN'


def test_verifier_no_constructor_and_tamper(monkeypatch):
    points={0:(0,0),1:(3,4)}
    z=p.assess_primary_futlyar(points,[[0],[1]])['certificate']
    def forbidden(*args,**kwargs): raise AssertionError('producer invoked')
    monkeypatch.setattr(p,'solve_exact_center',forbidden)
    monkeypatch.setattr(p,'verify_independent_cover_2d',forbidden)
    assert p.verify_primary_futlyar_certificate(points,z)
    assert p.assess_primary_futlyar_certificate(points,z,max_work=1)['status']=='UNKNOWN'
    for key,value in [('radius_squared',F(25,4)),('centers',[(0,0),(3,3)]),('extra',1)]:
        bad=deepcopy(z); bad[key]=value
        assert not p.verify_primary_futlyar_certificate(points,bad)
    bad=deepcopy(z); bad['point_checks'][0]['distances_squared'][0]=False
    assert not p.verify_primary_futlyar_certificate(points,bad)
    bad=deepcopy(z); bad['blocks'][0]['result']['certificate']['center']=(1,1)
    assert not p.verify_primary_futlyar_certificate(points,bad)
    assert not p.verify_primary_futlyar_certificate({**points,2:(10,10)},z)


def test_exact_radical_screen_equality_and_sign():
    b=Budget()
    assert p._radius_screen(F(3),dict(kind='ellipse',trace=4,discriminant=4),b)
    assert not p._radius_screen(F(29,10),dict(kind='ellipse',trace=4,discriminant=4),b)
    assert not p._radius_screen(F(1),dict(kind='ellipse',trace=4,discriminant=0),b)
    assert p._radius_screen(F(3),dict(kind='ellipse',trace=4,discriminant=2),b)


def test_nonrational_contact4_center_explicitly_unsupported():
    points={0:(0,0),1:(1,0),2:(0,1),3:(2,F(3,2)),4:(100,100)}
    a=p.assess_primary_futlyar(points,[[0,1,2,3],[4]])
    assert a['status']=='UNSUPPORTED'
    assert 'not proven rational' in a['reason']


def test_certificate_bool_numeric_pollution_and_fraction_points():
    points={0:(F(1,3),F(2,5)),1:(F(7,3),F(2,5))}
    answer=p.assess_primary_futlyar(points,[[0],[1]])
    assert answer['status']=='TRUE'
    assert answer['radius_squared']==4
    z=answer['certificate']
    bad=deepcopy(z); bad['size_descriptors'][0]['squared']=False
    assert not p.verify_primary_futlyar_certificate(points,bad)
    bad=deepcopy(z); bad['radius_squared']=True
    assert not p.verify_primary_futlyar_certificate(points,bad)
    bad=deepcopy(z); bad['point_checks'][0]['label']=False
    assert not p.verify_primary_futlyar_certificate(points,bad)
    bad=deepcopy(z); bad['point_checks'][0]['owner']=False
    assert not p.verify_primary_futlyar_certificate(points,bad)


def test_rational_contact4_center_accepted_if_constant_normalization():
    points={0:(-1,-1),1:(1,-1),2:(1,1),3:(-1,1),4:(10,0)}
    answer=p.assess_primary_futlyar(points,[[0,1,2,3],[4]])
    assert answer['status']=='TRUE'
    assert answer['centers'][0]==(F(0),F(0))
    assert answer['certificate']['blocks'][0]['result']['profile']=='contact4'
