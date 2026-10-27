from fractions import Fraction as F
import pytest
from kozlov_lab.kozlov2025.real_algebraic import (Root, AN, Budget, BudgetExceeded,
    isolate_real_roots, mul, divrem, gcd, derivative, eval)


def test_true_cubic_and_roundtrip():
    b=Budget()
    r=Root((-256,-1440,864,27),F(87,50),F(7,4),b)
    a=r.num((0,1))
    assert (27*a**3+864*a**2-1440*a-256).sign()==0
    assert F(87,50)<a<F(7,4)
    assert (a/a).equals(1)
    rr=Root.from_data(r.to_data(),b)
    assert AN.from_data(a.to_data(),rr)>1


def test_repeated_and_rational_roots():
    b=Budget()
    roots=isolate_real_roots(mul((-1,0,1),(-1,1),b),b)
    assert len(roots)==2
    assert roots[0].num((0,1)).equals(-1)
    assert roots[1].num((0,1)).equals(1)


def test_reducible_denominator_only_other_root_zero():
    b=Budget()
    r=Root((-1,0,1),0,2,b)
    a=r.num((1,),(1,1))
    assert a.equals(F(1,2))
    assert (1/a).equals(2)
    with pytest.raises(ZeroDivisionError):
        r.num((1,),(-1,1))


def test_poly_arithmetic():
    b=Budget()
    assert divrem((-1,0,1),(-1,1),b)==((F(1),F(1)),(F(0),))
    assert gcd((-1,0,1),(-1,1),b)==(F(-1),F(1))
    assert eval(derivative((1,2,3),b),2,b)==14


@pytest.mark.parametrize('bad',[True,1.2,'1',None])
def test_strict_scalar(bad):
    with pytest.raises(ValueError):
        Root((-1,1),bad,2,Budget())


def test_serialization_pollution_bounds_and_interval():
    b=Budget()
    data=Root((-1,1),0,2,b).to_data()
    polluted=dict(data,root_count=1)
    with pytest.raises(ValueError): Root.from_data(polluted,b)
    for x in (True,1.0,'1'):
        bad=dict(data,lo=[x,1])
        with pytest.raises(ValueError): Root.from_data(bad,b)
    with pytest.raises(BudgetExceeded): Root.from_data(dict(data,lo=[1<<16385,1]),b)
    with pytest.raises(ValueError): Root((-1,0,1),-2,2,b)
    with pytest.raises(ValueError): Root((1,-2,1),0,2,b)
    with pytest.raises(ValueError): AN.from_data({'numerator':[[1,1]],'denominator':[[1,1]],'extra':0},Root((-1,1),0,2,b))


def test_budget_exhaustion_is_not_no_roots():
    with pytest.raises(BudgetExceeded): isolate_real_roots((-2,0,0,1),Budget(work=1))
    with pytest.raises(BudgetExceeded): Root((-2,0,0,1),1,2,Budget(seconds=0))


def test_cubic_all_three_roots_and_shared_budget():
    b=Budget()
    roots=isolate_real_roots((-256,-1440,864,27),b)
    assert len(roots)==3
    numbers=[r.num((0,1)) for r in roots]
    assert numbers[0]<-30
    assert -1<numbers[1]<0
    assert F(87,50)<numbers[2]<F(7,4)
    with pytest.raises(ValueError): numbers[0]+numbers[1]
    before=b.used
    assert (numbers[2]**2).sign()==1
    assert b.used>before


def test_nonzero_sign_needs_refinement():
    b=Budget()
    r=Root((-2,0,0,1),1,2,b)
    a=r.num((0,1))
    assert (a-F(5,4)).sign()==1
    assert (a-F(13,10)).sign()==-1
    assert (a**3).equals(2)
    assert (a**-3).equals(F(1,2))

def test_unit_inverse_normalizes_repeated_mixed_arithmetic():
    r=Root((-2,0,0,1),1,2,Budget())
    a=r.num((0,1))
    x=a
    for k in range(1,41):
        x=((x+F(k,7))*(a+1))/(a+1)-F(k,7)
        assert x.denominator==(F(1),)
        assert x.equals(a)
    assert (1/a).denominator==(F(1),)


def test_nonunit_local_cancellation_and_unit_in_reducible_ring():
    r=Root((-1,0,1),0,2,Budget())
    x=r.num((2,2),(4,4))
    assert x.equals(F(1,2))
    assert x.denominator==(F(1),)
    y=r.num((1,),(1,1))
    assert y.denominator==(F(1),F(1))
    for _ in range(20):
        y=y*r.num((1,1))/r.num((1,1))
        assert y.equals(F(1,2))
    assert r.num((1,),(2,1)).equals(F(1,3))
    assert r.num((1,),(2,1)).denominator==(F(1),)
