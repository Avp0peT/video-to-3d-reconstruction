from copy import deepcopy
from fractions import Fraction as F
import itertools

import pytest

from kozlov_lab.registration import (
    solve_fixed_orientation_translation as solve,
    assess_fixed_orientation_certificate as assess,
)


def case(kernel):
    source = {i: (100*i, 31*i*i) for i in range(len(kernel))}
    target = {i+100: tuple(F(source[i][k])-F(p[k]) for k in (0, 1))
              for i, p in enumerate(kernel)}
    mapping = {i: i+100 for i in source}
    return source, target, mapping


@pytest.mark.parametrize("kernel,center,radius,support_size", [
    ([(3, 4)], (3, 4), 0, 1),
    ([(3, 4), (3, 4), (3, 4)], (3, 4), 0, 1),
    ([(0, 0), (4, 0)], (2, 0), 4, 2),
    ([(0, 0), (4, 0), (1, 1)], (2, 0), 4, 2),
    ([(0, 0), (4, 0), (2, 3)], (2, F(5, 6)), F(169, 36), 3),
    ([(0, 0), (2, 0), (4, 0), (2, 0)], (2, 0), 4, 2),
])
def test_hand_cases(kernel, center, radius, support_size):
    a, b, m = case(kernel)
    out = solve(a, b, m)
    assert out["status"] == "EXACT_OPTIMUM", out
    c = out["certificate"]
    assert c["center"] == center and c["radius_squared"] == radius
    assert len(c["support"]) == support_size
    assert len(c["kernel"]) == len(kernel)
    assert assess(a, b, m, c)["status"] == "VERIFIED"


def test_rational_orientation_reflection_and_nonidentity_labels():
    a = {-4: (0, 0), 22: (3, 2), 2**63-1: (1, -2)}
    m = {-4: 8, 22: -9, 2**63-1: 5}
    for q in (((F(3, 5), -F(4, 5)), (F(4, 5), F(3, 5))), ((1, 0), (0, -1))):
        b = {m[i]: tuple(sum(q[k][j]*p[j] for j in (0, 1))+[7, -2][k]
                           for k in (0, 1)) for i, p in a.items()}
        out = solve(a, b, m, orientation=q)
        assert out["status"] == "EXACT_OPTIMUM"
        assert out["certificate"]["translation"] == (7, -2)
        assert out["certificate"]["radius_squared"] == 0
        assert assess(a, b, m, out["certificate"], orientation=q)["status"] == "VERIFIED"


def test_verifier_does_not_search(monkeypatch):
    import kozlov_lab.registration.fixed_orientation as mod
    a, b, m = case([(0, 0), (4, 0), (2, 3)])
    cert = solve(a, b, m)["certificate"]
    def forbidden(*args, **kwargs):
        raise AssertionError("search invoked")
    monkeypatch.setattr(mod, "solve_fixed_orientation_translation", forbidden)
    monkeypatch.setattr(mod, "_support", forbidden)
    monkeypatch.setattr(mod, "combinations", forbidden)
    assert assess(a, b, m, cert)["status"] == "VERIFIED"


@pytest.mark.parametrize("field,value", [
    ("radius_squared", False), ("radius_squared", F(999)),
    ("center", (0, 0)), ("translation", (1, 2)),
    ("weights", (F(1), F(0), F(0))), ("support", (0, 0)),
    ("kernel", {}), ("residuals_squared", {}), ("schema", "bad"),
])
def test_tampering(field, value):
    a, b, m = case([(0, 0), (4, 0), (2, 3)])
    cert = deepcopy(solve(a, b, m)["certificate"])
    cert[field] = value
    assert assess(a, b, m, cert)["status"] == "INVALID"


def test_strict_certificate_and_complete_input():
    a, b, m = case([(0, 0), (4, 0)])
    cert = solve(a, b, m)["certificate"]
    for mutation in (lambda c: c.update(extra=0),
                     lambda c: c["source"].update({0: (False, 0)}),
                     lambda c: c["mapping"].update({0: F(100)})):
        c = deepcopy(cert); mutation(c)
        assert assess(a, b, m, c)["status"] == "INVALID"
    changed = dict(a); changed[0] = (1, 0)
    assert assess(changed, b, m, cert)["status"] == "INVALID"


@pytest.mark.parametrize("a,b,m,q", [
    ({}, {}, {}, None), ({0: (0, 0)}, {1: (0, 0)}, None, None),
    ({0: (False, 0)}, {1: (0, 0)}, {0: 1}, None),
    ({0: (0.0, 0)}, {1: (0, 0)}, {0: 1}, None),
    ({2**63: (0, 0)}, {1: (0, 0)}, {2**63: 1}, None),
    ({0: (2**256, 0)}, {1: (0, 0)}, {0: 1}, None),
    ({0: (0, 0), 2: (0, 0)}, {1: (0, 0), 3: (1, 0)}, {0: 1, 2: 3}, None),
    ({0: (0, 0)}, {1: (0, 0)}, {0: 1}, ((2, 0), (0, 1))),
    ({0: (0, 0)}, {1: (0, 0)}, {0: 1}, ((True, 0), (0, 1))),
])
def test_unsupported(a, b, m, q):
    assert solve(a, b, m, orientation=q)["status"] == "UNSUPPORTED"


def test_budget_and_boundaries():
    a, b, m = case([(0, 0), (4, 0)])
    assert solve(a, b, m, max_work=0)["status"] == "UNKNOWN"
    assert solve(a, b, m, seconds=0)["status"] == "UNKNOWN"
    cert = solve(a, b, m)["certificate"]
    assert assess(a, b, m, cert, max_work=0)["status"] == "UNKNOWN"
    assert solve(a, b, m, max_work=True)["status"] == "UNSUPPORTED"
    a, b, m = case([(0, 0)]*32)
    assert solve(a, b, m)["status"] == "EXACT_OPTIMUM"
    a, b, m = case([(0, 0)]*33)
    assert solve(a, b, m)["status"] == "UNSUPPORTED"


def test_permutation_and_shift_invariance():
    a, b, m = case([(0, 0), (4, 0), (2, 3)])
    original = solve(a, b, m)["certificate"]
    for order in itertools.permutations(a):
        assert solve({i: a[i] for i in order}, b, m)["certificate"] == original
    shifted = {i: (p[0]+9, p[1]-7) for i, p in b.items()}
    c = solve(a, shifted, m)["certificate"]
    assert c["radius_squared"] == original["radius_squared"]
    assert c["translation"] == (original["translation"][0]+9, original["translation"][1]-7)
