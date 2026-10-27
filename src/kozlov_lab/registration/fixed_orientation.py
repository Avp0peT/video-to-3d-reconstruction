"""S-2009 p102 fixed correspondence/orientation optimal translation.

The source minimum-circle reduction is implemented by project-owned finite
support enumeration. Rational reflections use the same reduction. No search
over orientation, correspondence, similarity or affine transformations occurs.
"""
from fractions import Fraction as F
from itertools import combinations
import math
import time

SCHEMA = "fixed-orientation-translation-v1"
MAX_STORED_BITS = 16384


class _Exhausted(Exception):
    pass


class _Budget:
    def __init__(self, work, seconds):
        if (type(work) is not int or not 0 <= work <= 2000000
                or type(seconds) not in (int, float) or not math.isfinite(seconds)
                or not 0 <= seconds <= 30):
            raise ValueError("bounded work 0..2000000 and seconds 0..30 required")
        self.maximum, self.used = work, 0
        self.deadline = time.monotonic() + seconds

    def spend(self, amount=1):
        self.used += amount
        if self.used > self.maximum or time.monotonic() >= self.deadline:
            raise _Exhausted("work/time budget exhausted")


def _fraction(v, bits=256):
    if type(v) not in (int, F):
        raise ValueError("int/Fraction required; bool and float rejected")
    v = F(v)
    if max(v.numerator.bit_length(), v.denominator.bit_length()) > bits:
        raise ValueError("rational bit bound exceeded")
    return v


def _label(i):
    if type(i) is not int or not -(2**63) <= i < 2**63:
        raise ValueError("int64 label required")
    return i


def _points(raw):
    if type(raw) is not dict or not 1 <= len(raw) <= 32:
        raise ValueError("complete labelled 1..32 point dictionary required")
    out = {}
    for i, p in raw.items():
        _label(i)
        if type(p) not in (list, tuple) or len(p) != 2:
            raise ValueError("two coordinates required")
        out[i] = tuple(_fraction(v) for v in p)
    if len(set(out.values())) != len(out):
        raise ValueError("duplicate original coordinates")
    return out


def _input(source, target, mapping, orientation):
    a, b = _points(source), _points(target)
    if type(mapping) is not dict:
        raise ValueError("explicit source-to-target bijection required")
    for i, j in mapping.items():
        _label(i); _label(j)
    if set(mapping) != set(a) or set(mapping.values()) != set(b) or len(a) != len(b):
        raise ValueError("complete bijection required")
    q = ((1, 0), (0, 1)) if orientation is None else orientation
    if type(q) not in (tuple, list) or len(q) != 2:
        raise ValueError("2x2 orientation required")
    if any(type(row) not in (tuple, list) or len(row) != 2 for row in q):
        raise ValueError("2x2 orientation required")
    q = tuple(tuple(_fraction(v) for v in row) for row in q)
    if any(sum(q[k][i]*q[k][j] for k in (0, 1)) != (1 if i == j else 0)
           for i in (0, 1) for j in (0, 1)):
        raise ValueError("exact rational orthogonal orientation required")
    return a, b, dict(mapping), q


def _apply(q, p):
    return tuple(sum(q[k][j]*p[j] for j in (0, 1)) for k in (0, 1))


def _norm(p, c):
    return sum((p[k]-c[k])**2 for k in (0, 1))


def _tree(v, budget, depth=0):
    budget.spend()
    if depth > 12:
        raise ValueError("certificate nesting bound")
    if type(v) in (int, F):
        _fraction(v, MAX_STORED_BITS)
    elif type(v) is str:
        if len(v) > 256: raise ValueError("string bound")
    elif type(v) in (list, tuple):
        if len(v) > 64: raise ValueError("array bound")
        for x in v: _tree(x, budget, depth+1)
    elif type(v) is dict:
        if len(v) > 64: raise ValueError("dictionary bound")
        for k, x in v.items():
            if type(k) is int: _label(k)
            elif type(k) is not str or len(k) > 64: raise ValueError("key bound")
            _tree(x, budget, depth+1)
    else:
        raise ValueError("pure scalar bounded certificate required; bool rejected")


def _support(points, ids):
    p = [points[i] for i in ids]
    if len(ids) == 1:
        return p[0], F(0), (F(1),)
    if len(ids) == 2:
        c = tuple((p[0][k]+p[1][k])/2 for k in (0, 1))
        return c, _norm(p[0], c), (F(1, 2), F(1, 2))
    a, b, d = p
    u, v = tuple(b[k]-a[k] for k in (0, 1)), tuple(d[k]-a[k] for k in (0, 1))
    det = u[0]*v[1]-u[1]*v[0]
    if not det: return None
    h, j = _norm(b, a)/2, _norm(d, a)/2
    off = ((h*v[1]-u[1]*j)/det, (u[0]*j-h*v[0])/det)
    c = tuple(a[k]+off[k] for k in (0, 1))
    beta = (off[0]*v[1]-off[1]*v[0])/det
    gamma = (u[0]*off[1]-u[1]*off[0])/det
    weights = (1-beta-gamma, beta, gamma)
    if min(weights) < 0: return None
    return c, _norm(a, c), weights


def _verify(a, b, mapping, q, cert, budget):
    _tree(cert, budget)
    keys = {"schema", "source", "target", "mapping", "orientation", "kernel",
            "center", "translation", "radius_squared", "support", "weights", "residuals_squared"}
    if type(cert) is not dict or set(cert) != keys or cert["schema"] != SCHEMA:
        return False
    if (cert["source"] != a or cert["target"] != b or cert["mapping"] != mapping
            or cert["orientation"] != q): return False
    # Labels are checked independently: Python equality accepts Fraction labels.
    for name in ("source", "target", "mapping", "kernel", "residuals_squared"):
        if type(cert[name]) is not dict or any(type(i) is not int for i in cert[name]): return False
    if any(type(j) is not int for j in cert["mapping"].values()): return False
    c = cert["center"]; t = cert["translation"]; r = cert["radius_squared"]
    if type(c) is not tuple or len(c) != 2 or type(t) is not tuple or len(t) != 2:
        return False
    if type(r) not in (int, F) or r < 0 or t != tuple(-x for x in c): return False
    kernel, residuals = {}, {}
    for i in sorted(a):
        budget.spend(30)
        moved = _apply(q, a[i])
        # source is transformed toward target; kernel = Q source - target.
        kernel[i] = tuple(moved[k]-b[mapping[i]][k] for k in (0, 1))
        placed = tuple(moved[k]+t[k] for k in (0, 1))
        residuals[i] = _norm(placed, b[mapping[i]])
        if residuals[i] > r: return False
    if cert["kernel"] != kernel or cert["residuals_squared"] != residuals: return False
    ids, weights = cert["support"], cert["weights"]
    if (type(ids) is not tuple or not 1 <= len(ids) <= 3 or len(set(ids)) != len(ids)
            or any(type(i) is not int or i not in a for i in ids)
            or type(weights) is not tuple or len(weights) != len(ids)
            or any(type(w) not in (int, F) or w < 0 for w in weights)
            or sum(weights) != 1): return False
    if any(_norm(kernel[i], c) != r for i in ids): return False
    if any(sum(w*kernel[i][k] for i, w in zip(ids, weights)) != c[k] for k in (0, 1)):
        return False
    budget.spend(0)
    return True


def assess_fixed_orientation_certificate(source, target, mapping, certificate, *, orientation=None,
                                         max_work=2000000, seconds=30):
    """Independent no-search complete-input certificate assessment."""
    budget = None
    try:
        budget = _Budget(max_work, seconds); budget.spend()
        a, b, m, q = _input(source, target, mapping, orientation)
    except _Exhausted as exc:
        return dict(status="UNKNOWN", reason=str(exc), work_used=budget.used)
    except (ValueError, TypeError, ArithmeticError) as exc:
        return dict(status="UNSUPPORTED", reason=str(exc), work_used=budget.used if budget else 0)
    try:
        valid = _verify(a, b, m, q, certificate, budget)
        return dict(status="VERIFIED" if valid else "INVALID", work_used=budget.used)
    except _Exhausted as exc:
        return dict(status="UNKNOWN", reason=str(exc), work_used=budget.used)
    except (ValueError, TypeError, KeyError, IndexError, ArithmeticError) as exc:
        return dict(status="INVALID", reason=str(exc), work_used=budget.used)


def verify_fixed_orientation_certificate(source, target, mapping, certificate, **kwargs):
    return assess_fixed_orientation_certificate(source, target, mapping, certificate, **kwargs)["status"] == "VERIFIED"


def solve_fixed_orientation_translation(source, target, mapping, *, orientation=None,
                                       max_work=2000000, seconds=30):
    """Exact minimax translation of Q(source) toward target with explicit pairing."""
    budget = None; validated = False
    try:
        budget = _Budget(max_work, seconds); budget.spend()
        a, b, m, q = _input(source, target, mapping, orientation); validated = True
        kernel = {i: tuple(_apply(q, a[i])[k]-b[m[i]][k] for k in (0, 1)) for i in sorted(a)}
        best = None
        for size in (1, 2, 3):
            for ids in combinations(sorted(a), size):
                budget.spend(100)
                candidate = _support(kernel, ids)
                if candidate is None: continue
                c, r, weights = candidate
                _tree((c, r, weights), budget)
                if best is not None and r >= best[1]: continue
                contained = True
                for p in kernel.values():
                    budget.spend(15)
                    if _norm(p, c) > r:
                        contained = False; break
                if contained: best = (c, r, ids, weights)
        if best is None: raise ArithmeticError("no enclosing support found")
        c, r, ids, weights = best
        cert = dict(schema=SCHEMA, source=a, target=b, mapping=m, orientation=q,
                    kernel=kernel, center=c, translation=tuple(-x for x in c),
                    radius_squared=r, support=ids, weights=weights,
                    residuals_squared={i: _norm(kernel[i], c) for i in a})
        if not _verify(a, b, m, q, cert, budget): raise ArithmeticError("internal certificate failure")
        return dict(status="EXACT_OPTIMUM", certificate=cert, work_used=budget.used)
    except _Exhausted as exc:
        return dict(status="UNKNOWN", reason=str(exc), work_used=budget.used)
    except (ValueError, TypeError, ArithmeticError) as exc:
        return dict(status="INVALID" if validated else "UNSUPPORTED", reason=str(exc),
                    work_used=budget.used if budget else 0)
