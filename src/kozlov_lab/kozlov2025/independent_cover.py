"""Supplied independent cover, S-2025 PDF4/p149 and PDF10/p155. Stdlib only.

Verifies a given cover of a complete finite 2D set, not optimal search,
m-ostov, ellipse centres, futlyar, fits, semantic parts or 3D reconstruction.
The independent certificate verifier never invokes hull/SAT or the solver.
"""
from fractions import Fraction as F
import itertools
import math
import time

__all__ = ["verify_independent_cover_2d", "verify_cover_certificate"]
MAX_COORDINATE_BITS = 2048
MAX_AXIS_BITS = 8192
MAX_WITNESS_BITS = 32768


def _fraction(value, maximum=MAX_COORDINATE_BITS):
    if type(value) not in (int, float, F) or type(value) is float and not math.isfinite(value):
        raise ValueError("finite integer/binary64/Fraction required; bool is unsupported")
    result = F(value)
    if max(result.numerator.bit_length(), result.denominator.bit_length()) > maximum:
        raise ValueError("rational bit budget exceeded")
    return result


class BudgetExceeded(Exception):
    pass


class Budget:
    def __init__(self, work=2000000, seconds=60):
        self.maximum = work
        self.used = 0
        self.deadline = time.monotonic()+seconds
    def spend(self, amount=1):
        self.used += amount
        if self.used > self.maximum or time.monotonic() >= self.deadline:
            raise BudgetExceeded("explicit work/time budget exhausted")


def normalize(points, parts):
    if (not isinstance(points, dict) or not 1 <= len(points) <= 4096
            or not isinstance(parts, (list, tuple)) or not 1 <= len(parts) <= 32):
        raise ValueError("nonempty image <=4096 points and 1..32 proposed parts required")
    normalized = {}
    for label, pair in points.items():
        if type(label) is not int or not isinstance(pair,(list,tuple)) or len(pair) != 2:
            raise ValueError("integer unique labels and 2D coordinates required")
        row = tuple(_fraction(v) for v in pair)
        normalized[label] = row
    if len(set(normalized.values())) != len(normalized):
        raise ValueError("initial image has duplicate coordinates; no automatic merging")
    groups = []
    seen = set()
    for part in parts:
        if not isinstance(part, (list, tuple)) or not part or any(type(i) is not int for i in part):
            raise ValueError("nonempty explicit label lists required")
        group = set(part)
        if len(group) != len(part) or group & seen or not group <= normalized.keys():
            raise ValueError("duplicate or foreign membership")
        seen |= group
        groups.append(tuple(sorted(group)))
    missing = set(normalized)-seen
    if missing:
        return normalized, groups, sorted(missing)
    return normalized, groups, []


def orient(a, b, c):
    return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])


def hull(labels, points, budget):
    order = sorted(labels, key=lambda i: points[i])
    if len(order) == 1:
        return order
    halves = []
    for sequence in (order, list(reversed(order))):
        half = []
        for ident in sequence:
            budget.spend()
            while len(half) >= 2:
                budget.spend()
                if orient(points[half[-2]], points[half[-1]], points[ident]) > 0:
                    break
                half.pop()
            half.append(ident)
        halves.append(half)
    return halves[0][:-1]+halves[1][:-1]


def edges(ids):
    return list(zip(ids, ids[1:]+ids[:1])) if len(ids) > 1 else []


def combination_for_point(z, ids, points, budget):
    """Find a <=3-vertex convex-combination witness on the computed hull."""
    for i in ids:
        budget.spend()
        if points[i] == z:
            return [(i, F(1))]
    for a, b in edges(ids):
        budget.spend()
        pa, pb = points[a], points[b]
        if orient(pa, pb, z) == 0:
            axis = 0 if pb[0] != pa[0] else 1
            t = (z[axis]-pa[axis])/(pb[axis]-pa[axis])
            if 0 <= t <= 1:
                return [(a, 1-t), (b, t)]
    if len(ids) >= 3:
        for a, b, c in ((ids[0], ids[k], ids[k+1]) for k in range(1, len(ids)-1)):
            budget.spend()
            pa, pb, pc = points[a], points[b], points[c]
            denominator = orient(pa, pb, pc)
            w = orient(pa, z, pc)/denominator
            u = orient(pa, pb, z)/denominator
            if w >= 0 and u >= 0 and w+u <= 1:
                return [(a, 1-w-u), (b, w), (c, u)]
    return None


def pair_certificate(left, right, points, budget):
    axes = [(F(1), F(0)), (F(0), F(1))]
    for a, b in edges(left)+edges(right):
        dx, dy = (points[b][k]-points[a][k] for k in (0, 1))
        axes.extend([(-dy, dx), (dx, dy)])
    for n in axes:
        budget.spend(len(left)+len(right))
        al = [sum(n[k]*points[i][k] for k in (0, 1)) for i in left]
        bl = [sum(n[k]*points[i][k] for k in (0, 1)) for i in right]
        if max(al) < min(bl):
            return {"kind": "strict_separation", "axis": n}
        if max(bl) < min(al):
            return {"kind": "strict_separation", "axis": tuple(-v for v in n)}
    for side, other in ((left, right), (right, left)):
        for ident in side:
            witness = combination_for_point(points[ident], other, points, budget)
            if witness:
                first, second = ([(ident, F(1))], witness) if side is left else (witness, [(ident, F(1))])
                return {"kind": "intersection", "point": points[ident], "left_weights": first, "right_weights": second}
    for a, b in edges(left):
        for c, d in edges(right):
            budget.spend()
            pa, pb, pc, pd = (points[i] for i in (a, b, c, d))
            r = tuple(pb[k]-pa[k] for k in (0, 1))
            s = tuple(pd[k]-pc[k] for k in (0, 1))
            delta = tuple(pc[k]-pa[k] for k in (0, 1))
            cross = lambda x, y: x[0]*y[1]-x[1]*y[0]
            denominator = cross(r, s)
            if denominator:
                t, u = cross(delta, s)/denominator, cross(delta, r)/denominator
                if 0 <= t <= 1 and 0 <= u <= 1:
                    z = tuple(pa[k]+t*r[k] for k in (0, 1))
                    return {"kind": "intersection", "point": z,
                            "left_weights": [(a, 1-t), (b, t)], "right_weights": [(c, 1-u), (d, u)]}
    raise ArithmeticError("no independently checkable intersection or separation witness; unknown")


def verify_independent_cover_2d(points, parts, *, max_work=2000000, seconds=60):
    result = {"status": "UNSUPPORTED_INPUT", "decision_complete": False, "pair_certificates": [],
              "schema": "independent-cover-certificate-v1",
              "contract": "independent-cover-v1", "source_pages": {"pdf":[4,10],"journal":[149,155]},
              "source_scope": "supplied 2D finite image; not a semantic object or 3D structure"}
    if (type(max_work) is not int or not 0 <= max_work <= 2000000
            or type(seconds) not in (int,float) or not 0 <= seconds <= 60 or not math.isfinite(seconds)):
        result.update(error="work/time budget invalid",work_used=0)
        return result
    budget = Budget(max_work, seconds)
    try:
        budget.spend(0)
        points, parts, missing = normalize(points, parts)
        budget.spend(0)
        if missing:
            result.update(status="INCOMPLETE_COVER", decision_complete=True, coverage_complete=False, missing_labels=missing)
            return result
        result["coverage_complete"] = True
        hulls = [hull(part, points, budget) for part in parts]
        result["hull_labels"] = hulls
        for i, j in itertools.combinations(range(len(parts)), 2):
            certificate = pair_certificate(hulls[i], hulls[j], points, budget)
            certificate.update(left_part=i, right_part=j)
            result["pair_certificates"].append(certificate)
            if certificate["kind"] == "intersection":
                budget.spend(0)
                result.update(status="NOT_INDEPENDENT", decision_complete=True, pairwise_independence=False)
                return result
        budget.spend(0)
        result.update(status="VALID_INDEPENDENT_COVER", decision_complete=True, pairwise_independence=True)
    except BudgetExceeded as exc:
        result.update(status="UNKNOWN_BUDGET", error=str(exc))
    except ArithmeticError as exc:
        result.update(status="UNKNOWN_UNCERTIFIED", error=str(exc))
    except (ValueError, OverflowError, TypeError, AttributeError, KeyError, IndexError) as exc:
        result.update(status="UNSUPPORTED_INPUT", error=str(exc))
    finally:
        result["work_used"] = budget.used
    return result


def verify_cover_certificate(points, parts, result):
    """Recompute witnesses using every supplied raw point; fail closed on malformed input."""
    try:
        if (not isinstance(result,dict) or not isinstance(points,dict) or not 1<=len(points)<=4096
                or not isinstance(parts,(list,tuple)) or not 1<=len(parts)<=32):return False
        if (result.get('schema') != 'independent-cover-certificate-v1'
                or result.get('contract') != 'independent-cover-v1'
                or result.get('coverage_complete') is not True
                or result.get('decision_complete') is not True
                or result.get('missing_labels', []) != []):return False
        status=result.get('status')
        if status not in ('VALID_INDEPENDENT_COVER','NOT_INDEPENDENT'):return False
        if result.get('pairwise_independence') is not (status == 'VALID_INDEPENDENT_COVER'):return False
        raw={}
        for label,pair in points.items():
            if type(label) is not int or not isinstance(pair,(list,tuple)) or len(pair)!=2:return False
            raw[label]=tuple(_fraction(v) for v in pair)
        points=raw
        if len(set(points.values()))!=len(points):return False
        sets=[]
        for part in parts:
            if not isinstance(part,(list,tuple)) or not part or any(type(i) is not int for i in part):return False
            group=set(part)
            if len(group)!=len(part):return False
            sets.append(group)
        if set.union(*sets)!=set(points) or sum(map(len,sets))!=len(points):return False
        certificates=result.get('pair_certificates')
        if not isinstance(certificates,list) or len(certificates)>496:return False
        pairs={}
        for cert in certificates:
            if not isinstance(cert,dict):return False
            i, j = cert["left_part"], cert["right_part"]
            if type(i) is not int or type(j) is not int or not 0 <= i < j < len(sets) or (i, j) in pairs:
                return False
            if cert["kind"] == "strict_separation":
                n = cert["axis"]
                if not isinstance(n,(list,tuple)) or len(n) != 2:
                    return False
                n=tuple(_fraction(v,MAX_AXIS_BITS) for v in n)
                if not any(n):
                    return False
                a = [sum(F(n[k])*points[p][k] for k in (0, 1)) for p in sets[i]]
                b = [sum(F(n[k])*points[p][k] for k in (0, 1)) for p in sets[j]]
                if not max(a) < min(b):
                    return False
            elif cert["kind"] == "intersection":
                coordinate=cert["point"]
                if not isinstance(coordinate,(list,tuple)) or len(coordinate)!=2:return False
                z = tuple(_fraction(v,MAX_WITNESS_BITS) for v in coordinate)
                for group, weights in ((sets[i], cert["left_weights"]), (sets[j], cert["right_weights"])):
                    if not isinstance(weights,(list,tuple)) or not 1<=len(weights)<=3:return False
                    parsed=[]
                    for row in weights:
                        if not isinstance(row,(list,tuple)) or len(row)!=2:return False
                        label,weight=row
                        if type(label) is not int or label not in group:return False
                        weight=_fraction(weight,MAX_WITNESS_BITS)
                        if weight<0:return False
                        parsed.append((label,weight))
                    if sum(w for _, w in parsed) != 1:
                        return False
                    if tuple(sum(w*points[label][k] for label, w in parsed) for k in (0, 1)) != z:
                        return False
            else:
                return False
            pairs[i, j] = cert["kind"]
        if result.get("status") == "VALID_INDEPENDENT_COVER":
            return (result.get("decision_complete") is True
                    and set(pairs) == set(itertools.combinations(range(len(sets)), 2))
                    and all(kind == "strict_separation" for kind in pairs.values()))
        if result.get("status") == "NOT_INDEPENDENT":
            return result.get("decision_complete") is True and "intersection" in pairs.values()
    except (KeyError, IndexError, ValueError, TypeError, ArithmeticError, AttributeError):
        return False
    return False
