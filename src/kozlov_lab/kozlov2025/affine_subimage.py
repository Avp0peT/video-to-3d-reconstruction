"""S-2025 PDF2 / journal147: exact affine equivalence to a nonempty subset.

Complete rational supplied images only. No futlyar containment or strict fits.
"""
from dataclasses import dataclass
from fractions import Fraction as F
from itertools import permutations

from .independent_cover import Budget, BudgetExceeded

VERSION = 'affine-subimage-v1'
MAX_BITS = 256
MAX_WITNESS_BITS = 8192


@dataclass(frozen=True)
class AffineSubimageCertificate:
    source: tuple
    target: tuple
    matrix: tuple
    translation: tuple
    mapping: tuple
    version: str = VERSION


@dataclass(frozen=True)
class AffineSubimageResult:
    status: str
    reason: str
    certificate: AffineSubimageCertificate | None = None
    candidates_tried: int = 0
    complete: bool = False
    work_used: int = 0


def _q(x, bits=MAX_BITS):
    if type(x) not in (int, F):
        raise ValueError('int/Fraction required; bool/float rejected')
    x = F(x)
    if max(x.numerator.bit_length(), x.denominator.bit_length()) > bits:
        raise ValueError('rational bit bound exceeded')
    return x


def _rows(points, maximum):
    if type(points) is not dict or not 1 <= len(points) <= maximum:
        raise ValueError('complete nonempty labelled dict exceeds size profile')
    rows = []
    for label, point in points.items():
        if type(label) is not int or label.bit_length() > 64:
            raise ValueError('64-bit integer labels required')
        if type(point) not in (tuple, list) or len(point) != 2:
            raise ValueError('exact two-dimensional coordinates required')
        rows.append((label, tuple(_q(x) for x in point)))
    if len({p for _, p in rows}) != len(rows):
        raise ValueError('duplicate coordinates rejected; no merging')
    return tuple(sorted(rows))


def _budget(budget):
    if budget is None:
        return Budget(work=2000000, seconds=30)
    if not isinstance(budget, Budget):
        raise ValueError('shared Budget required')
    return budget


def _sub(a, b):
    return a[0]-b[0], a[1]-b[1]


def _det(u, v):
    return u[0]*v[1]-u[1]*v[0]


def _linear(matrix, p):
    return tuple(row[0]*p[0]+row[1]*p[1] for row in matrix)


def _map(matrix, translation, p):
    return tuple(x+y for x, y in zip(_linear(matrix, p), translation))


def _solve(u, v, du, dv):
    d = _det(u, v)
    return (( (du[0]*v[1]-dv[0]*u[1])/d,
              (dv[0]*u[0]-du[0]*v[0])/d),
            ( (du[1]*v[1]-dv[1]*u[1])/d,
              (dv[1]*u[0]-du[1]*v[0])/d))


def find_affine_subimage_2d(source, target, *, budget=None):
    """Decide whether some invertible affine T carries all source into target.

    FALSE is exhaustive; a budget stop is UNKNOWN. A TRUE witness decides the
    existential predicate even though further candidate maps are not searched.
    """
    sr, tr = _rows(source, 12), _rows(target, 32)
    b = _budget(budget)
    initial, tried = b.used, 0
    def result(status, reason, certificate=None, complete=False):
        return AffineSubimageResult(status, reason, certificate, tried,
                                    complete, b.used-initial)
    try:
        b.spend(len(sr)+len(tr))
        if len(sr) > len(tr):
            return result('FALSE', 'cardinality forbids an injection', complete=True)
        p0 = sr[0][1]
        rank, basis = 0, [p0]
        if len(sr) > 1:
            basis.append(sr[1][1])
            u = _sub(basis[1], p0)
            rank = 1
            for _, p in sr[2:]:
                b.spend()
                if _det(u, _sub(p, p0)):
                    rank = 2
                    basis.append(p)
                    break
        lookup = {p: label for label, p in tr}
        for anchors in permutations(tr, rank+1):
            b.spend(1+rank*16)
            tried += 1
            a0 = anchors[0][1]
            if rank == 0:
                matrix = ((F(1), F(0)), (F(0), F(1)))
            elif rank == 1:
                du = _sub(anchors[1][1], a0)
                matrix = _solve(u, (-u[1], u[0]), du, (-du[1], du[0]))
            else:
                v = _sub(basis[2], p0)
                du, dv = _sub(anchors[1][1], a0), _sub(anchors[2][1], a0)
                if not _det(du, dv):
                    continue
                matrix = _solve(u, v, du, dv)
            translation = _sub(a0, _linear(matrix, p0))
            if any(max(x.numerator.bit_length(), x.denominator.bit_length())
                   > MAX_WITNESS_BITS for row in matrix for x in row):
                raise BudgetExceeded('witness bit bound exceeded')
            mapping = []
            for label, p in sr:
                b.spend(8)
                image = _map(matrix, translation, p)
                if image not in lookup:
                    break
                mapping.append((label, lookup[image]))
            if len(mapping) == len(sr):
                cert = AffineSubimageCertificate(sr, tr, matrix, translation, tuple(mapping))
                # Recompute with independent verification, sharing the same budget.
                status = verify_affine_subimage_certificate(source, target, cert, budget=b)
                if status == 'TRUE':
                    return result('TRUE', 'complete supplied source embeds in target', cert, True)
                if status == 'UNKNOWN':
                    return result('UNKNOWN', 'shared certificate verification budget exhausted')
                return result('UNKNOWN', 'constructed certificate failed independent verification')
        return result('FALSE', 'all affine-basis anchors exhausted', complete=True)
    except BudgetExceeded as exc:
        return result('UNKNOWN', str(exc))


def verify_affine_subimage_certificate(source, target, certificate, *, budget=None):
    """Independent determinant/full-input injection check; no search or solve."""
    sr, tr = _rows(source, 12), _rows(target, 32)
    b = _budget(budget)
    try:
        b.spend(len(sr)+len(tr))
        c = certificate
        if type(c) is not AffineSubimageCertificate or c.version != VERSION:
            return 'FALSE'
        # Strict canonical types avoid bool/int equality hiding forged labels.
        if (type(c.source) is not tuple or type(c.target) is not tuple
                or c.source != sr or c.target != tr):
            return 'FALSE'
        for rows in (c.source, c.target):
            if any(type(row) is not tuple or len(row) != 2 or type(row[0]) is not int
                   or type(row[1]) is not tuple or len(row[1]) != 2
                   or any(type(x) is not F for x in row[1]) for row in rows):
                return 'FALSE'
        if (type(c.matrix) is not tuple or len(c.matrix) != 2
                or any(type(row) is not tuple or len(row) != 2 for row in c.matrix)
                or type(c.translation) is not tuple or len(c.translation) != 2):
            return 'FALSE'
        for x in (*c.matrix[0], *c.matrix[1], *c.translation):
            if type(x) is not F:
                return 'FALSE'
            _q(x, MAX_WITNESS_BITS)
        b.spend(8)
        m = c.matrix
        if m[0][0]*m[1][1]-m[0][1]*m[1][0] == 0:
            return 'FALSE'
        if type(c.mapping) is not tuple or len(c.mapping) != len(sr):
            return 'FALSE'
        if any(type(row) is not tuple or len(row) != 2
               or any(type(x) is not int for x in row) for row in c.mapping):
            return 'FALSE'
        if tuple(i for i, _ in c.mapping) != tuple(i for i, _ in sr):
            return 'FALSE'
        labels = [j for _, j in c.mapping]
        targets = dict(tr)
        if len(set(labels)) != len(labels) or not set(labels) <= targets.keys():
            return 'FALSE'
        for (i, p), (_, j) in zip(sr, c.mapping):
            b.spend(8)
            # Deliberately inline arithmetic rather than calling search helpers.
            image = (m[0][0]*p[0]+m[0][1]*p[1]+c.translation[0],
                     m[1][0]*p[0]+m[1][1]*p[1]+c.translation[1])
            if image != targets[j]:
                return 'FALSE'
        return 'TRUE'
    except BudgetExceeded:
        return 'UNKNOWN'
    except (ValueError, TypeError, AttributeError, IndexError):
        return 'FALSE'
