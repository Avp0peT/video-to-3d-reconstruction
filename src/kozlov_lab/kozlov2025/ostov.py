"""Exact collinear branch of S-2025 m-ostov, PDF4/p149 and PDF10/p155.

Complete supplied images only; general noncollinear centres remain unsupported.
"""
import hashlib
import itertools
import json
import math
import time
from fractions import Fraction

from .cover_enumeration import enumerate_independent_covers_2d
from .independent_cover import normalize, orient, verify_cover_certificate

__all__ = ['enumerate_collinear_m_ostovs_2025', 'verify_collinear_ostov_result']


def _digest(points):
    encoded = [[hex(i), [[str(v.numerator), str(v.denominator)] for v in points[i]]]
               for i in sorted(points)]
    return hashlib.sha256(json.dumps(encoded, separators=(',', ':')).encode()).hexdigest()


def _input(points):
    if isinstance(points, dict) and len(points) > 7:
        raise ValueError('size exceeds complete 1..7 point domain; no sampling')
    labels = sorted(points) if isinstance(points, dict) and all(type(i) is int for i in points) else []
    raw, _, _ = normalize(points, [labels])
    if len(labels) > 1:
        if any(orient(raw[labels[0]], raw[labels[1]], raw[i]) != 0 for i in labels):
            return raw, None
    axis = 0 if len(labels) == 1 or len({p[0] for p in raw.values()}) > 1 else 1
    return raw, sorted(labels, key=lambda i: raw[i][axis])


def _geometry(points, parts):
    rows = []
    for part in parts:
        # Lexicographic extrema are segment extrema for every collinear direction.
        u, v = min(part, key=lambda i: points[i]), max(part, key=lambda i: points[i])
        centre = tuple((points[u][k]+points[v][k])/2 for k in (0, 1))
        square = sum((points[u][k]-points[v][k])**2 for k in (0, 1))
        rows.append({'labels': list(part), 'endpoint_labels': [u, v], 'center': centre,
                     'piece_size': {'squared': square, 'nonnegative_root': True}})
    return rows, {'squared': max(r['piece_size']['squared'] for r in rows), 'nonnegative_root': True}


def enumerate_collinear_m_ostovs_2025(points, *, max_candidates=877,
                                     total_max_work=2000000, seconds=30):
    """Enumerate every cover witness; do not deduplicate equal centre sets.

    Exact Fraction coordinates describe stored binary64 values, not measurements.
    Work is cooperative solver/geometry work, not Python instructions.
    """
    start = time.monotonic()
    result = {'schema': 'collinear-m-ostov-2025-v1', 'contract': 'ostov-v1',
              'status': 'UNSUPPORTED_INPUT', 'complete': False, 'ostovs': [],
              'source_pages': {'pdf': [4, 10], 'journal': [149, 155]},
              'scope': 'all independent covers of complete supplied collinear 1..7 point image',
              'work_used': 0}
    def finish(status, reason):
        result.update(status=status, reason=reason, elapsed_seconds=time.monotonic()-start)
        return result
    if (type(max_candidates) is not int or not 0 <= max_candidates <= 877
            or type(total_max_work) is not int or not 0 <= total_max_work <= 2000000
            or type(seconds) not in (int, float) or not math.isfinite(seconds) or not 0 <= seconds <= 30):
        return finish('UNSUPPORTED_INPUT', 'invalid bounded budget')
    deadline = start+seconds
    result['budgets'] = dict(max_candidates=max_candidates, total_max_work=total_max_work, seconds=seconds)
    if time.monotonic() >= deadline:
        return finish('PARTIAL_UNKNOWN', 'time budget exhausted before validation')
    try:
        raw, order = _input(points)
    except (ValueError, TypeError, ArithmeticError, AttributeError, KeyError, IndexError) as exc:
        return finish('UNSUPPORTED_INPUT', str(exc))
    result.update(source_coordinate_sha256=_digest(raw), declared_labels=sorted(raw), points=len(raw))
    if order is None:
        return finish('UNSUPPORTED_DOMAIN', 'general noncollinear exact minimum-ellipse centres not implemented')
    prework = len(raw)
    result['work_used'] = prework
    if prework > total_max_work or time.monotonic() >= deadline:
        return finish('PARTIAL_UNKNOWN', 'validation budget exhausted')
    enumeration = enumerate_independent_covers_2d(raw, max_candidates=max_candidates,
        total_max_work=total_max_work-prework, seconds=max(0, deadline-time.monotonic()))
    result['cover_enumeration'] = enumeration
    result['work_used'] += enumeration['work_used']
    for witness in enumeration['independent_covers']:
        # Charge construction and raw-point certificate verification conservatively.
        m = len(witness['parts'])
        cost = len(raw)*(1+m+m*(m-1)//2)+m
        if result['work_used']+cost > total_max_work or time.monotonic() >= deadline:
            return finish('PARTIAL_UNKNOWN', 'shared geometry budget exhausted')
        result['work_used'] += cost
        rows, size = _geometry(raw, witness['parts'])
        result['ostovs'].append({'m': len(rows), 'candidate_index': witness['candidate_index'],
            'parts': witness['parts'], 'pieces': rows, 'centers': [r['center'] for r in rows],
            'cover_size': size, 'cover_certificate': witness['certificate']})
    if not enumeration['complete'] or time.monotonic() >= deadline:
        return finish('PARTIAL_UNKNOWN', 'cover enumeration incomplete or shared time exhausted')
    n = len(raw)
    counts = {m: sum(row['m'] == m for row in result['ostovs']) for m in range(1, n+1)}
    result['counts_by_m'] = counts
    if len(result['ostovs']) != 2**(n-1) or any(counts[m] != math.comb(n-1, m-1) for m in counts):
        return finish('UNKNOWN_UNCERTIFIED', 'independent interval-cover count disagrees')
    oracle_work = len(raw)*2**(len(raw)-1)
    if result['work_used']+oracle_work > total_max_work:
        return finish('PARTIAL_UNKNOWN', 'shared final-oracle budget exhausted')
    result['work_used'] += oracle_work
    result['complete'] = True
    result['status'] = 'COMPLETE'
    if not verify_collinear_ostov_result(raw, result):
        result['complete'] = False
        return finish('UNKNOWN_UNCERTIFIED', 'independent result verification failed')
    if time.monotonic() >= deadline:
        result['complete'] = False
        return finish('PARTIAL_UNKNOWN', 'shared time exhausted during final verification')
    return finish('COMPLETE', 'all cover witnesses and exact segment centres verified')


def verify_collinear_ostov_result(points, result):
    """Independent raw-point and cut-gap oracle; invokes no enumeration/geometry solver."""
    try:
        raw, order = _input(points)
        if (order is None or result.get('schema') != 'collinear-m-ostov-2025-v1'
                or result.get('contract') != 'ostov-v1'
                or result.get('source_coordinate_sha256') != _digest(raw)
                or result.get('declared_labels') != sorted(raw) or result.get('points') != len(raw)
                or type(result.get('complete')) is not bool):
            return False
        if result['complete'] and result.get('status') != 'COMPLETE':
            return False
        seen = set()
        counts = {m: 0 for m in range(1, len(raw)+1)}
        for row in result['ostovs']:
            parts = row['parts']
            if (row['cover_certificate'].get('status') != 'VALID_INDEPENDENT_COVER'
                    or not verify_cover_certificate(raw, parts, row['cover_certificate'])):
                return False
            signature = frozenset(frozenset(p) for p in parts)
            if signature in seen or row['m'] != len(parts) or len(row['pieces']) != len(parts):
                return False
            seen.add(signature)
            counts[len(parts)] += 1
            centers, squares = [], []
            for part, piece in zip(parts, row['pieces']):
                u, v = piece['endpoint_labels']
                if piece['labels'] != part or u not in part or v not in part:
                    return False
                pu, pv = raw[u], raw[v]
                axis = 0 if pu[0] != pv[0] else 1
                if any(orient(pu, pv, raw[i]) != 0 or not min(pu[axis], pv[axis]) <= raw[i][axis] <= max(pu[axis], pv[axis]) for i in part):
                    return False
                # Coincident endpoint identity cannot cover multiple distinct points.
                if u == v and len(part) != 1:
                    return False
                center = tuple((pu[k]+pv[k])/2 for k in (0, 1))
                square = sum((pu[k]-pv[k])**2 for k in (0, 1))
                if tuple(piece['center']) != center or piece['piece_size'] != {'squared': square, 'nonnegative_root': True}:
                    return False
                centers.append(center)
                squares.append(square)
            if (list(map(tuple, row['centers'])) != centers or len(set(centers)) != len(parts)
                    or row['cover_size'] != {'squared': max(squares), 'nonnegative_root': True}):
                return False
        if result['complete']:
            expected = set()
            for cuts in itertools.product((False, True), repeat=len(order)-1):
                groups = [[order[0]]]
                for ident, cut in zip(order[1:], cuts):
                    if cut:
                        groups.append([])
                    groups[-1].append(ident)
                expected.add(frozenset(frozenset(g) for g in groups))
            if seen != expected or result.get('counts_by_m') != counts:
                return False
            enumeration = result['cover_enumeration']
            if (enumeration.get('complete') is not True
                    or enumeration.get('source_coordinate_sha256') != _digest(raw)):
                return False
            bindings = {(w['candidate_index'], frozenset(frozenset(p) for p in w['parts']))
                        for w in enumeration['independent_covers']}
            if len(bindings) != len(seen) or any((r['candidate_index'], frozenset(frozenset(p) for p in r['parts'])) not in bindings for r in result['ostovs']):
                return False
        return True
    except (ValueError, TypeError, ArithmeticError, AttributeError, KeyError, IndexError):
        return False
