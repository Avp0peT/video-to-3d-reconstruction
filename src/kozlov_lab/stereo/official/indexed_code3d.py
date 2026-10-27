"""Complete, bounded rational 3D indexed code from S-1996, pp.325-326.

This module compares the source's indexed volume ratios. It does not fit affine
coordinates or implement AlgT'/AlgSo. See contracts_indexed_code3d_v1.md.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
import itertools
import math
import numbers
from types import MappingProxyType
from typing import Any

MAX_POINTS = 12
MAX_ENTRIES = 250_000
MAX_COORDINATE_BITS = 2048
MAX_SEARCH_POINTS = 8
MAX_PERMUTATIONS = 100_000


class CodeInputError(ValueError):
    """Input cannot represent a labelled finite rational point set."""


class CodeBudgetError(ValueError):
    """A complete construction or requested search exceeds a count bound."""


def _coordinate(value: Any) -> Fraction:
    if isinstance(value, bool):
        raise CodeInputError('boolean coordinate')
    if isinstance(value, numbers.Integral):
        answer = Fraction(int(value))
    elif isinstance(value, Fraction):
        answer = value
    elif isinstance(value, numbers.Real):
        if getattr(getattr(value, 'dtype', None), 'itemsize', 0) > 8:
            raise CodeInputError('floating precision exceeds binary64')
        converted = float(value)
        if not math.isfinite(converted):
            raise CodeInputError('nonfinite coordinate')
        answer = Fraction.from_float(converted)
    else:
        raise CodeInputError('coordinates require int, Fraction or finite binary64 float')
    if max(abs(answer.numerator).bit_length(), answer.denominator.bit_length()) > MAX_COORDINATE_BITS:
        raise CodeBudgetError('coordinate numerator/denominator exceeds 2048 bits')
    return answer


def _input(points, labels):
    try:
        n = len(points)
        if not 1 <= n <= MAX_POINTS:
            raise CodeBudgetError('expected 1..12 points')
        if isinstance(points, (str, bytes)):
            raise CodeInputError('points must be rows')
        rows = []
        for i in range(n):
            row = points[i]
            if isinstance(row, (str, bytes)) or len(row) != 3:
                raise CodeInputError('each point needs three coordinates')
            rows.append(tuple(_coordinate(row[j]) for j in range(3)))
        if len(set(rows)) != n:
            raise CodeInputError('duplicate coordinates would change the labelled set')
        if labels is None:
            ids = tuple(range(n))
        else:
            if isinstance(labels, (str, bytes)) or len(labels) != n:
                raise CodeInputError('labels must match the number of points')
            ids = []
            for value in labels:
                if isinstance(value, bool) or not isinstance(value, numbers.Integral):
                    raise CodeInputError('labels must be signed 64-bit integers')
                value = int(value)
                if not -(2**63) <= value < 2**63:
                    raise CodeInputError('label outside signed 64-bit range')
                ids.append(value)
            ids = tuple(ids)
            if len(set(ids)) != n:
                raise CodeInputError('duplicate label')
        return tuple(rows), ids
    except (TypeError, IndexError, KeyError) as exc:
        raise CodeInputError('invalid point/label shape') from exc


def _sub(a, b):
    return tuple(x-y for x, y in zip(a, b))


def _cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def _dot(a, b):
    return sum(x*y for x, y in zip(a, b))


def _six_volume(a, b, c, d):
    return abs(_dot(_sub(b, a), _cross(_sub(c, a), _sub(d, a))))


def _domain(points, labels, full_rank):
    if not full_rank:
        return {'theorem3_applies': False, 'reason': 'contained_in_one_plane',
                'parallel_planes_witness': None, 'complete': True, 'normals_checked': 0}
    # For a rank-three set in two parallel planes, the union of within-plane
    # differences has rank exactly two. Its two independent vectors occur in
    # this exhaustive pair-vector list. See the independent contract proof.
    directions = [_sub(points[j], points[i]) for i, j in itertools.combinations(range(len(points)), 2)]
    seen = set()
    for first, second in itertools.combinations(directions, 2):
        normal = _cross(first, second)
        pivot = next((v for v in normal if v), None)
        if pivot is None:
            continue
        normal = tuple(v/pivot for v in normal)
        if normal in seen:
            continue
        seen.add(normal)
        projection = tuple(_dot(normal, p) for p in points)
        offsets = tuple(sorted(set(projection)))
        if len(offsets) <= 2:
            return {'theorem3_applies': False, 'reason': 'contained_in_two_parallel_planes',
                    'parallel_planes_witness': {'normal': normal, 'offsets': offsets,
                        'groups': tuple(tuple(label for label, v in zip(labels, projection) if v == offset)
                                        for offset in offsets)},
                    'complete': True, 'normals_checked': len(seen)}
    return {'theorem3_applies': True, 'reason': 'no_one_or_two_parallel_plane_cover',
            'parallel_planes_witness': None, 'complete': True, 'normals_checked': len(seen)}


def _json(value):
    if isinstance(value, Fraction):
        return {'numerator_hex': hex(value.numerator), 'denominator_hex': hex(value.denominator)}
    if isinstance(value, Mapping):
        return {k: _json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json(v) for v in value]
    return value


def _freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(v) for v in value)
    return value


def _domain_copy(value):
    if isinstance(value, Mapping):
        return {k: _domain_copy(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(_domain_copy(v) for v in value)
    return value


@dataclass(frozen=True)
class RatioEntry3D:
    numerator: tuple[int, int, int, int]
    denominator: tuple[int, int, int, int]
    value: Fraction


@dataclass(frozen=True, init=False)
class IndexedBodyCode:
    labels: tuple[int, ...]
    points: tuple[tuple[Fraction, Fraction, Fraction], ...]
    quadruple_volumes: tuple[tuple[tuple[int, int, int, int], Fraction], ...]
    entries: tuple[RatioEntry3D, ...]
    status: str
    complete: bool
    _theorem_domain: Mapping

    def __init__(self, *args, **kwargs):
        # In particular dataclasses.replace cannot remove indexed entries and
        # leave a false DEFINED/complete certificate attached to the object.
        raise CodeInputError('construct complete codes with indexed_body_code(), not directly')

    @property
    def theorem_domain(self):
        return _domain_copy(self._theorem_domain)

    def to_dict(self):
        return _json({'contract': 'indexed-body-code-v1', 'number_codec': 'rational-hex-v1',
            'source_id': 'S-1996', 'source_pages': [325, 326],
            'arithmetic': 'exact_rational_of_supplied_values',
            'status': self.status, 'complete': self.complete,
            'labels': self.labels, 'points': self.points,
            'quadruple_volumes': [{'quadruple': q, 'six_volume': v} for q, v in self.quadruple_volumes],
            'entries': [{'numerator': e.numerator, 'denominator': e.denominator, 'value': e.value}
                        for e in self.entries],
            'counts': {'points': len(self.labels), 'quadruples': len(self.quadruple_volumes),
                       'nonzero_denominators': sum(v != 0 for _, v in self.quadruple_volumes),
                       'entries': len(self.entries)},
            'theorem_domain': self.theorem_domain, 'affine_equivalence': None})


def _constructed(ids, coordinates, volumes, entries, status, complete, domain):
    result = object.__new__(IndexedBodyCode)
    for key, value in dict(labels=ids, points=coordinates, quadruple_volumes=volumes,
                           entries=entries, status=status, complete=complete,
                           _theorem_domain=_freeze(domain)).items():
        object.__setattr__(result, key, value)
    return result


def indexed_body_code(points, labels=None, *, max_entries=10_000):
    """Build every labelled ratio, or reject before materializing excess rows."""
    if type(max_entries) is not int or not 1 <= max_entries <= MAX_ENTRIES:
        raise CodeInputError('max_entries must be an integer in [1,250000]')
    coordinates, ids = _input(points, labels)
    lookup = dict(zip(ids, coordinates))
    volumes = tuple((quad, _six_volume(*(lookup[i] for i in quad)))
                    for quad in itertools.combinations(sorted(ids), 4))
    denominators = tuple((quad, volume) for quad, volume in volumes if volume)
    required = len(denominators)*(len(volumes)-1)
    if required > max_entries:
        raise CodeBudgetError(f'complete code requires {required} entries; budget is {max_entries}')
    domain = _domain(coordinates, ids, bool(denominators))
    if not denominators:
        return _constructed(ids, coordinates, volumes, (), 'UNDEFINED_PLANAR', False, domain)
    entries = tuple(RatioEntry3D(p, q, vp/vq) for p, vp in volumes
                    for q, vq in denominators if p != q)
    assert len(entries) == required
    return _constructed(ids, coordinates, volumes, entries, 'DEFINED', True, domain)


def _codes(left, right):
    if not isinstance(left, IndexedBodyCode) or not isinstance(right, IndexedBodyCode):
        raise CodeInputError('comparison requires constructed IndexedBodyCode objects')


def _result(equivalent, reason, *, left, right, mapping=None, checked=0, exhausted=False, counterexample=None,
            comparison_scope='supplied_bijection'):
    return {'status': {True: 'EQUIVALENT', False: 'NOT_EQUIVALENT', None: 'UNKNOWN'}[equivalent],
            'equivalent': equivalent, 'reason': reason, 'decision_complete': equivalent is not None,
            'comparison_scope': comparison_scope,
            'search_exhausted': exhausted, 'permutations_checked': checked,
            'mapping': None if mapping is None else [[i, mapping[i]] for i in left.labels],
            'permutation': None if mapping is None else [right.labels.index(mapping[i]) for i in left.labels],
            'counterexample': _json(counterexample), 'affine_equivalence': None,
            'theorem3_domain': {'left': left.theorem_domain['theorem3_applies'],
                               'right': right.theorem_domain['theorem3_applies']}}


def code_equivalence_for_mapping(left, right, mapping):
    """Independently verify every materialized ratio under one label bijection."""
    _codes(left, right)
    if not isinstance(mapping, dict) or any(type(k) is not int or type(v) is not int for k, v in mapping.items()):
        raise CodeInputError('mapping must be an integer-label dictionary')
    if set(mapping) != set(left.labels) or len(mapping) != len(right.labels) or set(mapping.values()) != set(right.labels):
        raise CodeInputError('mapping must be a bijection of the complete label sets')
    if left.status != 'DEFINED' or right.status != 'DEFINED':
        return _result(None, 'code_undefined_planar', left=left, right=right)
    target = {(e.numerator, e.denominator): e.value for e in right.entries}
    if len(left.entries) != len(target):
        return _result(False, 'defined_index_domain_size_mismatch', left=left, right=right, checked=1)
    for e in left.entries:
        key = (tuple(sorted(mapping[i] for i in e.numerator)), tuple(sorted(mapping[i] for i in e.denominator)))
        if key not in target or target[key] != e.value:
            return _result(False, 'indexed_ratio_mismatch', left=left, right=right, checked=1,
                           counterexample={'numerator': e.numerator, 'denominator': e.denominator,
                               'left': e.value, 'right': target.get(key)})
    return _result(True, 'all_materialized_indexed_ratios_match', left=left, right=right, mapping=mapping, checked=1)


def _factorized_match(volumes, target, mapping):
    # A common nonzero reference checks all indexed ratios by exact products.
    # Zero numerators and the complete denominator domain are retained.
    reference = next((q, v) for q, v in volumes if v)
    target_reference = target[tuple(sorted(mapping[i] for i in reference[0]))]
    if not target_reference:
        return False
    for q, value in volumes:
        other = target[tuple(sorted(mapping[i] for i in q))]
        if value*target_reference != other*reference[1]:
            return False
    return True


def find_code_equivalence(left, right, *, max_permutations=10_000):
    """Bounded exhaustive bijections; unfinished negative search is UNKNOWN."""
    _codes(left, right)
    if type(max_permutations) is not int or not 1 <= max_permutations <= MAX_PERMUTATIONS:
        raise CodeInputError('max_permutations must be an integer in [1,100000]')
    if max(len(left.labels), len(right.labels)) > MAX_SEARCH_POINTS:
        raise CodeBudgetError('exhaustive search is limited to 8 points; supply a mapping instead')
    if left.status != 'DEFINED' or right.status != 'DEFINED':
        return _result(None, 'code_undefined_planar', left=left, right=right,
                       comparison_scope='existential_bijection')
    if len(left.labels) != len(right.labels):
        return _result(False, 'label_cardinality_mismatch', left=left, right=right,
                       comparison_scope='existential_bijection')
    if len(left.entries) != len(right.entries):
        return _result(False, 'defined_index_domain_size_mismatch', left=left, right=right,
                       comparison_scope='existential_bijection')
    target = dict(right.quadruple_volumes)
    checked = 0
    for candidate in itertools.permutations(right.labels):
        if checked == max_permutations:
            return _result(None, 'permutation_budget_exhausted', left=left, right=right, checked=checked,
                           comparison_scope='existential_bijection')
        mapping = dict(zip(left.labels, candidate))
        checked += 1
        if _factorized_match(left.quadruple_volumes, target, mapping):
            verified = code_equivalence_for_mapping(left, right, mapping)
            if verified['equivalent'] is not True:
                raise ArithmeticError('factorized comparison disagrees with full indexed records')
            verified.update(permutations_checked=checked,
                            comparison_scope='existential_bijection',
                            search_exhausted=checked == math.factorial(len(left.labels)),
                            reason='exact_indexed_bijection_found')
            return verified
    return _result(False, 'all_bijections_exhausted', left=left, right=right, checked=checked, exhausted=True,
                   comparison_scope='existential_bijection')
