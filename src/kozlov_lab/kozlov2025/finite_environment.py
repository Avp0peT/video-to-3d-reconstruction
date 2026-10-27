"""S-2025 PDF3--4: bounded complete parts and label-disjoint groups.

S* is described by cardinality, never enumerated. No fits or compatibility.
"""
from dataclasses import dataclass

from .affine_subimage import _rows
from .independent_cover import Budget, BudgetExceeded

VERSION = 'finite-environment-v1'


@dataclass(frozen=True)
class EnvironmentParts:
    points: tuple
    parts: tuple
    star_cardinality: int
    version: str = VERSION


@dataclass(frozen=True)
class SeparateGroupResult:
    status: str
    group: tuple
    reason: str
    overlap: tuple = ()


@dataclass(frozen=True)
class SeparateGroupsCertificate:
    points: tuple
    include_empty: bool
    groups: tuple
    version: str = VERSION


@dataclass(frozen=True)
class SeparateGroupsEnumeration:
    status: str
    complete: bool
    groups: tuple
    certificate: SeparateGroupsCertificate | None
    reason: str
    work_used: int


def _input(points):
    return _rows(points, 7)


def _policy(include_empty):
    if type(include_empty) is not bool:
        raise ValueError('explicit bool empty-group policy required')


def _budget(budget):
    if budget is None:
        return Budget(2000000, 30)
    if not isinstance(budget, Budget):
        raise ValueError('shared Budget required')
    return budget


def enumerate_environment_parts(points):
    """All nonempty parts, in ascending subsetmask of sorted external labels."""
    rows = _input(points)
    labels = tuple(i for i, _ in rows)
    parts = tuple(tuple(labels[k] for k in range(len(labels)) if mask & (1 << k))
                  for mask in range(1, 1 << len(labels)))
    return EnvironmentParts(rows, parts, 1 << len(parts))


def verify_environment_parts(points, certificate):
    """Independent set-of-subsets check using inclusion/exclusion recursion."""
    rows = _input(points)
    if type(certificate) is not EnvironmentParts or certificate.version != VERSION:
        return False
    if certificate.points != rows or type(certificate.parts) is not tuple:
        return False
    expected = {()}
    for label, _ in rows:
        expected |= {part+(label,) for part in tuple(expected)}
    expected.remove(())
    positions = {label: k for k, (label, _) in enumerate(rows)}
    def mask(part):
        return sum(1 << positions[label] for label in part)
    return (_canonical_rows(certificate.points)
            and all(_canonical_part(p) for p in certificate.parts)
            and len(certificate.parts) == len(expected)
            and set(certificate.parts) == expected
            and tuple(sorted(certificate.parts, key=mask)) == certificate.parts
            and type(certificate.star_cardinality) is int
            and certificate.star_cardinality == 2**len(expected))


def _canonical_part(part):
    return (type(part) is tuple and len(part) > 0
            and all(type(i) is int for i in part)
            and tuple(sorted(set(part))) == part)


def _canonical_rows(rows):
    from fractions import Fraction
    return (type(rows) is tuple and all(type(row) is tuple and len(row) == 2
            and type(row[0]) is int and type(row[1]) is tuple and len(row[1]) == 2
            and all(type(x) is Fraction for x in row[1]) for row in rows))


def verify_separate_group(points, group, *, include_empty=False):
    """Given group: disjoint identities, never convex-hull disjointness."""
    rows = _input(points)
    _policy(include_empty)
    if type(group) not in (tuple, list):
        raise ValueError('explicit group of nonempty label parts required')
    labels, seen, parts = {i for i, _ in rows}, set(), []
    for part in group:
        if type(part) not in (tuple, list) or not part or any(type(i) is not int for i in part):
            raise ValueError('nonempty integer label parts required')
        members = set(part)
        if len(members) != len(part) or not members <= labels:
            raise ValueError('duplicate/foreign label inside part')
        if seen & members:
            return SeparateGroupResult('FALSE', (), 'parts share point identities', tuple(sorted(seen & members)))
        seen |= members
        parts.append(tuple(sorted(members)))
    canonical = tuple(sorted(parts))
    if not canonical and not include_empty:
        return SeparateGroupResult('EXCLUDED_BY_POLICY', (), 'empty group excluded explicitly')
    return SeparateGroupResult('TRUE', canonical, 'all member label sets are pairwise disjoint')


def enumerate_separate_groups(points, *, include_empty=False, budget=None):
    """Complete S+ via RGS of environment plus unused sentinel block zero."""
    rows = _input(points)
    _policy(include_empty)
    b = _budget(budget)
    initial, groups = b.used, []
    labels = tuple(i for i, _ in rows)
    def walk(code, maximum):
        b.spend()
        if len(code) == len(labels):
            parts = tuple(sorted(tuple(labels[k] for k, v in enumerate(code) if v == block)
                                 for block in range(1, maximum+1)))
            if parts or include_empty:
                groups.append(parts)
            return
        for block in range(maximum+2):
            walk(code+(block,), max(maximum, block))
    try:
        b.spend(len(rows))
        walk((), 0)
        canonical = tuple(sorted(groups))
        cert = SeparateGroupsCertificate(rows, include_empty, canonical)
        status = verify_separate_groups_certificate(points, cert, budget=b)
        if status != 'TRUE':
            return SeparateGroupsEnumeration('UNKNOWN', False, canonical, None,
                'shared independent verification incomplete or invalid', b.used-initial)
        return SeparateGroupsEnumeration('COMPLETE', True, canonical, cert,
            'all unused/block partitions independently verified', b.used-initial)
    except BudgetExceeded as exc:
        return SeparateGroupsEnumeration('PARTIAL_UNKNOWN', False, tuple(sorted(groups)), None,
                                        str(exc), b.used-initial)


def verify_separate_groups_certificate(points, certificate, *, budget=None):
    """Independent lowest-label subset recursion; never calls RGS enumerator."""
    rows = _input(points)
    b = _budget(budget)
    try:
        b.spend(len(rows))
        c = certificate
        if (type(c) is not SeparateGroupsCertificate or c.version != VERSION
                or type(c.include_empty) is not bool or not _canonical_rows(c.points)
                or c.points != rows or type(c.groups) is not tuple):
            return 'FALSE'
        for group in c.groups:
            b.spend(1+len(group))
            if type(group) is not tuple or any(not _canonical_part(p) for p in group):
                return 'FALSE'
            if tuple(sorted(group)) != group:
                return 'FALSE'
        if tuple(sorted(set(c.groups))) != c.groups:
            return 'FALSE'
        labels = tuple(i for i, _ in rows)
        def recurse(remaining):
            b.spend()
            if not remaining:
                return {()}
            first, rest = remaining[0], remaining[1:]
            # First identity unused, or in one block with any subset of rest.
            output = recurse(rest)
            for mask in range(1 << len(rest)):
                block = (first,)+tuple(x for k, x in enumerate(rest) if mask & (1 << k))
                left = tuple(x for k, x in enumerate(rest) if not mask & (1 << k))
                for suffix in recurse(left):
                    b.spend()
                    output.add(tuple(sorted((block,)+suffix)))
            return output
        expected = recurse(labels)
        if not c.include_empty:
            expected.remove(())
        return 'TRUE' if set(c.groups) == expected else 'FALSE'
    except BudgetExceeded:
        return 'UNKNOWN'
    except (TypeError, ValueError, AttributeError):
        return 'FALSE'
