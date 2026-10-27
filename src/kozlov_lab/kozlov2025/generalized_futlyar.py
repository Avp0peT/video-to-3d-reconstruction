"""Exact supplied-placement predicate, S-2025 PDF13 / journal158.

No affine-placement search, primary radius, optimal fits or ellipse computation.
See docs/contracts_generalized_futlyar_v1.md for the bounded rational domain.
"""
from dataclasses import dataclass
from fractions import Fraction

MAX_POINTS = 4096
MAX_CENTERS = 32
MAX_BITS = 256
MAX_WORK = 262144


@dataclass(frozen=True)
class FutlyarCertificate:
    points: tuple
    centers: tuple
    radius: Fraction
    blocks: tuple
    distances_squared: tuple
    center_distances_squared: tuple
    version: str = "generalized-futlyar-v1"


@dataclass(frozen=True)
class FutlyarResult:
    status: str
    reason: str
    certificate: FutlyarCertificate | None = None
    witness: tuple = ()
    work_required: int = 0


def _rational(value):
    if type(value) not in (int, Fraction):
        raise ValueError("exact int or Fraction required; bool/float/string rejected")
    value = Fraction(value)
    if max(value.numerator.bit_length(), value.denominator.bit_length()) > MAX_BITS:
        raise ValueError("rational coordinate/radius bit limit exceeded")
    return value


def _rows(values, maximum):
    if type(values) is not dict or not 1 <= len(values) <= maximum:
        raise ValueError("nonempty labelled dict exceeds supported size")
    rows = []
    for label, point in values.items():
        if type(label) is not int or label.bit_length() > 64:
            raise ValueError("int labels of at most 64 bits required")
        if type(point) not in (tuple, list) or len(point) != 2:
            raise ValueError("two rational coordinates required")
        rows.append((label, tuple(_rational(x) for x in point)))
    return tuple(sorted(rows))


def _input(points, centers, radius, max_work):
    if type(max_work) is not int or not 0 <= max_work <= MAX_WORK:
        raise ValueError("work budget must be an integer in 0..MAX_WORK")
    points = _rows(points, MAX_POINTS)
    centers = _rows(centers, MAX_CENTERS)
    radius = _rational(radius)
    if radius <= 0:
        raise ValueError("supported radius domain is positive rationals")
    if len({p for _, p in points}) != len(points):
        raise ValueError("points must be a complete finite set; duplicate coordinates rejected")
    if len(centers) > len(points):
        raise ValueError("S-2025 requires number of centers <= number of points")
    return points, centers, radius


def _d2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def verify_generalized_futlyar_2d(points, centers, radius, *, max_work=MAX_WORK):
    """Validate all supplied points. Invalid-domain input raises ValueError.

    TRUE has a certificate; FALSE has an exact failing-condition witness;
    UNKNOWN means insufficient work; UNSUPPORTED preserves unresolved semantics.
    Work is the number of rational squared-distance evaluations, preflighted.
    """
    pts, ctrs, r = _input(points, centers, radius, max_work)
    if len(ctrs) == 1:
        return FutlyarResult("UNSUPPORTED", "single-center minimum distance unresolved")
    if len({p for _, p in ctrs}) != len(ctrs):
        return FutlyarResult("UNSUPPORTED", "coincident centers unresolved")
    work = len(pts) * len(ctrs) + len(ctrs) * (len(ctrs) - 1) // 2
    if work > max_work:
        return FutlyarResult("UNKNOWN", "work budget insufficient", work_required=work)
    center_d = tuple((a, b, _d2(pa, pb))
                     for i, (a, pa) in enumerate(ctrs) for b, pb in ctrs[i + 1:])
    for a, b, distance in center_d:
        if r * r > distance:
            return FutlyarResult("FALSE", "radius exceeds center separation",
                                 witness=(a, b, distance), work_required=work)
    groups = {label: [] for label, _ in ctrs}
    distances = []
    for label, point in pts:
        row = tuple((center, _d2(point, pos)) for center, pos in ctrs)
        distances.append((label, row))
        minimum = min(d for _, d in row)
        closest = tuple(c for c, d in row if d == minimum)
        if len(closest) != 1:
            return FutlyarResult("FALSE", "nearest center is not unique",
                                 witness=(label, closest, minimum), work_required=work)
        if minimum >= r * r:
            return FutlyarResult("FALSE", "point distance is not strictly below radius",
                                 witness=(label, closest[0], minimum), work_required=work)
        groups[closest[0]].append(label)
    for center, group in groups.items():
        if not group:
            return FutlyarResult("FALSE", "empty block", witness=(center,), work_required=work)
    blocks = tuple((center, tuple(group)) for center, group in groups.items())
    cert = FutlyarCertificate(pts, ctrs, r, blocks, tuple(distances), center_d)
    return FutlyarResult("TRUE", "all supplied-placement conditions certified", cert,
                         work_required=work)


def verify_generalized_futlyar_certificate(points, centers, radius, certificate,
                                         *, max_work=MAX_WORK):
    """Independently check a TRUE certificate against the complete caller input.

    Does not call the producer or trust cached distances, membership or metadata.
    Returns TRUE/FALSE/UNKNOWN/UNSUPPORTED; malformed certificates are FALSE.
    Invalid caller input still raises ValueError.
    """
    pts, ctrs, r = _input(points, centers, radius, max_work)
    if len(ctrs) == 1 or len({p for _, p in ctrs}) != len(ctrs):
        return "UNSUPPORTED"
    work = len(pts) * len(ctrs) + len(ctrs) * (len(ctrs) - 1) // 2
    if work > max_work:
        return "UNKNOWN"
    if type(certificate) is not FutlyarCertificate:
        return "FALSE"
    # Type equality matters: Python otherwise lets bool/float impersonate exact
    # rational payloads. All nested certificate data must have canonical types.
    def same(actual, expected):
        if type(actual) is not type(expected):
            return False
        if type(expected) is tuple:
            return len(actual) == len(expected) and all(
                same(a, b) for a, b in zip(actual, expected))
        return actual == expected
    try:
        if (not same(certificate.points, pts)
                or not same(certificate.centers, ctrs)
                or not same(certificate.radius, r)
                or not same(certificate.version, "generalized-futlyar-v1")):
            return "FALSE"
        pairs = []
        for i, (a, pa) in enumerate(ctrs):
            for b, pb in ctrs[i + 1:]:
                distance = sum((x - y) * (x - y) for x, y in zip(pa, pb))
                if distance < r * r:
                    return "FALSE"
                pairs.append((a, b, distance))
        expected_distances = []
        expected_blocks = {c: [] for c, _ in ctrs}
        for label, point in pts:
            row = tuple((c, sum((x-y)*(x-y) for x, y in zip(point, pos)))
                        for c, pos in ctrs)
            winner = [(c, d) for c, d in row if all(d < other for k, other in row if k != c)]
            if len(winner) != 1 or winner[0][1] >= r * r:
                return "FALSE"
            expected_blocks[winner[0][0]].append(label)
            expected_distances.append((label, row))
        blocks = tuple((c, tuple(ids)) for c, ids in expected_blocks.items())
        if any(not ids for _, ids in blocks):
            return "FALSE"
        if (not same(certificate.blocks, blocks)
                or not same(certificate.distances_squared, tuple(expected_distances))
                or not same(certificate.center_distances_squared, tuple(pairs))):
            return "FALSE"
    except (ValueError, TypeError, AttributeError):
        return "FALSE"
    return "TRUE"
