"""AlgT' / AlgS0 — SOURCE-EXACT constructions for *already labelled* parallel projections.

Source
------
V. N. Kozlov, "Matematicheskoe modelirovanie zritel'nogo vospriyatiya",
Matematicheskie voprosy kibernetiki 6 (1996) 321-338, pages 331-338.
Step extraction / implementation boundary: ``docs/specs/stereo_algorithms_1996.md``.

What this module is
-------------------
Given one and the same label set with two **injective, exactly compatible** parallel
projections (two ``label -> rational 2D point`` maps), it builds

* ``construct_representative``  -> AlgT' : an affine representative body in R^3, and
* ``construct_output_image``    -> AlgS0 : a chosen output image in R^2,

executing the source's per-point branch structure (step 1 / step 2 with Lemma 6 /
step 2a / step 2b) and recording a per-point branch witness.

What it is NOT
--------------
* The correspondence is **given**. Unknown-correspondence search (p338, ``n!`` candidates)
  is a different procedure and is *not* solved here.  Noisy / perspective (pinhole)
  photographs are outside the source's model and outside this module.
* Nothing here is a metric reconstruction.  ``scale_status`` is ``unknown``: exactness is
  asserted only about the *supplied rational coordinates*, never about real units.
* Theorem 6 (p337) has its own ``obytemnoe telo`` domain; a successful construction is
  **not** evidence for Theorem 6.
* Branch selection is by the source's own image-space predicates (4-point marked affine
  equivalence, single-view collinearity).  The independent algebraic control of
  specification section 8 is deliberately **not** used as a solver: it lives only in the
  test/RUN layer so that product and control cannot share a solve.

SOURCE-DERIVED supplements (implemented here, *not* verbatim in the source)
--------------------------------------------------------------------------
* ``U1`` — a setwise line-to-line constraint leaves one degree of freedom (spec 7).  Two
  distinct base points of the intersection line are paired through the ``abc``-induced
  correspondence with their parameter preserved, and ``e``'s correspondence is added;
  that fixes the target plane map.  Base points / parameters can be varied and the result
  is unchanged.
* ``U2`` — AlgS0's target-plane transfer may need a **rank-deficient** affine map (``e0``
  can land on ``L0``).  Rank-deficient maps are allowed; labels are never merged away.
* ``ROLE SUBSTITUTION`` — Lemma 6 is worded for the triangle ``(a,b,c)`` and the face
  ``(c,d,e)``, so it names ``a``/``b`` as the free vertices of the triangle.  Step 2b
  renames the face to ``aed``/``bed``; the shared vertex is then no longer ``c``, and the
  auxiliary edge used by Lemma 6 cases 2/3/4 must be derived from the face's *local* known
  vertex.  Using the global ``a``/``b`` after a rename makes the second point of the
  intersection line come out equal to the known one (this was the R1 defect, see the R2
  ERRATA).

R2 status separation: a genuine input rejection raises ``ValidationError`` from
:func:`prepare_views`, while a failure of an internal construction step marks the affected
label ``UNSUPPORTED`` (overall ``UNSUPPORTED_BRANCH``).  Internal failures are never
reported as invalid input.

Fixed recorded initial choice (spec item 3)
-------------------------------------------
Auxiliary direction is vertical; the first image's ``abc`` is put at ``z = 0`` and ``e``
at ``z = 1``; the representative point of label ``i`` is ``(u_i, t_i)`` where ``u_i`` is
the first image coordinate after the ``abc -> (0,0),(1,0),(0,1)`` normalisation and
``t_i`` is the scalar with ``q_i - u_i = t_i * (q_e - u_e)``.  This gauge is a recorded
choice, not a recovered metric pose.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "SOURCE",
    "STATUS_OK",
    "STATUS_INVALID_INPUT",
    "STATUS_NOT_OFF_PLANE",
    "STATUS_INCOMPATIBLE",
    "STATUS_BUDGET_EXCEEDED",
    "STATUS_UNSUPPORTED",
    "BRANCH_INITIALIZATION",
    "BRANCH_STEP1",
    "BRANCH_LEMMA6_SINGLE_VIEW_COLLINEAR",
    "BRANCH_LEMMA6_COMMON_POINT_A",
    "BRANCH_LEMMA6_COMMON_POINT_B",
    "BRANCH_LEMMA6_COMMON_POINT_RENAMED",
    "BRANCH_LEMMA6_PARALLEL_EDGES",
    "BRANCH_LEMMA6_GENERAL_INTERSECTION",
    "BRANCH_STEP2A_CE",
    "BRANCH_STEP2A_AE",
    "BRANCH_STEP2A_BE",
    "BRANCH_STEP2B_REDUCE_AED",
    "BRANCH_STEP2B_REDUCE_BED",
    "BRANCH_STEP2B_EDGE_TEST",
    "Point2",
    "Point3",
    "PointWitness",
    "ConstructionResult",
    "Prepared",
    "ValidationError",
    "BudgetExceeded",
    "prepare_views",
    "construct_representative",
    "construct_output_image",
]

SOURCE = "S-1996 (Kozlov 1996, pp.331-338)"

# --- statuses -------------------------------------------------------------------------
STATUS_OK = "OK"
STATUS_INVALID_INPUT = "INVALID_INPUT"
STATUS_NOT_OFF_PLANE = "NOT_OFF_PLANE_PREREQUISITE_UNMET"
STATUS_INCOMPATIBLE = "INCOMPATIBLE_CORRESPONDENCE"
STATUS_BUDGET_EXCEEDED = "BUDGET_EXCEEDED"
STATUS_UNSUPPORTED = "UNSUPPORTED_BRANCH"

# --- branch names (each maps to a source step; see the branch map in the run evidence) --
BRANCH_INITIALIZATION = "INITIALIZATION_RECORDED_CHOICE"
BRANCH_STEP1 = "STEP1_IN_PLANE"
BRANCH_LEMMA6_SINGLE_VIEW_COLLINEAR = "STEP2_LEMMA6_6_1_SINGLE_VIEW_COLLINEAR"
BRANCH_LEMMA6_COMMON_POINT_A = "STEP2_LEMMA6_6_2_COMMON_POINT_A"
BRANCH_LEMMA6_COMMON_POINT_B = "STEP2_LEMMA6_6_2_COMMON_POINT_B"
BRANCH_LEMMA6_COMMON_POINT_RENAMED = "STEP2_LEMMA6_6_2_COMMON_POINT_RENAMED_FACE"
BRANCH_LEMMA6_PARALLEL_EDGES = "STEP2_LEMMA6_6_3_PARALLEL_EDGES"
BRANCH_LEMMA6_GENERAL_INTERSECTION = "STEP2_LEMMA6_6_4_GENERAL_INTERSECTION"
BRANCH_STEP2A_CE = "STEP2A_TRUE_COLLINEAR_CE"
BRANCH_STEP2A_AE = "STEP2A_TRUE_COLLINEAR_AE"
BRANCH_STEP2A_BE = "STEP2A_TRUE_COLLINEAR_BE"
BRANCH_STEP2B_REDUCE_AED = "STEP2B_REDUCE_FACE_AED"
BRANCH_STEP2B_REDUCE_BED = "STEP2B_REDUCE_FACE_BED"
BRANCH_STEP2B_EDGE_TEST = "STEP2B_EDGE_TEST_AE_BE"

Point2 = tuple[Fraction, Fraction]
Point3 = tuple[Fraction, Fraction, Fraction]

# --- explicit input budget (spec item 1) ----------------------------------------------
MIN_LABELS = 4
MAX_LABELS = 32
MAX_ABS_NUMERATOR = 10 ** 12
MAX_DENOMINATOR = 10 ** 12
DEFAULT_OP_BUDGET = 2_000_000


class ValidationError(Exception):
    """Input rejected by the declared contract -- rejection, never quantisation.

    ``kind`` separates the three failure expressions the contract requires:
    ``INVALID_INPUT`` (labels / values / structure), ``INCOMPATIBLE_CORRESPONDENCE`` (the
    given correspondence cannot come from one body's two parallel projections) and -- for
    construction results rather than exceptions -- ``UNSUPPORTED_BRANCH`` (legal input whose
    branch could not be constructed).
    """

    def __init__(self, message: str, kind: str = "INVALID_INPUT") -> None:
        super().__init__(message)
        self.kind = kind


class BudgetExceeded(Exception):
    """Raised when the declared exact-operation budget is exhausted."""


class _Budget:
    __slots__ = ("left", "spent")

    def __init__(self, total: int) -> None:
        self.left = int(total)
        self.spent = 0

    def spend(self, amount: int = 1) -> None:
        self.left -= amount
        self.spent += amount
        if self.left < 0:
            raise BudgetExceeded(f"exact operation budget exhausted after {self.spent} operations")


# --- exact 2D/3D arithmetic -----------------------------------------------------------
ZERO = Fraction(0)
ONE = Fraction(1)


def _sub2(u, v):
    return (u[0] - v[0], u[1] - v[1])


def _add2(u, v):
    return (u[0] + v[0], u[1] + v[1])


def _mul2(u, s):
    return (u[0] * s, u[1] * s)


def _cross2(u, v):
    return u[0] * v[1] - u[1] * v[0]


def _parallel2(u, v) -> bool:
    return _cross2(u, v) == 0


def _affine_solve(src: Sequence[Point2], dst: Sequence[Sequence[Fraction]]):
    """Exact affine map (x,y) -> A(x,y)+b with src[k] -> dst[k], len(src)==3.

    ``src`` must be non-collinear (that is the only degeneracy that matters here: a
    collinear target set is fine and yields a rank-deficient map, which is exactly what
    the U2 supplement needs).  Returns a callable.
    """
    (x1, y1), (x2, y2), (x3, y3) = src
    det = x1 * (y2 - y3) - y1 * (x2 - x3) + (x2 * y3 - x3 * y2)
    if det == 0:
        raise ValidationError("affine solve needs a non-collinear source triple")
    d = len(dst[0])
    inv = Fraction(1, 1) / det
    rows = []
    for k in range(d):
        t1 = Fraction(dst[0][k])
        t2 = Fraction(dst[1][k])
        t3 = Fraction(dst[2][k])
        a = inv * ((y2 - y3) * t1 + (y3 - y1) * t2 + (y1 - y2) * t3)
        b = inv * ((x3 - x2) * t1 + (x1 - x3) * t2 + (x2 - x1) * t3)
        c = inv * ((x2 * y3 - x3 * y2) * t1 + (x3 * y1 - x1 * y3) * t2 + (x1 * y2 - x2 * y1) * t3)
        rows.append((a, b, c))

    def f(p):
        x, y = p[0], p[1]
        return tuple(row[0] * x + row[1] * y + row[2] for row in rows)

    return f


def _line_point_dir(p: Point2, q: Point2):
    return (p, _sub2(q, p))


def _line_intersect(l1, l2):
    """Intersection of two 2D lines given as (point, direction); None if parallel."""
    p, u = l1
    q, v = l2
    den = _cross2(u, v)
    if den == 0:
        return None
    w = _sub2(q, p)
    t = _cross2(w, v) / den
    return (_add2(p, _mul2(u, t)), t)


def _on_line(l, x: Point2) -> bool:
    p, u = l
    return _cross2(u, _sub2(x, p)) == 0


def _fixed_point_on_line(f, line):
    """Exact point of ``line`` that an affine map ``f`` fixes, or None if inconsistent.

    Used by Lemma 6 case 4 for the source's symmetric renaming: the second point of the
    intersection line is the point of ab whose two images coincide, which is exactly a fixed
    point of the face map on the image of ab.
    """
    base, direction = line
    fbase = f(base)
    d0 = _sub2(fbase, base)
    fd = _sub2(f(_add2(base, direction)), fbase)
    a = _sub2(direction, fd)
    for i in (0, 1):
        if a[i] != 0:
            s = d0[i] / a[i]
            if s * a[(i + 1) % 2] != d0[(i + 1) % 2]:
                return None
            return _add2(base, _mul2(direction, s))
    return None


def _same_line(l1, l2) -> bool:
    return _on_line(l1, l2[0]) and _parallel2(l1[1], l2[1])


# --- input validation (spec item 1) ---------------------------------------------------
def _as_fraction(value: Any, ctx: str) -> Fraction:
    if isinstance(value, bool):
        raise ValidationError(f"{ctx}: bool is not a rational coordinate")
    if isinstance(value, float):
        raise ValidationError(f"{ctx}: float is rejected (no silent quantisation)")
    if isinstance(value, int):
        f = Fraction(value)
    elif isinstance(value, Fraction):
        f = value
    else:
        raise ValidationError(f"{ctx}: unsupported coordinate type {type(value).__name__!r}")
    if abs(f.numerator) > MAX_ABS_NUMERATOR:
        raise ValidationError(f"{ctx}: |numerator| exceeds {MAX_ABS_NUMERATOR}")
    if f.denominator > MAX_DENOMINATOR:
        raise ValidationError(f"{ctx}: denominator exceeds {MAX_DENOMINATOR}")
    return f


def _as_point2(value: Any, ctx: str) -> Point2:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValidationError(f"{ctx}: expected a 2-tuple of rationals")
    return (_as_fraction(value[0], f"{ctx}[0]"), _as_fraction(value[1], f"{ctx}[1]"))


@dataclass(frozen=True)
class Prepared:
    """Validated, normalised input plus the one recorded gauge choice."""

    labels: tuple[str, ...]
    first: Mapping[str, Point2]        # raw first-image coordinates
    second: Mapping[str, Point2]       # raw second-image coordinates
    norm_first: Mapping[str, Point2]   # after abc -> (0,0),(1,0),(0,1)
    norm_second: Mapping[str, Point2]
    anchors: tuple[str, str, str, str]  # (a, b, c, e)
    difference: Mapping[str, Point2]   # q_i - p_i (normalised)
    v: Point2                          # q_e - p_e, the recorded non-zero direction
    t: Mapping[str, Fraction]          # q_i - p_i = t_i * v
    normalisation_first: tuple[tuple[Fraction, Fraction, Fraction], ...]
    normalisation_second: tuple[tuple[Fraction, Fraction, Fraction], ...]
    budget: _Budget = field(repr=False, compare=False, default=None)

    @property
    def a(self) -> str:
        return self.anchors[0]

    @property
    def b(self) -> str:
        return self.anchors[1]

    @property
    def c(self) -> str:
        return self.anchors[2]

    @property
    def e(self) -> str:
        return self.anchors[3]


def _normalisation(pa: Point2, pb: Point2, pc: Point2):
    """Exact affine map sending pa,pb,pc -> (0,0),(1,0),(0,1) as (a,b,c) coefficients."""
    A = _sub2(pb, pa)
    B = _sub2(pc, pa)
    det = _cross2(A, B)
    if det == 0:
        raise ValidationError("abc must be a triangle in each image")
    inv = Fraction(1, 1) / det
    # rows: [m00, m01, m10, m11] then translation is applied as m * (x - pa)
    m00 = B[1] * inv
    m01 = -B[0] * inv
    m10 = -A[1] * inv
    m11 = A[0] * inv
    return ((m00, m01, m10, m11), pa)


def _apply_normalisation(norm, p: Point2) -> Point2:
    (m00, m01, m10, m11), pa = norm
    x = p[0] - pa[0]
    y = p[1] - pa[1]
    return (m00 * x + m01 * y, m10 * x + m11 * y)


def _normalisation_as_rows(norm) -> tuple[tuple[Fraction, Fraction, Fraction], ...]:
    (m00, m01, m10, m11), pa = norm
    return (
        (m00, m01, -(m00 * pa[0] + m01 * pa[1])),
        (m10, m11, -(m10 * pa[0] + m11 * pa[1])),
    )


# --- preparation ----------------------------------------------------------------------
def prepare_views(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    a: str,
    b: str,
    c: str,
    e: str,
    *,
    op_budget: int = DEFAULT_OP_BUDGET,
) -> Prepared:
    """Validate the labelled pair and fix the recorded gauge.  Rejects, never quantises."""
    for name, view in (("first", first), ("second", second)):
        if not isinstance(view, Mapping):
            raise ValidationError(f"{name}: expected a mapping label -> 2D point")
    labels = tuple(first.keys())
    if len(labels) < MIN_LABELS:
        raise ValidationError(f"need at least {MIN_LABELS} labels, got {len(labels)}")
    if len(labels) > MAX_LABELS:
        raise ValidationError(
            f"n={len(labels)} exceeds the supported maximum n={MAX_LABELS}"
        )
    if set(second.keys()) != set(labels):
        only_first = sorted(set(labels) - set(second.keys()))
        only_second = sorted(set(second.keys()) - set(labels))
        raise ValidationError(
            f"label sets differ (first-only={only_first}, second-only={only_second})"
        )
    for lab in labels:
        if not isinstance(lab, str) or lab == "":
            raise ValidationError(f"illegal label {lab!r}: labels must be non-empty strings")
    for lab in (a, b, c, e):
        if not isinstance(lab, str) or lab not in labels:
            raise ValidationError(f"anchor label {lab!r} is not present in both images")
    if len({a, b, c, e}) != 4:
        raise ValidationError("a, b, c, e must be four distinct labels")

    p_raw = {lab: _as_point2(first[lab], f"first[{lab!r}]") for lab in labels}
    q_raw = {lab: _as_point2(second[lab], f"second[{lab!r}]") for lab in labels}
    for name, view in (("first", p_raw), ("second", q_raw)):
        seen: dict[Point2, str] = {}
        for lab in labels:
            pt = view[lab]
            if pt in seen:
                raise ValidationError(
                    f"{name}: labels {seen[pt]!r} and {lab!r} share coordinates {pt}"
                )
            seen[pt] = lab

    budget = _Budget(op_budget)
    nf = _normalisation(p_raw[a], p_raw[b], p_raw[c])
    ns = _normalisation(q_raw[a], q_raw[b], q_raw[c])
    budget.spend(2 * len(labels))
    p = {lab: _apply_normalisation(nf, p_raw[lab]) for lab in labels}
    q = {lab: _apply_normalisation(ns, q_raw[lab]) for lab in labels}

    diff = {lab: _sub2(q[lab], p[lab]) for lab in labels}
    v = diff[e]
    t: dict[str, Fraction] = {}
    if v != (ZERO, ZERO):
        for lab in labels:
            d = diff[lab]
            # solve d = t * v exactly on a non-zero component; any inconsistency is a rejection
            if v[0] != 0:
                ti = d[0] / v[0]
                if ti * v[1] != d[1]:
                    raise ValidationError(
                        f"label {lab!r}: the given correspondence is not compatible with a "
                        "single body under two parallel projections",
                        kind=STATUS_INCOMPATIBLE,
                    )
            else:
                ti = d[1] / v[1]
                if ti * v[0] != d[0]:
                    raise ValidationError(
                        f"label {lab!r}: the given correspondence is not compatible with a "
                        "single body under two parallel projections",
                        kind=STATUS_INCOMPATIBLE,
                    )
            t[lab] = ti
    else:
        for lab in labels:
            t[lab] = ZERO

    return Prepared(
        labels=labels,
        first=p_raw,
        second=q_raw,
        norm_first=p,
        norm_second=q,
        anchors=(a, b, c, e),
        difference=diff,
        v=v,
        t=t,
        normalisation_first=_normalisation_as_rows(nf),
        normalisation_second=_normalisation_as_rows(ns),
        budget=budget,
    )


# --- source predicate: "d lies in the plane of triangle (l1,l2,l3)" -------------------
def _in_plane(prep: Prepared, l4: str, l1: str, l2: str, l3: str) -> bool:
    """SOURCE-EXACT two-view predicate (p332).

    ``S1(l1,l2,l3,l4) ~ S2(l1,l2,l3,l4)``: the four-point configurations are affinely
    equivalent.  If the first triple is a triangle this is an exact affine check; if it is
    collinear the source's own fallback applies -- the four points must be collinear in
    both images.
    """
    prep.budget.spend(8)
    p = prep.norm_first
    q = prep.norm_second
    a1, a2, a3 = p[l1], p[l2], p[l3]
    b1, b2, b3 = q[l1], q[l2], q[l3]
    if _cross2(_sub2(a2, a1), _sub2(a3, a1)) != 0:
        f = _affine_solve((a1, a2, a3), (b1, b2, b3))
        return tuple(f(p[l4])) == tuple(q[l4])
    if _cross2(_sub2(b2, b1), _sub2(b3, b1)) != 0:
        return False
    return _on_line(_line_point_dir(a1, a2), p[l4]) and _on_line(_line_point_dir(b1, b2), q[l4])


def _collinear_view(prep: Prepared, view: str, l1: str, l2: str, l3: str) -> bool:
    pts = prep.norm_first if view == "first" else prep.norm_second
    prep.budget.spend(2)
    return _cross2(_sub2(pts[l2], pts[l1]), _sub2(pts[l3], pts[l1])) == 0


# --- "d lies on the body line through m and k" (two distinct probe planes) ------------
def _on_body_line(prep: Prepared, d: str, m: str, k: str):
    """d in line(m,k): d must lie in two *different* planes through that line.

    Returns ``(verdict, probes)``; ``verdict`` is ``None`` when fewer than two usable probe
    labels exist (branch then cannot be decided and the caller must not guess).
    """
    probes = [x for x in (prep.a, prep.b, prep.c, prep.e) if x not in (m, k, d)]
    if len(probes) < 2:
        return None, ()
    u, w = probes[0], probes[1]
    ok = _in_plane(prep, d, m, k, u) and _in_plane(prep, d, m, k, w)
    return bool(ok), (u, w)


# --- Lemma 6 (p332-333): the images of the intersection line --------------------------
@dataclass(frozen=True)
class _LineChoice:
    branch: str
    point: Point2
    direction: Point2
    detail: Mapping[str, Any]

    def at(self, s: Fraction) -> Point2:
        return _add2(self.point, _mul2(self.direction, s))


def _lemma6_line(prep: Prepared, d: str, c_role: str, e_role: str) -> _LineChoice:
    """Position of the intersection line L = plane(abc) ^ plane(c_role, d, e_role).

    Returns the line in normalised first-image coordinates.  Both images give the *same*
    normalised line, because the abc-induced map between the two images is the identity in
    this frame (source Lemma 6, case 1 and its transport argument).
    """
    p = prep.norm_first
    q = prep.norm_second
    tri1 = not _collinear_view(prep, "first", c_role, d, e_role)
    tri2 = not _collinear_view(prep, "second", c_role, d, e_role)

    # 6.1 -- exactly one image shows (c,d,e) as a triangle
    if tri1 and not tri2:
        return _LineChoice(
            BRANCH_LEMMA6_SINGLE_VIEW_COLLINEAR,
            q[c_role],
            _sub2(q[d], q[c_role]),
            {"triangle_view": "first", "collinear_view": "second", "transport": "identity"},
        )
    if tri2 and not tri1:
        return _LineChoice(
            BRANCH_LEMMA6_SINGLE_VIEW_COLLINEAR,
            p[c_role],
            _sub2(p[d], p[c_role]),
            {"triangle_view": "second", "collinear_view": "first", "transport": "identity"},
        )
    if not tri1 and not tri2:
        raise ValidationError("_lemma6_line called for a face that is collinear in both images")

    # ROLE SUBSTITUTION.  Lemma 6's wording is written for the triple (a,b,c) as the plane
    # triangle and (c,d,e) as the face, so it names a and b as the two free vertices.  When
    # step 2b renames the face to (a,e,d) or (b,e,d) the shared vertex is no longer c, and the
    # free vertices of the abc triangle are *not* the global a/b any more.  Using the global
    # a/b after a rename is what makes the second point of L come out equal to the known one;
    # the auxiliary edge must therefore be derived from the face's known vertex c_role.
    aux = tuple(x for x in (prep.a, prep.b, prep.c) if x != c_role)
    aux_edge = (aux[0], aux[1])

    # 6.2 -- both triangles: one of the two free vertices may lie in the plane of the face
    for probe in aux:
        if _in_plane(prep, probe, c_role, d, e_role):
            if c_role == prep.c:
                branch = (BRANCH_LEMMA6_COMMON_POINT_A if probe == prep.a
                          else BRANCH_LEMMA6_COMMON_POINT_B)
            else:
                branch = BRANCH_LEMMA6_COMMON_POINT_RENAMED
            return _LineChoice(
                branch,
                p[c_role],
                _sub2(p[probe], p[c_role]),
                {"probe": probe, "face_known_vertex": c_role, "free_vertices": list(aux)},
            )

    # 6.3 -- L is parallel to de and to the auxiliary edge.  The premise must be tested as a
    #        premise, not via "the two image directions look parallel": an image-space
    #        parallelism is *implied by* but not equivalent to it (R2 counterexample: the
    #        images of de and of ac are parallel while de is not parallel to plane(abc)).
    #          de parallel to plane(abc)   <=> t_d == t_e   (t is the recorded height)
    #          aux parallel to plane(face) <=> the face map fixes the image of the virtual
    #              point face_known_vertex + aux_direction: that point lies in plane(abc), so
    #              its two images coincide, and "it lies in plane(face)" is exactly the
    #              source's 4-point predicate, i.e. F(image) == image.
    aux_vec = _sub2(p[aux_edge[0]], p[aux_edge[1]])
    face_map_probe = _affine_solve((p[c_role], p[d], p[e_role]), (q[c_role], q[d], q[e_role]))
    virtual = _add2(p[c_role], aux_vec)
    if prep.t[d] == prep.t[e_role] and tuple(face_map_probe(virtual)) == tuple(virtual):
        return _LineChoice(
            BRANCH_LEMMA6_PARALLEL_EDGES,
            p[c_role],
            _sub2(p[d], p[e_role]),
            {
                "premise_de_parallel_plane_abc": "t_d == t_e",
                "premise_aux_parallel_plane_face": f"face map fixes {c_role} + {aux_edge}",
                "aux_edge": list(aux_edge),
                "virtual_point_image": _ser(virtual),
                "de_image_direction_first": _sub2(p[d], p[e_role]),
                "aux_edge_image_direction_first": aux_vec,
            },
        )

    # 6.4 -- general: align the two abc's (identity in this frame) and take the second point
    #        of L from the two images of the edge de.  When those two images coincide / are
    #        parallel the source's symmetric renaming applies, so the second point is taken
    #        from the auxiliary edge instead (no consistent one => report, never guess).
    #
    #        SOURCE-DERIVED completion: for an exact solver the robust form of "x = de ^ L"
    #        is the fixed point of the face map on the image of de.  If x = de ^ plane(abc)
    #        then F(x) = x for the affine map F carrying (c_role,d,e) of image 1 onto image 2,
    #        because F(x) = F(p_d) + lam*(F(p_e)-F(p_d)) = x + v*(t_d + lam*(t_e-t_d)) = x.
    de_images = (_line_point_dir(p[d], p[e_role]), _line_point_dir(q[d], q[e_role]))
    inter = _line_intersect(de_images[0], de_images[1])
    if inter is not None:
        x_pt = inter[0]
        method = "de_intersection"
    else:
        face_map = _affine_solve((p[c_role], p[d], p[e_role]), (q[c_role], q[d], q[e_role]))
        y = _fixed_point_on_line(face_map, de_images[0])
        method = "de_coincident_fixed_point"
        if y is None:
            y = _fixed_point_on_line(face_map, _line_point_dir(p[aux_edge[0]], p[aux_edge[1]]))
            method = f"aux_edge_fixed_point({aux_edge[0]}{aux_edge[1]})"
        if y is None:
            raise ValidationError(
                "_lemma6_line case 4: the two images of de are parallel and the auxiliary edge "
                f"{aux_edge[0]}{aux_edge[1]} yields no consistent second point of L"
            )
        x_pt = y
    if x_pt == p[c_role]:
        raise ValidationError(
            "_lemma6_line case 4: the second point coincides with the known point, so L is "
            "not determined"
        )
    return _LineChoice(
        BRANCH_LEMMA6_GENERAL_INTERSECTION,
        x_pt,
        _sub2(p[c_role], x_pt),
        {"second_point": x_pt, "second_point_method": method},
    )


# --- result records -------------------------------------------------------------------
@dataclass(frozen=True)
class PointWitness:
    """Per-label branch witness.  ``completeness`` is CONSTRUCTED or UNSUPPORTED."""

    label: str
    branch: str
    route: tuple[str, ...]
    source_view: str | None
    detail: Mapping[str, Any]
    completeness: str = "CONSTRUCTED"
    note: str = ""


@dataclass
class ConstructionResult:
    algorithm: str
    status: str
    points: dict[str, tuple[Fraction, ...]]
    witnesses: dict[str, PointWitness]
    anchors_out: dict[str, tuple[Fraction, ...]]
    v: Point2
    t: Mapping[str, Fraction]
    base_parameters: tuple[Fraction, Fraction]
    unsupported: list[str] = field(default_factory=list)
    failure_reason: str = ""
    scale_status: str = "unknown"
    note: str = ""

    @property
    def ok(self) -> bool:
        return self.status == STATUS_OK and not self.unsupported


def _ratio_on_line(x: Point2, base: Point2, other: Point2) -> Fraction:
    """lambda with x = base + lambda * (other - base); requires collinearity."""
    d = _sub2(other, base)
    if d[0] != 0:
        lam = (x[0] - base[0]) / d[0]
        if base[1] + lam * d[1] != x[1]:
            raise ValidationError("_ratio_on_line: point is not on the line")
        return lam
    if d[1] != 0:
        lam = (x[1] - base[1]) / d[1]
        if base[0] + lam * d[0] != x[0]:
            raise ValidationError("_ratio_on_line: point is not on the line")
        return lam
    raise ValidationError("_ratio_on_line: degenerate direction")


@dataclass(frozen=True)
class _OffPlane:
    point: tuple[Fraction, ...]
    branch: str
    route: tuple[str, ...]
    source_view: str | None
    detail: Mapping[str, Any]
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.point is not None


# --- generic tuple ops (work for 2- and 3-tuples) ------------------------------------
def _subn(u, v):
    return tuple(a - b for a, b in zip(u, v))


def _addn(u, v):
    return tuple(a + b for a, b in zip(u, v))


def _muln(u, s):
    return tuple(a * s for a in u)


# --- shared branch resolution: input images + Lemma 6 (identical for both algorithms) --
@dataclass(frozen=True)
class _Decision:
    """Source branch chosen for one off-plane label plus the data needed to build it."""

    kind: str                      # STEP2 | STEP2A | UNSUPPORTED
    branch: str
    route: tuple[str, ...]
    source_view: str | None
    face: tuple[str, str] = ("", "")
    ratio: Fraction | None = None
    probes: tuple = ()
    line: _LineChoice | None = None
    base_points: tuple = ()
    line_source_coord: Point2 | None = None
    line_source_e: Point2 | None = None
    detail: Mapping[str, Any] = field(default_factory=dict)
    reason: str = ""


def _resolve_offplane(prep: Prepared, d: str, base_parameters) -> _Decision:
    """Choose the source branch for an off-plane label.

    Order follows p334 exactly: step 2 (Lemma 6) is tried on the face (c,d,e) whenever that
    triple is a triangle in at least one image.  Only when it is collinear in *both* images
    come step 2a and 2b, and then the point is tested against the edges in the source's order
    ce, ae, be before the face is rotated to aed and finally bed, re-entering step 2.
    """
    p = prep.norm_first
    s1, s2 = base_parameters

    def step2(m: str, route: tuple) -> _Decision:
        lc = _lemma6_line(prep, d, m, prep.e)
        tri2 = not _collinear_view(prep, "second", m, d, prep.e)
        src_view = "second" if tri2 else "first"
        coord = prep.norm_second if src_view == "second" else prep.norm_first
        b1 = lc.at(s1)
        b2 = lc.at(s2)
        detail = {
            "line": _ser(lc),
            "base_s": (str(s1), str(s2)),
            "base_points": (_ser(b1), _ser(b2)),
        }
        if b1 == b2:
            return _Decision("UNSUPPORTED", lc.branch, route, src_view, (m, prep.e), detail=detail,
                             reason="U1 base points coincide; choose different base parameters")
        # U1 needs two base points of L plus e's correspondence to fix the target plane map.
        # The guard below fires only if e's *source-image* point lies on the image of L.  That
        # is provably impossible in this branch, under the contract's premises:
        #   (P1) abc is a triangle in both images  => each projection is injective on plane(abc)
        #        and maps the line L = plane(abc) ^ plane(face) bijectively onto its image;
        #   (P2) the face (c_role, d, e) is a triangle in the source view used here.
        # Suppose e_src were on that image.  Then some y in L has the same source image as e,
        # so the body direction e - y is parallel to the source projection direction.  Both y
        # (in L) and e lie in the face plane, hence that direction lies in the face plane's
        # direction space, i.e. the face plane is parallel to the source direction - and then
        # the source view shows c_role, d, e collinear, contradicting (P2).  The guard is kept
        # as a defensive assertion; it is not counted as a covered source branch.
        if _cross2(_sub2(b2, b1), _sub2(coord[prep.e], b1)) == 0:
            return _Decision(
                "UNSUPPORTED", lc.branch, route, src_view, (m, prep.e), detail=detail,
                reason="U1 base-line degeneracy: the source-image point of e lies on the "
                       "intersection line, so line + e does not fix the target plane map "
                       "(SOURCE-DERIVED U1 is not applicable to this configuration)",
            )
        return _Decision("STEP2", lc.branch, route, src_view, (m, prep.e), line=lc,
                         base_points=(b1, b2), line_source_coord=coord[d],
                         line_source_e=coord[prep.e], detail=detail)

    def edge(m: str, route: tuple, probes) -> _Decision:
        lam = _ratio_on_line(p[d], p[m], p[prep.e])
        branch = {
            prep.c: BRANCH_STEP2A_CE,
            prep.a: BRANCH_STEP2A_AE,
            prep.b: BRANCH_STEP2A_BE,
        }[m]
        return _Decision("STEP2A", branch, route, None, (m, prep.e), ratio=lam, probes=probes)

    # -- p334 step 2 on the face (c,d,e): a triangle in at least one image means Lemma 6
    if (not _collinear_view(prep, "first", prep.c, d, prep.e)
            or not _collinear_view(prep, "second", prep.c, d, prep.e)):
        return step2(prep.c, ())
    # -- p334 step 2a: c, d, e lie on one straight line in the body, so d lies on the edge ce
    verdict, probes = _on_body_line(prep, d, prep.c, prep.e)
    if verdict:
        return edge(prep.c, (), probes)
    # -- p334 step 2b: d on the edge ae or be is still step 2a (checked before any rotation)
    for m in (prep.a, prep.b):
        verdict2, probes2 = _on_body_line(prep, d, m, prep.e)
        if verdict2:
            return edge(m, (BRANCH_STEP2B_EDGE_TEST,), probes2)
    # -- p334 step 2b reduction: re-enter step 2 on the face aed, then on bed
    if (not _collinear_view(prep, "first", prep.a, d, prep.e)
            or not _collinear_view(prep, "second", prep.a, d, prep.e)):
        return step2(prep.a, (BRANCH_STEP2B_REDUCE_AED,))
    if (not _collinear_view(prep, "first", prep.b, d, prep.e)
            or not _collinear_view(prep, "second", prep.b, d, prep.e)):
        return step2(prep.b, (BRANCH_STEP2B_REDUCE_AED, BRANCH_STEP2B_REDUCE_BED))
    return _Decision(
        "UNSUPPORTED", BRANCH_STEP2B_REDUCE_BED,
        (BRANCH_STEP2B_REDUCE_AED, BRANCH_STEP2B_REDUCE_BED), None, (prep.b, prep.e),
        reason="STEP2B face rotation exhausted: the face (c,d,e) is collinear in both images, "
               "the point lies on none of the edges ce, ae, be, and neither aed nor bed becomes "
               "a triangle in either image",
    )


# --- AlgT' : off-plane construction of one point (target = 3D representative body) -----
def _tprime_offplane(prep: Prepared, d: str, base_parameters) -> _OffPlane:
    dec = _resolve_offplane(prep, d, base_parameters)
    p = prep.norm_first
    target3 = {
        prep.a: (p[prep.a][0], p[prep.a][1], ZERO),
        prep.b: (p[prep.b][0], p[prep.b][1], ZERO),
        prep.c: (p[prep.c][0], p[prep.c][1], ZERO),
        prep.e: (p[prep.e][0], p[prep.e][1], ONE),
    }
    if dec.kind == "STEP2A":
        base3 = target3[dec.face[0]]
        end3 = target3[dec.face[1]]
        out = _addn(base3, _muln(_subn(end3, base3), dec.ratio))
        return _OffPlane(out, dec.branch, dec.route, None,
                         {"ratio_lambda": str(dec.ratio), "probes": dec.probes, "face": dec.face})
    if dec.kind == "STEP2":
        b1, b2 = dec.base_points
        f = _affine_solve(
            (b1, b2, dec.line_source_e),
            ((b1[0], b1[1], ZERO), (b2[0], b2[1], ZERO), target3[prep.e]),
        )
        return _OffPlane(tuple(f(dec.line_source_coord)), dec.branch, dec.route, dec.source_view,
                         dict(dec.detail))
    return _OffPlane(None, dec.branch, dec.route, dec.source_view, dict(dec.detail), dec.reason)


# --- AlgS0 : off-plane construction of one point (target = chosen output image) -------
def _s0_offplane(prep: Prepared, d: str, anchors_out, base_parameters) -> _OffPlane:
    """AlgS0's own target construction, built independently of AlgT'.

    It reads only the two images, the chosen output anchors and the shared branch decision
    (input-space skeleton: which face, which edge, which Lemma 6 case).  It never consults
    a 3D body.  Rank-deficient maps and coincident output positions are allowed (U2).
    """
    p = prep.norm_first
    A, B, C, E = anchors_out
    n_map = _affine_solve((p[prep.a], p[prep.b], p[prep.c]), (A, B, C))
    dec = _resolve_offplane(prep, d, base_parameters)
    if dec.kind == "STEP2A":
        # source 2a: an affine map of S1 with c1 -> c0 and e1 -> e0; the point on the edge
        # keeps its affine ratio.  In AlgS0 the image of e is the *given* anchor E (not the
        # plane map's value at p_e), which is what makes this independent of AlgT'.
        base2 = n_map(p[dec.face[0]])
        end2 = tuple(E)
        out = _addn(base2, _muln(_subn(end2, base2), dec.ratio))
        return _OffPlane(out, dec.branch, dec.route, None,
                         {"ratio_lambda": str(dec.ratio), "probes": dec.probes, "face": dec.face,
                          "c_role_image": "abc -> A,B,C", "e_image": "given anchor E",
                          "rank_status": "rank-deficient transfer allowed (U2)"})
    if dec.kind == "STEP2":
        b1, b2 = dec.base_points
        # U1: L2 -> L0 pointwise (parameter preserved) plus e2 -> e0 = E.
        # L0 is the image of L1 under the affine map abc -> A,B,C.
        # U2: the three target points may be collinear; the map is then rank-deficient and
        # that is allowed -- the source points stay non-collinear, so the solve is unique.
        f = _affine_solve(
            (b1, b2, dec.line_source_e),
            (tuple(n_map(b1)), tuple(n_map(b2)), tuple(E)),
        )
        detail = dict(dec.detail)
        detail["L0_from"] = "abc -> A,B,C"
        detail["rank_status"] = "rank-deficient transfer allowed (U2)"
        return _OffPlane(tuple(f(dec.line_source_coord)), dec.branch, dec.route, dec.source_view,
                         detail)
    return _OffPlane(None, dec.branch, dec.route, dec.source_view, dict(dec.detail), dec.reason)


def _ser(x):
    if isinstance(x, _LineChoice):
        return {"point": _ser(x.point), "direction": _ser(x.direction), "branch": x.branch,
                "detail": {k: _ser(v) for k, v in x.detail.items()}}
    if isinstance(x, tuple):
        return [_ser(v) for v in x]
    if isinstance(x, Fraction):
        return f"{x.numerator}/{x.denominator}" if x.denominator != 1 else str(x.numerator)
    return x


# --- public constructions -------------------------------------------------------------
def _base_pair(base_parameters) -> tuple[Fraction, Fraction]:
    if not isinstance(base_parameters, (tuple, list)) or len(base_parameters) != 2:
        raise ValidationError("base_parameters must be a 2-tuple of rationals")
    s1 = _as_fraction(base_parameters[0], "base_parameters[0]")
    s2 = _as_fraction(base_parameters[1], "base_parameters[1]")
    if s1 == s2:
        raise ValidationError("base_parameters must be two distinct values")
    return (s1, s2)


def construct_representative(prep: Prepared, *, base_parameters=(0, 1)) -> ConstructionResult:
    """AlgT' (source pp.333-334): build the affine representative body.

    Never runs the algebraic control of specification section 8; the per-point result comes
    from the source's own branches (recorded in the witnesses).
    """
    bp = _base_pair(base_parameters)
    p = prep.norm_first
    algorithm = "AlgT'"
    points: dict[str, tuple[Fraction, ...]] = {}
    witnesses: dict[str, PointWitness] = {}
    unsupported: list[str] = []
    anchor3 = {
        prep.a: (p[prep.a][0], p[prep.a][1], ZERO),
        prep.b: (p[prep.b][0], p[prep.b][1], ZERO),
        prep.c: (p[prep.c][0], p[prep.c][1], ZERO),
        prep.e: (p[prep.e][0], p[prep.e][1], ONE),
    }
    if prep.v == (ZERO, ZERO):
        return ConstructionResult(
            algorithm=algorithm,
            status=STATUS_NOT_OFF_PLANE,
            points={},
            witnesses={},
            anchors_out={},
            v=prep.v,
            t=prep.t,
            base_parameters=bp,
            failure_reason=(
                "v = q_e - p_e is zero: after aligning the two abc's the point e coincides "
                "with itself, i.e. the two marked images are affinely equivalent. The "
                "source's non-planar premise (e outside the plane of triangle abc) is NOT "
                "met, so no representative body is reported and no anchor is substituted."
            ),
        )
    try:
        for lab in prep.labels:
            if lab == prep.e:
                points[lab] = anchor3[lab]
                witnesses[lab] = PointWitness(
                    lab, BRANCH_INITIALIZATION, (), None,
                    {"recorded_choice": "first image abc at z=0, e at z=1, auxiliary direction vertical"},
                )
                continue
            if _in_plane(prep, lab, prep.a, prep.b, prep.c):
                f = _affine_solve(
                    (p[prep.a], p[prep.b], p[prep.c]),
                    (anchor3[prep.a], anchor3[prep.b], anchor3[prep.c]),
                )
                points[lab] = tuple(f(p[lab]))
                witnesses[lab] = PointWitness(lab, BRANCH_STEP1, (), None,
                                              {"plane_affine_map": "abc -> a'b'c'"})
                continue
            try:
                off = _tprime_offplane(prep, lab, bp)
            except ValidationError as exc:
                # An internal construction failure is NOT an input-validity rejection.  The
                # label is reported UNSUPPORTED, the branch gate does not pass, and no point
                # is invented for it.
                unsupported.append(lab)
                witnesses[lab] = PointWitness(
                    lab, "INTERNAL_CONSTRUCTION_FAILURE", (), None, {},
                    completeness="UNSUPPORTED", note=str(exc))
                continue
            if off.ok:
                points[lab] = off.point
                witnesses[lab] = PointWitness(lab, off.branch, off.route, off.source_view, off.detail)
            else:
                unsupported.append(lab)
                witnesses[lab] = PointWitness(lab, off.branch, off.route, off.source_view, off.detail,
                                              completeness="UNSUPPORTED", note=off.reason)
    except BudgetExceeded as exc:
        return ConstructionResult(
            algorithm=algorithm, status=STATUS_BUDGET_EXCEEDED, points=points, witnesses=witnesses,
            anchors_out=anchor3, v=prep.v, t=prep.t, base_parameters=bp,
            unsupported=[lab for lab in prep.labels if lab not in points],
            failure_reason=(
                f"exact-operation budget exhausted: {exc}. Labels without a constructed point "
                "are reported as unresolved; no absent or substituted match is recorded."
            ),
        )
    except ValidationError as exc:
        # safety net only: arguments were validated above, so reaching here means an internal
        # construction failure.  It must NOT be reported as invalid input.
        unresolved = [lab for lab in prep.labels if lab not in points]
        for lab in unresolved:
            unsupported.append(lab)
            witnesses[lab] = PointWitness(lab, "INTERNAL_CONSTRUCTION_FAILURE", (), None, {},
                                          completeness="UNSUPPORTED", note=str(exc))
        return ConstructionResult(
            algorithm=algorithm, status=STATUS_UNSUPPORTED, points=points, witnesses=witnesses,
            anchors_out=anchor3, v=prep.v, t=prep.t, base_parameters=bp,
            unsupported=sorted(set(unsupported)),
            failure_reason="INTERNAL_CONSTRUCTION_FAILURE (not an input-validity rejection): "
                           f"{exc}",
        )

    status = STATUS_OK if not unsupported else STATUS_UNSUPPORTED
    reason = "" if not unsupported else (
        "declared UNSUPPORTED for labels " + ", ".join(sorted(unsupported))
        + "; their branch gate does not pass"
    )
    return ConstructionResult(
        algorithm=algorithm, status=status, points=points, witnesses=witnesses,
        anchors_out=anchor3, v=prep.v, t=prep.t, base_parameters=bp,
        unsupported=sorted(unsupported), failure_reason=reason,
        note="exactness is asserted only for the supplied rational coordinates; scale_status=unknown",
    )


def construct_output_image(prep: Prepared, anchors, *, base_parameters=(0, 1)) -> ConstructionResult:
    """AlgS0 (source pp.335-336): build the chosen output image.

    ``anchors`` is (A, B, C, E): four distinct rational 2D points with ABC non-collinear,
    the images of a, b, c, e.  Rank-deficient target maps are allowed (SOURCE-DERIVED U2);
    coincident output positions are kept per label, never merged.
    """
    bp = _base_pair(base_parameters)
    algorithm = "AlgS0"
    if not isinstance(anchors, (tuple, list)) or len(anchors) != 4:
        raise ValidationError("anchors must be (A, B, C, E)")
    A = _as_point2(anchors[0], "A")
    B = _as_point2(anchors[1], "B")
    C = _as_point2(anchors[2], "C")
    E = _as_point2(anchors[3], "E")
    if len({A, B, C, E}) != 4:
        raise ValidationError("output anchors A, B, C, E must be pairwise distinct")
    if _cross2(_sub2(B, A), _sub2(C, A)) == 0:
        raise ValidationError("output anchors A, B, C must not be collinear")

    p = prep.norm_first
    n_map = _affine_solve((p[prep.a], p[prep.b], p[prep.c]), (A, B, C))
    points: dict[str, tuple[Fraction, ...]] = {}
    witnesses: dict[str, PointWitness] = {}
    unsupported: list[str] = []
    anchor_out = {prep.a: A, prep.b: B, prep.c: C, prep.e: E}
    if prep.v == (ZERO, ZERO):
        return ConstructionResult(
            algorithm=algorithm, status=STATUS_NOT_OFF_PLANE, points={}, witnesses={},
            anchors_out=anchor_out, v=prep.v, t=prep.t, base_parameters=bp,
            failure_reason=(
                "v = 0: the two marked images are affinely equivalent, so the source's "
                "non-planar premise is not met and no output image is reported."
            ),
        )
    try:
        for lab in prep.labels:
            if lab == prep.e:
                points[lab] = E
                witnesses[lab] = PointWitness(
                    lab, BRANCH_INITIALIZATION, (), None,
                    {"recorded_choice": "e0 is the given output anchor E"},
                )
                continue
            if _in_plane(prep, lab, prep.a, prep.b, prep.c):
                points[lab] = tuple(n_map(p[lab]))
                witnesses[lab] = PointWitness(lab, BRANCH_STEP1, (), None,
                                              {"plane_affine_map": "abc -> A,B,C"})
                continue
            try:
                off = _s0_offplane(prep, lab, (A, B, C, E), bp)
            except ValidationError as exc:
                unsupported.append(lab)
                witnesses[lab] = PointWitness(
                    lab, "INTERNAL_CONSTRUCTION_FAILURE", (), None, {},
                    completeness="UNSUPPORTED", note=str(exc))
                continue
            if off.ok:
                points[lab] = off.point
                witnesses[lab] = PointWitness(lab, off.branch, off.route, off.source_view, off.detail)
            else:
                unsupported.append(lab)
                witnesses[lab] = PointWitness(lab, off.branch, off.route, off.source_view, off.detail,
                                              completeness="UNSUPPORTED", note=off.reason)
    except BudgetExceeded as exc:
        return ConstructionResult(
            algorithm=algorithm, status=STATUS_BUDGET_EXCEEDED, points=points, witnesses=witnesses,
            anchors_out=anchor_out, v=prep.v, t=prep.t, base_parameters=bp,
            unsupported=[lab for lab in prep.labels if lab not in points],
            failure_reason=f"exact-operation budget exhausted: {exc}",
        )
    except ValidationError as exc:
        unresolved = [lab for lab in prep.labels if lab not in points]
        for lab in unresolved:
            unsupported.append(lab)
            witnesses[lab] = PointWitness(lab, "INTERNAL_CONSTRUCTION_FAILURE", (), None, {},
                                          completeness="UNSUPPORTED", note=str(exc))
        return ConstructionResult(
            algorithm=algorithm, status=STATUS_UNSUPPORTED, points=points, witnesses=witnesses,
            anchors_out=anchor_out, v=prep.v, t=prep.t, base_parameters=bp,
            unsupported=sorted(set(unsupported)),
            failure_reason="INTERNAL_CONSTRUCTION_FAILURE (not an input-validity rejection): "
                           f"{exc}",
        )

    status = STATUS_OK if not unsupported else STATUS_UNSUPPORTED
    reason = "" if not unsupported else (
        "declared UNSUPPORTED for labels " + ", ".join(sorted(unsupported))
        + "; their branch gate does not pass"
    )
    return ConstructionResult(
        algorithm=algorithm, status=status, points=points, witnesses=witnesses,
        anchors_out=anchor_out, v=prep.v, t=prep.t, base_parameters=bp,
        unsupported=sorted(unsupported), failure_reason=reason,
        note="labels are preserved even when two labels share an output position (U2); "
             "scale_status=unknown",
    )
