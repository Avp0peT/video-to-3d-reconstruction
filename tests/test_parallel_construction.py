"""C-LABELLED-STEREO: branch-by-branch tests for AlgT' / AlgS0 (Kozlov 1996, pp.331-338).

Independence rules honoured here
--------------------------------
* The **truth** used for the geometric assertions is a fixed rational 3D body plus two
  directional projections, written down independently of the product.  The product never
  sees the truth.
* The affine map that carries the truth onto the product output is solved *in this file* by
  barycentric coordinates / Gaussian elimination -- a different route from the product's
  Cramer-style solver, so product and test do not share a solve.
* Specification section 8's algebraic control is re-derived here and used only as a
  *secondary* cross-check, after the truth assertions.  Two identical formulas agreeing is
  not evidence by itself.
* Branch names are asserted per point, and the geometry of the point is asserted as well:
  a name without a correct coordinate is not coverage.
"""

from __future__ import annotations

import copy
import json
from fractions import Fraction as F
from pathlib import Path

import pytest

from kozlov_lab.stereo.official import parallel_construction as pc

ROOT = Path(__file__).resolve().parents[1]
Z = F(0)
O = F(1)
Q = F(1, 4)


# --------------------------------------------------------------------------------------
# test-side exact linear algebra (independent implementation)
# --------------------------------------------------------------------------------------
def _sub(u, v):
    return tuple(a - b for a, b in zip(u, v))


def _det3(rows):
    (a, b, c), (d, e, f), (g, h, i) = rows
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


def _solve3(rows, rhs):
    """Exact 3x3 solve by Gauss-Jordan elimination (independent of the product solver)."""
    m = [list(rows[i]) + [rhs[i]] for i in range(3)]
    for col in range(3):
        piv = next((r for r in range(col, 3) if m[r][col] != 0), None)
        assert piv is not None, "singular system"
        m[col], m[piv] = m[piv], m[col]
        pv = m[col][col]
        m[col] = [x / pv for x in m[col]]
        for r in range(3):
            if r != col and m[r][col] != 0:
                fac = m[r][col]
                m[r] = [m[r][k] - fac * m[col][k] for k in range(4)]
    return [m[i][3] for i in range(3)]


def affine2(src, dst):
    """Affine R^2 -> R^d through three correspondences, via barycentric coordinates."""
    s0, s1, s2 = src
    u = _sub(s1, s0)
    w = _sub(s2, s0)
    det = u[0] * w[1] - u[1] * w[0]
    assert det != 0, "source triple must be a triangle"

    def f(p):
        v = _sub(p, s0)
        a = (v[0] * w[1] - v[1] * w[0]) / det
        b = (u[0] * v[1] - u[1] * v[0]) / det
        return tuple(
            dst[0][k] + a * (dst[1][k] - dst[0][k]) + b * (dst[2][k] - dst[0][k])
            for k in range(len(dst[0]))
        )

    return f


def affine3(src4, dst4):
    """Affine R^3 -> R^3 through four non-coplanar correspondences (Gauss-Jordan route).

    ``p = base + sum_k alpha_k * cols_k`` with ``alpha = M^-1 (p - base)``, and the image is
    ``dst4[0] + sum_k alpha_k * (dst4[k+1] - dst4[0])``.
    """
    base = src4[0]
    cols = [[src4[i][j] - base[j] for j in range(3)] for i in (1, 2, 3)]
    m = [[cols[c][r] for c in range(3)] for r in range(3)]
    assert _det3(m) != 0, "four source points must not be coplanar"
    delta = [_sub(dst4[i], dst4[0]) for i in (1, 2, 3)]

    def f(p):
        v = [p[k] - base[k] for k in range(3)]
        alpha = _solve3(m, v)
        return tuple(
            dst4[0][j] + sum(alpha[k] * delta[k][j] for k in range(3)) for j in range(3)
        )

    return f


def assert_marked_affine(truth, out, labels):
    """The output must be one single affine image of the truth body (marked correspondence)."""
    quad = None
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            for k in range(j + 1, len(labels)):
                for l in range(k + 1, len(labels)):
                    pts = [truth[labels[t]] for t in (i, j, k, l)]
                    cols = [[pts[t][c] - pts[0][c] for c in range(3)] for t in (1, 2, 3)]
                    if _det3([[cols[c][r] for c in range(3)] for r in range(3)]) != 0:
                        quad = (i, j, k, l)
                        break
                if quad:
                    break
            if quad:
                break
        if quad:
            break
    assert quad is not None, "truth has no four non-coplanar points"
    sel = [labels[t] for t in quad]
    phi = affine3([truth[l] for l in sel], [out[l] for l in sel])
    for lab in labels:
        assert phi(truth[lab]) == out[lab], f"label {lab} is not on the single affine image"


def views_from_truth(body, v):
    """Two directional projections of a body already written in the recorded gauge.

    First projection is (x, y), second is (x + z*vx, y + z*vy); the auxiliary direction is
    vertical.  This is the *test's* construction of the input, not the product's.
    """
    vx, vy = v
    first = {k: (p[0], p[1]) for k, p in body.items()}
    second = {k: (p[0] + p[2] * vx, p[1] + p[2] * vy) for k, p in body.items()}
    return first, second


def s0_expected(body, anchors):
    """Expected output image: the truth body projected along the direction the anchors pin.

    With N the affine map (0,0),(1,0),(0,1) -> A,B,C and w = E - N(p_e), the target image is
    x -> N(x, y) + z * w.  Linear in the 3D point, so it is a genuine parallel projection;
    it maps the truth anchors onto A, B, C, E by construction.
    """
    A, B, C, E = anchors
    nmap = affine2(((Z, Z), (O, Z), (Z, O)), (A, B, C))
    pe = body["e"]
    w = _sub(E, nmap((pe[0], pe[1])))
    out = {}
    for lab, p in body.items():
        n = nmap((p[0], p[1]))
        out[lab] = tuple(n[i] + p[2] * w[i] for i in range(2))
    return out


def spec8_body(first, second, anchors=("a", "b", "c", "e")):
    """Specification section 8's independent algebraic control (test-side only)."""
    a, b, c, e = anchors
    tri = ((Z, Z), (O, Z), (Z, O))
    nf = affine2((first[a], first[b], first[c]), tri)
    ns = affine2((second[a], second[b], second[c]), tri)
    p = {k: nf(v) for k, v in first.items()}
    q = {k: ns(v) for k, v in second.items()}
    v = _sub(q[e], p[e])
    body = {}
    for k in first:
        d = _sub(q[k], p[k])
        t = d[0] / v[0] if v[0] != 0 else d[1] / v[1]
        body[k] = (p[k][0], p[k][1], t)
    return body


def spec8_output(body, anchors):
    return s0_expected(body, anchors)


ANCHORS = ((Z, Z), (O, Z), (Z, O), (F(2), O))


# --------------------------------------------------------------------------------------
# fixed rational truth bodies, one per source branch
# --------------------------------------------------------------------------------------
ABC = {"a": (Z, Z, Z), "b": (O, Z, Z), "c": (Z, O, Z)}


def truth(**extra):
    body = dict(ABC)
    body.update(extra)
    return body


CASES = {
    # name: (v, body, expected branch for the off-plane label, expected route)
    "step1": ((1, 2), truth(d=(F(2), F(3), Z), e=(Q, Q, O)), pc.BRANCH_STEP1, ()),
    "lemma6_6_1_single_view_collinear": (
        (1, 2), truth(d=(F(1, 2), F(-1, 2), F(1, 2)), e=(Q, Q, O)),
        pc.BRANCH_LEMMA6_SINGLE_VIEW_COLLINEAR, ()),
    "lemma6_6_2_common_point_a": (
        (1, 2), truth(d=(Q, F(3), O), e=(Q, Q, O)), pc.BRANCH_LEMMA6_COMMON_POINT_A, ()),
    "lemma6_6_2_common_point_b": (
        (1, 2), truth(d=(F(3, 4), F(-1, 4), O), e=(Q, Q, O)),
        pc.BRANCH_LEMMA6_COMMON_POINT_B, ()),
    "lemma6_6_2_common_point_renamed_face": (
        (1, 2), truth(d=(O, F(3), F(2)), e=(Q, F(3, 2), O)),
        pc.BRANCH_LEMMA6_COMMON_POINT_RENAMED, (pc.BRANCH_STEP2B_REDUCE_AED,)),
    "lemma6_6_4_rename_ab": (
        (1, 2), truth(d=(O, F(1, 2), O), e=(Q, Q, O)),
        pc.BRANCH_LEMMA6_GENERAL_INTERSECTION, ()),
    "step2a_ce": (
        (1, 2), truth(d=(F(1, 8), F(5, 8), F(1, 2)), e=(Q, Q, O)), pc.BRANCH_STEP2A_CE, ()),
    "step2a_ae": (
        (0, 2), truth(d=(Z, F(3, 2), F(1, 2)), e=(Z, F(3), O)), pc.BRANCH_STEP2A_AE,
        (pc.BRANCH_STEP2B_EDGE_TEST,)),
    "step2a_be": (
        (1, -1), truth(d=(F(5, 8), F(3, 8), F(1, 2)), e=(Q, F(3, 4), O)), pc.BRANCH_STEP2A_BE,
        (pc.BRANCH_STEP2B_EDGE_TEST,)),
    "step2b_reduce_face_aed": (
        (1, 2), truth(d=(O, F(3), F(3)), e=(Q, F(3, 2), O)),
        pc.BRANCH_LEMMA6_GENERAL_INTERSECTION, (pc.BRANCH_STEP2B_REDUCE_AED,)),
}

WITNESS = (ROOT / "outputs" / "delegation" / "STEREO_SOURCE" / "20261007T140000Z_main"
           / "PROJECTION_WITNESS.json")


def witness_inputs():
    w = json.loads(WITNESS.read_text(encoding="utf-8"))
    first = {k: tuple(F(x) for x in v) for k, v in w["first"].items()}
    second = {k: tuple(F(x) for x in v) for k, v in w["second"].items()}
    body = {k: tuple(F(x) for x in v) for k, v in w["body"].items()}
    target = {k: tuple(F(x) for x in v) for k, v in w["target"].items()}
    return first, second, body, target


# --------------------------------------------------------------------------------------
# branch tests: AlgT' truth, AlgS0 truth, AlgS0-against-witness, branch names
# --------------------------------------------------------------------------------------
@pytest.mark.gate
@pytest.mark.parametrize("name", sorted(CASES))
def test_branch_constructs_the_truth_body_and_target_image(name):
    v, body, branch, route = CASES[name]
    labels = tuple(body)
    first, second = views_from_truth(body, v)
    prep = pc.prepare_views(first, second, "a", "b", "c", "e")

    rep = pc.construct_representative(prep)
    assert rep.status == pc.STATUS_OK, (name, rep.status, rep.failure_reason)
    assert rep.unsupported == []
    # exact coordinates in the recorded gauge: the truth is already in that gauge
    for lab in labels:
        assert rep.points[lab] == body[lab], (name, lab, rep.points[lab], body[lab])
    assert_marked_affine(body, rep.points, labels)
    assert_reprojections(first, second, rep.points, labels)
    assert rep.witnesses["d"].branch == branch, (name, rep.witnesses["d"].branch)
    assert rep.witnesses["d"].route == route, (name, rep.witnesses["d"].route)
    assert rep.witnesses["d"].completeness == "CONSTRUCTED"

    out = pc.construct_output_image(prep, ANCHORS)
    assert out.status == pc.STATUS_OK, (name, out.status, out.failure_reason)
    want = s0_expected(body, ANCHORS)
    for lab in labels:
        assert out.points[lab] == want[lab], (name, lab, out.points[lab], want[lab])
    assert out.witnesses["d"].branch == branch, (name, out.witnesses["d"].branch)
    assert out.witnesses["d"].route == route, (name, out.witnesses["d"].route)

    if branch == pc.BRANCH_LEMMA6_COMMON_POINT_RENAMED:
        # Independent construction: cde projects to a line in both views, while
        # plane(aed) is y = 3*z/2 and contains b.  Its intersection with abc is ab.
        assert len(set(first.values())) == len(labels)
        assert len(set(second.values())) == len(labels)
        assert first["e"] == (Q, F(3, 2)) and second["e"] == (F(5, 4), F(7, 2))
        assert first["d"] == (O, F(3)) and second["d"] == (F(3), F(7))
        assert rep.points["d"] == (O, F(3), F(2))
        assert out.points["d"] == (F(9, 2), F(2))
        for result in (rep, out):
            detail = result.witnesses["d"].detail["line"]["detail"]
            assert detail["face_known_vertex"] == "a"
            assert detail["probe"] == "b"
            assert detail["free_vertices"] == ["b", "c"]

    # secondary cross-check: spec section 8's algebraic control, computed here only
    ctrl = spec8_body(first, second)
    for lab in labels:
        assert ctrl[lab] == rep.points[lab], (name, lab, "control", ctrl[lab])
    ctrl_out = spec8_output(ctrl, ANCHORS)
    for lab in labels:
        assert ctrl_out[lab] == out.points[lab], (name, lab, "control", ctrl_out[lab])


def test_witness_artifact_is_reproduced_exactly():
    """The pre-existing PROJECTION_WITNESS (independent script, recorded hash) is reproduced.

    It exercises Lemma 6 case 3 (d) and case 4 (g, h) and an AlgS0 target whose cde face maps
    onto a line, with two non-anchor labels sharing an output position.
    """
    first, second, body, target = witness_inputs()
    prep = pc.prepare_views(first, second, "a", "b", "c", "e")
    assert prep.v == (O, F(2))
    rep = pc.construct_representative(prep)
    assert rep.status == pc.STATUS_OK
    for lab in body:
        assert rep.points[lab] == body[lab], (lab, rep.points[lab], body[lab])
    assert rep.witnesses["d"].branch == pc.BRANCH_LEMMA6_PARALLEL_EDGES
    methods = {rep.witnesses[l].detail["line"]["detail"]["second_point_method"] for l in ("g", "h")}
    assert methods == {"de_intersection"}
    for lab in ("g", "h"):
        assert rep.witnesses[lab].branch == pc.BRANCH_LEMMA6_GENERAL_INTERSECTION

    out = pc.construct_output_image(prep, ((Z, Z), (O, Z), (Z, O), (F(2), O)))
    assert out.status == pc.STATUS_OK
    for lab in target:
        assert out.points[lab] == target[lab], (lab, out.points[lab], target[lab])
    assert out.points["g"] == out.points["h"]
    assert len(out.points) == len(target) == 7


def test_witness_output_cde_is_degenerate_but_anchors_stay_distinct():
    """U2: the target cde face collapses onto a line while A, B, C, E stay distinct."""
    first, second, _body, target = witness_inputs()
    prep = pc.prepare_views(first, second, "a", "b", "c", "e")
    out = pc.construct_output_image(prep, ((Z, Z), (O, Z), (Z, O), (F(2), O)))
    c, d, e = out.points["c"], out.points["d"], out.points["e"]
    cross = (d[0] - c[0]) * (e[1] - c[1]) - (d[1] - c[1]) * (e[0] - c[0])
    assert cross == 0
    for lab in ("a", "b", "c", "e"):
        assert out.points[lab] == out.anchors_out[lab]
    assert len({out.points[l] for l in ("a", "b", "c", "e")}) == 4


@pytest.mark.gate
def test_all_required_source_branches_are_reached_with_geometric_assertions():
    """Reachability: every required branch must be *executed* and its point must be correct.

    A predicted-UNSUPPORTED test passing is not branch completion, so this test additionally
    requires the observed branch to carry a correct exact coordinate.
    """
    reached = set()
    for name, (v, body, branch, route) in CASES.items():
        first, second = views_from_truth(body, v)
        prep = pc.prepare_views(first, second, "a", "b", "c", "e")
        rep = pc.construct_representative(prep)
        assert rep.status == pc.STATUS_OK, name
        assert rep.points["d"] == body["d"], name
        reached.add(rep.witnesses["d"].branch)
    first, second, body, _target = witness_inputs()
    prep = pc.prepare_views(first, second, "a", "b", "c", "e")
    rep = pc.construct_representative(prep)
    for lab in ("d", "g", "h"):
        assert rep.points[lab] == body[lab]
        reached.add(rep.witnesses[lab].branch)

    required = {
        pc.BRANCH_STEP1,
        pc.BRANCH_LEMMA6_SINGLE_VIEW_COLLINEAR,
        pc.BRANCH_LEMMA6_COMMON_POINT_A,
        pc.BRANCH_LEMMA6_COMMON_POINT_B,
        pc.BRANCH_LEMMA6_COMMON_POINT_RENAMED,
        pc.BRANCH_LEMMA6_PARALLEL_EDGES,
        pc.BRANCH_LEMMA6_GENERAL_INTERSECTION,
        pc.BRANCH_STEP2A_CE,
        pc.BRANCH_STEP2A_AE,
        pc.BRANCH_STEP2A_BE,
    }
    assert required <= reached, sorted(required - reached)
    # no witness may carry a branch name outside the declared source vocabulary
    known = required | {
        pc.BRANCH_INITIALIZATION,
        pc.BRANCH_LEMMA6_COMMON_POINT_RENAMED,
        pc.BRANCH_STEP2B_REDUCE_AED,
        pc.BRANCH_STEP2B_REDUCE_BED,
        pc.BRANCH_STEP2B_EDGE_TEST,
    }
    for lab in ("a", "b", "c", "d", "e", "g", "h"):
        assert rep.witnesses[lab].branch in known


@pytest.mark.gate
def test_every_declared_route_marker_is_reached():
    """REDUCE_FACE_AED and REDUCE_FACE_BED are both reachable and both carry exact geometry."""
    routes = set()
    values = []
    bodies = []
    for perm, name in ROLE_PERMS:
        bodies.append((f"bed:{name}", _role_case(perm, v=(Z, F(2)), vertices=BED_VERTICES)))
        bodies.append((f"cex:{name}", _role_case(perm)))
    for name, (body, (first, second)) in bodies:
        prep = pc.prepare_views(first, second, "a", "b", "c", "e")
        rep = pc.construct_representative(prep)
        assert rep.status == pc.STATUS_OK, (name, rep.failure_reason)
        assert_marked_affine(body, rep.points, tuple(body))
        values.append(True)
        routes.add(rep.witnesses["d"].route)
    assert any(pc.BRANCH_STEP2B_REDUCE_AED in r for r in routes), routes
    assert any(pc.BRANCH_STEP2B_REDUCE_BED in r for r in routes), routes
    assert all(values)


# --------------------------------------------------------------------------------------
# predicates, planar bodies, incompatibility, input hygiene
# --------------------------------------------------------------------------------------
def test_e_inside_the_drawn_triangle_but_off_plane_is_accepted():
    """`off plane` is a two-view predicate: e may sit inside the drawn triangle in both images."""
    v = (F(1, 2), F(-1, 8))
    body = truth(d=(O, F(2), F(1, 2)), e=(Q, Q, O))
    first, second = views_from_truth(body, v)
    for view in (first, second):
        x, y = view["e"]
        assert x > 0 and y > 0 and x + y < 1, (view["e"], "e must be inside triangle abc")
    prep = pc.prepare_views(first, second, "a", "b", "c", "e")
    assert prep.v != (Z, Z), "e must still be off the abc plane"
    assert prep.t["e"] == O
    rep = pc.construct_representative(prep)
    assert rep.status == pc.STATUS_OK
    for lab in body:
        assert rep.points[lab] == body[lab]


def _kernel_of_projection(rows):
    (a, b, c), (d, e, f) = rows
    return (b * f - c * e, c * d - a * f, a * e - b * d)


def test_planar_body_from_two_different_directions_is_reported_not_reconstructed():
    """A planar body's two directional images are affinely equivalent: premise unmet."""
    p1, p2 = ((1, 0, 0), (0, 0, 1)), ((1, 1, 0), (0, 1, 1))
    k1, k2 = _kernel_of_projection(p1), _kernel_of_projection(p2)
    assert k1 != (0, 0, 0) and k2 != (0, 0, 0)
    # the two kernels are different *directions* (not just different scalings)
    assert k1[1] * k2[2] - k1[2] * k2[1] != 0 or k1[2] * k2[0] - k1[0] * k2[2] != 0
    plane = {"a": (0, 0, 0), "b": (1, 0, 0), "c": (0, 0, 1), "e": (2, 0, 3), "d": (3, 0, 5)}
    first, second = {}, {}
    for lab, (x, y, z) in plane.items():
        first[lab] = (F(x), F(z))
        second[lab] = (F(x + y), F(z + y))
    assert first == second, "a planar body gives a'-equivalent images for both directions"
    prep = pc.prepare_views(first, second, "a", "b", "c", "e")
    assert prep.v == (Z, Z)
    rep = pc.construct_representative(prep)
    assert rep.status == pc.STATUS_NOT_OFF_PLANE
    assert rep.points == {} and rep.anchors_out == {}
    assert "premise" in rep.failure_reason
    out = pc.construct_output_image(prep, ANCHORS)
    assert out.status == pc.STATUS_NOT_OFF_PLANE
    assert out.points == {}


def test_wrong_label_swap_and_incompatible_sets_are_rejected():
    first, second, _body, _target = witness_inputs()
    swapped = dict(second)
    swapped["g"], swapped["h"] = second["h"], second["g"]
    with pytest.raises(pc.ValidationError):
        pc.prepare_views(first, swapped, "a", "b", "c", "e")
    with pytest.raises(pc.ValidationError):
        pc.prepare_views(first, {k: v for k, v in second.items() if k != "h"},
                         "a", "b", "c", "e")
    with pytest.raises(pc.ValidationError):
        pc.prepare_views(first, second, "a", "b", "z", "e")


def test_anchor_degeneracy_in_algS0_is_rejected():
    first, second, _body, _target = witness_inputs()
    prep = pc.prepare_views(first, second, "a", "b", "c", "e")
    with pytest.raises(pc.ValidationError):
        pc.construct_output_image(prep, ((Z, Z), (O, Z), (F(2), Z), (F(3), O)))
    with pytest.raises(pc.ValidationError):
        pc.construct_output_image(prep, ((Z, Z), (Z, Z), (Z, O), (F(2), O)))


def test_inputs_are_never_mutated():
    first, second, body, _target = witness_inputs()
    f0, s0 = copy.deepcopy(first), copy.deepcopy(second)
    prep = pc.prepare_views(first, second, "a", "b", "c", "e")
    pc.construct_representative(prep)
    pc.construct_output_image(prep, ANCHORS)
    assert first == f0 and second == s0
    for name, (v, truth_body, _b, _r) in CASES.items():
        f1, s1 = views_from_truth(truth_body, v)
        g1, t1 = copy.deepcopy(f1), copy.deepcopy(s1)
        p = pc.prepare_views(f1, s1, "a", "b", "c", "e")
        pc.construct_representative(p)
        pc.construct_output_image(p, ANCHORS)
        assert f1 == g1 and s1 == t1, name


# --------------------------------------------------------------------------------------
# U1: base-point / parameter invariance, and the controlled negative example
# --------------------------------------------------------------------------------------
@pytest.mark.gate
def test_algT_is_independent_of_the_chosen_intersection_line_bases():
    """U1: changing the two base points and their parameters must not change the result."""
    first, second, body, _target = witness_inputs()
    prep = pc.prepare_views(first, second, "a", "b", "c", "e")
    base_sets = [(0, 1), (2, 5), (-3, 7), (F(1, 3), F(4, 7))]
    results = []
    recorded = []
    for bp in base_sets:
        rep = pc.construct_representative(prep, base_parameters=bp)
        assert rep.status == pc.STATUS_OK
        results.append(rep.points)
        recorded.append(tuple(tuple(x) for x in rep.witnesses["d"].detail["base_points"]))
    for other in results[1:]:
        assert other == results[0]
    assert results[0] == body
    assert len(set(recorded)) > 1, "base points must actually differ for the check to bite"


def test_negative_control_setwise_line_correspondence_is_not_enough():
    """U1 is load-bearing: fixing L only as a *set* leaves a degree of freedom.

    The wrong constructor below is written entirely in this test with the test's own affine
    solver: it fixes one base point of L, then absorbs the leftover freedom with a constraint
    point that is *not* on L.  The correct parameter-preserving construction must differ from
    it, and only the correct one may match the truth.
    """
    first, second, body, _target = witness_inputs()
    prep = pc.prepare_views(first, second, "a", "b", "c", "e")
    # independent value of L for this case: the abc plane is z = 0 and cde is y + 3z/4 = 1
    line_a, line_b = (Z, O), (O, O)
    lc = pc._lemma6_line(prep, "d", "c", "e")
    assert pc._on_line((lc.point, lc.direction), line_a)
    assert pc._on_line((lc.point, lc.direction), line_b), "independent L disagrees"

    qe, qd = prep.norm_second["e"], prep.norm_second["d"]
    pe = prep.norm_first["e"]
    dst_line = (line_a + (Z,), line_b + (Z,))
    correct = affine2((line_a, line_b, qe), (dst_line[0], dst_line[1], (pe[0], pe[1], O)))
    off = (line_b[0] + (line_b[1] - line_a[1]), line_b[1] - (line_b[0] - line_a[0]))
    assert not pc._on_line((lc.point, lc.direction), off), "the negative-control point must be off L"
    wrong = affine2((line_a, off, qe), (dst_line[0], (off[0], off[1], Z), (pe[0], pe[1], O)))

    assert correct(qd) == body["d"]
    assert wrong(qd) != correct(qd)
    assert wrong(qd) != body["d"]
    assert pc.construct_representative(prep).points["d"] == body["d"]


# --------------------------------------------------------------------------------------
# general gauge: affine equivalence only, and exact re-projection of both input images
# --------------------------------------------------------------------------------------
def assert_reprojections(first, second, points, labels):
    """Independent affine re-projection check of both input images through the output."""
    a, b, c, e = "a", "b", "c", "e"
    phi1 = affine2((first[a], first[b], first[c]),
                   tuple(tuple(points[l][:2]) for l in (a, b, c)))
    for lab in labels:
        assert phi1(first[lab]) == tuple(points[lab][:2]), lab
    nf = affine2((first[a], first[b], first[c]), ((Z, Z), (O, Z), (Z, O)))
    ns = affine2((second[a], second[b], second[c]), ((Z, Z), (O, Z), (Z, O)))
    v = _sub(ns(second[e]), nf(first[e]))
    assert v != (Z, Z)
    proj2 = {lab: (points[lab][0] + points[lab][2] * v[0],
                   points[lab][1] + points[lab][2] * v[1]) for lab in labels}
    phi2 = affine2((second[a], second[b], second[c]),
                   tuple(tuple(proj2[l]) for l in (a, b, c)))
    for lab in labels:
        assert phi2(second[lab]) == tuple(proj2[lab]), lab


def test_body_in_an_arbitrary_gauge_is_recovered_only_up_to_marked_affine():
    """A truth body outside the recorded gauge: output must be one single affine image of it."""
    gauge_src = ((Z, Z, Z), (O, Z, Z), (Z, O, Z), (Z, Z, O))
    gauge_dst = ((Z, Z, Z), (O, Z, Z), (Z, O, Z), (Q, F(1, 2), F(3, 2)))
    gauge = affine3(gauge_src, gauge_dst)
    canon = truth(d=(F(1, 8), F(5, 8), F(1, 2)), e=(Q, Q, O), f=(F(2), F(3), Z))
    body = {k: gauge(p) for k, p in canon.items()}
    labels = tuple(body)
    first, second = views_from_truth(body, (F(2), F(3)))
    prep = pc.prepare_views(first, second, "a", "b", "c", "e")
    # the gauge scales the third coordinate, so the recorded direction is a multiple of it
    assert prep.v == (F(3), F(9, 2))
    assert prep.t["e"] == O

    rep = pc.construct_representative(prep)
    assert rep.status == pc.STATUS_OK, rep.failure_reason
    assert rep.unsupported == []
    assert rep.points != body, "this truth is deliberately not in the recorded gauge"
    assert_marked_affine(body, rep.points, labels)
    # the product's own image must re-project onto both input images; the raw truth need not
    # (it is a different, non-canonical representative whose projections differ affinely)
    assert_reprojections(first, second, rep.points, labels)
    out = pc.construct_output_image(prep, ANCHORS)
    assert out.status == pc.STATUS_OK
    want = s0_expected(rep.points, ANCHORS)
    for lab in labels:
        assert out.points[lab] == want[lab], lab


# --------------------------------------------------------------------------------------
# budget and illegal values
# --------------------------------------------------------------------------------------
def test_exact_operation_budget_exhaustion_is_explicit_and_not_a_fake_result():
    first, second, body, _target = witness_inputs()
    prep = pc.prepare_views(first, second, "a", "b", "c", "e")
    prep.budget.left = 4
    rep = pc.construct_representative(prep)
    assert rep.status == pc.STATUS_BUDGET_EXCEEDED
    assert set(rep.points) | set(rep.unsupported) == set(body)
    assert set(rep.points) & set(rep.unsupported) == set()
    assert "budget" in rep.failure_reason
    for lab, pt in rep.points.items():
        assert pt == body[lab], "no substituted or wrong point may be recorded"
    with pytest.raises(pc.BudgetExceeded):
        pc.prepare_views(first, second, "a", "b", "c", "e", op_budget=0)


def test_illegal_inputs_are_rejected_without_quantisation():
    first, second, _body, _target = witness_inputs()
    as_float = {k: (float(v[0]), float(v[1])) for k, v in first.items()}
    with pytest.raises(pc.ValidationError):
        pc.prepare_views(as_float, second, "a", "b", "c", "e")
    as_bool = dict(first, a=(True, F(0)))
    with pytest.raises(pc.ValidationError):
        pc.prepare_views(as_bool, second, "a", "b", "c", "e")
    dup = dict(first, zz=first["g"])
    with pytest.raises(pc.ValidationError):
        pc.prepare_views(dup, dict(second, zz=second["g"]), "a", "b", "c", "e")
    big = dict(first, zz=(F(10 ** 13), Z))
    with pytest.raises(pc.ValidationError):
        pc.prepare_views(big, dict(second, zz=(Z, Z)), "a", "b", "c", "e")
    small = {k: first[k] for k in ("a", "b", "c")}
    with pytest.raises(pc.ValidationError):
        pc.prepare_views(small, {k: second[k] for k in ("a", "b", "c")}, "a", "b", "c", "e")
    wide = {f"p{i}": (F(i), F(i + 1)) for i in range(33)}
    with pytest.raises(pc.ValidationError):
        pc.prepare_views(wide, wide, "p0", "p1", "p2", "p3")


def test_scale_status_is_unknown_and_no_metric_claim_is_made():
    first, second, _body, _target = witness_inputs()
    prep = pc.prepare_views(first, second, "a", "b", "c", "e")
    rep = pc.construct_representative(prep)
    assert rep.scale_status == "unknown"
    assert "scale_status=unknown" in rep.note


# ======================================================================================
# R2: mainline counterexamples, reachable BED, role substitution, U1 domain
# ======================================================================================
COUNTEREXAMPLES = (ROOT / "outputs" / "delegation" / "ABC_REVIEW" / "20261008_main"
                   / "COUNTEREXAMPLES.json")


def _cex_block(name):
    data = json.loads(COUNTEREXAMPLES.read_text(encoding="utf-8"))[name]
    truth = {k: tuple(F(x) for x in v) for k, v in data["truth"].items()}
    first = {k: tuple(F(x) for x in v) for k, v in data["first"].items()}
    second = {k: tuple(F(x) for x in v) for k, v in data["second"].items()}
    return data, truth, first, second


@pytest.mark.gate
def test_mainline_counterexample_valid_input_is_no_longer_rejected():
    """The mainline counterexample: a *valid* input that R1 wrongly rejected.

    Truth a=(0,0,0), b=(1,0,0), c=(0,1,0), e=(1/4,3/2,1), d=(5/4,7/2,1) with images (x,y)
    and (x+z, y+2z).  Both algorithms must return OK and the exact expected geometry: the
    representative d is the truth, and with A,B,C,E = (0,0),(1,0),(0,1),(2,1) the output d
    is (3,3).  The R1 failure came from fixing the global a/b after the face was renamed.
    """
    data, truth, first, second = _cex_block("c_counterexample")
    labels = tuple(truth)
    assert data["independent_facts"]["first_injective"]
    assert data["independent_facts"]["second_injective"]
    assert data["independent_facts"]["expected_representative_d"] == ["5/4", "7/2", "1"]
    assert data["independent_facts"]["expected_output_d"] == ["3", "3"]

    prep = pc.prepare_views(first, second, "a", "b", "c", "e")
    assert prep.v == (O, F(2))
    assert prep.t["d"] == O

    rep = pc.construct_representative(prep)
    assert rep.status == pc.STATUS_OK, (rep.status, rep.failure_reason)
    assert rep.unsupported == []
    for lab in labels:
        assert rep.points[lab] == truth[lab], (lab, rep.points[lab], truth[lab])
    assert_marked_affine(truth, rep.points, labels)
    assert_reprojections(first, second, rep.points, labels)
    w = rep.witnesses["d"]
    assert w.completeness == "CONSTRUCTED"
    assert w.branch == pc.BRANCH_LEMMA6_GENERAL_INTERSECTION
    assert w.route == (pc.BRANCH_STEP2B_REDUCE_AED,)
    # the auxiliary edge after the rename to (a,e,d) is bc, not the global ab: this is the
    # exact record that the R1 defect (hard-coded a/b) is gone.
    assert w.detail["line"]["detail"]["second_point_method"] == "aux_edge_fixed_point(bc)"

    out = pc.construct_output_image(prep, ANCHORS)
    assert out.status == pc.STATUS_OK, (out.status, out.failure_reason)
    assert out.points["d"] == (F(3), F(3))
    want = s0_expected(truth, ANCHORS)
    for lab in labels:
        assert out.points[lab] == want[lab], (lab, out.points[lab], want[lab])


@pytest.mark.gate
def test_bed_face_rotation_is_a_reachable_success_branch():
    """BED is reachable and succeeds: aed is collinear in *both* images, bed is not.

    A 3D face need not project to a triangle, so the aed route can be unavailable while the
    configuration is perfectly legal.  The geometry is asserted against the 3D truth, not
    just the branch name.
    """
    data, truth, first, second = _cex_block("c_bed_valid_case")
    labels = tuple(truth)
    prep = pc.prepare_views(first, second, "a", "b", "c", "e")
    assert prep.v == (Z, F(2))
    assert prep.t["d"] == F(3)

    rep = pc.construct_representative(prep)
    assert rep.status == pc.STATUS_OK, (rep.status, rep.failure_reason)
    for lab in labels:
        assert rep.points[lab] == truth[lab], (lab, rep.points[lab], truth[lab])
    assert_marked_affine(truth, rep.points, labels)
    assert_reprojections(first, second, rep.points, labels)
    w = rep.witnesses["d"]
    assert w.route == (pc.BRANCH_STEP2B_REDUCE_AED, pc.BRANCH_STEP2B_REDUCE_BED)
    assert w.completeness == "CONSTRUCTED"

    out = pc.construct_output_image(prep, ANCHORS)
    assert out.status == pc.STATUS_OK, (out.status, out.failure_reason)
    want = s0_expected(truth, ANCHORS)
    for lab in labels:
        assert out.points[lab] == want[lab], (lab, out.points[lab], want[lab])
    assert data["representative"]["status"] == "OK"


CANON_VERTICES = {
    "A": (Z, Z, Z),
    "B": (O, Z, Z),
    "C": (Z, O, Z),
    "E": (F(1, 4), F(3, 2), O),
    "D": (F(5, 4), F(7, 2), O),
}
ROLE_PERMS = [(("A", "B", "C"), "identity"), (("B", "C", "A"), "rotate-left"),
              (("C", "A", "B"), "rotate-right"), (("A", "C", "B"), "swap-bc"),
              (("C", "B", "A"), "swap-ac")]


def _role_case(perm, v=(O, F(2)), vertices=None):
    """Assign the canonical vertices to the roles a, b, c (kept as a triangle), e, d."""
    verts = CANON_VERTICES if vertices is None else vertices
    va, vb, vc = perm
    body = {"a": verts[va], "b": verts[vb], "c": verts[vc], "e": verts["E"], "d": verts["D"]}
    return body, views_from_truth(body, v)


@pytest.mark.gate
def test_abc_role_permutations_recover_the_same_body():
    """Every legal a/b/c role assignment must give a marked affine image of the same truth.

    The recorded gauge changes with the roles, so the assertion is the marked-affine one plus
    the two re-projections, not coordinate equality.  This is what catches a hard-coded
    global a/b in Lemma 6.
    """
    for perm, name in ROLE_PERMS:
        body, (first, second) = _role_case(perm)
        labels = tuple(body)
        prep = pc.prepare_views(first, second, "a", "b", "c", "e")
        rep = pc.construct_representative(prep)
        assert rep.status == pc.STATUS_OK, (name, rep.status, rep.failure_reason)
        assert rep.unsupported == [], name
        assert_marked_affine(body, rep.points, labels)
        assert_reprojections(first, second, rep.points, labels)
        assert rep.witnesses["d"].completeness == "CONSTRUCTED", name
        out = pc.construct_output_image(prep, ANCHORS)
        assert out.status == pc.STATUS_OK, (name, out.status, out.failure_reason)
        # role-dependent gauge: the output is the anchor-pinned projection of the produced body
        for lab in labels:
            assert out.points[lab] == s0_expected(rep.points, ANCHORS)[lab], (name, lab)


BED_VERTICES = {
    "A": (Z, Z, Z), "B": (O, Z, Z), "C": (Z, O, Z),
    "E": (Z, F(3), O), "D": (Z, F(5), F(3)),
}


@pytest.mark.gate
def test_degenerate_face_times_case4_combinations_stay_exact():
    """Degenerate face rotation combined with Lemma 6 case 4, across role assignments.

    For the identity role assignment aed is collinear in both images, so BED is used.
    The coincident de images have a unique face-map fixed point (0, 2), giving the
    de_coincident_fixed_point method.  The separate mainline counterexample checks
    the auxiliary-edge fallback; this BED example does not establish that coverage.
    """
    for perm, name in ROLE_PERMS:
        body, (first, second) = _role_case(perm, v=(Z, F(2)), vertices=BED_VERTICES)
        labels = tuple(body)
        prep = pc.prepare_views(first, second, "a", "b", "c", "e")
        rep = pc.construct_representative(prep)
        assert rep.status == pc.STATUS_OK, (name, rep.status, rep.failure_reason)
        assert rep.unsupported == [], (name, rep.unsupported)
        assert_marked_affine(body, rep.points, labels)
        assert_reprojections(first, second, rep.points, labels)
        if name == "identity":
            # the recorded gauge is the identity only for this role assignment
            assert rep.points["d"] == BED_VERTICES["D"]
            assert rep.witnesses["d"].detail["line"]["detail"]["second_point_method"] == "de_coincident_fixed_point"
        out = pc.construct_output_image(prep, ANCHORS)
        assert out.status == pc.STATUS_OK, (name, out.status, out.failure_reason)
        for lab in labels:
            assert out.points[lab] == s0_expected(rep.points, ANCHORS)[lab], (name, lab)


@pytest.mark.gate
def test_u1_base_line_is_never_degenerate_in_the_reachable_domain(record_property):
    """U1's guard is provably unreachable in the STEP2 branch (premises in the contract).

    This asserts the premise on the final bytes: for every reachable branch case the recorded
    U1 base pair plus e's source image form a triangle, and no witness carries the U1
    degeneracy rejection.  It is an assertion about the reachable domain, not a claim that a
    predicted-UNSUPPORTED test counts as branch coverage.
    """
    checks = 0
    cases = []
    for name, (v, body, _b, _r) in CASES.items():
        cases.append((name, body, v))
    for perm, name in ROLE_PERMS:
        body, _views = _role_case(perm)
        cases.append((f"perm:{name}", body, (O, F(2))))
        body_bed, _views = _role_case(perm, v=(Z, F(2)), vertices=BED_VERTICES)
        cases.append((f"bed:{name}", body_bed, (Z, F(2))))
    _data, truth_cex, first_cex, second_cex = _cex_block("c_counterexample")
    cases.append(("counterexample", truth_cex, (O, F(2))))

    for name, body, v in cases:
        first, second = views_from_truth(body, v)
        prep = pc.prepare_views(first, second, "a", "b", "c", "e")
        rep = pc.construct_representative(prep)
        for lab, w in rep.witnesses.items():
            assert "degeneracy" not in (w.note or ""), (name, lab, w.note)
            detail = w.detail.get("line", {}) if isinstance(w.detail, dict) else {}
            if not detail:
                continue
            base = w.detail.get("base_points")
            if not base:
                continue
            b1 = tuple(F(x) for x in base[0])
            b2 = tuple(F(x) for x in base[1])
            coord = (prep.norm_second if w.source_view == "second" else prep.norm_first)[prep.e]
            cross = ((b2[0] - b1[0]) * (coord[1] - b1[1])
                     - (b2[1] - b1[1]) * (coord[0] - b1[0]))
            assert cross != 0, (name, lab, "U1 base line degenerated")
            checks += 1
    record_property("u1_input_configurations", len(cases))
    record_property("u1_non_degenerate_baseline_checks", checks)
    assert checks >= 12, checks


def test_failure_expressions_are_distinguishable():
    """Wrong labels, incompatible correspondence and unsupported branches are separate."""
    first, second, _body, _target = witness_inputs()
    with pytest.raises(pc.ValidationError) as label_err:
        pc.prepare_views(first, second, "a", "b", "zzz", "e")
    assert label_err.value.kind == pc.STATUS_INVALID_INPUT

    swapped = dict(second)
    swapped["g"], swapped["h"] = second["h"], second["g"]
    with pytest.raises(pc.ValidationError) as incompat_err:
        pc.prepare_views(first, swapped, "a", "b", "c", "e")
    assert incompat_err.value.kind == pc.STATUS_INCOMPATIBLE

    assert pc.STATUS_UNSUPPORTED == "UNSUPPORTED_BRANCH"
    assert pc.STATUS_INVALID_INPUT != pc.STATUS_INCOMPATIBLE
    assert pc.STATUS_INCOMPATIBLE != pc.STATUS_UNSUPPORTED
    assert hasattr(pc, "BRANCH_LEMMA6_COMMON_POINT_RENAMED")
