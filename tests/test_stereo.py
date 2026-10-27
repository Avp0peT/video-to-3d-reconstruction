"""Phase 6 property tests: 3D model, projection, controls, oracle, taxonomy, naming.

The controls are the important ones.  The directive's sanity rule is that a known-correspondence,
zero-noise, well-conditioned configuration must reconstruct stably **before** any correspondence
work is believed, so Controls A/B/C are asserted here rather than only in an experiment script.
"""

from __future__ import annotations

import ast
import collections
import dataclasses
import pathlib

import numpy as np
import pytest

from kozlov_lab.stereo import (
    AffineTransform3D,
    CorrespondenceVerdict,
    DegeneracyKind,
    EngineAgreement,
    PointSet3D,
    RecoveryOutcome,
    build_scene,
    classify_geometry,
    engines_agree,
    is_a_equivalent,
    kozlov_body_code,
    match_correspondences,
    oracle_correspondence,
    parallel_project,
    profile_relative_difference,
    reconstruction_outcome,
    relative_3d_error,
    reprojection_error,
    rmse_3d,
    scene_names,
    triangulate_dlt,
    triangulate_many,
    volume_ratio_profile,
)
from kozlov_lab.stereo.baseline.camera import look_at_camera

SRC = pathlib.Path(__file__).resolve().parent.parent / "src" / "kozlov_lab" / "stereo"


def _pair(baseline: float = 4.0, focal: float = 500.0):
    left = look_at_camera([0, 0, 12], [0, 0, 0], [0, 1, 0], focal=focal, name="L")
    right = look_at_camera([baseline, 0, 12], [0, 0, 0], [0, 1, 0], focal=focal, name="R")
    return left, right


# --------------------------------------------------------------------------- #
# G6.2 — 3D model and projection
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", scene_names())
def test_scenes_are_finite_point_sets(name):
    body = build_scene(name)
    assert body.n >= 4
    assert np.all(np.isfinite(body.as_array()))
    assert body.diameter() > 0


def test_point_set_order_independence():
    body = build_scene("irregular")
    shuffled = PointSet3D(body.as_array()[::-1])
    assert shuffled == body
    assert shuffled.signature() == body.signature()


def test_affine_transform_requires_invertibility():
    singular = AffineTransform3D(np.diag([1.0, 0.0, 1.0]), np.zeros(3))
    assert singular.is_invertible is False
    rotation = AffineTransform3D.rotation("z", 30.0)
    assert rotation.is_invertible and rotation.det == pytest.approx(1.0)
    shear = AffineTransform3D.shear(0.5)
    assert shear.det == pytest.approx(1.0)
    scaling = AffineTransform3D.scaling(2.0, 3.0, 5.0)
    assert scaling.det == pytest.approx(30.0)


@pytest.mark.parametrize(
    "transform",
    [
        AffineTransform3D.translation_of(1, -2, 3),
        AffineTransform3D.rotation("x", 37.0),
        AffineTransform3D.rotation("y", 11.0),
        AffineTransform3D.scaling(2.0, 0.5, 1.5),
        AffineTransform3D.shear(0.4, -0.3, 0.2),
        AffineTransform3D.random_invertible(7),
    ],
)
def test_bodies_stay_affinely_equivalent_under_3d_affine_maps(transform):
    """Théorème 3's equivalence: an invertible affine map must not change the class."""
    body = build_scene("irregular").as_array()
    moved = transform.apply(body)
    assert is_a_equivalent(body, moved)["equivalent"] is True
    # and a genuinely different body must not be reported as equivalent
    assert is_a_equivalent(body, build_scene("cube").as_array())["equivalent"] is False


def test_parallel_projection_is_the_sources_projection_type():
    body = build_scene("cube").as_array()
    image = parallel_project(body, [1.0, 0.0, 0.2])
    assert image.shape == (body.shape[0], 2)
    # shifting along the projection direction must not change the image
    shifted = body + np.array([1.0, 0.0, 0.2]) * 7.0
    assert np.allclose(parallel_project(shifted, [1.0, 0.0, 0.2]), image, atol=1e-9)
    with pytest.raises(ValueError):
        parallel_project(body, [0.0, 0.0, 0.0])


# --------------------------------------------------------------------------- #
# G6.3/G6.4 — Controls A, B, C
# --------------------------------------------------------------------------- #
def test_control_a_known_correspondence_zero_noise():
    """3D error and reprojection error must both be at machine precision."""
    body = build_scene("irregular").as_array()
    left, right = _pair()
    image_left, image_right = left.project(body), right.project(body)
    points, failed = triangulate_many(left.projection, image_left, right.projection, image_right)
    assert failed == []
    assert rmse_3d(points, body) < 1e-10
    assert reprojection_error(left, points, image_left) < 1e-10
    assert reprojection_error(right, points, image_right) < 1e-10


def test_control_b_reprojection_consistency_and_equivalence_class():
    body = build_scene("irregular").as_array()
    left, right = _pair()
    image_left, image_right = left.project(body), right.project(body)
    points, _ = triangulate_many(left.projection, image_left, right.projection, image_right)
    assert is_a_equivalent(points, body)["equivalent"] is True


def test_control_c_identical_cameras_are_refused():
    """A zero baseline must be **refused**, not answered with a fabricated depth."""
    left, _ = _pair()
    body = build_scene("irregular").as_array()
    image = left.project(body)
    with pytest.raises(ValueError, match="rank-deficient"):
        triangulate_dlt(left.projection, image[0], left.projection, image[0])


def test_degeneracy_taxonomy_is_three_valued_and_ordered():
    body = build_scene("irregular").as_array()
    kind, evidence = classify_geometry([[0, 0, 12], [4, 0, 12]], body)
    assert kind is DegeneracyKind.OK
    assert "baseline" in evidence
    # D1 takes precedence over everything: with one camera nothing else is meaningful
    zero, _ = classify_geometry([[0, 0, 12], [0, 0, 12]], body)
    assert zero is DegeneracyKind.D1_ZERO_BASELINE
    tiny, _ = classify_geometry([[0, 0, 12], [1e-9, 0, 12]], body)
    assert tiny is DegeneracyKind.D2_SMALL_BASELINE
    planar, _ = classify_geometry(
        [[0, 0, 12], [4, 0, 12]], np.array([[i, j, 0.0] for i in range(3) for j in range(3)])
    )
    assert planar is DegeneracyKind.D5_PLANAR_STRUCTURE


# --------------------------------------------------------------------------- #
# G6.5/G6.6 — oracle and the pruned engine
# --------------------------------------------------------------------------- #
def test_oracle_recovers_the_correspondence_up_to_the_index_convention():
    """The generator's permutation and the recovered mapping differ by **inversion**, not by error.

    With ``right_shuffled = right[perm]`` the 3D point ``i`` sits at shuffled index ``perm⁻¹[i]``,
    so the recovered mapping equals ``perm⁻¹``.  Comparing it with ``perm`` directly would report a
    correct recovery as a failure — a mistake worth pinning with a test, since the same
    orientation trap appeared in Phase 4 with ``psi``.
    """
    body = build_scene("irregular").as_array()
    left, right = _pair()
    image_left, image_right = left.project(body), right.project(body)
    perm = [3, 0, 5, 1, 7, 2, 6, 4]
    shuffled = image_right[perm]
    result = oracle_correspondence(left, right, image_left, shuffled)
    inverse = {i: perm.index(i) for i in range(len(perm))}
    assert result.verdict is CorrespondenceVerdict.UNIQUE
    assert dict(result.mapping) == inverse


def test_pruned_engine_agrees_with_the_oracle():
    body = build_scene("irregular").as_array()
    left, right = _pair()
    image_left, image_right = left.project(body), right.project(body)
    perm = [3, 0, 5, 1, 7, 2, 6, 4]
    shuffled = image_right[perm]
    oracle = oracle_correspondence(left, right, image_left, shuffled)
    pruned = match_correspondences(left, right, image_left, shuffled, candidates_per_point=3)
    assert pruned.mapping == oracle.mapping
    assert pruned.verdict is oracle.verdict
    assert pruned.diagnostics["nodes"] < oracle.diagnostics["n_bijections_examined"]


def test_oracle_is_capped_because_it_is_a_reference_not_a_method():
    body = build_scene("irregular", n=9).as_array()
    left, right = _pair()
    image_left, image_right = left.project(body), right.project(body)
    with pytest.raises(ValueError, match="capped at n"):
        oracle_correspondence(left, right, image_left, image_right)


def test_ambiguous_is_a_result_not_a_failure():
    """A cube admits several equally valid correspondences; that is a property of the object."""
    body = build_scene("cube").as_array()
    left, right = _pair()
    image_left, image_right = left.project(body), right.project(body)
    perm = [4, 1, 6, 0, 5, 3, 7, 2]
    result = oracle_correspondence(left, right, image_left, image_right[perm])
    # the symmetry of the cube is expected to produce alternatives
    assert result.verdict in (CorrespondenceVerdict.AMBIGUOUS, CorrespondenceVerdict.UNIQUE)
    if result.verdict is CorrespondenceVerdict.AMBIGUOUS:
        assert result.n_valid_mappings > 1


# --------------------------------------------------------------------------- #
# 6C — the source construction is a separate layer
# --------------------------------------------------------------------------- #
def test_body_code_grows_like_the_fourth_power_of_the_quadruple_count():
    """``|T_A| = q(N-1)`` with ``N = C(n,4)`` — the 3D lift of the Phase 3 formula."""
    body = build_scene("cube").as_array()
    code = kozlov_body_code(body)
    assert code["n_quadruples"] == 70  # C(8,4)
    assert code["n_admissible_denominators"] + code["n_degenerate_coplanar_quadruples"] == 70
    assert code["n_entries"] == code["n_admissible_denominators"] * (70 - 1)
    assert "Kozlov" in code["source"] or "Козлов" in code["source"]


def test_only_official_may_use_the_authors_name():
    """G6 naming rule, the same one Phases 3–5 enforce."""
    offenders: list[str] = []
    for directory in ("baseline",):
        for path in (SRC / directory).rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    lowered = node.name.lower()
                    if "kozlov" in lowered or "official" in lowered:
                        offenders.append(f"{directory}/{path.name}::{node.name}")
    assert not offenders, f"baseline must not borrow the author's name: {offenders}"


def test_legacy_code3d_does_not_define_alg_t_or_alg_s0():
    """Keep the later source-exact constructions out of the legacy numeric module.

    AlgT' and AlgS0 are now implemented in ``official.parallel_construction``.  This regression
    guard preserves the module boundary instead of retaining the obsolete Phase-6A claim that the
    algorithms do not exist anywhere in the project.
    """
    text = (SRC / "official" / "kozlov_code3d.py").read_text(encoding="utf-8")
    for forbidden in ("def alg_t", "def algt", "def alg_s0", "def alg_so", "def algso"):
        assert forbidden not in text, f"{forbidden!r} must remain in parallel_construction.py"


# --------------------------------------------------------------------------- #
# 6E–6H — the engine comparison, and the two defects it exposed
# --------------------------------------------------------------------------- #
def _case(name: str, seed: int = 20261007):
    """Body, camera pair, both images, the right image shuffled, and the true mapping."""
    body = build_scene(name).as_array()
    n = body.shape[0]
    left, right = _pair()
    image_left, image_right = left.project(body), right.project(body)
    perm = list(np.random.default_rng(seed + n).permutation(n))
    truth = tuple(perm.index(i) for i in range(n))
    return body, left, right, image_left, image_right[perm], truth


def _with_mapping(result, permutation):
    return dataclasses.replace(result, mapping={i: int(c) for i, c in enumerate(permutation)})


def test_optimum_set_contains_the_true_correspondence():
    """Defect 1: a *ratio* tie-window excludes a correct answer on an exact fit.

    The first implementation called two correspondences equivalent when
    ``cost <= best_cost * 1.05``.  An exact synthetic fit has ``best_cost ~ 1e-26``, so that window
    separated floating-point noise, the **true** mapping fell outside the "optimum set", and
    ``engines_agree`` would have reported a correct engine as a disagreement.  The true mapping
    must be in the optimum set, on every scene, or the comparison built on top of it is dishonest.
    """
    for name in ("irregular", "pyramid", "tetrahedron", "cube", "cuboid"):
        body, left, right, image_left, image_right, truth = _case(name)
        oracle = oracle_correspondence(left, right, image_left, image_right)
        assert truth in oracle.good_mappings, f"{name}: the true correspondence is not an optimum"
        assert oracle.mapping is not None
        assert tuple(oracle.mapping[i] for i in range(len(truth))) in oracle.good_mappings


def test_tie_window_scales_with_the_image_not_with_the_optimum():
    """The falsifying counterpart of the ratio rule: the window must not track ``best_cost``.

    ``best_cost`` for an exact fit is numerical noise; the window must be derived from the image
    extent instead.  Requiring the window to exceed the optimum by orders of magnitude is exactly
    what a 5 % ratio rule cannot satisfy, so this test fails on the old implementation.
    """
    for name in ("irregular", "cube"):
        _, left, right, image_left, image_right, _ = _case(name)
        oracle = oracle_correspondence(left, right, image_left, image_right)
        window = oracle.diagnostics["tie_cost_window"]
        assert window > 0
        assert window > oracle.cost * 1e6, (
            f"{name}: the tie window {window:.3e} tracks the noise-level optimum "
            f"{oracle.cost:.3e} — it is behaving as a ratio of the optimum again"
        )


def test_a_tie_is_not_a_disagreement():
    """The original harness comparison was verdict-blind and manufactured a false defect.

    ``pruned.mapping == oracle.mapping`` is wrong when the body admits several cost-0 optima: both
    engines are correct and may return different representatives.

    The difference is **constructed** here rather than relied on.  Which member of a zero-cost tie
    an engine returns is decided by floating-point noise — measured: with numpy 2.4.4 the pruned
    engine returned a different optimum from the oracle on the cube, with numpy 2.5.3 the same one,
    while the optimum *set* was identical both times.  A test asserting a particular member would
    therefore be asserting a coin flip; the test asserts membership instead.
    """
    body, left, right, image_left, image_right, truth = _case("cube")
    oracle = oracle_correspondence(left, right, image_left, image_right)
    assert len(oracle.good_mappings) > 1, "the tie this test is about does not exist"
    alternative = next(p for p in oracle.good_mappings if p != oracle.permutation)
    other = _with_mapping(oracle, alternative)
    # the raw comparison really does report "different" — that is the defect being pinned
    assert other.mapping != oracle.mapping
    assert engines_agree(oracle, other) is EngineAgreement.AMBIGUOUS_CONSISTENT

    # and the pruned engine's answer is an optimum too, whichever member it happens to pick
    pruned = match_correspondences(left, right, image_left, image_right, candidates_per_point=3)
    assert pruned.permutation in oracle.good_mappings
    assert engines_agree(oracle, pruned) in (
        EngineAgreement.EXACT, EngineAgreement.AMBIGUOUS_CONSISTENT
    )


def test_agreement_flags_a_genuinely_wrong_answer():
    """The negative control: without this, "the engines agree" could not fail.

    A non-optimal answer and a missing answer must both be reported as failures; otherwise the
    ``AMBIGUOUS_CONSISTENT`` result above is a claim with no falsifying counterpart.
    """
    body, left, right, image_left, image_right, truth = _case("irregular")
    oracle = oracle_correspondence(left, right, image_left, image_right)
    assert oracle.good_mappings == (truth,)
    wrong = tuple(reversed(range(len(truth))))
    assert wrong != truth
    assert engines_agree(oracle, _with_mapping(oracle, wrong)) is EngineAgreement.DISAGREEMENT
    assert (
        engines_agree(oracle, dataclasses.replace(oracle, mapping=None))
        is EngineAgreement.PRUNED_FAILED
    )


def test_zero_reprojection_does_not_imply_a_correct_correspondence():
    """The phase's sharpest stereo finding, stated as a property of the **optimum set**.

    On an axis-aligned cube the pixel evidence admits multiple zero-cost bijections. Some recover
    an affine-equivalent unordered body and some do not. A pipeline that reports only the
    reprojection error therefore cannot distinguish all wrong answers from valid ones.

    The assertion is about the set, not about which member an engine returns — that choice is
    floating-point noise (see :func:`test_a_tie_is_not_a_disagreement`), and the set is the fact
    that matters: the pixels cannot tell the two apart.
    """
    body, left, right, image_left, image_right, truth = _case("cube")
    oracle = oracle_correspondence(left, right, image_left, image_right)
    assert len(oracle.good_mappings) > 1
    assert truth in oracle.good_mappings
    assert oracle.reprojection_error < 1e-9

    outcomes = collections.Counter()
    for permutation in oracle.good_mappings:
        result = _with_mapping(oracle, permutation)
        outcomes[
            reconstruction_outcome(
                result, left, right, image_left, image_right, body, true_mapping=truth
            )
        ] += 1
    assert outcomes[RecoveryOutcome.EXACT_MATCH] == 1
    assert outcomes[RecoveryOutcome.INVALID_RECONSTRUCTION] > 1, (
        "expected the zero-cost set to contain wrong reconstructions; if it does not, the "
        "ambiguity is benign and this phase's headline finding does not hold"
    )


def test_recovery_outcome_separates_symmetry_from_error(record_testsuite_property):
    """Three outcomes that a boolean would merge, each produced by a real configuration.

    ``EXACT_MATCH`` (the truth), ``EQUIVALENT_VIA_SYMMETRY`` (a different mapping that still
    recovers the body) and ``INVALID_RECONSTRUCTION`` (a different mapping that does not) are
    different results with different consequences, and the third one is the dangerous one.
    """
    body, left, right, image_left, image_right, truth = _case("tetrahedron")
    oracle = oracle_correspondence(left, right, image_left, image_right)
    assert truth in oracle.good_mappings
    other = next(p for p in oracle.good_mappings if p != truth)
    cases = {
        truth: RecoveryOutcome.EXACT_MATCH,
        other: RecoveryOutcome.EQUIVALENT_VIA_SYMMETRY,
    }
    for permutation, expected in cases.items():
        outcome = reconstruction_outcome(
            _with_mapping(oracle, permutation), left, right, image_left, image_right,
            body, true_mapping=truth,
        )
        assert outcome is expected, f"{permutation} -> {outcome}, expected {expected}"

    cube_body, lc, rc, ilc, irc, cube_truth = _case("cube")
    cube_oracle = oracle_correspondence(lc, rc, ilc, irc)
    cube_outcomes = collections.Counter(
        reconstruction_outcome(
            _with_mapping(cube_oracle, p), lc, rc, ilc, irc, cube_body, true_mapping=cube_truth
        )
        for p in cube_oracle.good_mappings
    )
    record_testsuite_property("cube_recovery_outcomes", repr({
        outcome.value: count for outcome, count in cube_outcomes.items()
    }))
    assert RecoveryOutcome.INVALID_RECONSTRUCTION in cube_outcomes
    # The body is an unordered set: non-truth mappings can still reconstruct the same body.
    # The previous row-order comparison incorrectly excluded these witnesses.
    assert RecoveryOutcome.EQUIVALENT_VIA_SYMMETRY in cube_outcomes


def test_invariant_profile_separates_what_reprojection_cannot():
    """6O: the affine invariant detects the wrong reconstruction where the residual does not.

    The reconstruction is selected by *searching the optimum set for an invalid one*, so the test
    does not depend on which member the engine returns.
    """
    body, left, right, image_left, image_right, _ = _case("cube")
    oracle = oracle_correspondence(left, right, image_left, image_right)
    found: tuple[np.ndarray, tuple[int, ...]] | None = None
    for permutation in oracle.good_mappings:
        points, failed = triangulate_many(
            left.projection, image_left, right.projection, image_right[list(permutation)]
        )
        if not failed and not is_a_equivalent(points, body)["equivalent"]:
            found = (points, permutation)
            break
    assert found is not None, "no zero-cost yet invalid reconstruction found"
    wrong, permutation = found

    # the fit is perfect in **both** views ... and the body is wrong.
    # NOTE the right-hand comparison must use the right image *in the recovered order*: comparing
    # it against the unshuffled image is the same ordering trap this phase has already hit twice.
    assert reprojection_error(left, wrong, image_left) < 1e-9
    assert reprojection_error(right, wrong, image_right[list(permutation)]) < 1e-9
    difference = profile_relative_difference(volume_ratio_profile(wrong), volume_ratio_profile(body))
    assert difference > 1e-6, "the invariant cannot see a difference the residual hides"


def test_relative_3d_error_is_scale_free():
    """6M: accuracy is quoted as a fraction of the object's own size, so it survives a rescaling."""
    body = build_scene("irregular").as_array()
    scaled = body * 1000.0
    assert relative_3d_error(body + 0.001, body) == pytest.approx(
        relative_3d_error(scaled + 1.0, scaled), rel=1e-9
    )
    assert rmse_3d(scaled + 1.0, scaled) == pytest.approx(1000.0 * rmse_3d(body + 0.001, body))


def test_admissibility_of_a_quadruple_is_decided_dimensionlessly():
    """The Phase 3 defect, reproduced in 3D and fixed: a literal ``V > 0`` test is unusable.

    The cube has 12 exactly-coplanar quadruples, so its body admits 58 denominators.  A *correct*
    reconstruction of that cube perturbs those quadruples to ``V ~ 1e-17``; under a literal
    ``V > 0`` test it would admit all 70, report a profile of a different length, and the invariant
    would appear to "detect" a difference that is purely arithmetic.  With the dimensionless test
    the two agree to machine precision.
    """
    body, left, right, image_left, image_right, truth = _case("cube")
    assert len(volume_ratio_profile(body)) == 58  # 70 quadruples - 12 coplanar

    points, failed = triangulate_many(
        left.projection, image_left, right.projection, image_right[list(truth)]
    )
    assert failed == []
    assert len(volume_ratio_profile(points)) == len(volume_ratio_profile(body))
    assert profile_relative_difference(
        volume_ratio_profile(points), volume_ratio_profile(body)
    ) < 1e-12


def test_an_abstention_is_not_reported_as_a_positive_result():
    """``is_a_equivalent`` abstains when every quadruple is coplanar; that must not read as success.

    A planar body has no control quadruple, so the equivalence test returns
    ``{"equivalent": None, "undecided": True}``. Historically it returned ``equivalent=True``;
    that flag could turn an abstention into a positive result. The reconstruction outcome must
    still explicitly distinguish an abstention from both a positive and a negative decision.
    """
    planar = np.array([
        [0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0],
        [0.0, 2.0, 0.0], [2.0, 3.0, 0.0], [4.0, 2.0, 0.0],
    ])
    left, right = _pair()
    image_left, image_right = left.project(planar), right.project(planar)
    probe = is_a_equivalent(
        triangulate_many(left.projection, image_left, right.projection, image_right)[0], planar
    )
    assert probe.get("undecided") is True, "the fixture no longer triggers the abstention"

    oracle = oracle_correspondence(left, right, image_left, image_right)
    assert (
        reconstruction_outcome(oracle, left, right, image_left, image_right, planar)
        is RecoveryOutcome.UNDECIDED
    )
    # and it must not be counted among the positive outcomes
    positive = {RecoveryOutcome.EXACT_MATCH, RecoveryOutcome.EQUIVALENT_VIA_SYMMETRY}
    assert RecoveryOutcome.UNDECIDED not in positive


def test_the_sorted_profile_is_a_detector_not_a_criterion(record_testsuite_property):
    """Compare the profile with geometric bijection witnesses on the cube's optimum set.

    The historical positive miss count came from a row-order bug in the geometric oracle:
    relabelled affine-equivalent bodies were counted as wrong reconstructions. Retain the actual
    sample counts without requiring a miss. Zero misses on this sample would not establish that
    a sorted volume-ratio bag is a complete criterion for arbitrary point sets. Equivalence here
    is checked by an affine map and a full point bijection, not by an indexed-code comparison.
    """
    body, left, right, image_left, image_right, _ = _case("cube")
    oracle = oracle_correspondence(left, right, image_left, image_right)
    reference = volume_ratio_profile(body)
    counts = collections.Counter({name: 0 for name in (
        "triangulation_failed", "undecided", "equivalent_same_profile",
        "equivalent_different_profile", "inequivalent_same_profile",
        "inequivalent_different_profile",
    )})
    for permutation in oracle.good_mappings:
        points, failed = triangulate_many(
            left.projection, image_left, right.projection, image_right[list(permutation)]
        )
        if failed:
            counts["triangulation_failed"] += 1
            continue
        result = is_a_equivalent(points, body)
        if result.get("undecided"):
            counts["undecided"] += 1
            continue
        try:
            difference = profile_relative_difference(volume_ratio_profile(points), reference)
        except ValueError:
            difference = float("inf")
        profile_agrees = difference <= 1e-6
        if result["equivalent"]:
            # Validate the geometric witness independently of the invariant summary.
            bijection = result["permutation"]
            assert sorted(bijection) == list(range(len(body)))
            moved = points @ np.asarray(result["matrix"]).T + result["translation"]
            assert np.abs(moved - body[bijection]).max() <= result["tolerance"]
            counts["equivalent_same_profile" if profile_agrees else
                   "equivalent_different_profile"] += 1
        else:
            counts["inequivalent_same_profile" if profile_agrees else
                   "inequivalent_different_profile"] += 1
    record_testsuite_property("cube_profile_comparison_counts", repr(dict(counts)))
    assert sum(counts.values()) == len(oracle.good_mappings)
    assert counts["equivalent_same_profile"] > 1, "expected relabelled affine witnesses"
    assert counts["equivalent_different_profile"] == 0, "the profile violated affine invariance"
    assert counts["inequivalent_different_profile"] > 0, "expected a detectable wrong body"


# --------------------------------------------------------------------------- #
# 6J — the degeneracy taxonomy
# --------------------------------------------------------------------------- #
def test_d7_names_the_axis_aligned_degeneracy():
    """Two cube vertices lie on one ray from the left camera, so the pixel evidence is lossy."""
    cube = build_scene("cube").as_array()
    kind, evidence = classify_geometry([[0, 0, 12], [4, 0, 12]], cube)
    assert kind is DegeneracyKind.D7_COINCIDENT_RAYS
    assert evidence["n_coincident_ray_pairs"] >= 1
    # the same cube, rotated and translated off the axis, is no longer ray-degenerate
    moved = AffineTransform3D(
        np.asarray(AffineTransform3D.rotation("y", 37.0).matrix), np.array([1.0, 2.0, -3.0])
    ).apply(cube)
    assert classify_geometry([[0, 0, 12], [4, 0, 12]], moved)[0] is not (
        DegeneracyKind.D7_COINCIDENT_RAYS
    )
    assert classify_geometry([[0, 0, 12], [4, 0, 12]], build_scene("irregular").as_array())[0] is (
        DegeneracyKind.OK
    )


def test_symmetry_alone_does_not_make_the_correspondence_ambiguous():
    """Measured, and it reverses the natural assumption: D6 is benign, alignment is not.

    A square pyramid rotated 30° about its axis is exactly as symmetric as the unrotated one
    (``D6`` both times), but its correspondence is **unique**, while the axis-aligned pyramid has
    four zero-cost answers.  The ambiguity comes from the alignment with the epipolar geometry
    (``D8``), not from the symmetry.  Reporting ``D6`` above ``D8`` would hide the real cause.
    """
    axis_aligned = build_scene("pyramid").as_array()
    rotated = AffineTransform3D.rotation("z", 30.0).apply(axis_aligned)
    assert classify_geometry([[0, 0, 12], [4, 0, 12]], rotated)[0] is (
        DegeneracyKind.D6_SYMMETRIC_OBJECT
    )
    assert classify_geometry([[0, 0, 12], [4, 0, 12]], axis_aligned)[0] is (
        DegeneracyKind.D8_EPIPOLAR_PLANE_MULTIPLICITY
    )

    left, right = _pair()
    for body, expected_unique in ((axis_aligned, False), (rotated, True)):
        image_left, image_right = left.project(body), right.project(body)
        perm = list(np.random.default_rng(11 + len(body)).permutation(body.shape[0]))
        oracle = oracle_correspondence(left, right, image_left, image_right[perm])
        assert (len(oracle.good_mappings) == 1) is expected_unique


def test_d3_is_not_a_restatement_of_d2():
    """The old D3 compared ``baseline / scale`` again, so it could never see a real epipole point.

    Two assertions, and both are needed: a point genuinely on the baseline is labelled ``D3`` while
    the baseline ratio is *large* (so the old criterion could not have fired), and a body whose
    ratio falls in the window the old rule used for D3 is **not** labelled ``D3``.
    """
    on_baseline = np.array([[2.0, 0.0, 12.0], [5.0, 1.0, 9.0], [3.0, 2.0, 10.0], [6.0, 3.0, 8.0]])
    kind, evidence = classify_geometry([[0, 0, 12], [4, 0, 12]], on_baseline)
    assert kind is DegeneracyKind.D3_POINT_NEAR_EPIPOLE
    assert evidence["min_epipole_angle_degrees"] <= 1e-2
    assert evidence["baseline_over_scale"] > 1e-2  # the old test required the opposite

    # baseline / scale = 8e-3 sits inside the old "D3" window [1e-3, 1e-2)
    small = np.array([[500.0, 0.0, 0.0], [0.0, 500.0, 0.0], [0.0, 0.0, 400.0], [300.0, 300.0, 100.0]])
    assert classify_geometry([[0, 0, 12], [4, 0, 12]], small)[0] is not (
        DegeneracyKind.D3_POINT_NEAR_EPIPOLE
    )


def test_every_taxonomy_branch_is_reachable():
    """A category that no configuration reaches is an untested category — Phase 4's lesson.

    Each of D1, D2, D3, D4, D5, D6, D7, D8 and ``OK`` is produced here by a real configuration, so
    none of them is dead code and none of them is exempt from being wrong.
    """
    irregular = build_scene("irregular").as_array()
    far = irregular + np.array([0.0, 0.0, 500.0])
    cases: list[tuple[DegeneracyKind, np.ndarray, list[list[float]]]] = [
        (DegeneracyKind.OK, irregular, [[0, 0, 12], [4, 0, 12]]),
        (DegeneracyKind.D1_ZERO_BASELINE, irregular, [[0, 0, 12], [0, 0, 12]]),
        (DegeneracyKind.D2_SMALL_BASELINE, irregular, [[0, 0, 12], [1e-9, 0, 12]]),
        (DegeneracyKind.D3_POINT_NEAR_EPIPOLE,
         np.array([[2.0, 0.0, 12.0], [5.0, 1.0, 9.0], [3.0, 2.0, 10.0], [6.0, 3.0, 8.0]]),
         [[0, 0, 12], [4, 0, 12]]),
        (DegeneracyKind.D4_PARALLEL_RAYS, far, [[0, 0, 12], [4, 0, 12]]),
        (DegeneracyKind.D5_PLANAR_STRUCTURE,
         np.array([[i, j, 0.0] for i in range(3) for j in range(3)]), [[0, 0, 12], [4, 0, 12]]),
        (DegeneracyKind.D6_SYMMETRIC_OBJECT,
         AffineTransform3D.rotation("z", 30.0).apply(build_scene("pyramid").as_array()),
         [[0, 0, 12], [4, 0, 12]]),
        (DegeneracyKind.D7_COINCIDENT_RAYS, build_scene("cube").as_array(), [[0, 0, 12], [4, 0, 12]]),
        (DegeneracyKind.D8_EPIPOLAR_PLANE_MULTIPLICITY,
         build_scene("pyramid").as_array(), [[0, 0, 12], [4, 0, 12]]),
    ]
    seen = {}
    for expected, body, centres in cases:
        kind, evidence = classify_geometry(centres, body)
        seen[kind] = seen.get(kind, 0) + 1
        assert kind is expected, f"expected {expected}, got {kind}"
        assert isinstance(evidence, dict) and evidence
    missing = set(DegeneracyKind) - set(seen)
    assert not missing, f"unreached taxonomy values: {sorted(k.value for k in missing)}"
