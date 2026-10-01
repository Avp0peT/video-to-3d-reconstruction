"""Tests for the teaching affine code (Sprint 3C).

Scope: this file validates the *construction* — its affine invariance, its
relabelling invariance, its canonical form, its refusals and its quality
reporting.  What it deliberately does **not** validate is any claim about
Kozlov's textbook: see ``tests/test_teaching_code_isolation.py`` for the
separation rules and ``docs/specs/official_image_code.md`` for the open question.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from kozlov_lab.config import COLLINEAR_TOL
from kozlov_lab.datasets import shapes
from kozlov_lab.datasets.corpus import DATASETS, dataset_names
from kozlov_lab.geometry import AffineFrame, AffineTransform, PointSet2D, triangle_flatness
from kozlov_lab.teaching import (
    CODE_FORMAT_VERSION,
    DISCLAIMER_EN,
    DISCLAIMER_RU,
    DISCLAIMER_ZH,
    TeachingCodingParameters,
    build_canonical_frame_code,
    build_teaching_frame_code,
    build_teaching_image_code,
    compare_teaching_codes,
    recommended_decimals,
    teaching_codes_equal,
    worst_frame_condition_number,
)

#: A quantisation that is comfortably safe for every dataset used here.
SAFE_PARAMS = TeachingCodingParameters(decimals=6)


def _families(x: np.ndarray) -> dict[str, AffineTransform]:
    centre = x.mean(axis=0)
    size = float(np.abs(x).max()) or 1.0
    rng = np.random.default_rng(20261003)
    reflection = np.array([[-0.6, 0.8], [0.8, 0.6]])
    return {
        "identity": AffineTransform.identity(),
        "translation": AffineTransform.translation_of(0.37 * size, -0.21 * size),
        "rotation": AffineTransform.rotation(37.0, center=centre),
        "reflection": AffineTransform(reflection, centre - reflection @ centre),
        "similarity": AffineTransform.similarity(2.5, 41.0, center=centre),
        "scale_nonuniform": AffineTransform.scaling(2.0, 0.5, center=centre),
        "shear": AffineTransform.shear(0.8, -0.2),
        "affine_general": AffineTransform(np.array([[1.3, 0.45], [-0.2, 0.9]]), np.array([-1.0, 2.0])),
        "random": AffineTransform.random_invertible(rng, max_condition=20.0),
    }


# --------------------------------------------------------------------------- #
# the property the whole construction exists for
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("key", dataset_names())
def test_teaching_code_is_affine_invariant(key):
    """``C_teach(X) == C_teach(T(X))`` for every transform family."""
    x = DATASETS[key].points
    reference = build_teaching_image_code(x, SAFE_PARAMS)
    assert reference.quality()["quantization_is_safe"]

    for family, transform in _families(x).items():
        moved = transform.apply(x)
        code = build_teaching_image_code(moved, SAFE_PARAMS)
        assert code.equivalent_to(reference), (
            f"dataset {key}, family {family}: codes differ\n"
            f"  {compare_teaching_codes(x, moved, SAFE_PARAMS).reason}"
        )
        assert code.digest() == reference.digest()


def test_teaching_code_distinguishes_a_valid_affine_map_from_a_general_distortion():
    """Falsifiability: the invariance test must be able to fail.

    Compare the image, an affine image (equal), and a *non-affine* distortion of a
    single point (not equal).  Without the third case, an implementation returning
    a constant code would pass everything above.
    """
    x = DATASETS["A"].points
    reference = build_teaching_image_code(x, SAFE_PARAMS)

    affine_image = AffineTransform.random_invertible(99, max_condition=20.0).apply(x)
    assert build_teaching_image_code(affine_image, SAFE_PARAMS).equivalent_to(reference)

    distorted = x.copy()
    distorted[2] = distorted[2] + np.array([0.15, -0.25])  # one point moved
    assert not build_teaching_image_code(distorted, SAFE_PARAMS).equivalent_to(reference)

    # and the comparison explains why
    comparison = compare_teaching_codes(x, distorted, SAFE_PARAMS)
    assert not comparison.equivalent
    assert "descriptor mismatch" in comparison.reason
    assert not comparison.weak_evidence


def test_teaching_code_distinguishes_same_cardinality_different_shape():
    """Two 9-point images that are not affinely equivalent must differ."""
    x = DATASETS["A"].points
    other = shapes.random_points(x.shape[0], seed=555).as_array()
    comparison = compare_teaching_codes(x, other, SAFE_PARAMS)
    assert not comparison.equivalent
    assert "descriptor mismatch" in comparison.reason


def test_cardinality_mismatch_is_reported_as_such():
    x = DATASETS["A"].points
    comparison = compare_teaching_codes(x, np.delete(x, 0, axis=0), SAFE_PARAMS)
    assert not comparison.equivalent
    assert "cardinality mismatch" in comparison.reason


# --------------------------------------------------------------------------- #
# relabelling invariance — the reason the canonical form is label-free
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("key", dataset_names())
def test_teaching_code_is_invariant_under_relabelling(key):
    x = DATASETS[key].points
    reference = build_teaching_image_code(x, SAFE_PARAMS)
    rng = np.random.default_rng(4242)
    for _ in range(8):
        perm = rng.permutation(x.shape[0])
        code = build_teaching_image_code(x[perm], SAFE_PARAMS)
        assert code.equivalent_to(reference)
        assert code.canonical_text() == reference.canonical_text()


def test_canonical_text_contains_no_point_indices():
    """The canonical form must not leak arbitrary labels.

    A regression guard with teeth: if someone "helpfully" adds the frame indices
    or the condition number to the canonical text, relabelling invariance breaks
    and this test says why.
    """
    x = DATASETS["B"].points
    code = build_teaching_image_code(x, SAFE_PARAMS)
    text = code.canonical_text()

    assert "triple=" not in text and "order=" not in text, (
        "the canonical form must not contain point indices: they are arbitrary labels"
    )
    assert "cond=" not in text, (
        "the canonical form must not contain the condition number: it differs in the last "
        "bits between two equivalent images"
    )
    assert CODE_FORMAT_VERSION in text

    # the diagnostics are still available — just not in the canonical form
    frame = code.frames[0]
    assert "frame=" in frame.describe() and "cond=" in frame.describe()
    assert "triple" in frame.as_dict()

    # ... and a relabelled image produces byte-identical canonical text
    rng = np.random.default_rng(7)
    perm = rng.permutation(len(x))
    assert build_teaching_image_code(x[perm], SAFE_PARAMS).canonical_bytes() == code.canonical_bytes()


def test_canonical_form_is_deterministic():
    x = DATASETS["C"].points
    first = build_teaching_image_code(x, SAFE_PARAMS)
    second = build_teaching_image_code(x, SAFE_PARAMS)
    assert first.canonical_bytes() == second.canonical_bytes()
    assert first.digest() == second.digest()
    assert hash(first) == hash(second)
    assert first == second


# --------------------------------------------------------------------------- #
# canonicalisation over frame orderings
# --------------------------------------------------------------------------- #
def test_canonical_frame_code_is_ordering_invariant():
    x = DATASETS["B"].points
    base = (0, 1, 2)
    bodies = {
        build_canonical_frame_code(x, (base[p[0]], base[p[1]], base[p[2]])).canonical_body()
        for p in itertools.permutations(range(3))
    }
    assert len(bodies) == 1

    raw = {
        build_teaching_frame_code(x, (base[p[0]], base[p[1]], base[p[2]])).canonical_body()
        for p in itertools.permutations(range(3))
    }
    assert len(raw) == 6, "the raw descriptors must differ, else canonicalisation is untested"


def test_frame_points_reduce_to_the_unit_triangle():
    """The frame's own points always read as ``(0,0), (1,0), (0,1)``."""
    x = DATASETS["A"].points
    code = build_canonical_frame_code(x, (0, 1, 2))
    frame = AffineFrame.from_indices(x, code.ordering)
    assert frame.coordinates_xy(frame.a) == pytest.approx((0.0, 0.0), abs=1e-12)
    assert frame.coordinates_xy(frame.b) == pytest.approx((1.0, 0.0), abs=1e-12)
    assert frame.coordinates_xy(frame.c) == pytest.approx((0.0, 1.0), abs=1e-12)
    # and those three constants are present in the descriptor
    quantized = set(code.quantized_points)
    assert (0, 0) in quantized and (10**code.decimals, 0) in quantized
    assert (0, 10**code.decimals) in quantized


def test_three_point_images_are_all_equivalent():
    """Mathematically correct, and worth pinning.

    Any three non-collinear points are affinely equivalent to any other three, so
    with ``n = 3`` every image *must* share the same code.  A "suspiciously good"
    result here is the right answer, not a bug — which is why the test states it
    explicitly.
    """
    triangles = [
        np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        np.array([[1.0, 2.0], [3.5, 2.2], [2.9, 4.1]]),
        np.array([[-2.0, 0.5], [0.3, -1.0], [1.1, 2.7]]),
    ]
    codes = [build_teaching_image_code(t, SAFE_PARAMS) for t in triangles]
    for code in codes[1:]:
        assert code.equivalent_to(codes[0])

    # with only three points there is nothing beyond the frame, so the descriptor
    # degenerates to exactly the three constants
    assert len(codes[0].frames) == 1
    assert len(codes[0].frames[0].quantized_points) == 3


def test_four_point_quadrilaterals_differ():
    """Non-vacuity for the previous test: with four points the code discriminates."""
    a = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])       # square
    b = np.array([[0.0, 0.0], [2.0, 0.0], [2.5, 1.0], [0.2, 1.3]])       # irregular
    assert not build_teaching_image_code(a, SAFE_PARAMS).equivalent_to(
        build_teaching_image_code(b, SAFE_PARAMS)
    )


def test_include_frame_points_does_not_change_the_verdict():
    """Both settings are complete invariants; they only differ in redundancy."""
    x = DATASETS["A"].points
    with_points = build_teaching_image_code(
        x, TeachingCodingParameters(decimals=6, include_frame_points=True)
    )
    without = build_teaching_image_code(
        x, TeachingCodingParameters(decimals=6, include_frame_points=False)
    )
    assert len(with_points.frames[0].quantized_points) == x.shape[0]
    assert len(without.frames[0].quantized_points) == x.shape[0] - 3

    moved = AffineTransform.shear(0.9, -0.3).apply(x)
    assert build_teaching_image_code(
        moved, TeachingCodingParameters(decimals=6, include_frame_points=False)
    ).equivalent_to(without)


# --------------------------------------------------------------------------- #
# refusals and quality reporting
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", [1, 2, 3, 5, 8])
def test_one_dimensional_images_are_refused_not_given_an_empty_code(n):
    """An empty code would compare equal to every other empty code.

    That is a silent false positive in the recogniser, so the correct behaviour is
    a refusal.
    """
    degenerate = shapes.collinear_chain(n).as_array()
    with pytest.raises(ValueError, match="no affine frame"):
        build_teaching_image_code(degenerate, SAFE_PARAMS)

    comparison = compare_teaching_codes(degenerate, DATASETS["A"].points, SAFE_PARAMS)
    assert not comparison.equivalent
    assert comparison.reason.startswith("not codeable")


def test_collinear_triples_are_excluded_from_the_frame_count():
    x = DATASETS["A"].points
    code = build_teaching_image_code(x, SAFE_PARAMS)
    n = x.shape[0]
    total = n * (n - 1) * (n - 2) // 6
    assert len(code) == total - code.n_collinear_triples
    assert code.n_collinear_triples == 2  # dataset A's designed structure
    assert code.n_triples_available == len(code)


def test_truncation_is_reported_and_downgrades_the_claim():
    """A capped code is not a complete invariant, and must say so."""
    x = DATASETS["B"].points
    params = TeachingCodingParameters(decimals=6, max_frames=5)
    code = build_teaching_image_code(x, params)
    assert code.truncated
    assert len(code) == 5
    assert code.n_triples_available > 5
    assert not code.quality()["complete_invariant"]
    assert any("TRUNCATED_FRAMES" in warning for warning in code.warnings)

    # the comparison still runs, but flags the evidence as weak
    comparison = compare_teaching_codes(x, AffineTransform.shear(0.5, 0.1).apply(x), params)
    assert comparison.equivalent
    assert comparison.weak_evidence


def test_quantization_unsafe_is_detected():
    """Too fine a quantum must be reported, because it encodes noise as structure."""
    x = shapes.near_collinear_triangle(1e-8).as_array()
    feasible = recommended_decimals(x)
    assert feasible < 9, (
        f"a flat frame should need a coarse quantum, got {feasible}"
    )

    too_fine = build_teaching_image_code(
        x, TeachingCodingParameters(decimals=14, collinear_tol=COLLINEAR_TOL / 100)
    )
    assert not too_fine.quality()["quantization_is_safe"]
    assert any("QUANTIZATION_UNSAFE" in warning for warning in too_fine.warnings)

    coarse = build_teaching_image_code(
        x, TeachingCodingParameters(decimals=feasible, collinear_tol=COLLINEAR_TOL / 100)
    )
    assert coarse.quality()["quantization_is_safe"]


def test_recommended_decimals_is_monotone_in_the_conditioning():
    """A flatter frame must get a coarser (smaller) recommendation."""
    good = shapes.square(2.0, n_per_side=3).as_array()
    flat = shapes.near_collinear_triangle(1e-6).as_array()
    assert recommended_decimals(good) > recommended_decimals(flat)
    assert recommended_decimals(flat) >= 0

    # a frame-less image yields the floor rather than a bogus value
    assert recommended_decimals(shapes.collinear_chain(5).as_array()) == 0


def test_quantization_overflow_is_reported():
    """A tiny frame with far-away points breaks the integer snap itself.

    Note what triggers this: the snap applies to the *internal* coordinates, so the
    condition is "image extends far beyond its frame", not "the absolute image scale
    is large".  Here the frame is a unit triangle scaled by ``1e-9`` while a fourth
    point sits at distance ~1, giving ``|alpha| ~ 1e9`` and a lattice value ~``1e18``.
    """
    x = np.array([[0.0, 0.0], [1e-9, 0.0], [0.0, 1e-9], [1.0, 1.0]])
    code = build_teaching_image_code(x, TeachingCodingParameters(decimals=9))
    assert code.quantization_magnitude > 2**53, code.quantization_magnitude
    assert not code.quality()["quantization_representable"]
    assert any("QUANTIZATION_OVERFLOW" in warning for warning in code.warnings)


def test_large_absolute_scale_alone_does_not_trigger_the_overflow_warning():
    """The mirror-image control: a big image with a proportionate frame is fine.

    Without this, the overflow check could be implemented on the input coordinates
    and still pass the test above.
    """
    x = np.array([[0.0, 0.0], [1e12, 0.0], [0.0, 1e12], [3e11, 4e11]])
    code = build_teaching_image_code(x, TeachingCodingParameters(decimals=9))
    assert code.quantization_magnitude < 2**53
    assert code.quality()["quantization_representable"]
    assert not any("QUANTIZATION_OVERFLOW" in warning for warning in code.warnings)


def test_worst_frame_condition_number_matches_a_brute_force_scan():
    x = DATASETS["C"].points
    brute = max(
        AffineFrame.from_indices(x, (i, j, k)).condition_number
        for i, j, k in itertools.combinations(range(len(x)), 3)
        if triangle_flatness(x[i], x[j], x[k]) > COLLINEAR_TOL
    )
    assert worst_frame_condition_number(x) == pytest.approx(brute, rel=1e-9)


# --------------------------------------------------------------------------- #
# serialisation and the disclaimer
# --------------------------------------------------------------------------- #
def test_as_dict_carries_the_three_language_disclaimer():
    x = DATASETS["A"].points
    payload = build_teaching_image_code(x, SAFE_PARAMS).as_dict()
    assert payload["disclaimer_en"] == DISCLAIMER_EN
    assert payload["disclaimer_zh"] == DISCLAIMER_ZH
    assert payload["disclaimer_ru"] == DISCLAIMER_RU
    assert payload["quality"]["disclaimer_zh"] == DISCLAIMER_ZH
    assert payload["format"] == CODE_FORMAT_VERSION


def test_comparison_describe_mentions_weak_evidence_when_applicable():
    x = DATASETS["B"].points
    params = TeachingCodingParameters(decimals=6, max_frames=5)
    comparison = compare_teaching_codes(x, x, params)
    assert "weak evidence" in comparison.describe()
    assert comparison.as_dict()["weak_evidence"] is True


def test_codes_of_different_parameters_never_compare_equal():
    """Parameters are part of the canonical text, so they cannot be conflated."""
    x = DATASETS["A"].points
    fine = build_teaching_image_code(x, TeachingCodingParameters(decimals=9))
    coarse = build_teaching_image_code(x, TeachingCodingParameters(decimals=6))
    assert not fine.equivalent_to(coarse)
    assert fine != coarse


def test_image_code_rejects_nonsense_input():
    with pytest.raises((ValueError, TypeError)):
        build_teaching_image_code([], SAFE_PARAMS)
    with pytest.raises((ValueError, TypeError)):
        build_teaching_image_code([[0.0, float("nan")], [1.0, 0.0], [0.0, 1.0]], SAFE_PARAMS)
    with pytest.raises(ValueError):
        TeachingCodingParameters(decimals=-1)
    with pytest.raises(ValueError):
        TeachingCodingParameters(collinear_tol=0.0)
    with pytest.raises(ValueError):
        TeachingCodingParameters(max_frames=0)


def test_pointset_input_is_accepted():
    x = DATASETS["A"].points
    from_set = build_teaching_image_code(PointSet2D(x), SAFE_PARAMS)
    from_array = build_teaching_image_code(x, SAFE_PARAMS)
    assert from_set.equivalent_to(from_array)
