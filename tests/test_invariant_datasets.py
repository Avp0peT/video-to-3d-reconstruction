"""Multi-shape invariant validation, precision warnings, and permutation laws.

This module implements Sprint 3B.  Its purpose is to remove the single-shape
dependency of the Phase 0–2 evidence: every claim in the invariant table is
re-verified on five deliberately **asymmetric** configurations
(:mod:`kozlov_lab.datasets.corpus`).

Design rules applied throughout (they are the reason these tests are not
tautological):

* every dataset's structural premise (asymmetry, presence/absence of collinear
  triples and parallel pairs) is **asserted**, so a dataset that stops having the
  intended structure fails loudly instead of quietly turning a row vacuous;
* "preserved" is only accepted together with a witness, and every preservation
  claim is paired with a falsifying configuration elsewhere in the file;
* precision claims are compared against a *reported bound*, never against a
  globally hard-coded ``1e-9`` (see :mod:`kozlov_lab.precision`).
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from kozlov_lab.config import COLLINEAR_TOL, MATCH_TOL
from kozlov_lab.datasets.corpus import (
    DATASETS,
    build_dataset,
    dataset_names,
    measure_asymmetry,
)
from kozlov_lab.geometry import (
    AffineFrame,
    AffineTransform,
    PointSet2D,
    affine_coordinates,
    collinearity_sine,
    invariant_report,
    is_non_collinear,
    report_for_transform,
)
from kozlov_lab.precision import (
    precision_of_invariance,
    precision_report,
)
from kozlov_lab.teaching import (
    TeachingCodingParameters,
    build_canonical_frame_code,
    build_teaching_frame_code,
    build_teaching_image_code,
    teaching_codes_equal,
)

# --------------------------------------------------------------------------- #
# fixtures and helpers
# --------------------------------------------------------------------------- #
#: Transform families used for every dataset.  Rotation/reflection are
#: built about the dataset's own centroid (see ``_family_transforms``) so the
#: probes stay in a numerically comfortable range for every shape.
FAMILY_NAMES: tuple[str, ...] = (
    "translation",
    "rotation",
    "reflection",
    "similarity",
    "scale_nonuniform",
    "shear",
    "affine_general",
)

#: Datasets that were *designed* to carry a witness for each relation row.
EXPECTED_NON_VACUOUS: dict[str, set[str]] = {
    "A": {"collinearity", "parallelism"},
    "B": {"collinearity", "parallelism"},
    "C": {"collinearity"},
    "D": {"collinearity"},
    "E": set(),
}

#: Which metric rows each family must *break*, per the theory table.
#:
#: Written per family rather than as one flat list, because the families differ:
#: a similarity preserves angles and length ratios while breaking distances, and a
#: flat list here would produce a test that fails on correct code.
MUST_BREAK: dict[str, tuple[str, ...]] = {
    "translation": (),
    "rotation": (),
    "reflection": (),
    "similarity": ("distances",),
    "scale_nonuniform": ("distances", "angles", "length_ratios"),
    "shear": ("distances", "angles", "length_ratios"),
    "affine_general": ("distances", "angles", "length_ratios"),
}


def _family_transforms(x: np.ndarray) -> dict[str, AffineTransform]:
    centre = x.mean(axis=0)
    size = float(np.abs(x).max()) or 1.0
    return {
        "translation": AffineTransform.translation_of(0.37 * size, -0.21 * size),
        "rotation": AffineTransform.rotation(37.0, center=centre),
        "reflection": AffineTransform(
            np.array([[-0.6, 0.8], [0.8, 0.6]]), centre - np.array([[-0.6, 0.8], [0.8, 0.6]]) @ centre
        ),
        "similarity": AffineTransform.similarity(2.5, 41.0, center=centre),
        "scale_nonuniform": AffineTransform.scaling(2.0, 0.5, center=centre),
        "shear": AffineTransform.shear(0.8, -0.2),
        "affine_general": AffineTransform(
            np.array([[1.3, 0.45], [-0.2, 0.9]]), np.array([-1.0, 2.0])
        ),
    }


def _check_dataset_table(key: str) -> None:
    """The shared body of the five ``test_invariant_table_*`` tests."""
    dataset = DATASETS[key]
    x = dataset.points
    non_vacuous = EXPECTED_NON_VACUOUS[key]
    transforms = _family_transforms(x)

    for family, transform in transforms.items():
        y, report = report_for_transform(x, transform)
        check = report.theory_check(transform)

        # 1. no critical disagreement with the theory table
        assert not check.critical, (
            f"dataset {key}, family {family}: {[d.as_dict() for d in check.critical]}\n"
            f"{check.markdown_table()}"
        )

        # 2. relation rows are non-vacuous exactly where the dataset was designed
        for relation in ("collinearity", "parallelism"):
            entry = report.get(relation)
            if relation in non_vacuous:
                assert entry.verdict == "preserved", (
                    f"dataset {key}, family {family}: {relation} should be a non-vacuous "
                    f"preservation, got {entry.verdict} (size={entry.value_before})"
                )
                assert entry.value_before > 0, f"dataset {key}: {relation} relation is empty"
            else:
                assert entry.verdict == "vacuous", (
                    f"dataset {key}, family {family}: {relation} was designed to be vacuous "
                    f"on this dataset, got {entry.verdict} (size={entry.value_before})"
                )

        # 3. the core invariant: measured deviation within the reported bound
        affine_row = report.get("affine_coordinates")
        assert affine_row.verdict == "preserved"
        assert affine_row.delta <= max(affine_row.tolerance, 1e-12)

        # 4. falsifiability: each family must really break the rows the theory says
        #    it breaks, and every violation must name a witness
        for violated_key in MUST_BREAK[family]:
            entry = report.get(violated_key)
            assert entry.verdict == "violated", (
                f"dataset {key}, family {family}: {violated_key} should be violated "
                f"(delta={entry.delta:.3e}, tol={entry.tolerance:.3e})"
            )
            assert entry.witness is not None, "a violation must name a witness"

        # ... and a family that preserves a metric must do so with a witness present
        for preserved_key in ("angles", "length_ratios"):
            if preserved_key in MUST_BREAK[family] or family == "similarity" and preserved_key == "length_ratios":
                continue
            entry = report.get(preserved_key)
            if entry.verdict == "preserved":
                assert entry.witness_count > 0, (
                    f"dataset {key}, family {family}: {preserved_key} 'preserved' without "
                    "any witness is an empty claim"
                )


# --------------------------------------------------------------------------- #
# dataset premises — "asymmetric" must be a measured fact, not a label
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("key", dataset_names())
def test_dataset_is_asymmetric_and_profile_matches_design(key):
    """Verify the two premises of every dataset before using it.

    If this fails, the tables below lose their power: a symmetric shape masks
    correspondence, orientation, permutation and reflection bugs.
    """
    profile = DATASETS[key].profile()
    asym = DATASETS[key].asymmetry()

    assert profile["is_asymmetric"], f"dataset {key} has symmetry {asym['symmetric_under']}"
    assert asym["candidates"] >= 7, "the symmetry scan must check a meaningful candidate list"
    assert profile["n_points"] >= 5, "an invariant table needs more than a triangle"

    expected = EXPECTED_NON_VACUOUS[key]
    if "collinearity" in expected:
        assert profile["n_collinear_triples"] > 0, f"dataset {key} lost its collinear triples"
    else:
        assert profile["n_collinear_triples"] == 0, f"dataset {key} is not in general position"
    if "parallelism" in expected:
        assert profile["n_parallel_pairs"] > 0, f"dataset {key} lost its parallel pairs"
    else:
        assert profile["n_parallel_pairs"] == 0, f"dataset {key} is not in general position"


def test_general_position_dataset_really_is_general_position():
    """Dataset E is the vacuity control, so its vacuity has to be real.

    A margin check, not just a threshold check: the flattest triple must be well
    above ``COLLINEAR_TOL``, otherwise the row is vacuous only by a hair and the
    test would be luck rather than design.
    """
    x = DATASETS["E"].points
    from kozlov_lab.geometry import triangle_flatness

    flattest = min(
        triangle_flatness(x[i], x[j], x[k]) for i, j, k in itertools.combinations(range(len(x)), 3)
    )
    assert flattest > COLLINEAR_TOL * 100, (
        f"dataset E's flattest triple is only {flattest:.3e}; vacuity would be marginal"
    )


# --------------------------------------------------------------------------- #
# the five required table tests
# --------------------------------------------------------------------------- #
def test_invariant_table_triangle():
    """Dataset A — asymmetric triangle with internal points."""
    _check_dataset_table("A")


def test_invariant_table_asymmetric_quad():
    """Dataset B — asymmetric (trapezoid) quadrilateral."""
    _check_dataset_table("B")


def test_invariant_table_pentagon():
    """Dataset C — irregular pentagon."""
    _check_dataset_table("C")


def test_invariant_table_star():
    """Dataset D — irregular (concave) star-like configuration."""
    _check_dataset_table("D")


def test_invariant_table_random_general_position():
    """Dataset E — random general position (the vacuity control)."""
    _check_dataset_table("E")


@pytest.mark.parametrize("key", dataset_names())
def test_every_family_agrees_with_the_theory_on_every_dataset(key):
    """Explicit cross-check of the theory table on the full dataset x family grid."""
    x = DATASETS[key].points
    for family, transform in _family_transforms(x).items():
        _, report = report_for_transform(x, transform)
        check = report.theory_check(transform)
        assert check.headline().startswith("理论预测与实测一致"), (
            f"{key}/{family}: {check.banner()}"
        )
        comparable = sum(1 for row in check.rows if row.expected != "unknown")
        assert comparable == 7, "every family must be comparable on all seven properties"


# --------------------------------------------------------------------------- #
# permutation behaviour
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("key", dataset_names())
def test_permutation_invariance_of_pointset_semantics(key):
    """Shuffling the points must change nothing mathematical.

    Checked at three levels: set identity, the invariant verdicts, and the
    teaching code.  The transform is applied *before* shuffling so that the
    point correspondence is exercised too.
    """
    x = DATASETS[key].points
    image = PointSet2D(x, name=key)
    transform = AffineTransform.random_invertible(31337, max_condition=20.0)
    y = transform.apply(x)
    reference = invariant_report(
        x, y, kind=transform.classify(), invariant_group=transform.invariant_group()
    )
    reference_code = build_teaching_image_code(x, TeachingCodingParameters(decimals=6))

    rng = np.random.default_rng(4242)
    for _ in range(12):
        perm = rng.permutation(x.shape[0])
        shuffled = PointSet2D(x[perm], name=key)

        # 1. set semantics
        assert shuffled == image
        assert shuffled.signature() == image.signature()
        assert shuffled.multiset_signature() == image.multiset_signature()
        np.testing.assert_allclose(shuffled.sorted_points(), image.sorted_points())

        # 2. invariant verdicts are unchanged (the *measurements* follow the permutation)
        shuffled_report = invariant_report(
            x[perm],
            y[perm],
            kind=transform.classify(),
            invariant_group=transform.invariant_group(),
        )
        assert shuffled_report.verdicts() == reference.verdicts()

        # 3. the teaching code is invariant under relabelling
        assert build_teaching_image_code(x[perm], TeachingCodingParameters(decimals=6)).equivalent_to(
            reference_code
        )


def _basis_matrix_of_permutation(perm: tuple[int, int, int]) -> np.ndarray:
    """Analytic ``T`` with ``[q-p, r-p] = [u, v] @ T`` for the ordered triple ``perm``.

    Derived, not looked up: writing the permuted spanning vectors as integer
    combinations of ``u = b - a`` and ``v = c - a`` and reading off the
    coefficients.  Column ``j`` of ``T`` holds the coefficients of the ``j``-th
    spanning vector, in the basis ``a -> (0,0), b -> (1,0), c -> (0,1)``.
    """
    names = ["a", "b", "c"]
    p, q, r = (_BASIS_REPR[names[i]] for i in perm)
    return np.column_stack([q - p, r - p])


def _origin_shift_of_permutation(perm: tuple[int, int, int]) -> np.ndarray:
    """``a - a'`` in the ``(u, v)`` basis, where ``a'`` is the permuted frame origin.

    **The term everyone forgets.**  The clean law ``z' = T^{-1} z`` only holds for
    permutations that keep ``A`` as the origin.  In general the coordinates are
    taken from a *different* base point, so

        P - a' = (P - a) + (a - a') = M z + M delta,
        z' = (M T)^{-1} (M z + M delta) = T^{-1} (z + delta),

    with ``delta`` the ``(u,v)``-coordinates of ``a - a'``.
    """
    names = ["a", "b", "c"]
    return _BASIS_REPR["a"] - _BASIS_REPR[names[perm[0]]]


#: Coordinates of ``a``, ``b``, ``c`` in the ``(u, v)`` basis centred at ``a``.
_BASIS_REPR: dict[str, np.ndarray] = {
    "a": np.array([0.0, 0.0]),
    "b": np.array([1.0, 0.0]),
    "c": np.array([0.0, 1.0]),
}


@pytest.mark.parametrize("key", ["A", "B", "E"])
def test_affine_coordinate_frame_permutation_behavior(key):
    """Permuting the frame must produce the *documented* coordinate relation.

    Easy to get wrong, so stated explicitly: this test does **not** assert that the
    coordinates stay the same.  Permuting ``(A, B, C)`` changes ``[B-A, C-A]`` by an
    integer unimodular matrix ``T``, and in general also moves the origin, hence

        M' = M T,   z' = T^{-1} (z + delta),   det M' = det M * det T,

    where ``delta`` is the ``(u, v)``-coordinates of ``a - a'`` (zero for the two
    permutations that keep ``A`` first — which is why the naive
    ``z' = T^{-1} z`` looks right until it isn't).

    Three analytically derived relations are checked:

    1. ``z' == T^{-1} (z + delta)`` to floating-point accuracy;
    2. ``det T == sign(permutation)`` and ``|det M'| == |det M|``, so a permutation
       never destroys the frame;
    3. the barycentric triple ``(1 - alpha - beta, alpha, beta)`` is *permuted* the
       same way as the frame.  This one is origin-independent and is exactly what
       the image-code layer relies on when it canonicalises over frame orderings.
    """
    x = DATASETS[key].points
    base = (0, 3, 5) if x.shape[0] > 5 else (0, 1, 2)
    u = x[base[1]] - x[base[0]]
    v = x[base[2]] - x[base[0]]
    det_base = float(u[0] * v[1] - u[1] * v[0])
    assert abs(det_base) > 0

    reference: dict[tuple[int, int, int], np.ndarray] = {}
    for perm in itertools.permutations(range(3)):
        ordering = (base[perm[0]], base[perm[1]], base[perm[2]])
        frame = AffineFrame.from_indices(x, ordering)  # type: ignore[arg-type]
        assert frame.is_valid(), f"permuted frame {ordering} must still be a valid frame"
        reference[tuple(perm)] = np.array(
            [frame.coordinates(point).as_tuple() for point in x]
        )

        t_matrix = _basis_matrix_of_permutation(tuple(perm))
        expected_det = _permutation_sign(tuple(perm))
        assert np.linalg.det(t_matrix) == pytest.approx(expected_det, abs=1e-12)
        assert abs(np.linalg.det(frame.matrix)) == pytest.approx(abs(det_base), rel=1e-9)
        # T is unimodular: the frame's validity cannot depend on the ordering
        assert abs(np.linalg.det(t_matrix)) == pytest.approx(1.0, abs=1e-12)

    base_coords = reference[(0, 1, 2)]

    def barycentric(z: np.ndarray) -> np.ndarray:
        return np.column_stack([1.0 - z[:, 0] - z[:, 1], z[:, 0], z[:, 1]])

    for perm, coords in reference.items():
        t_matrix = _basis_matrix_of_permutation(perm)
        delta = _origin_shift_of_permutation(perm)

        # relation 1: z' = T^{-1} (z + delta)
        expected = np.linalg.solve(t_matrix, (base_coords + delta).T).T
        np.testing.assert_allclose(
            coords, expected, rtol=0, atol=1e-11,
            err_msg=f"permutation {perm} on dataset {key}",
        )

        # relation 3: the barycentric triple w.r.t. the *permuted* frame equals the
        # base barycentric triple reordered by the same permutation.
        # Written in this direction on purpose: applying the permutation to both
        # sides (which looks symmetric for transpositions) is wrong for 3-cycles,
        # and that is exactly the case a careless test would miss.
        np.testing.assert_allclose(
            barycentric(coords), barycentric(base_coords)[:, list(perm)],
            rtol=0, atol=1e-10,
            err_msg=f"barycentric relation failed for permutation {perm} on dataset {key}",
        )

    # non-vacuity: permuting really does change the coordinates (except for the
    # two permutations that keep A as the origin, where only the axes swap)
    assert not np.allclose(reference[(0, 1, 2)], reference[(1, 0, 2)]), (
        "swapping A and B must change the coordinates, otherwise this test proves nothing"
    )


def _permutation_sign(perm: tuple[int, int, int]) -> float:
    a, b, c = perm
    inversions = (a > b) + (a > c) + (b > c)
    return -1.0 if inversions % 2 else 1.0


def test_frame_permutation_does_not_change_the_canonical_frame_code():
    """The raw coordinates change; the *canonical* frame code does not.

    This is the coupling between the geometry layer and the teaching code:
    permuting a frame is a symmetry of the code, and the minimum over the six
    orderings is what implements it (step 3 of the de-labelling recipe).
    """
    x = DATASETS["B"].points
    base = (0, 1, 2)

    # the canonical code is the same for every ordering of the same triple
    canonical = {
        build_canonical_frame_code(x, (base[p[0]], base[p[1]], base[p[2]])).canonical_body()
        for p in itertools.permutations(range(3))
    }
    assert len(canonical) == 1, "the cannot be order dependent"
    # passing the triple in a different order is also invisible (it is sorted first)
    assert (
        build_canonical_frame_code(x, (base[2], base[0], base[1])).canonical_body()
        == next(iter(canonical))
    )

    # non-vacuity: the six *raw* descriptors genuinely differ, so this test is not
    # passing merely because canonicalisation was unnecessary
    raw = {
        build_teaching_frame_code(x, (base[p[0]], base[p[1]], base[p[2]])).canonical_body()
        for p in itertools.permutations(range(3))
    }
    assert len(raw) == 6, "the six orderings should give six different raw descriptors"

    # ... and the raw coordinates change too
    z_base = np.array([AffineFrame.from_indices(x, base).coordinates(p).as_tuple() for p in x])
    z_swapped = np.array(
        [
            AffineFrame.from_indices(x, (base[1], base[0], base[2])).coordinates(p).as_tuple()
            for p in x
        ]
    )
    assert not np.allclose(z_base, z_swapped), "swapping A and B must change the coordinates"

    # the canonical *image* code is likewise invariant under relabelling
    image_code = build_teaching_image_code(x, TeachingCodingParameters(decimals=6))
    rng = np.random.default_rng(11)
    for _ in range(5):
        perm = rng.permutation(len(x))
        assert build_teaching_image_code(
            x[perm], TeachingCodingParameters(decimals=6)
        ).equivalent_to(image_code)


def test_canonical_frame_code_rejects_degenerate_triples():
    """Degenerate frames must be refused, not silently coded.

    Dataset A's collinear triples are ``(0, 4, 5)`` (``A`` and the two points on
    the median from ``A``) and ``(0, 1, 3)`` (``A``, ``B`` and the point on
    segment ``AB``).  Both are asserted to be non-collinear *as triples* by the
    frame constructor, so the refusal is the frame's own check firing.
    """
    x = DATASETS["A"].points
    for degenerate in ((0, 4, 5), (0, 1, 3)):
        with pytest.raises(ValueError, match="collinear"):
            build_canonical_frame_code(x, degenerate)
    with pytest.raises(ValueError):
        build_canonical_frame_code(x, (0, 0, 1))  # repeated index
    with pytest.raises(ValueError):
        build_canonical_frame_code(x, (0, 1))  # too few indices

    # control: a non-degenerate triple must *not* be refused, otherwise the
    # assertions above could pass for the wrong reason
    assert build_canonical_frame_code(x, (0, 1, 2)) is not None


# --------------------------------------------------------------------------- #
# precision warnings
# --------------------------------------------------------------------------- #
def test_near_collinear_warning():
    """A flat frame must produce a warning *before* it produces a wrong number."""
    from kozlov_lab.datasets import shapes

    # comfortably conditioned: no warning, 1e-9 attainable
    good = AffineFrame.from_indices(shapes.square(2.0, n_per_side=3).as_array(), (0, 3, 5))
    good_report = precision_report(good.condition_number)
    assert good_report.level == "ok"
    assert good_report.warnings == ()
    assert good_report.meets_readme_requirement

    for sine in (1e-6, 1e-8):
        x = shapes.near_collinear_triangle(sine).as_array()
        frame = AffineFrame.from_indices(x, (0, 1, 2))
        assert frame.is_valid(), "these frames are accepted, so the warning must be issued"

        report = precision_report(frame.condition_number)
        assert report.level != "ok", f"flatness={sine:g} must not look fine"
        assert not report.meets_readme_requirement
        codes = {warning.code for warning in report.warnings}
        assert "NEAR_COLLINEAR_FRAME" in codes
        # the warning is a real message, in all three languages, with the numbers in it
        warning = next(w for w in report.warnings if w.code == "NEAR_COLLINEAR_FRAME")
        for lang in ("zh", "ru", "en"):
            assert warning.text(lang).strip()
        assert warning.detail["condition_number"] == pytest.approx(frame.condition_number)
        assert report.significant_decimal_digits < 9, "fewer than 9 usable digits remain"

    # below the collinearity threshold the frame is refused outright, not warned about
    x = shapes.near_collinear_triangle(COLLINEAR_TOL / 10.0).as_array()
    assert not is_non_collinear(x[0], x[1], x[2])
    with pytest.raises(ValueError):
        AffineFrame.from_indices(x, (0, 1, 2)).coordinates(x[0])


def test_large_translation_small_frame_precision_warning():
    """The magnitude failure mode must be reported, and must match reality.

    A tiny image under a huge translation loses precision in the *input*, so the
    conditioning bound cannot see it.  Two levels are checked:

    * ``|t|`` large but the image still resolvable → TRANSLATION_MAGNITUDE_DOMINATES;
    * ``|t|`` beyond the resolution limit → IMAGE_STRUCTURE_DESTROYED, level unusable.

    In the first case the warning is tied back to the measured deviation, so the
    warning cannot drift away from the phenomenon it describes.
    """
    from kozlov_lab.datasets import shapes

    base = shapes.square(2.0, n_per_side=3).as_array()

    # --- level 1: resolvable but dominated ---------------------------------
    # Iterated over several magnitudes on purpose: a single configuration can be
    # *luckily* exact (rounding cancels over a small point set — observed at
    # |t| = 1e3), so a single sample cannot support a claim about the phenomenon.
    base = shapes.square(2.0, n_per_side=3).as_array()
    scale = 1e-9
    x = base * scale
    frame = AffineFrame.from_indices(x, (0, 3, 5))
    frame_scale = float(np.linalg.norm(frame.b - frame.a))

    worst_deviation = 0.0
    n_warned = 0
    # ratios eps*|t|/|B-A| are 1.1e-2, 1.1e-1 here; the >= 1 regime (structure
    # destroyed) is covered by the level-2 block below
    for translation in (1e5, 1e6):
        t = AffineTransform.translation_of(translation, -translation)
        delta = frame.invariance(t, x)
        report = precision_of_invariance(
            delta,
            frame_scale=frame_scale,
            coordinate_scale=float(np.abs(t.apply(x)).max()),
        )
        codes = {warning.code for warning in report.warnings}
        assert "TRANSLATION_MAGNITUDE_DOMINATES" in codes, (
            f"|t| = {translation:g}: {report.describe()}"
        )
        assert report.level != "ok"
        assert not report.meets_readme_requirement, (
            "1e-9 is not attainable in this configuration, so the report must not claim it"
        )
        assert report.bound_from_magnitude is not None
        assert report.total_bound >= report.bound_from_magnitude
        # the warning is tied to the measurement: it must not promise a bound the
        # library then violates by orders of magnitude
        assert delta.max_delta <= report.total_bound * 10.0, (
            f"measured {delta.max_delta:.3e} far exceeds the reported bound "
            f"{report.total_bound:.3e}: the reported bound would be a false promise"
        )
        worst_deviation = max(worst_deviation, delta.max_delta)
        n_warned += 1

    assert n_warned == 2
    assert worst_deviation > MATCH_TOL, (
        f"every configuration was luckily exact (worst {worst_deviation:.3e}); the warning "
        "would then be describing a non-problem and this test would be mis-framed"
    )

    # --- level 2: structure destroyed --------------------------------------
    destroyed_scale = 1e-12
    x2 = base * destroyed_scale
    frame2 = AffineFrame.from_indices(x2, (0, 3, 5))
    t2 = AffineTransform.translation_of(1e6, -1e6)
    frame_scale = float(np.linalg.norm(frame2.b - frame2.a))
    report2 = precision_report(
        3.37, frame_scale=frame_scale, coordinate_scale=float(np.abs(t2.apply(x2)).max())
    )
    codes2 = {warning.code for warning in report2.warnings}
    assert "IMAGE_STRUCTURE_DESTROYED" in codes2, report2.describe()
    assert report2.level == "unusable"
    assert any(w.severity == "critical" for w in report2.warnings)

    # ... and the geometry layer really does refuse to answer there
    moved = t2.apply(x2)
    assert np.unique(moved, axis=0).shape[0] < x2.shape[0], "structure should have collapsed"
    with pytest.raises(ValueError):
        frame2.invariance(t2, x2)


def test_precision_report_is_informative_not_constant():
    """Guard against a report that always says the same thing.

    A helper that always returned ``level="ok"`` (or always ``"unusable"``) would
    satisfy the two tests above only by accident; this one requires the three
    levels to be reachable.
    """
    levels = {
        precision_report(3.37).level,
        precision_report(1e8).level,
        precision_report(1e14).level,
        precision_report(3.37, frame_scale=2e-9, coordinate_scale=1e6).level,
    }
    assert {"ok", "caution", "unusable"} <= levels, f"levels reachable: {levels}"


# --------------------------------------------------------------------------- #
# datasets are usable end to end
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("key", dataset_names())
def test_dataset_round_trip_and_coding(key):
    """Every dataset must survive JSON round-trip and be codeable."""
    image = build_dataset(key)
    assert PointSet2D.from_json(image.to_json()) == image

    code = build_teaching_image_code(image.as_array(), TeachingCodingParameters(decimals=6))
    assert len(code) > 0
    assert code.n_points == image.n
    assert code.quality()["quantization_is_safe"]

    # an affine image of the dataset must produce the same code
    transform = AffineTransform.random_invertible(2026, max_condition=20.0)
    moved = transform.apply(image.as_array())
    assert teaching_codes_equal(
        image.as_array(), moved, TeachingCodingParameters(decimals=6)
    ), f"dataset {key}: teaching code is not affine invariant"


def test_collinear_dataset_is_not_codeable():
    """A 1-D image must be *refused*, not given an empty code.

    An empty code would compare equal to every other empty code, which would be a
    silent false positive in the recogniser.
    """
    from kozlov_lab.datasets import shapes

    with pytest.raises(ValueError, match="no affine frame"):
        build_teaching_image_code(shapes.collinear_chain(6).as_array())


def test_relation_verdict_is_permutation_invariant_on_all_datasets():
    """The *relation* verdict must be permutation invariant — the vertex sine is not.

    Restated here at dataset level because the image-code layer enumerates
    collinearity over an arbitrary index order: a criterion that depended on which
    point is named ``A`` would make the code depend on the labelling.

    The control at the end builds a near-degenerate triple where the three vertex
    sines differ by ~150 orders of magnitude, so this test cannot pass by accident
    on "everything is collinear" or "nothing is collinear".
    """
    for key in dataset_names():
        x = DATASETS[key].points
        for i, j, k in itertools.combinations(range(len(x)), 3):
            verdicts = {
                is_non_collinear(x[p], x[q], x[r])
                for p, q, r in itertools.permutations((i, j, k))
            }
            assert len(verdicts) == 1, f"dataset {key}: triple ({i},{j},{k}) is order dependent"

    # control: a triple whose vertex sines disagree wildly must still get one verdict
    a = np.array([0.0, 0.0])
    b = np.array([3.47370303e-152, 0.0])
    c = np.array([0.0, 1.0])
    sines = [
        collinearity_sine(a, b, c),
        collinearity_sine(b, a, c),
        collinearity_sine(c, a, b),
    ]
    assert max(sines) == pytest.approx(1.0)
    assert min(sines) < 1e-100
    assert len({is_non_collinear(*p) for p in itertools.permutations((a, b, c))}) == 1


def test_measure_asymmetry_detects_a_known_symmetric_shape():
    """Negative control for :func:`measure_asymmetry`.

    A symmetry detector that never fires makes every dataset "asymmetric" by
    construction and the whole premise of Sprint 3B vacuous.
    """
    square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    result = measure_asymmetry(square)
    assert not result["is_asymmetric"]
    assert result["symmetric_under"], "a square is symmetric, so something must be detected"

    scalene = np.array([[0.0, 0.0], [1.3, 0.1], [0.4, 1.7]])
    assert measure_asymmetry(scalene)["is_asymmetric"]


def test_verdicts_do_not_depend_on_the_image_scale():
    """Rescaling the image must not change a single verdict.

    Only the verdicts are compared, not the tolerances: the tolerance has an
    absolute floor (``max(d_max, 1)``) by design, so for sub-unit images it is
    *relatively* looser.  Asserting tolerance equality here would be asserting a
    property the design deliberately does not have.
    """
    shear = AffineTransform.shear(0.8, -0.2)
    for key in dataset_names():
        base = DATASETS[key].points
        reference = report_for_transform(base, shear)[1].verdicts()
        for scale in (1e-6, 1e-3, 1e3, 1e6):
            scaled = AffineTransform.similarity(scale, 0.0).apply(base)
            moved = shear.apply(scaled)
            record = invariant_report(
                scaled, moved, kind=shear.classify(), invariant_group=shear.invariant_group()
            )
            assert record.verdicts() == reference, f"dataset {key} at scale {scale:g}"
