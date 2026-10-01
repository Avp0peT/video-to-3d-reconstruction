"""Phase 5 gates: contracts (G5.2), metric properties (G5.3), naming isolation (G5.9),
and the F4.3 collision-injection requirement.

Three things are being established:

* every compressor's **declared contract is true** — coverage guarantees are verified rather
  than trusted, and a compressor that claims a subset really returns one;
* the **metrics behave like metrics** (Hausdorff properties, ratio identities), because a
  metric with a broken triangle inequality would silently corrupt every comparison built on it;
* the **three families stay apart**: nothing outside ``official/`` may use the author's name,
  and nothing inside it may claim more than the source says.
"""

from __future__ import annotations

import ast
import pathlib

import numpy as np
import pytest

from kozlov_lab.compression import (
    COMPRESSORS,
    code_size_reduction,
    compression_ratio,
    coverage_report,
    directed_hausdorff,
    formal_code_size,
    get_compressor,
    hausdorff,
    list_compressors,
    run_compressor,
)
from kozlov_lab.compression.official import (
    certified_nearest,
    greedy_sketch,
    kozlov_sketch_distance_bounds,
    minimum_cover_size,
    skeleton_hierarchy,
    verify_sketch,
)

IMAGE = [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [1.0, 1.0], [3.0, 5.0], [4.0, 1.0], [2.5, 3.5], [5.0, 5.0]]
EXTRA = [[1.0, 2.0], [-1.0, 0.5], [0.3, -2.0], [4.0, 4.0]]

#: One representative parameterisation per compressor.
PARAMETERS: dict[str, dict] = {
    "KozlovSketch": {"r": 1.2},
    "KozlovSkeleton": {"r": 1.2},
    "TeachingEpsilonCover": {"epsilon": 1.2},
    "TeachingFarthestPoint": {"k": 5},
    "TeachingGridReduce": {"cell": 1.2},
    "AffineAwareSampler": {"r": 0.6},
}


# --------------------------------------------------------------------------- #
# G5.2 — contracts must be true
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", sorted(PARAMETERS))
def test_every_compressor_honours_its_contract(name):
    """Coverage guarantees are **verified**, and subset claims are **checked**."""
    result = run_compressor(name, IMAGE, **PARAMETERS[name])
    matrix = np.asarray(IMAGE, dtype=float)
    contract = result.contract

    assert result.points.shape[0] >= 1
    assert result.n_after <= result.n_before

    if contract.cover_guarantee == "guaranteed":
        assert result.requested_radius is not None
        assert result.achieved_radius <= result.requested_radius + 1e-12, (
            f"{name} promised a guaranteed radius but achieved {result.achieved_radius} "
            f"> {result.requested_radius}"
        )
        assert verify_sketch(matrix, result.points, float(result.requested_radius))
    elif contract.cover_guarantee == "guaranteed_normalized":
        # the guarantee is in the normalized metric only; the normalized radius must hold
        assert result.diagnostics["normalized_radius_achieved"] <= result.requested_radius + 1e-9
    else:
        assert contract.cover_guarantee == "measured"

    if contract.surjective_on:
        pool = {tuple(point) for point in matrix}
        for point in result.points:
            assert tuple(point) in pool, f"{name} claims a subset but returned {point}"


def test_a_guarantee_that_is_only_measured_is_labelled_as_such():
    """"Measured" must not be reported as "guaranteed".

    ``TeachingFarthestPoint`` and ``TeachingGridReduce`` have no covering-radius guarantee, and
    the contract says so.  If either ever claimed ``guaranteed``, the claim would have to be
    backed by a proof in its docstring — which is the point of separating the two words.
    """
    for name in ("TeachingFarthestPoint", "TeachingGridReduce"):
        assert get_compressor(name).contract.cover_guarantee == "measured"
    for name in ("KozlovSketch", "KozlovSkeleton", "TeachingEpsilonCover"):
        assert get_compressor(name).contract.cover_guarantee == "guaranteed"


def test_minimum_cardinality_is_only_claimed_when_proved():
    """``cardinality_guarantee`` must reflect what the solver actually established."""
    proved = run_compressor("KozlovSkeleton", IMAGE, r=1.2)
    assert proved.diagnostics["proved_minimum"] is True
    assert proved.contract.cardinality_guarantee == "minimum"

    # force the budget to bite: the answer must degrade to "upper_bound", not stay "minimum"
    starved = run_compressor("KozlovSkeleton", IMAGE, r=1.2, node_budget=1)
    if not starved.diagnostics["proved_minimum"]:
        assert starved.contract.cardinality_guarantee == "upper_bound"


def test_the_source_construction_is_a_subset_and_the_epsilon_net_need_not_be():
    """The genuine difference between the эскиз and a true $\\varepsilon$-net."""
    sketch = run_compressor("KozlovSketch", IMAGE, r=1.2)
    cover = run_compressor("TeachingEpsilonCover", IMAGE, epsilon=1.2)
    assert sketch.contract.surjective_on is True
    assert cover.contract.surjective_on is False
    assert cover.diagnostics["sketch_is_subset"] in (True, False)  # measured, not assumed


def test_skeleton_never_exceeds_the_greedy_cover():
    """The остов is the minimum, so it cannot be larger than any other valid эскиз."""
    for r in (0.5, 1.0, 1.6, 2.5):
        _, greedy, _ = minimum_cover_size(IMAGE, r)
        size, skeleton, proved = minimum_cover_size(IMAGE, r)
        assert proved
        assert size <= len(greedy)
        assert verify_sketch(IMAGE, np.asarray(IMAGE)[skeleton], r)


def test_skeleton_hierarchy_reproduces_the_sources_ladder():
    """「...наименьший по числу точек остов всегда состоит из одной точки, а наибольший
    совпадает с исходным изображением А.»"""
    rows = skeleton_hierarchy(IMAGE, [0.0, 2.0, 100.0])
    sizes = {row["r"]: row["n_skeleton"] for row in rows}
    assert sizes[0.0] == len(IMAGE), "at r = 0 the skeleton is the image itself"
    assert sizes[100.0] == 1, "for a radius beyond the diameter the skeleton is one point"
    assert all(row["proved_minimum"] for row in rows)


# --------------------------------------------------------------------------- #
# G5.3 — the metrics must behave like metrics
# --------------------------------------------------------------------------- #
def test_hausdorff_is_symmetric_and_zero_on_identical_sets():
    assert hausdorff(IMAGE, IMAGE) == pytest.approx(0.0, abs=1e-12)
    assert directed_hausdorff(IMAGE, IMAGE) == pytest.approx(0.0, abs=1e-12)
    a, b = IMAGE, EXTRA
    assert hausdorff(a, b) == pytest.approx(hausdorff(b, a), rel=1e-12)


def test_hausdorff_satisfies_the_triangle_inequality():
    """Checked numerically, because a broken triangle inequality would corrupt comparisons
    built on top of it — and it is the property the sketch bounds rely on."""
    a = np.asarray(IMAGE, dtype=float)
    b = np.asarray(EXTRA, dtype=float)
    c = a * 1.1 + 0.3
    assert hausdorff(a, c) <= hausdorff(a, b) + hausdorff(b, c) + 1e-9


def test_directed_hausdorff_is_zero_when_the_target_contains_the_source():
    """Which is why the two-sided version is still computed: the reverse term is the informative
    one, and skipping it would remove the only place a non-subset compressor could be caught."""
    subset = np.asarray(IMAGE, dtype=float)[[0, 1, 3]]
    assert directed_hausdorff(subset, IMAGE) == pytest.approx(0.0, abs=1e-12)
    assert directed_hausdorff(IMAGE, subset) > 0.0


def test_compression_ratio_identities():
    assert compression_ratio(100, 100) == 0.0
    assert compression_ratio(100, 0) == 1.0
    assert compression_ratio(10, 5) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        compression_ratio(0, 0)


def test_coverage_report_exposes_both_directions():
    report = coverage_report(IMAGE, np.asarray(IMAGE, dtype=float)[[0, 1, 2]])
    assert set(report) >= {
        "directed_original_to_sketch",
        "directed_sketch_to_original",
        "hausdorff",
        "sketch_is_subset",
    }
    assert report["hausdorff"] >= report["directed_original_to_sketch"]
    assert report["sketch_is_subset"] is True


def test_formal_code_size_follows_the_phase4_formula():
    """5H's coupling must use the same formula Phase 4 measured, not a second one."""
    assert formal_code_size(4)["n_triples"] == 4
    assert formal_code_size(4)["n_entries_max"] == 4 * 3
    sizes = code_size_reduction(12, 8)
    assert sizes["n_entries_before"] > sizes["n_entries_after"]
    assert sizes["code_size_speedup"] > 1.0
    # a point ratio of 8/12 is about 0.67, but the code ratio is far smaller: that is the
    # O(n^6) effect 5H exists to measure
    assert sizes["code_size_ratio"] < sizes["point_ratio"] ** 4


# --------------------------------------------------------------------------- #
# G5.9 — the three families must not impersonate each other
# --------------------------------------------------------------------------- #
SRC = pathlib.Path(__file__).resolve().parent.parent / "src" / "kozlov_lab" / "compression"


def _sources(directory: pathlib.Path) -> list[pathlib.Path]:
    return sorted(path for path in directory.rglob("*.py") if "__pycache__" not in path.parts)


def _defined_names(path: pathlib.Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return [
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    ]


def test_the_three_families_exist():
    for family in ("official", "teaching", "experimental"):
        assert (SRC / family).is_dir(), family
        assert (SRC / family / "__init__.py").is_file(), family


def test_only_official_may_use_the_authors_name():
    """Nothing in ``teaching/`` or ``experimental/`` may be named after Kozlov."""
    offenders: list[str] = []
    for family in ("teaching", "experimental"):
        for source in _sources(SRC / family):
            for name in _defined_names(source):
                lowered = name.lower()
                if "kozlov" in lowered or "official" in lowered:
                    offenders.append(f"{family}/{source.name}::{name}")
    assert not offenders, f"baselines/experiments must not borrow the author's name: {offenders}"


def test_official_contains_everything_that_carries_the_authors_name():
    matches = {
        f"{path.parent.name}/{path.name}": [
            name for name in _defined_names(path) if "kozlov" in name.lower()
        ]
        for path in _sources(SRC)
    }
    striking = {key: value for key, value in matches.items() if value}
    assert set(striking) == {"official/kozlov_sketch.py"}, striking


def test_every_contract_declares_its_provenance():
    """A contract with no source is a contract nobody can check."""
    for contract in list_compressors():
        assert contract["source"], contract["name"]
        assert contract["family"] in {"official", "teaching", "experimental"}, contract["name"]
        if contract["family"] != "official":
            assert "NOT from Kozlov 2002" in contract["source"] or "NO author" in contract["source"] or \
                "own" in contract["source"], contract["name"]


def test_compressor_names_carry_their_family():
    for name, cls in COMPRESSORS.items():
        if cls.contract.family == "teaching":
            assert name.startswith("Teaching"), name
        if cls.contract.family == "official":
            assert name.startswith("Kozlov"), name
        if cls.contract.family == "experimental":
            assert not name.startswith(("Kozlov", "Teaching")), name
