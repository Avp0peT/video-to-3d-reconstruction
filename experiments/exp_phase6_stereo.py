"""Phase 6 experiments S1–S8 (6S).

=========================================  =======================================================
S1  known correspondence, zero noise       the sanity rule: must reconstruct at machine precision
S2  unknown correspondence, zero noise     how many zero-cost answers exist, and how many are wrong
S3  noise sweep                            sigma -> relative 3D error, with the reprojection error
S4  baseline sweep                         baseline -> depth conditioning (a hypothesis, measured)
S5  symmetry vs alignment                  symmetry is benign; alignment is not (measured)
S6  degeneracy taxonomy coverage           every D-branch produced by a real configuration
S7  tie-breaking                           the residual cannot break a zero-cost tie; the code can
S8  invariants vs coordinates              the affine invariant sees what the residual hides
=========================================  =======================================================

This is the **stereo** form of the three-valued problem the earlier phases kept meeting: "the
residual fits", "the body was recovered" and "the answer was determined" are three different
questions.  Every row records the degeneracy label and the conditioning, because a 3D error from
nearly-parallel rays is a number with no content on its own.

The phase's sanity rule is asserted first and gates the exit code: a known-correspondence,
zero-noise, well-conditioned configuration must reconstruct at machine precision **before** any
result about correspondence is interpreted.

Run::

    python experiments/exp_phase6_stereo.py
    python experiments/exp_phase6_stereo.py --quick
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from kozlov_lab.stereo import (  # noqa: E402
    AffineTransform3D,
    DegeneracyKind,
    RecoveryOutcome,
    build_scene,
    classify_geometry,
    engines_agree,
    is_a_equivalent,
    match_correspondences,
    oracle_correspondence,
    profile_relative_difference,
    reconstruction_outcome,
    relative_3d_error,
    reprojection_error,
    rmse_3d,
    scene_names,
    triangulate_many,
    volume_ratio_profile,
)
from kozlov_lab.stereo.baseline.camera import look_at_camera  # noqa: E402
from _common import print_header, write_csv, write_json  # noqa: E402

FIELDNAMES = (
    "experiment", "scene", "orientation", "n_points", "baseline", "sigma", "degeneracy",
    "correspondence", "n_good_mappings", "n_valid_outcomes", "n_invalid_outcomes",
    "n_undecided_outcomes", "outcome",
    "engine_agreement", "truth_is_an_optimum", "pruned_is_an_optimum", "rmse_3d",
    "relative_3d_error", "reprojection_error", "profile_difference", "ray_angle_deg",
    "condition_number", "mean_depth", "nodes", "elapsed_ms", "note",
)

#: two orientations: the axis-aligned default, and one rotated out of alignment with the baseline.
#: The contrast between them is the whole point of S5, so it is a first-class axis here rather
#: than something the reader has to infer.
GENERIC = (
    AffineTransform3D.rotation("z", 30.0).matrix @ AffineTransform3D.rotation("y", 37.0).matrix
)
ORIENTATIONS: dict[str, np.ndarray] = {"axis_aligned": np.eye(3), "generic": GENERIC}


def _pair(baseline: float, focal: float = 500.0):
    return (
        look_at_camera([0, 0, 12], [0, 0, 0], [0, 1, 0], focal=focal),
        look_at_camera([baseline, 0, 12], [0, 0, 0], [0, 1, 0], focal=focal),
    )


def _row(**kwargs) -> dict:
    base = {name: "" for name in FIELDNAMES}
    base.update(kwargs)
    return base


def _shuffled(n: int, seed: int) -> tuple[list[int], tuple[int, ...]]:
    perm = list(np.random.default_rng(seed).permutation(n))
    return perm, tuple(perm.index(i) for i in range(n))


def _orient(body: np.ndarray, orientation: str) -> np.ndarray:
    return body @ ORIENTATIONS[orientation].T


def _conditioning(body: np.ndarray, centres) -> tuple[float, float, str]:
    """(min ray angle between the two cameras, condition number, degeneracy label)."""
    kind, evidence = classify_geometry(centres, body)
    return (
        float(evidence.get("min_ray_angle_degrees", float("nan"))),
        float(evidence.get("condition_number", float("nan"))),
        kind.value,
    )


def _outcome_breakdown(oracle, left, right, image_left, image_right, body, truth) -> collections.Counter:
    """Classify **every** member of the optimum set, not just the one an engine returns.

    Which member a solver returns is floating-point noise (measured: numpy 2.4.4 and 2.5.3 return
    different optima for the same cube while the optimum set is identical).  So the quantity worth
    reporting is the set's composition — how many equally-perfect answers are right and how many
    are wrong.  That is a property of the configuration; the engine's pick is not.
    """
    counter: collections.Counter = collections.Counter()
    for permutation in oracle.good_mappings:
        result = dataclasses.replace(oracle, mapping={i: int(c) for i, c in enumerate(permutation)})
        outcome = reconstruction_outcome(
            result, left, right, image_left, image_right, body, true_mapping=truth
        )
        counter[outcome.value] += 1
    return counter


# --------------------------------------------------------------------------- #
# S1/S2 — the sanity rule, then the correspondence result
# --------------------------------------------------------------------------- #
def s1_s2(scenes, rows):
    """S1 first, deliberately: the sanity rule must be read before any correspondence result."""
    for scene in scenes:
        for orientation in ORIENTATIONS:
            body = _orient(build_scene(scene).as_array(), orientation)
            n = body.shape[0]
            left, right = _pair(4.0)
            image_left, image_right = left.project(body), right.project(body)
            perm, truth = _shuffled(n, 20261007 + n)
            shuffled = image_right[perm]
            centres = [left.centre.tolist(), right.centre.tolist()]
            angle, condition, label = _conditioning(body, centres)

            # -- S1: known correspondence ---------------------------------- #
            started = time.perf_counter()
            points, failed = triangulate_many(
                left.projection, image_left, right.projection, image_right
            )
            rows.append(_row(
                experiment="S1_known_correspondence", scene=scene, orientation=orientation,
                n_points=n, baseline=4.0, sigma=0.0, degeneracy=label,
                correspondence="known", n_good_mappings=1, n_valid_outcomes=1,
                n_invalid_outcomes=0, n_undecided_outcomes=0, outcome=RecoveryOutcome.EXACT_MATCH.value,
                truth_is_an_optimum=True,
                rmse_3d=rmse_3d(points, body),
                relative_3d_error=relative_3d_error(points, body),
                reprojection_error=max(reprojection_error(left, points, image_left),
                                       reprojection_error(right, points, image_right)),
                ray_angle_deg=angle, condition_number=condition,
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
                note="" if not failed else f"triangulation failed at {len(failed)} points",
            ))

            # -- S2: unknown correspondence -------------------------------- #
            started = time.perf_counter()
            oracle = oracle_correspondence(left, right, image_left, shuffled)
            pruned = match_correspondences(
                left, right, image_left, shuffled, candidates_per_point=3
            )
            elapsed = (time.perf_counter() - started) * 1000.0

            breakdown = _outcome_breakdown(
                oracle, left, right, image_left, shuffled, body, truth
            )
            recovered, failed_unknown = triangulate_many(
                left.projection, image_left, right.projection,
                shuffled[list(oracle.permutation)] if oracle.permutation else shuffled,
            )
            rows.append(_row(
                experiment="S2_unknown_correspondence", scene=scene, orientation=orientation,
                n_points=n, baseline=4.0, sigma=0.0, degeneracy=label, correspondence="unknown",
                n_good_mappings=len(oracle.good_mappings),
                n_valid_outcomes=sum(
                    breakdown[k] for k in (
                        RecoveryOutcome.EXACT_MATCH.value,
                        RecoveryOutcome.EQUIVALENT_VIA_SYMMETRY.value,
                    )
                ),
                n_invalid_outcomes=breakdown[RecoveryOutcome.INVALID_RECONSTRUCTION.value],
                n_undecided_outcomes=breakdown[RecoveryOutcome.UNDECIDED.value],
                outcome=(oracle.verdict if oracle.mapping is None else
                         reconstruction_outcome(oracle, left, right, image_left, shuffled, body,
                                                true_mapping=truth).value),
                engine_agreement=engines_agree(oracle, pruned).value,
                truth_is_an_optimum=bool(truth in oracle.good_mappings),
                pruned_is_an_optimum=bool(
                    pruned.permutation is not None
                    and pruned.permutation in oracle.good_mappings
                ),
                rmse_3d="" if failed_unknown else rmse_3d(recovered, body),
                relative_3d_error="" if failed_unknown else relative_3d_error(recovered, body),
                reprojection_error=oracle.reprojection_error,
                ray_angle_deg=angle, condition_number=condition,
                nodes=pruned.diagnostics.get("nodes", ""), elapsed_ms=elapsed,
                note=f"oracle n!={oracle.diagnostics['n_bijections_examined']}",
            ))


# --------------------------------------------------------------------------- #
# S3 — noise
# --------------------------------------------------------------------------- #
def s3_noise(scenes, rows, sigmas, *, repeats: int):
    """The 3D error must grow with noise, and the reprojection error records how much of it is noise.

    Each ``(scene, sigma)`` is measured over **``repeats`` independent draws** and the **median** is
    reported, with the range kept in ``note``.  A single draw is not a measurement: the first
    version took one sample per point and produced a *non-monotone* curve whose final dip was
    simply the draw, which is exactly the "do not quote a single sample" rule this project already
    learned in Phase 2 (three of fifteen configurations there had errors that cancelled by luck).

    The per-draw seed is built from **integers**, never from ``hash()`` of a tuple containing a
    string: Python randomises string hashing per process, so an earlier version of this function
    produced *different numbers on every run* while still writing a seed into its provenance
    sidecar.  A run that is labelled reproducible and is not is worse than an unlabelled one.
    """
    scene_ordinal = {name: index for index, name in enumerate(scene_names())}
    for scene in scenes:
        body = build_scene(scene).as_array()
        left, right = _pair(4.0)
        clean_left, clean_right = left.project(body), right.project(body)
        scale = float(np.ptp(np.vstack([clean_left, clean_right]), axis=0).max()) or 1.0
        centres = [left.centre.tolist(), right.centre.tolist()]
        angle, condition, label = _conditioning(body, centres)
        for sigma in sigmas:
            medians: list[float] = []
            residuals: list[float] = []
            depths: list[float] = []
            for repeat in range(repeats):
                rng = np.random.default_rng(
                    [scene_ordinal[scene], int(round(sigma * 10_000)), repeat]
                )
                noisy_left = clean_left + rng.normal(scale=sigma * scale, size=clean_left.shape)
                noisy_right = clean_right + rng.normal(scale=sigma * scale, size=clean_right.shape)
                points, _failed = triangulate_many(
                    left.projection, noisy_left, right.projection, noisy_right
                )
                medians.append(relative_3d_error(points, body))
                residuals.append(max(reprojection_error(left, points, noisy_left),
                                     reprojection_error(right, points, noisy_right)))
                # ``mean |X_hat|`` is the diagnostic that *explains* the non-monotonicity: at high
                # noise the cloud drifts away from the origin, so the pointwise distance to a body
                # centred near the origin can shrink again even though the fit got worse.  Without
                # this column the explanation in the report would be an assertion; with it, the
                # reader can check the drift against the artifact.
                depths.append(float(np.linalg.norm(points, axis=1).mean()))
            rows.append(_row(
                experiment="S3_noise_sweep", scene=scene, orientation="axis_aligned",
                n_points=body.shape[0], baseline=4.0, sigma=sigma, degeneracy=label,
                correspondence="known",
                relative_3d_error=float(np.median(medians)),
                reprojection_error=float(np.median(residuals)),
                mean_depth=float(np.median(depths)),
                ray_angle_deg=angle, condition_number=condition,
                note=(f"{repeats} draws: rel3D {min(medians):.2e}..{max(medians):.2e}; "
                      "known correspondence, so a large error here is conditioning, not matching"),
            ))


# --------------------------------------------------------------------------- #
# S4 — baseline length
# --------------------------------------------------------------------------- #
def s4_baseline(rows, baselines):
    body = build_scene("irregular").as_array()
    for baseline in baselines:
        left, right = _pair(baseline)
        image_left, image_right = left.project(body), right.project(body)
        centres = [left.centre.tolist(), right.centre.tolist()]
        angle, condition, label = _conditioning(body, centres)
        note = ""
        try:
            points, failed = triangulate_many(
                left.projection, image_left, right.projection, image_right
            )
            if failed:
                note = f"refused at {len(failed)} points rather than inventing a depth"
            rmse = rmse_3d(points, body)
            rel = relative_3d_error(points, body)
            reproj: object = max(reprojection_error(left, points, image_left),
                                 reprojection_error(right, points, image_right))
        except ValueError as exc:  # noqa: BLE001 - a refusal is a result
            rmse = rel = reproj = ""
            note = f"refused: {type(exc).__name__}"
        rows.append(_row(
            experiment="S4_baseline_sweep", scene="irregular", orientation="axis_aligned",
            n_points=body.shape[0], baseline=baseline, sigma=0.0, degeneracy=label,
            correspondence="known", rmse_3d=rmse, relative_3d_error=rel,
            reprojection_error=reproj, ray_angle_deg=angle, condition_number=condition, note=note,
        ))


# --------------------------------------------------------------------------- #
# S5 — symmetry vs alignment
# --------------------------------------------------------------------------- #
def s5_symmetry(rows):
    """Both orientations of each scene, plus the pyramid contrast that isolates the cause.

    Measured claim: a square pyramid rotated 30° about its own axis is exactly as symmetric as the
    unrotated one, but only the axis-aligned one has an ambiguous correspondence.  So the cause is
    the alignment with the epipolar geometry, not the symmetry.
    """
    for scene in scene_names():
        for orientation in ORIENTATIONS:
            body = _orient(build_scene(scene).as_array(), orientation)
            n = body.shape[0]
            left, right = _pair(4.0)
            image_left, image_right = left.project(body), right.project(body)
            perm, truth = _shuffled(n, 11 + n)
            shuffled = image_right[perm]
            centres = [left.centre.tolist(), right.centre.tolist()]
            angle, condition, label = _conditioning(body, centres)
            oracle = oracle_correspondence(left, right, image_left, shuffled)
            breakdown = _outcome_breakdown(oracle, left, right, image_left, shuffled, body, truth)
            rows.append(_row(
                experiment="S5_symmetry_vs_alignment", scene=scene, orientation=orientation,
                n_points=n, baseline=4.0, sigma=0.0, degeneracy=label, correspondence="unknown",
                n_good_mappings=len(oracle.good_mappings),
                n_valid_outcomes=sum(
                    breakdown[k] for k in (
                        RecoveryOutcome.EXACT_MATCH.value,
                        RecoveryOutcome.EQUIVALENT_VIA_SYMMETRY.value,
                    )
                ),
                n_invalid_outcomes=breakdown[RecoveryOutcome.INVALID_RECONSTRUCTION.value],
                n_undecided_outcomes=breakdown[RecoveryOutcome.UNDECIDED.value],
                outcome=str(oracle.verdict), truth_is_an_optimum=bool(truth in oracle.good_mappings),
                reprojection_error=oracle.reprojection_error,
                ray_angle_deg=angle, condition_number=condition,
                note="D6 = symmetric but benign; D7/D8 = aligned, ambiguous",
            ))


# --------------------------------------------------------------------------- #
# S6 — taxonomy coverage
# --------------------------------------------------------------------------- #
def s6_taxonomy(rows):
    """Every branch produced by a real configuration, so none of them is dead code."""
    irregular = build_scene("irregular").as_array()
    cases = (
        ("irregular", DegeneracyKind.OK, irregular, [[0, 0, 12], [4, 0, 12]]),
        ("zero_baseline", DegeneracyKind.D1_ZERO_BASELINE, irregular, [[0, 0, 12], [0, 0, 12]]),
        ("tiny_baseline", DegeneracyKind.D2_SMALL_BASELINE, irregular, [[0, 0, 12], [1e-9, 0, 12]]),
        ("point_on_baseline", DegeneracyKind.D3_POINT_NEAR_EPIPOLE,
         np.array([[2.0, 0.0, 12.0], [5.0, 1.0, 9.0], [3.0, 2.0, 10.0], [6.0, 3.0, 8.0]]),
         [[0, 0, 12], [4, 0, 12]]),
        ("far_body", DegeneracyKind.D4_PARALLEL_RAYS, irregular + np.array([0.0, 0.0, 500.0]),
         [[0, 0, 12], [4, 0, 12]]),
        ("planar_grid", DegeneracyKind.D5_PLANAR_STRUCTURE,
         np.array([[i, j, 0.0] for i in range(3) for j in range(3)]), [[0, 0, 12], [4, 0, 12]]),
        ("pyramid_rot_z30", DegeneracyKind.D6_SYMMETRIC_OBJECT,
         AffineTransform3D.rotation("z", 30.0).apply(build_scene("pyramid").as_array()),
         [[0, 0, 12], [4, 0, 12]]),
        ("cube", DegeneracyKind.D7_COINCIDENT_RAYS, build_scene("cube").as_array(),
         [[0, 0, 12], [4, 0, 12]]),
        ("pyramid", DegeneracyKind.D8_EPIPOLAR_PLANE_MULTIPLICITY,
         build_scene("pyramid").as_array(), [[0, 0, 12], [4, 0, 12]]),
    )
    for label, expected, body, centres in cases:
        kind, evidence = classify_geometry(centres, body)
        rows.append(_row(
            experiment="S6_degeneracy_taxonomy", scene=label, orientation="",
            n_points=len(body), degeneracy=kind.value, correspondence="",
            outcome="MATCHES" if kind is expected else f"EXPECTED {expected.value}",
            ray_angle_deg=evidence.get("min_ray_angle_degrees", ""),
            condition_number=evidence.get("condition_number", ""),
            n_good_mappings=evidence.get("n_coincident_ray_pairs", ""),
            note=(f"coincident_pairs={evidence.get('n_coincident_ray_pairs', 0)} "
                  f"epipolar_pairs={evidence.get('n_epipolar_plane_pairs', 0)} "
                  f"symmetric_pairs={evidence.get('n_symmetric_pairs', 0)}"),
        ))


# --------------------------------------------------------------------------- #
# S7 — can anything break a zero-cost tie?
# --------------------------------------------------------------------------- #
def s7_tie_breaking(rows):
    """The residual cannot: every member of the optimum set fits to machine precision.

    So a reprojection-only pipeline, confronted with the cube, returns *some* zero-cost answer and
    has no way to prefer the true one.  Quantified: the share of the optimum set that is wrong.
    """
    for scene in scene_names():
        for orientation in ORIENTATIONS:
            body = _orient(build_scene(scene).as_array(), orientation)
            n = body.shape[0]
            left, right = _pair(4.0)
            image_left, image_right = left.project(body), right.project(body)
            perm, truth = _shuffled(n, 7 + n)
            shuffled = image_right[perm]
            oracle = oracle_correspondence(left, right, image_left, shuffled)
            if oracle.mapping is None:
                continue
            breakdown = _outcome_breakdown(oracle, left, right, image_left, shuffled, body, truth)
            good = len(oracle.good_mappings)
            valid = sum(breakdown[k] for k in (
                RecoveryOutcome.EXACT_MATCH.value, RecoveryOutcome.EQUIVALENT_VIA_SYMMETRY.value))
            # the invariant check: does the code flag a member the residual cannot separate?
            best = max(
                (profile_relative_difference(
                    volume_ratio_profile(points), volume_ratio_profile(body)
                ) for points in [_reconstruct(left, right, image_left, shuffled, p)
                                 for p in oracle.good_mappings] if points is not None),
                default=0.0,
            )
            rows.append(_row(
                experiment="S7_tie_breaking", scene=scene, orientation=orientation,
                n_points=n, baseline=4.0, sigma=0.0, correspondence="unknown",
                n_good_mappings=good, n_valid_outcomes=valid,
                n_invalid_outcomes=breakdown[RecoveryOutcome.INVALID_RECONSTRUCTION.value],
                outcome=("UNRESOLVABLE_BY_RESIDUAL" if good > 1 else "UNIQUE"),
                reprojection_error=oracle.reprojection_error,
                profile_difference=best,
                note=f"wrong share = {(good - valid) / good:.0%}" if good else "",
            ))


def _reconstruct(left, right, image_left, image_right, permutation):
    points, failed = triangulate_many(
        left.projection, image_left, right.projection, image_right[list(permutation)]
    )
    return None if failed else points


# --------------------------------------------------------------------------- #
# S8 — invariants vs coordinates
# --------------------------------------------------------------------------- #
def s8_invariants(rows, *, cap: int | None = None):
    """The affine invariant separates a wrong reconstruction that has zero reprojection error."""
    for scene in scene_names():
        body = build_scene(scene).as_array()
        n = body.shape[0]
        left, right = _pair(4.0)
        image_left, image_right = left.project(body), right.project(body)
        perm, _ = _shuffled(n, 3 + n)
        shuffled = image_right[perm]
        oracle = oracle_correspondence(left, right, image_left, shuffled)
        members = oracle.good_mappings
        truncated = cap is not None and len(members) > cap
        if truncated:
            # deterministic subsample: the optimum set of a symmetric body is huge, and the point
            # of this section is the *existence* of a wrong member with a perfect residual, not the
            # enumeration of all of them (S7 reports the full composition)
            members = members[:: max(1, len(members) // cap)][:cap]
        for permutation in members:
            points = _reconstruct(left, right, image_left, shuffled, permutation)
            if points is None:
                continue
            equivalent = is_a_equivalent(points, body)["equivalent"]
            try:
                difference: object = profile_relative_difference(
                    volume_ratio_profile(points), volume_ratio_profile(body)
                )
            except ValueError as exc:  # noqa: BLE001 - a degenerate reconstruction is a result
                difference = ""
                note = f"invariant undefined: {exc}"
            else:
                note = ("coordinate RMSE is only meaningful once the correspondence is right"
                        if not truncated else "deterministic subsample of the optimum set")
            rows.append(_row(
                experiment="S8_invariants_vs_coordinates", scene=scene, orientation="axis_aligned",
                n_points=n, baseline=4.0, sigma=0.0, correspondence="from the optimum set",
                outcome="VALID" if equivalent else "INVALID",
                rmse_3d=rmse_3d(points, body),
                relative_3d_error=relative_3d_error(points, body),
                reprojection_error=max(reprojection_error(left, points, image_left),
                                       reprojection_error(right, points, shuffled[list(permutation)])),
                profile_difference=difference, note=note,
            ))


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seed", type=int, default=20261007)
    args = parser.parse_args(argv)

    scenes = scene_names() if not args.quick else ["irregular", "cube"]
    sigmas = [0.0, 0.01, 0.05, 0.1, 0.5, 1.0]
    baselines = [1e-6, 1e-3, 0.05, 0.5, 1.0, 2.0, 4.0, 8.0]
    # >=3 independent draws per noisy point: the project's rule, and a single draw produced a
    # spurious non-monotone dip in the first version of this experiment.  9 draws in the full run
    # because the median of 5 was still visibly shaky in the extreme-noise regime.
    repeats_used = 3 if args.quick else 9

    print_header("Phase 6 — stereo experiments S1–S8")
    rows: list[dict] = []
    started = time.perf_counter()
    s1_s2(scenes, rows)
    print(f"  S1/S2 done ({len(rows)} rows)")
    s3_noise(scenes if not args.quick else ["irregular"], rows, sigmas, repeats=repeats_used)
    print(f"  S3 done ({len(rows)} rows)")
    s4_baseline(rows, baselines)
    print(f"  S4 done ({len(rows)} rows)")
    s5_symmetry(rows)
    print(f"  S5 done ({len(rows)} rows)")
    s6_taxonomy(rows)
    print(f"  S6 done ({len(rows)} rows)")
    s7_tie_breaking(rows)
    print(f"  S7 done ({len(rows)} rows)")
    s8_invariants(rows, cap=24 if args.quick else None)
    print(f"  S8 done ({len(rows)} rows)")
    elapsed = time.perf_counter() - started

    # -- the sanity rule, reported first ----------------------------------- #
    s1 = [r for r in rows if r["experiment"] == "S1_known_correspondence"]
    print("\n  S1 known correspondence, zero noise (the phase's sanity rule)")
    for row in s1:
        print(f"    {row['scene']:<12} {row['orientation']:<13} "
              f"rel3D={row['relative_3d_error']:.2e}  rpe={row['reprojection_error']:.2e}")
    worst = max(r["relative_3d_error"] for r in s1 if isinstance(r["relative_3d_error"], float))
    passed = worst < 1e-10
    print(f"    worst relative 3D error = {worst:.3e}  -> {'PASS' if passed else 'FAIL'}")

    s2 = [r for r in rows if r["experiment"] == "S2_unknown_correspondence"]
    if s2:
        print("\n  S2 unknown correspondence (the optimum set, not the engine's pick)")
        for row in s2:
            print(f"    {row['scene']:<12} {row['orientation']:<13} good={row['n_good_mappings']:>4} "
                  f"valid={row['n_valid_outcomes']:>3} invalid={row['n_invalid_outcomes']:>3} "
                  f"agree={row['engine_agreement']}")
        disagreements = [
            r for r in s2 if r["pruned_is_an_optimum"] is False
        ]
        print(f"    the pruned engine returned a non-optimum: {len(disagreements)}")
        truth_missing = [r for r in s2 if r["truth_is_an_optimum"] is False]
        print(f"    scenes where the true correspondence is NOT an optimum: {len(truth_missing)}")

    # -- S3: two monotonicity claims, checked separately --------------------- #
    s3 = [r for r in rows if r["experiment"] == "S3_noise_sweep"]

    def _non_monotone(scene: str, field: str) -> list[str]:
        series = sorted(
            (float(r["sigma"]), float(r[field])) for r in s3 if r["scene"] == scene
        )
        return [
            f"{scene}: sigma {s_prev}->{s_next} {field} {v_prev:.2e}->{v_next:.2e}"
            for (s_prev, v_prev), (s_next, v_next) in zip(series, series[1:])
            if v_next < v_prev
        ]

    scenes_in_s3 = sorted({r["scene"] for r in s3})
    dips = [d for scene in scenes_in_s3 for d in _non_monotone(scene, "relative_3d_error")]
    residual_dips = [d for scene in scenes_in_s3 for d in _non_monotone(scene, "reprojection_error")]
    print(f"\n  S3 noise ({repeats_used} draws per point, median reported; "
          f"{len(scenes_in_s3) * (len(sigmas) - 1)} steps)")
    steps = len(scenes_in_s3) * (len(sigmas) - 1)
    print(f"    reprojection error: {len(residual_dips)} non-monotone step(s) of {steps}")
    print(f"    pointwise 3D error: {len(dips)} non-monotone step(s) of {steps}")
    print("    => neither quantity is *unconditionally* monotone in the noise; the residual is the")
    print("       better behaved of the two (fewer dips, and a much smaller dip when it does dip).")
    print("       Do not upgrade that to \"the residual is monotone\": an earlier draft of this report")
    print("       said so, and the artifact below contradicts it.")
    for dip in dips:
        print(f"      {dip}")
    print("    => the pointwise 3D error stops being monotone once the reconstruction is no longer")
    print("       a *perturbation* of the body: the triangulated cloud drifts to large depth, so the")
    print("       pointwise distance to a reference near the origin shrinks again.  A 3D error is only")
    print("       an accuracy measure while the fit is good — which is why it must be quoted next to")
    print("       the residual, and why 6O compares invariants rather than coordinates.")
    # The drift is *read from the artifact*, not quoted from an earlier ad-hoc probe: the numbers
    # printed here are the `mean_depth` column of this very run, so a reader can check them.
    drift_scene = dips[0].split(":")[0] if dips else scenes_in_s3[0]
    drift = sorted(
        (float(r["sigma"]), float(r["mean_depth"]))
        for r in s3 if r["scene"] == drift_scene and r["mean_depth"] not in ("", "nan")
    )
    if drift:
        body_scale = float(
            np.linalg.norm(build_scene(drift_scene).as_array(), axis=1).mean()
        )
        print(f"       measured drift ({drift_scene}, mean |X_hat|, same runs as the table above):")
        print("         " + "  ".join(f"sigma={s:g}:{d:.2f}" for s, d in drift))
        print(f"       (the body's own mean |X| is {body_scale:.2f}, so the cloud ends up far outside it)")

    # every member of every optimum set must land in exactly one outcome bucket; a counter that
    # silently drops cases is how "0 unexplained" loses its meaning (Phase 4's F4.1 lesson)
    incomplete = [
        r for r in rows
        if r["experiment"] in ("S2_unknown_correspondence", "S5_symmetry_vs_alignment")
        and r["n_good_mappings"] != (
            r["n_valid_outcomes"] + r["n_invalid_outcomes"] + r["n_undecided_outcomes"]
        )
    ]
    print(f"\n  outcome partition incomplete in {len(incomplete)} row(s)  (expected 0)")
    for row in incomplete[:5]:
        print(f"    {row['experiment']} {row['scene']} {row['orientation']}")

    s7 = [r for r in rows if r["experiment"] == "S7_tie_breaking"]
    if s7:
        ambiguous = [r for r in s7 if r["n_good_mappings"] > 1]
        wrong = sum(r["n_invalid_outcomes"] for r in ambiguous)
        total = sum(r["n_good_mappings"] for r in ambiguous)
        print(f"\n  S7 tie-breaking: {len(ambiguous)} configurations have >1 zero-cost answer")
        if total:
            print(f"    of those {total} answers, {wrong} reconstruct a non-equivalent body "
                  f"({wrong / total:.0%}) — the residual cannot separate them")
        print(f"    worst invariant separation over the optimum sets: "
              f"{max((r['profile_difference'] for r in s7 if isinstance(r['profile_difference'], float)), default=0.0):.3e}")

    coverage = collections.Counter(
        r["degeneracy"] for r in rows if r["experiment"] == "S6_degeneracy_taxonomy"
    )
    mismatches = [
        r for r in rows
        if r["experiment"] == "S6_degeneracy_taxonomy" and r["outcome"] != "MATCHES"
    ]
    print(f"\n  S6 taxonomy: {len(coverage)} branches exercised, {len(mismatches)} mismatches")
    for name, count in sorted(coverage.items()):
        print(f"    {name:<32} {count}")

    # -- S8: how good a *detector* is the sorted invariant? ------------------ #
    def _separated(row: dict) -> bool:
        value = row["profile_difference"]
        if value == "":  # the profile was undefined: certainly not the same body
            return True
        return float(value) > 1e-6

    s8_invalid = [r for r in rows
                  if r["experiment"] == "S8_invariants_vs_coordinates" and r["outcome"] == "INVALID"]
    s8_separated = [r for r in s8_invalid if _separated(r)]
    s8_valid = [r for r in rows
                if r["experiment"] == "S8_invariants_vs_coordinates" and r["outcome"] == "VALID"]
    s8_false_alarm = [r for r in s8_valid if _separated(r)]
    print(f"\n  S8 invariant vs coordinates: {len(s8_valid)} valid, {len(s8_invalid)} invalid "
          f"reconstructions in the optimum sets")
    print(f"    the sorted profile separated {len(s8_separated)}/{len(s8_invalid)} invalid ones "
          f"and raised {len(s8_false_alarm)} false alarms on valid ones")
    print("    => a detector with a measurable miss rate, not a decision procedure "
          "(the decision is is_a_equivalent, which uses the indexed code)")

    target, meta = write_csv("phase6_stereo", rows, fieldnames=FIELDNAMES, seed=args.seed,
                             config={"quick": args.quick, "scenes": scenes, "sigmas": sigmas,
                                     "baselines": baselines,
                                     "orientations": list(ORIENTATIONS)})
    summary, summary_meta = write_json(
        "phase6_stereo_summary",
        {
            "experiment": "phase6_stereo",
            "title": "stereo experiments S1-S8",
            "s1_worst_relative_3d_error": worst,
            "s1_passed": bool(passed),
            "s2_pruned_returned_a_non_optimum": len(
                [r for r in s2 if r["pruned_is_an_optimum"] is False]
            ) if s2 else None,
            "s2_true_correspondence_not_an_optimum": len(
                [r for r in s2 if r["truth_is_an_optimum"] is False]
            ) if s2 else None,
            "s7_configurations_with_multiple_optima": len([r for r in s7 if r["n_good_mappings"] > 1]),
            "s7_optimum_answers": sum(r["n_good_mappings"] for r in s7 if r["n_good_mappings"] > 1),
            "s7_invalid_answers": sum(r["n_invalid_outcomes"] for r in s7 if r["n_good_mappings"] > 1),
            "s6_taxonomy_mismatches": len(mismatches),
            "outcome_partition_incomplete_rows": len(incomplete),
            "s3_repeats_per_point": repeats_used,
            "s3_residual_non_monotone_steps": len(residual_dips),
            "s3_pointwise_3d_non_monotone_steps": len(dips),
            "s8_valid_reconstructions": len(s8_valid),
            "s8_invalid_reconstructions": len(s8_invalid),
            "s8_invalid_separated_by_the_invariant": len(s8_separated),
            "s8_invalid_missed_by_the_invariant": len(s8_invalid) - len(s8_separated),
            "s8_false_alarms_on_valid_reconstructions": len(s8_false_alarm),
            "rows": len(rows),
            "elapsed_seconds": elapsed,
            "note": ("the sanity rule is asserted first: a known-correspondence, zero-noise, "
                     "well-conditioned configuration must reconstruct at machine precision before "
                     "any correspondence result is interpreted; the optimum-set composition is "
                     "reported instead of the engine's pick because the pick is float-noise "
                     "dependent"),
        },
        seed=args.seed, config={"quick": args.quick},
    )
    print(f"\n  rows={len(rows)}  elapsed={elapsed:.1f}s")
    for path in (target, meta, summary, summary_meta):
        print(f"  [written] {path}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
