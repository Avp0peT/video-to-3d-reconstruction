# Learned object tracking: local TAPNext++ experiment

2026-10-22. Experimental candidate generation, not identity/geometry acceptance.

## Sources and mechanism

- TAPNext++: https://tap-next-plus-plus.github.io/
- Pinned implementation: https://github.com/google-deepmind/tapnet/tree/730cda1c730877cfedbe01bf87fb1cadb78a565d
- CoTracker3 joint/online tracking reference: https://github.com/facebookresearch/co-tracker
- Track-to-camera/triangulation architecture reference: https://vggsfm.github.io/

The implementation runs one model with persistent temporal state and a fixed
query batch. SAM supplies the object domain. Invisible predictions, absent SAM
masks and outside-mask coordinates remain excluded candidates; nothing snaps a
point onto the silhouette or fills a NO_MATCH mask. Recovery is a proposed
identity, not proof of the same physical surface location.

## Reproduce on existing inputs

Run from the repository with root and src in PYTHONPATH and
PYTHONDONTWRITEBYTECODE=1. Use the existing light Python for orchestration.
Prepare accepts a fresh output, frozen INPUTS.json and MASKS.json source runs:

```powershell
python -B -X utf8 scripts/track_learned_object.py prepare --output <fresh-output> --source-run <texture-source> --mask-run <mask-source> --upstream outputs/toolchains/tapnet-730cda1 --checkpoint outputs/toolchains/tapnextpp-assets/tapnextpp_ckpt.pt --points 128
python -B -X utf8 scripts/track_learned_object.py run --output <fresh-output>
```

The actual worker uses the existing isolated SAM Python, threads=2, 256x256
model input, FP16 autocast, strict checkpoint loading and mmap. Its query
coordinates convert OpenCV centers using +0.5 before resize and -0.5 after
inverse resize. Certainty computation uses float32. The official weights have
SHA256 cb96a43444ccb4fbdb25d800b88c7ba196179a526e78f01b021a16b1c1eff6da.

## Actual evidence

All paths below are under
`outputs/delegation/NEW_VIDEO/20261019_new01/learned_tracking/`.

| Run | Work performed | Result |
|---|---|---|
| tapnextpp01 | 64 frames, one original query | 42 visible candidates, 18 uncertain, 4 masks unavailable; model 3.432 s |
| tapnextpp12801 | Same 64 frames, 128 queries | 5409 visible candidates, 2374 uncertain, 379 mask unavailable, 30 outside mask; model 2.967 s |
| reverse12801 | All 105 visible queries at raw600, reversed 600..555 | 3172 common-visible comparisons; 1757 within declared 4px heuristic, 1415 outside; 1658 unknown |
| geometry12801 | Native pair verification and mapping of visible candidates | Largest separate component 54 views, 305 points; NOT_QUALITY_ACCEPTED |

Single and batch runs completed with process-tree reclamation and source/code
hash checks. GPU allocator reserved peaks were 1,667,235,840 and 1,700,790,272
bytes. The Job commit peak was approximately 5.52 GiB for each. The 12 GiB Job
limit is a process-tree commit limit; the 7 GiB CUDA reserved check is a soft
frame-boundary check, not a driver-wide hard cap. Report rendering occurs after
the worker supervisor and is not covered by its 600 s limit.

These timings are single measurements. Changing query count changes attention
context and model outputs: query0 differs by a mean 17.50 px (maximum34.51 px)
between single and batch runs, and 3/64 status labels differ. Therefore this is
not an output-equivalent batching speedup claim.

The reverse check has median discrepancy3.40 px and p95=14.30 px. Query0 returns
to within3.43 px of its original raw555 query, but discrepancies elsewhere reach
9.11 px. Missing/occluded intervals remain unknown. A cycle pass is a heuristic
consistency observation, not physical identity ground truth.

Native geometry uses a separate learned-candidate schema, not the old measured
SIFT/balanced source schema. It writes no descriptors, verifies pair geometry
and performs one incremental-mapping call. Four components remain separate.
The largest has305 points for128 query IDs:68 query IDs split into multiple 3D
points; no 3D point mixes different query IDs. Its259 multiview-consistent and46
two-view provisional points are not quality accepted. Reprojection median1.31
px does not establish physical accuracy. Focal length841.30 px and radial
k=-0.10491 are estimated, not calibrated. Scale, surface and CAD remain unknown.

## Inspect

- `tapnextpp01/index.html`: original query with local texture crops.
- `tapnextpp12801/index.html`: candidate population and visibility states.
- `reverse12801/CYCLE_ASSESSMENT.json`: per-query, per-frame disagreement.
- `geometry12801/report01/index.html`: separate rotatable point clouds and PLY.
- `geometry12801/native/GEOMETRY_EVIDENCE.json`: ray/depth/reprojection evidence.

## Remaining correctness work

Use image-supported re-identification and cycle/geometry agreement to prevent
one query from silently representing different surface locations. Extend query
seeding to additional views only with explicit identity handling; the current
64-frame, first-view batch cannot cover all newly exposed object surfaces.

Read-only review found two publication-path gaps in the current version:
standalone `report` does not independently verify source SHA, and run's report
gate does not explicitly require a clean supervisor status/errors list. The
actual runs here used normal run with pre-render hash checks and completed,
error-free supervisor records. Preserve those records; close the general-path
gaps in a newly bound runtime revision before broader deployment.
