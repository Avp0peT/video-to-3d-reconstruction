# COLMAP to OpenMVS bridge

`scripts/prepare_mvs.py` is an independently bounded format bridge. It does not
perform SFM, dense reconstruction, mesh repair or CAD construction.

Validation state: **FORMAT_BRIDGE_VERIFIED** for the two prepared models listed
below (4 images / 312 points and 44 images / 2393 points). The earlier
`IMPLEMENTED_NOT_VERIFIED` state and the R1 batch that stopped in the verify stage
are superseded: that stop came from an over-strict correspondence rule, not from a
damaged conversion. See `outputs/delegation/MVS_B/<tag>_r2/HANDOFF.md`.

Run with the project's pinned pycolmap 4.2.1 Python. The script sets five numeric
runtime thread controls to 2 before importing NumPy/pycolmap. All native calls
run in task-owned child processes with a shared deadline (default 120 seconds),
a 200 MiB output threshold, task-specific stdout/stderr and argument arrays.
Output must be a new directory. Existing sparse models and their images are
hashed before and after execution; no source files or SQLite databases are
written. Timeout cleanup targets the task's own process tree only.

```powershell
& $PinnedPython -B scripts/prepare_mvs.py `
  --sparse $NativeModelDirectory `
  --image-root $NativeRunDirectory `
  --interface-colmap $PortableInterfaceCOLMAPExe `
  --output $FreshBridgeDirectory `
  --wall-seconds 120 --max-output-mib 200
```

The matching and acceptance logic lives in three pure data functions
(`point_records`, `compatibility_edges`, `correspondence`) that take plain records
and import their numerical dependencies on demand. Native processes are not needed
to test them; `MVS_BRIDGE_MODULE` points the same assertions at a mutant copy for
negative-control runs.

Intended stages, all required for PASS:

1. Reject duplicate-view tracks and broken image/point reciprocal associations.
2. Undistort into PINHOLE/SIMPLE_PINHOLE, long side at most 1024, copy policy
   `copy`, two native threads. Verify point identities and XYZ, poses, camera
   centers, image identities, and dimensionless normalized rays at 1e-5.
3. Import the undistorted model into an OpenMVS scene, then export it into a
   fresh COLMAP directory. This handles the pixel-center convention internally;
   the wrapper must not apply another half-pixel shift.
4. Independently check calibrated geometry, colors and per-point image
   membership. Point IDs may be renumbered and point geometries may repeat, so an
   **existence-of-perfect-bijection** check replaces the earlier one-candidate
   rule: a one-to-one assignment covering *every* point on both sides must exist,
   and each matched pair must satisfy the XYZ tolerance, an identical RGB triple
   and an identical complete view-member set. A unique spatial correspondence is
   **not** required. Duplicated geometry that is legitimate is returned as a
   deterministic witness plus explicit ambiguity groups; a pair sharing XYZ but
   differing in color or membership is never interchangeable. The maximum XYZ
   difference is bounded by 1e-6 times the maximum of 1 and the largest absolute
   input coordinate, allowing float32 vertex storage. Intrinsic matrix tolerance
   is 1e-7 pixel; pose tolerance is 1e-8 relative and absolute. These are
   representation checks, not physical accuracy limits.

The bijection is found with a maximum-bipartite-matching search, so overlapping
candidate sets are resolved by augmenting re-assignment instead of a greedy first
fit, and a case where every point has a neighbour but no perfect bijection exists
is reported as a failure. Ambiguity is *detected*, not assumed: an edge can only be
used by a perfect matching if it lies on an alternating cycle, so strongly
connected components of the directed matched/unmatched graph give the set of edges
that every candidate assignment may use. A source point is reported as
uniquely determined exactly when one such edge remains.

An exit code of zero from InterfaceCOLMAP alone does not certify conversion:
the expected output models and independent numerical comparisons must succeed.
`--internal verify --source-dir <dir>` re-checks an existing
undistorted/roundtrip pair without rerunning any external program and writes only
into the new `--output` directory.

The upstream [OpenMVS v2.4.0 InterfaceCOLMAP implementation](https://github.com/cdcseacave/openMVS/blob/v2.4.0/apps/InterfaceCOLMAP/InterfaceCOLMAP.cpp)
imports pinhole cameras, applies its own pixel-center conversion, and exports
newly projected feature positions with zero residual fields. Therefore an MVS
roundtrip cannot prove preservation of original SIFT coordinates or measured
reprojection errors. The immutable native model remains authoritative for
those. Rigs and frames are flattened to the per-image absolute camera poses;
this is not a rig/frame identity-preserving archival format.

No successful format bridge would establish complete object coverage, absolute
scale, material, surface fidelity, or readiness for a closed CAD solid.
