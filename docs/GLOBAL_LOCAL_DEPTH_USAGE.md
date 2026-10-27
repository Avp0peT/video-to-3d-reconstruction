# Whole-preserving local depth experiment

This explicit experimental mode is integrated into `scripts/run_region_mask_review.py`.
It is not a new default for all reconstruction, an OpenMVS per-pixel-prior interface,
the complete Cascade/ACMM algorithm, or an implementation of a Kozlov theorem.

## Automatic entry

```powershell
& C:/Python313/python.exe -B -X utf8 scripts/run_region_mask_review.py `
  --global-local-experiment <completed-regional-experiment-directory> `
  --reference-image-id <global-camera-ID> `
  --roi-spec <reference-domain-cell-subset.json> `
  --output <new-output-directory>
```

The equivalent standalone CLI is `scripts/refine_local_mvs_depth.py`, with
`--source-experiment` instead of `--global-local-experiment`. Run from the repository
in an environment containing NumPy, OpenCV and Pillow. The real delivery records
the exact verified versions and module paths; it installs nothing.

Do not reuse an existing output. A global camera ID is not a video decode index.
The explicit ROI must name the matching decode index and integer cell bounds.
No glove/finger label, fixed coordinate rectangle, or per-object anatomy is coded
into the method. Automatic region selection and reconstruction of every part are
not implemented by this entry.

## What executes

1. Verify the completed WHOLE depth bindings, historical source-plan hash, image,
   mask and validity bytes, and embedded camera identity/K/R/C.
2. Use two to five existing neighbor depth maps selected from the native reference
   neighbor list. Source images stay full-frame; no copied reference ROI is used.
3. In the reference ROI, search 96 inverse-depth planes within the original native
   depth range, using 5x5 foreground-only grayscale patches. Both selected sources
   must have NCC >= 0.75; low-variation patches and ambiguous hypotheses reject.
4. Require at least two of the same sources to support photometry AND source-depth
   agreement/roundtrip reprojection. Two geometric conflicts reject. Missing data
   remains unknown.
5. Preserve every original positive WHOLE depth exactly. Only validated holes are
   eligible for additive combination. Existing wrong depths are not repaired.
6. Export per-stage images, candidate arrays, combined depth, point clouds and a
   two-scene offline viewer. Bind/read back outputs and recheck inputs.

A supported reference depth has a bounded soft prior; a missing reference depth
does not. For holes, whole-model constraints are shared cameras and neighbor-depth
checks, not a fabricated reference prior. The default camera convention is axial
camera z, world-to-camera R and world-space camera centre C.

`RESULT.json` records geometry as unaccepted even when execution returns zero.
Point-cloud concatenation does not deduplicate observations or make a surface.
The original fused cloud is captured at execution time, not retroactively assigned
a historical hash. Depth and XYZ arrays are finite; `LOCAL_PROPOSAL.npz` photometric
diagnostics use documented +/-infinity for unavailable cost/gap values.

## Actual 2026-10-25 result and limitation

For the existing decode-840 side-branch evaluation ROI: 19,998 foreground pixels,
2,737 existing positive depths, two photometric candidates, zero accepted additions.
All existing depths remain. Only 203 pixels pass both the reference texture and
foreground-patch rules; all 203 already have WHOLE depth. Thus this particular
single-scale patch rule cannot supply new observations in the missing portion.
This does not prove the video or global/local reconstruction concept unusable.

The first supervised launch failed before calculation because the isolated child
could not import user-site NumPy. The second uses a recorded explicit CPU package
root, with no installations or parameter changes. Total launches=2; actual depth
searches=1. See the delivery's `HANDOFF.md` for exact evidence.

## Explicit alternative methods

The automatic entry accepts `--local-method regional-plane-v2` and
`--local-method silhouette-interval-v1`. The original small-patch method remains
the default. These are CPU experiments on an already calibrated source experiment,
not an automatic guarantee for every object or a new end-to-end default.

`regional-plane-v2` fits inverse-depth planes from existing geometrically supported
depth anchors in overlapping 32/64/96-pixel windows. A candidate stays inside its
anchor hull and the foreground; it must agree with original source depths. Whole
depths are preserved. This is multi-window plane inference, not the complete ACMM
image pyramid or TAPA-MVS implementation. In the fixed decode-840 evaluation ROI,
its first real run produced zero eligible fits and zero additions. That observation
does not establish why every window was rejected or prove local methods impossible.

`silhouette-interval-v1` runs a separate consumer and never inserts its estimates
into accepted depth. It samples reference-mask rays every three pixels at 128
inverse-depth positions within the original native bounds. All three admission
silhouettes must be known foreground; unknown/out-of-view samples cannot pass.
A ray needs a single surviving segment, with known-background brackets at both
ends. Range-clipped and disjoint segments stay unknown. Front/back midpoint
estimates and transition brackets are exported as separate point layers, alongside
the unchanged whole cloud. No mesh, gap filling, or hidden-concavity recovery is
performed. Sampling can miss thin intervals: neither midpoint nor sampled bracket
is a certified continuous visual hull. Masks, rigidity and cameras remain assumptions.

Both alternatives reserve the final native neighbor for diagnostics after proposal
construction. This exclusion applies only to the current proposal stage. The
historical camera/depth solution may already use that image, so it is not an
independent ground-truth evaluation. Real image comparisons, input bindings,
readback and uncertainty statuses are generated by the script.

```powershell
& C:/Python313/python.exe -B -X utf8 scripts/run_region_mask_review.py `
  --global-local-experiment <completed-regional-experiment-directory> `
  --reference-image-id <global-camera-ID> `
  --roi-spec <reference-domain-cell-subset.json> `
  --local-method silhouette-interval-v1 `
  --output <new-output-directory>
```

The standalone `refine_local_mvs_depth.py` offers the two depth methods only;
silhouette inference is explicitly separate from depth combination.

## Progressive use of additional calibrated views

`--local-method silhouette-multiview-v2` loads all native calibrated images and
silhouettes in the bound source plan. The first three original native neighbors
form a matched baseline. Remaining camera IDs are sorted; the final ID is reserved
for this stage's diagnostics, and all other views are added as constraints in that
fixed order. No source is dropped because its observed result is inconvenient.

Every stage uses the same three-pixel reference grid, 128 inverse-depth positions
and original range. Saved baseline/final sample sets and spans make set inclusion
inspectable: additional constraints cannot introduce surviving samples. The number
of bounded rays can increase or decrease, so it alone is not an accuracy metric.
Leave-one-out pressure is also reported without changing the adopted constraint
set. A high-pressure source is not automatically a bad source. Empty intersection
does not prove the physical object is absent.

A calibrated image with no native depth can still constrain silhouettes. Its depth
array is an explicitly unknown placeholder (`depth_available=false`), not a measured
depth; its K/R/C come from the historically bound camera plan. The overall model,
all old depth maps and old runs remain unchanged. This mode does not yet create or
accept surfaces. Masks, calibration and rigid alignment across the selected video
times still need evaluation. All source projection images and per-stage results
are generated automatically.

## Diagnose boundary alignment before changing geometry

```powershell
& C:/Python313/python.exe -B -X utf8 scripts/run_region_mask_review.py `
  --alignment-review <saved-multiview-real-run-directory> `
  --output <new-review-directory>
```

This explicit review reads the saved ROI, cameras, depth and full images/masks and
verifies their bindings. It performs two separate diagnostics: image gradients
along mask-boundary normals, and image-only forward/backward LK controls compared
with old-depth projections. The tracker is not initialized from 3D predictions.
Weak or ambiguous image edges and insufficient controls remain unknown.

Controls are selected in fixed spatial tiles. BODY controls exclude the ROI plus
a six-pixel halo; opposite train/test tiles have a four-pixel edge margin, keeping
their 7x7 comparison patches disjoint. The coarser optical-flow pyramid still uses
shared image information, so this is not an independent ground-truth evaluation.
A median 2D translation is fitted only from BODY training controls when support is
sufficient, evaluated on BODY test controls and reported separately for the ROI.
It is a diagnostic hypothesis, never a camera, mask or depth correction.

The review automatically exports all attempted/accepted counts, ambiguous/unknown
boundary states, per-view images and before/after hypothetical projection figures.
Image-edge agreement does not prove segmentation truth; flow agreement does not
prove the same physical point; a common image displacement does not uniquely
identify a focal length, pose, depth or deformation error. It never adopts geometry.

## Generate separate model layers automatically

```powershell
& C:/Python313/python.exe -B -X utf8 scripts/run_region_mask_review.py `
  --model-from-multiview <saved-multiview-real-run-directory> `
  --output <new-model-directory> --blender
```

The frozen image/depth/camera/mask bindings and temporal decode map are checked.
The entry runs neighboring image tracks with reverse-chain closure and fixed-camera
triangulation, then native depth-grid meshing with whole-triangle mask footprints.
Reference ROI seeds do not require old depth. Occlusion or failed identity stops
that tracking direction; missing observations are not interpolated.

Depth-supported faces and an independent orange voxel silhouette proposal are
saved separately as PLY/OBJ, with provenance arrays and offline viewers. The old
whole cloud remains byte-identical. The shell uses the old whole bounds plus 10%
per-axis span as a finite search domain; it is not depth fusion or observed shape.
Center sampling plus eight-corner disagreement records cannot guarantee thin-gap
or hidden-concavity recovery. UNKNOWN/ROI/domain boundaries are not capped.
The prior posthoc ROI remains an evaluation selection, not automatic semantic parts.

No smoothing, hole filling or cross-layer welding is performed. Separate view
patches may overlap or self-intersect; geometry, completeness, CAD and physical
scale remain unaccepted. `--blender` uses the installed exporter to save and reopen
the same geometry under supervision. Export success is not geometry acceptance.
CPU modeling has a 900-second cooperative limit; each Blender phase has its own
90-second supervised budget. No SAM3, GPU inference or OpenMVS rerun is invoked.

## Primary references

- [OpenMVS v2.4.0 dense implementation](https://raw.githubusercontent.com/cdcseacave/openMVS/v2.4.0/libs/MVS/SceneDensify.cpp)
- [Cascade Cost Volume, CVPR 2020](https://arxiv.org/abs/1912.06378)
- [ACMM, CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Xu_Multi-Scale_Geometric_Consistency_Guided_Multi-View_Stereo_CVPR_2019_paper.html)
- [CMU: Visual Hull Construction, Alignment and Refinement Across Time](https://publications.ri.cmu.edu/visual-hull-construction-alignment-and-refinement-across-time)
- [TAPA-MVS: Textureless-Aware PatchMatch Multi-View Stereo](https://arxiv.org/abs/1903.10929)
- [MIT: Image-Based Visual Hulls, SIGGRAPH 2000](https://people.csail.mit.edu/wojciech/IBVH/index.html)
- [OpenCV: Optical Flow tutorial](https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html)
- [OpenCV: Tracking API and LK initialization flags](https://docs.opencv.org/4.x/dc/d6b/group__video__track.html)
