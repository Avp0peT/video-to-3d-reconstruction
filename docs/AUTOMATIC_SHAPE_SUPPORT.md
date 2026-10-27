# Automatic local shape support review

The regional reconstruction entry `scripts/build_region_surface.py run` now adds
`shape-support-worker` after native reconstruction and candidate processing. It
also runs on an available raw surface when later candidate processing failed.
The first failure remains recorded. The stage adds no model inference or native
reconstruction; it reads existing bound images, cameras, masks, depths and meshes.

## Distinct measurements

- The native dense point cloud is the measured reconstruction input. Candidate
  subdivision vertices do not count as new observations.
- All registered native depth files are joined by image name, global image ID,
  dimensions, K, R and C; absent depths are unknown.
- Sixteen bounded diagnostic views are selected using four temporal strata and
  camera direction spread. Missing or low-depth views remain eligible, so the
  report cannot hide holes by selecting only well reconstructed views.
- Each input mask's bounding box is partitioned into 8 by 12 regions. Regions
  with fewer than 64 valid foreground pixels or less than 80% valid mask area
  are unknown. A regional warning does not label a point or face as wrong.
- Dense points are projected with the bound native camera. A nearest-pixel
  z-buffer prevents a farther hidden point from supplying support when a nearer
  point disagrees. Stored camera-Z agreement is checked within 2%; an 8-pixel
  neighborhood of agreeing points is recorded separately from mere projections.
- Raw and candidate opaque triangle silhouettes are compared in the same image
  domain. Low original coverage and new foreground loss are separate reasons.
  High whole-image overlap cannot cancel a local missing protrusion.
- Input masks touching the frame boundary are reported as clipping risk. This
  is not a segmentation error label: inspect the source image before attributing
  a missing tip to SAM, depth estimation or view coverage.

The policy is explicit in each `PLAN.json`. Thresholds are engineering warnings,
not a guarantee of physical density, accuracy, visibility, rigidity or topology.

## Result and automatic behavior

`RESULT.json` uses `shape-support-audit-v1`. `REVIEW_REQUIRED`, missing evidence,
source drift or a prior pipeline failure prevents automatic promotion of the
postprocessed candidate as the preferred viewer. Raw and candidate artifacts
remain available. Even `NO_FLAGGED_REGRESSION_NOT_QUALITY_ACCEPTANCE` does not
claim a complete object or accepted CAD geometry. Neither status changes masks,
vertices or faces, and no automatic hole filling occurs.

Run only this read-only review against an existing regional reconstruction:

```powershell
python -B -X utf8 scripts/build_region_surface.py shape-support-worker `
  --source <regional-surface-run> --candidate-run <local-candidate-run> `
  --output <new-output-directory>
```

Omit `--candidate-run` for raw-only diagnosis. The automatic pipeline uses a
600-second / 4-GiB supervised budget; the kernel has a 540-second deadline.
Source/mask/camera/mesh hashes and pixel-domain checks precede comparison.
Output includes source / depth-supported points / raw silhouette / candidate
silhouette panels and per-region machine-readable evidence. Unprojected views,
unseen surfaces and independent shape quality remain explicitly unassessed.
