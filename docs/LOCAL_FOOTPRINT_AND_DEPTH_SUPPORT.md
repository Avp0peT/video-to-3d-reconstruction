# Local footprint and missing-region depth review

These are explicit diagnostic/candidate modes. They do not accept an object,
change the default surface profile, or fill an unobserved part.

## Background-gap footprint

`scripts/build_region_surface.py face-local-worker` accepts
`--footprint-mode closed_cells`. Omission retains the existing `disk` mode.
The new mode checks all closed pixel cells intersecting the projected triangle,
including boundary contacts. Every cell must be valid and inside the bounded
background gap. A conservative distance margin covers the full cell, not just
its centre. The four actual interior pixel centres, two views separated by at
least 8 degrees, original-probe protection and positive-support veto remain.

```powershell
python -B -X utf8 scripts/build_region_surface.py face-local-worker `
  --source <regional-surface-run> --gate <v3-face-gate-run> `
  --output <fresh-output> --footprint-mode closed_cells
```

Use the existing process supervisor for real runs. The worker keeps its
420-second cooperative deadline. New metadata records the explicit mode.
This avoids some enclosing-circle false negatives; it does not establish
separation of anatomical parts or solve subpixel/unknown-depth observations.
The candidate is reversible, and no interpolated vertex is a new measurement.

## Missing-region depth support

```powershell
python -B -X utf8 scripts/audit_region_depth_support.py `
  --source <regional-surface-run> --depth-run <compatible-native-depth-run> `
  --region-json <frozen-grid-region.json> --output <fresh-depth-review>
python -B -X utf8 scripts/analyze_region_depth_ledger.py `
  --source <regional-surface-run> --ledger-run <fresh-depth-review> `
  --region-json <same-frozen-grid-region.json> --output <fresh-tier-review>
```

The grid JSON contains `decode_index` and `cells[].bounds=[x0,y0,x1,y1]` in
the bound native camera domain. It is a declared review region, not an anatomical
label or ground truth. Camera, image, mask, validity and depth identities are
checked. The analysis neither interpolates a new depth map nor launches MVS.

Sampling uses stride 2 and a fixed-seed uniform priority reservoir capped at
50,000 source-depth samples. Three other depth views with separated directions
are required for conditional support; the generating and target views do not
count toward those three. A target with no depth stays unknown.

Always inspect both tiers:

- `SUPPORT_ONLY`: conditional agreement; may contradict other stored depths.
- `NO_KNOWN_FRONT_CONFLICT`: no sampled view places the candidate in front of
  its known depth by more than 2%. This conservative secondary diagnostic is
  not a physical correctness test. Being behind another depth can be ordinary
  occlusion and is reported separately.

Counts include unique target pixels and actual raw-triangle missing-area
coverage. Multiple depth samples at the same location are not new independent
measurements. The second script validates the first ledger and writes separate
PLY/PNG tiers, bindings and a machine-readable matrix. Its 600-second bound is
cooperative; routine automation should supply an outer process supervisor.

## Current glove evidence

`outputs/delegation/SHAPE_REPAIR/20261024_targeted03/` records 67-view footprint
evaluation and the frozen frame-840 missing-region review. It remains
`REPAIR_INCOMPLETE_CANDIDATE_NOT_ADOPTED`. Small gap-raster improvements coexist
with small foreground losses. Conditional thumb candidates mostly disagree
with another depth estimate; the no-known-front-conflict tier covers only
3 pixels of the 11,478-pixel raw-missing region. This is not a complete thumb,
proof of deformation, or justification to force the candidates into a mesh.
