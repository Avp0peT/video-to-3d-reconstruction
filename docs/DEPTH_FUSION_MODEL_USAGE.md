# Native depth fusion model

`scripts/run_depth_fusion_model.py` authenticates a completed `global_local_model_pipeline`
source (`RESULT.json`, `PLAN.json`, `SURFACE.json`, `SURFACE_PROVENANCE.npz` and its parent
SHA256 manifest), loads its original calibrated images/masks/validity/native depths,
then automatically performs coverage classification, TSDF integration, open-surface
extraction, source comparison, per-view diagnostics, mesh export, and optional Blender
creation plus readback. It does not accept inferred silhouette geometry as native depth.

The default CLI now writes `raw/` plus `final/`: raw TSDF is diagnostic. The automatic
finalization preserves every original local triangle and admits added TSDF triangles
only after a complete projected pixel-square mask test, at least two complete
foreground views, at least two common near-band sources over the extraction cell,
and a supported centroid depth verdict. It fails on any decrease of projected
foreground coverage or increase in known-background projection. These are regression
checks, not independent shape acceptance. Old evidence conflicts remain disclosed.
Use `--raw-only` explicitly to omit this finalization; do not present that diagnostic
as a boundary-safe replacement model. One shared CLI deadline covers both stages.

To finalize an already finished raw result without repeating native depth fusion:

```powershell
C:\Python313\python.exe -B -X utf8 scripts/finalize_depth_fusion_model.py --fusion-run PATH_TO_RAW --output FRESH_FINAL_PATH --blender
```

The final Blender contains separate original-surface and audited-addition objects in
the same coordinate system. They are not silently welded, repaired or made watertight.
`combined_surface.obj` and `.ply` contain both surface layers. Global isolated raw
points remain in historical exports; they do not create new surface triangles here.

## Include the full calibrated parent rather than a temporal subset

Use `--full-depth-parent PATH_TO_COMPLETED_PARENT_SURFACE_ROOT` along with
`--source-model` naming the historical local comparison model. The adapter binds
every parent plan frame, frame-map entry, mask/validity, scene and existing native
depth. Shared pixels and K/R/C must match the local source; no pose estimation or
registration is silently performed. Historical parent depths may differ from
depths estimated in a temporal subset, even for the same image IDs. Their original
mask-consumption/quality status remains a limitation, not upgraded by reuse.

Up to 128 calibrated mask views and 64 available native-depth sources are supported.
All available depth sources are fused (no missing depth is replaced by zero evidence).
Cross-view diagnostic sampling chooses one disclosed regular stride to stay below
180,000 samples; it never takes a prefix of frames or points. Full-resolution native
depth integration and per-pixel availability are unaffected by this audit sampling.
Source bitsets use uint64 for TSDF's up-to-64 sources; the full-mask gate uses uint64
words for more than64 views. The real inventory and ID-to-bit order are saved.

The 67-view parent used in the October26 continuation spans registered video frames
0..1088; this is not a claim that all 1,089 video frames have camera poses. A wider
time range may also expose deformation or inconsistent calibration; more sources
do not by themselves prove a rigid complete surface.

Example in the repository, using the existing verified CPU Python dependencies:

```powershell
C:\Python313\python.exe -B -X utf8 scripts/run_depth_fusion_model.py --source-model outputs/delegation/GLOBAL_LOCAL/20261025_model07/real01 --output outputs/delegation/GLOBAL_LOCAL/my_fresh_fusion --blender
```

Use a **new output directory**. Existing output and historical sources are never
overwritten. For scheduled/production use, wrap this command in the repository's
`integrations.mvs_surface.supervision.run_supervised` process-tree guard and freeze
the input/code/runtime bindings first (the delivered `execute_once.py` is an example).
The CLI's deadline is cooperative; the external job supervisor supplies a hard
process-tree wall-time and committed-memory bound. Blender has its own two bounded
create/readback entries with isolated task temporary files.

Defaults: two threads, maximum grid axis 160, at least two native views per trusted
lattice sample, truncation band three voxels, shared pipeline deadline 720 seconds.
The grid AABB includes **all** positive valid foreground native-depth pixels, plus
four unpadded voxel pitches; it is not cropped to the old mesh. Each extraction cell
requires all eight corners to have at least two near-band votes. This need not be
the same two views throughout a cell; the intersection of source bitsets is saved.
Background conflicts and clipped-vote variance are saved, not interpreted as truth.
At most 200,000 mesh vertices and 400,000 faces are exported; excess fails rather than
silently truncating. Marching-cubes intermediate memory is bounded by the external
supervisor, not this final face-count check.

Outputs include:

- `PLAN.json`, `STAGES.json`, `RESULT.json`, or `FAILURE.json` on an incomplete run.
- `COVERAGE.{npz,json}`: full pixel availability and disclosed-stride cross-view evidence.
  Background pixels are not counted as object holes. Repeated pixel observations
  are not independent physical points or surface area.
- `FUSION.{npz,json}`: TSDF, vote weights, near votes, variance, cell/view provenance,
  and the actual generated mesh. Unobserved storage values are never evidence.
- `OLD_REVIEW` / `NEW_REVIEW`: indexed topology, four-point silhouette diagnostics,
  and centroid depth consistency. No full-face visibility or independent truth claim.
- `COMPARISON.json`, `PROJECTION_COMPARISON.json`, and `coverage/*.png`.
- `fused_surface.ply`, `fused_surface.obj`, `viewer/viewer.html`, and optional
  `blender/model.blend` (use the actual `BLENDER.json` path if the exporter changes).

Status `FUSED_SURFACE_GENERATED_NOT_GEOMETRY_ACCEPTED` means an inspectable surface
was generated, **not** a complete object, a watertight CAD solid, or physical accuracy
acceptance. Uniform blue is diagnostic color, not recovered texture. Missing depth
remains open. Grid/truncation smoothing can erase thin gaps; all source poses and
depths share possible systematic errors. Self-intersection testing and real-world
scale are still separate acceptance steps. The engineering fusion is not presented
as a theorem or implementation from Kozlov's parallel-projection research.


## Explicitly allow an inferred completion layer

`--allow-inferred-completion` enables a separate silhouette-based inference stage.
It requires `--full-depth-parent` and rejects `--raw-only`. Without this explicit
flag the existing raw/final behavior is unchanged, even when a full parent is supplied.

```powershell
C:\Python313\python.exe -B -X utf8 scripts/run_depth_fusion_model.py --source-model outputs/delegation/GLOBAL_LOCAL/20261025_model07/real01 --full-depth-parent outputs/delegation/DENSITY_MODEL/20261023_identity_surface02 --output outputs/delegation/GLOBAL_LOCAL/my_fresh_completion --allow-inferred-completion --blender --seconds 720
```

The sequence is full-source authentication, `raw/` native-depth fusion, `final/`
reviewed observed surface, `completion/` inferred candidate layers, then
`completed_model/` closed-envelope selection and layered delivery, followed by
automatic `strict_comparison/` audit and before/after comparison. The completion
stage receives the same `FULL_INPUTS.json` and `raw/` result; no additional camera
registration or replacement of missing depth is implied. Candidate generation runs
without Blender; the delivery stage creates and independently reads back the
presentation Blender when requested. One shared deadline covers all stages.
Both candidate generation and final delivery receive only the remaining seconds;
expiry fails rather than starting a fresh budget.

The root `index.html` recommends `strict_comparison/index.html` and retains links
to `completed_model/`, `completion/`, `raw/` and `final/`. The comparison uses the
already generated strict `q=0` and robust `q=0.1` candidates, the unchanged depth
layers and all supplied views; it does not rerun fusion or silently replace the
baseline. It receives `completed_model/` as its baseline delivery, runs exactly
once after that delivery, and uses only the remaining shared deadline.

The largest-component delivery selects the largest
exact-index component only when it contains at least 90 percent of candidate faces,
is already closed with consistent winding and nonzero volume, and remains at least
one grid spacing inside every volume boundary. Other components remain in a
separate diagnostic layer. No welding or hole filling is performed. Observed-depth
geometry remains separate from the inferred orange envelope.

With `--allow-inferred-completion`, exit code 0 additionally requires the actual delivery
result's `closed_estimated_outer_model=true`. If Blender is requested, its delivery
must pass; a failed earlier `final/` Blender export is never hidden by successful
completion or comparison. A requested comparison Blender export must also pass.
The sole no-geometry exception is an explicit comparison Blender result
`status=NOT_APPLICABLE, reason=strict_envelope_absent`, with `strict_closed=false`.
It means the diagnostic comparison exists but no strict envelope exists to export;
it is never recorded as a successful Blender run. Other `NOT_RUN` or unexplained
`NOT_APPLICABLE` results fail when Blender was requested, and this exception cannot
hide failure of the baseline or earlier observed-model Blender export.

`STRICT_COMPARISON_DELIVERED` means the comparison was delivered; its `strict_closed`
flag may be false when the strict candidate is open or evidence is insufficient.
That finding, or lack of improvement, is disclosed and is not itself a comparison
execution failure. It does not erase the baseline's separate closed-model result.
Comparison exceptions or failed requested exports still fail the CLI. The printed
summary includes `strict_comparison_status` and `strict_closed` alongside the actual
closed-envelope delivery status. Without `--allow-inferred-completion`, the original
raw/final route remains unchanged.

Closed estimated outer shape does not establish a complete measured
object, absence of self-intersections, physical accuracy, metric scale or CAD validity.

Observed support and silhouette inference remain labelled separately. Strict `q=0`
and robust low-quantile `q=0.1` envelopes are candidate outer shapes, not measured
hidden surfaces. The robust envelope explicitly tolerates minority silhouette
conflicts. Unknown/out-of-image/behind-camera projections are not background;
unsafe volume boundaries cannot establish a closed object. Optional declared field
smoothing can erase narrow gaps. A closed mesh or successful Blender readback does
not prove a complete object, hidden concavity, preserved finger gaps, physical
accuracy, or CAD acceptance. Use the saved per-view projection diagnostics and
completion metadata to inspect the resulting risks. The CLI top-level page links
to both evidence-supported and inferred outputs; it does not relabel inference as
new native observations.
