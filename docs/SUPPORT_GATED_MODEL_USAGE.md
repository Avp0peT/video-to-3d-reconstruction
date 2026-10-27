# Whole/local support-gated model

Run the existing calibrated model stage first. Then run this automatic, reversible
selection stage on its completed, sealed `real01` directory:

```powershell
& C:/Python313/python.exe -B -X utf8 scripts/run_support_gated_model.py `
  --source-model <model-run>/real01 --output <fresh-output-directory> --blender
```

The source run must have its parent `SHA256SUMS.txt`, unchanged data bindings,
original native depth maps/cameras/masks, and a surface with generating-view
provenance. The inferred silhouette shell is never a substitute for depth.
All operations are object-agnostic; there are no glove or finger coordinates.

Each original global point and depth-derived local vertex is classified against
the original views. Its own generating depth is excluded when known. A stable,
fully valid foreground 2x2 footprint is inverse-depth interpolated. Two different
depth views, at least one degree apart, must agree within 2% axial depth. The
global cloud's generating views are unknown, so this is correlated consistency,
not independent correctness or precision validation.

Depth in front of a candidate can occlude it. A candidate in front of stable
observed depth conflicts with free space. These are different states. Missing,
mixed-mask-boundary and discontinuous-depth evidence is unknown. An entirely
known-background footprint is a separately recorded mask diagnostic; it excludes
adoption but does not prove the physical point false. The supported model excludes
both conflicting and unverified points; the complete original audit remains.

Only exact XYZ equality shares an output index. All original indices, generating
view IDs, original colors and membership mappings remain in the NPZ ledger.
Nearby points are not snapped or averaged. Additional unique coordinates do not
prove additional physical coverage. Local original faces survive only when their
vertices and three edge midpoints plus barycenter pass. This is not exhaustive
per-pixel depth visibility; prior full mask-footprint evidence remains bound.
There is no new triangulation across rejected points, no hole filling and no
automatic adoption of a silhouette proposal. Exact pooling can still expose
nonmanifold/overlapping faces; topology and CAD are not accepted.

Outputs include the classified whole cloud (green support, red conflict, yellow
unverified), separately visible supported points, one indexed supported model,
per-view overlays, source ledgers, OBJ/PLY and optional Blender save/readback.
Mesh renderers may hide isolated points, so inspect the supported-point viewer too.
Original model files are never replaced. Physical scale remains unknown.

An explicit comparison policy keeps **all depth-derived local points** while
still screening the global cloud:

```powershell
& C:/Python313/python.exe -B -X utf8 scripts/run_support_gated_model.py `
  --source-model <model-run>/real01 --output <fresh-output-directory> `
  --local-policy all-points --blender
```

The default remains `supported-only`. `all-points` changes point admission only.
All local evidence labels are preserved, including conflict and unverified points;
they are not relabelled supported. `FUSION_PROVENANCE.npz` separately records
`local_admitted`, `local_evidence_supported` and `local_evidence_classification`.
Global admission still requires support. Original face vertex and four-sample
checks are unchanged, so additional points may be isolated with no additional
surface. Inspect `local_all_combined_points_viewer.html` for all admitted points
and `local_evidence_audit_viewer.html` for the unchanged audit colors. Mesh solid
rendering hides isolated points; Blender Edit Mode shows every retained vertex.
The status is `ALL_LOCAL_COMPARISON_NOT_GEOMETRY_ACCEPTED`, never automatic quality
or completeness acceptance. No inferred silhouette shell is added by this option.

For an explicit **all-input geometry comparison**, retain all global points,
all local points and all existing local faces:

```powershell
& C:/Python313/python.exe -B -X utf8 scripts/run_support_gated_model.py `
  --source-model <model-run>/real01 --output <fresh-output-directory> `
  --local-policy all-points --global-policy all-points --face-policy original `
  --blender
```

This mode is `ALL_ORIGINAL_COMPARISON_NOT_GEOMETRY_ACCEPTED`. Audit still runs
unchanged and its labels are preserved, but does not exclude points or original
faces. `face_counts` describes the evidence audit; `fusion_counts` describes
actual admission, including `restored_failed_face_audit_faces`. Invalid indices,
collapsed or zero-area faces, nonfinite inputs and budget breaches remain
structural errors or explicitly recorded omissions. `--face-policy original`
requires `--local-policy all-points`. Defaults and the previous local-only option
are unchanged. Other supported combinations are labelled a relaxed comparison.

The source global artifact is a point cloud: it has no original faces to restore.
No new triangles are created for it, and source holes remain possible. No inferred
silhouette shell is ingested. Restored faces can reintroduce bridges or other
conflicts. This comparison is not automatic geometry, completeness or CAD
acceptance. Use `all_original_points_viewer.html` for every admitted vertex and
`model_viewer.html` for the retained original surface. Isolated vertices appear
in Blender Edit Mode, not as solid surfaces.

The CLI has a 600-second cooperative CPU deadline. Use the project task supervisor
for hard process-tree memory/outer-time control. No SAM3, GPU inference, new MVS,
software installation or camera adjustment is performed by this stage.

Blender's volatile temporary directories are automatically routed to an owned
`outputs/runtime_scratch/blender_<unique-id>` tree outside the artifact run.
The child inherits the bounded Job. Artifact trees remain strictly monitored;
scratch disk use has bounded start/end snapshots, not continuous monitoring.
Temporary files are retained for audit, never silently deleted by this adapter.

Primary references consulted: [COLMAP MVS](https://doi.org/10.1007/978-3-319-46487-9_31)
for geometric consistency/occlusion distinctions and [Open3D integration](https://www.open3d.org/docs/latest/tutorial/t_reconstruction_system/integration.html)
for the separate concept of depth fusion. This implementation is a bounded
selection/pooling stage, not an implementation or benchmark of those full systems.
