# Explicit experimental native photo sources

`experimental-native-photo-source-v1` describes a real experimental native
reconstruction. It is not a `sfm-native-*` run and does not replace or relabel the
original T01/SfM records. The surface runner accepts it only through the explicit
`experimental-real` origin and the derived
`experimental-native-surface-input-v1` specification. The existing `real` origin
continues to require its original source contract.

The identity specification binds the producer PLAN, CONFIG, actual supervised
ENTRY, RESULT, closed SQLite database, every selected model file, original
accepted object-mask manifests, and an explicit input-index to manifest selection
for every requested image. The profile is `paired-cpu-pycolmap421-v1`, whose worker
source and native reader binary are checked. Actual argv, cwd, native calls,
effective options, exit code and confirmed reclamation must agree. It does not
require nonexistent per-call historical logs.

Every source and copied RGB/mask byte binding is reopened. Full native names join
input indices to actual database IDs; `index+1` is never an identity rule. Native
camera ID/model/dimensions must agree with the DB, while finite positive optimized
intrinsics may differ from initialization. All native 2D feature coordinates must
equal the corresponding DB float32 keypoints. Sparse points require finite XYZ,
at least two distinct registered view observations and complete reciprocal tracks.
Aliases, incomplete coverage and conflicting bindings are rejected.

Original accepted-mask identity is joined through the actual photo-IO canonical
frame, frame ID, original-image digest, canonical-image bytes, dimensions, selected
candidate path/digest and foreground count. No new candidate is selected. Requested
inputs, selected-model registrations and unregistered requested indices remain
separate facts; other native components are not silently merged.

An optional `historical_code_copies` list may bind an exact preserved copy of the
producer's two supervisor source files. Each copy must have exactly the bytes and
SHA256 declared at production. This is historical execution context only. Worker,
native binary, database, model, images and masks cannot use this replacement.
The current supervisor remains independently bound by the current surface plan's
CODE_FILES. Missing snapshots or mismatching archive bytes are rejected.

The `identity` prepare mode creates a new specification and independently rereads
its sources. The `warp` prepare mode requires an already completed real bridge; it
starts no native process. It reuses `prepare_mvs_masks` bridge identity and pose/
camera/scene checks and `warp_binary_mask` for inverse camera-domain transfer.
Every bridge file is bound, recorded stages must match PLAN with successful exits,
and the current bridge producer/tool identities must match. Source/undistorted
points, colors, tracks, all feature indices and normalized rays are checked.
The existing whole-point bijection matcher checks the roundtrip and actual MVSI
vertices against native XYZ/RGB/full viewing-name membership. Roundtrip calibration
and reprojected pixel conventions are rechecked. The shared bounded MVSI reader
validates the scene before retaining its vertex identity fields.

Surface admission repeats all identity and bridge checks, recomputes each warp,
and reopens every output PNG to require exact mask and validity pixels. Merely
declaring PASS or refreshing an altered file's digest cannot bypass these checks.
The surface plan binds the specification, original source chain, current adapter
and helper code, native tools and actual output pixels; execution recomputes the
same plan before launching.

No source-admission result certifies physical accuracy, scale, segmentation quality
or a complete object. `object_shape_status=NOT_EVALUATED` and
`full_object_accepted=false` remain explicit. Prepared partial surfaces remain
experimental artifacts and cannot claim the original unified pipeline passed.

The explicit `derive` mode additionally supports
`experimental-native-derived-photo-source-v1`. Its original v1 parent is fully
revalidated before the child is admitted; parent producer model counts remain
unchanged. The deterministic policy deletes whole points whose tracks repeat a
view or contain fewer than two unique views, and no other points. Every registered
view is retained. The shared `general_subset.compare` independently recomputes
point deletion IDs, removed observations, retained XYZ/RGB/tracks, all measured
2D coordinates and associations, cameras and poses. Camera, rig and frame identity
sets and records are additionally unchanged. A certificate self-report or updated
child file hash cannot authorize another subset or geometry. The derived identity
binds its parent specification, every child model file, certificate and current
adapter/helper code. It inherits the parent's exact RGB and accepted SAM mask
chain. Original v1 behavior and source-kind semantics remain unchanged; a derived
model is never represented as a new SfM run. `derive` writes a native model only,
with no extraction, matching, solving, bundle adjustment, SAM or MVS. The caller
must supervise this preparation independently of the historical producer.

A parent with these exact track defects can be authenticated only through the
explicit `identity --derivation-parent` preparation flag. Its result is
`DERIVATION_PARENT_AUTHENTICATED_NOT_BRIDGE_ADMITTED`, never a v1 admission PASS.
The default v1 validator still rejects it. This special authentication validates
the entire real producer chain and runs the shared relation against a disposable
in-memory deterministic clean copy, checking finite XYZ and measured coordinates,
full source reciprocity and retained structure. It never alters or writes the
parent. Only the independently checked derived identity is eligible for bridge
preparation.
