# Efficient point re-detection: evidence and next comparison

## Implementation update, 2026-10-22

The source-only review below is historical. TAPNext++ has now been run on the
onion video in both single-query and 128-query modes. See
`docs/LEARNED_POINT_TRACKING.md` for actual commands, evidence and limitations.
The published benchmark results below still are not local accuracy results.

Reviewed 2026-10-22. Status: source review and experiment design only. No new
tracker, weights, dependency installation, or model inference in this review.
The previous immutable texture_point_stop/real01 result remains unchanged.

## Requirement

Keep the identity of a point when its observation is lost. An invisible or
unconfirmed interval has no accepted measurement. When sufficiently supported
appearance returns, recover the original identity. Evaluate erroneous recovery
separately from tracking duration. A predicted coordinate behind an occluder is
not a measured point for reconstruction.

## Primary references

1. TAPNext++, CVPR 2026 Findings:
   https://tap-next-plus-plus.github.io/
   https://arxiv.org/abs/2604.10582
   Explicitly studies long sequences and re-detection after occlusion or frame
   exit. Introduces a re-detection-specific metric, AJ_RD. The official TAP
   repository links a PyTorch checkpoint and demo. These are published results,
   not results on this project's onion video or laptop.
2. Official TAP implementation and released assets:
   https://github.com/google-deepmind/tapnet
   The README distinguishes TAPNext, TAPNext++, TAPIR and their checkpoints.
   Do not use an older checkpoint and call it TAPNext++. The upstream coordinate
   conventions also require explicit conversion and synthetic mapping tests.
3. CoTracker3 official repository:
   https://github.com/facebookresearch/co-tracker
   Provides joint point tracking, point visibility, and online/offline models.
   Its README identifies online processing as more memory-efficient for long
   videos and recommends joint evaluation for faster inference.
4. TAPIR, ICCV 2023:
   https://openaccess.thecvf.com/content/ICCV2023/html/Doersch_TAPIR_Tracking_Any_Point_with_Per-Frame_Initialization_and_Temporal_Refinement_ICCV_2023_paper.html
   Separates per-frame candidate matching from temporal refinement. This is a
   useful architectural reference for recovering identity after local tracking
   fails; it is not proof that a particular candidate is correct.
5. TAPNext, ICCV 2025:
   https://tap-next.github.io/
   Online recurrent inference avoids iterative trajectory refinement. Its own
   project page reports long-sequence limitations for the original version;
   therefore reappearance testing should distinguish it from the newer ++ model.

The previously consulted book excerpt, PDF page 56 of book_selected_pages.txt,
discusses approximate affine correspondence. It does not specify a modern
point re-detection network or its runtime scheduling policy.

## Recommendation (engineering inference)

First evaluate TAPNext++ for this particular reappearance requirement. Use
CoTracker3 online as the joint-tracking comparator. Keep the current CPU texture
tracker as an inspectable baseline and as an optional local confirmation tool.
No claim is made that a neural tracker costs less than the current method for
one point; batch sizes, resolution, model load, and actual device determine that.

For an engineering fallback, cache per-frame image features and masks, use local
tracking while confident, and invoke coarse-to-fine re-detection only after
loss. Limit the re-detection frequency and candidate verification count. Record
missed/skipped search frames honestly. Continuous all-image multi-candidate ECC
for every point is not the proposed production design.

SAM3 remains the object-domain source. The tracker supplies point coordinates
and visibility estimates. Neither being inside the object mask nor receiving a
high tracker score alone proves the same physical point. Missing SAM masks
remain missing; a predicted point is not snapped onto the silhouette.

## Matched experiment to implement next

- Retain raw frames 555..618, their actual timestamps, and existing masks.
- Retain source point (410, 830), frame 555, with its original identity. Do not
  select a more favorable point after seeing a result.
- Freeze an official repository revision, matching model config/checkpoint and
  hashes before loading; isolate additions from the existing accepted SAM setup.
- Test the coordinate transform and visibility handling before inference. Keep
  raw/model coordinates separate and preserve an invertible resize/padding map.
- Compare the same one-point query first. Then test a fixed mask-interior batch
  to measure shared computation rather than repeatedly invoking a model once
  per point. Do not compare different point sets as if they were identical work.
- Record model loading separately from inference: total seconds, seconds/frame,
  memory peaks, query count, input dimensions, and real process exits.
- Present visible tracks, lost intervals, and re-detection events separately.
  Inspect nearby repeated texture and wrong-ID cases. No ground-truth accuracy
  claim without reference annotations; longer tracks alone do not mean better.
- Do not feed invisible-point predictions or unconfirmed re-detection candidates
  into triangulation. Geometry and surface acceptance remain separate stages.

Implementation of this comparison and true re-detection on the current video
are pending. Published FPS or benchmark gains are not projected onto this machine.

## Current-code cross-check and reconstruction reference (2026-10-22)

Read-only verification of `texture_point_stop/real01/TRACE.json` confirms 64
frames, 3 accepted positions including the seed, 2 accepted transitions, a stop
at raw frame 558 (`FORWARD_REJECTED`, `AFFINE_SHAPE_GATE`), and 60 frames not run
after that stop. `texture_point_trace.trace()` deliberately makes a permanent
accepted prefix. Thus this particular stop is not evidence that the point has
become occluded, and the current implementation cannot re-detect it later.

The official TAPNext++ PyTorch notebook is available at:
https://github.com/google-deepmind/tapnet/blob/main/colabs/torch_tapnextpp_demo.ipynb
It demonstrates frame-by-frame state reuse, separate visibility logits and
location certainty, and links matching released checkpoints. Inspect coordinate
conventions in that implementation rather than sharing a query tensor unchanged
with CoTracker. Its demonstrated evaluation supports first-query mode; do not
assume adding new points midway through an existing state is supported without
checking the API. New query batches can instead start from selected keyframes.

For the next stage, VGGSfM is a relevant primary architectural reference:
https://vggsfm.github.io/
https://github.com/facebookresearch/vggsfm
It connects learned 2D tracks to camera estimation, triangulation and bundle
adjustment. This is a route from correspondence to sparse geometry, not a claim
of a complete surface or CAD solid. Its README warns that default chunk sizes
target a 32 GB GPU. Do not transfer those defaults to this laptop.

The current repository already has an explicit-track consumer in
`integrations/demo_pipeline/connectivity_native.py`: it writes track IDs as
matches, runs native geometric verification and incremental reconstruction.
However, its accepted source arms are currently `balanced` and `sift_control`.
A learned-tracker integration must identify its own source with a reviewed
adapter/schema extension; it must not impersonate an existing arm. The consumer
also rejects predicted observations, missing original masks and coordinates
outside the mask. Preserve those distinctions. Model-predicted visibility does
not by itself prove physical point identity; validate image support and geometry
before publishing reconstruction observations.

Efficiency remains an experimental question: share feature extraction across a
batch, reuse temporal state, and retain only bounded diagnostic records. For
short single-point sequences the model loading cost may exceed the existing CPU
tracker. Compare total runtime as well as steady-state inference, and report
re-detection mistakes separately from the number of surviving points.

This supplementary review performed no tracker installation, checkpoint loading,
GPU inference or reconstruction. The recommendation is ready for implementation,
not an accepted improvement on this video.

## Follow-up evidence and efficiency references (2026-10-22)

The preceding implementation-pending statements describe the earlier review,
not the current state. `learned_tracking/tapnextpp01` and `tapnextpp12801`
now contain real runs on 64 frames, with one and 128 queries respectively.
`geometry_diagnostics01/identity_v2` contains fixed-camera robust triangulation,
and `surface_identity01/identity_export` contains 128 unique query points,
54 registered views, and 4606 retained observations. These are experimental
correspondences, not ground-truth physical identities. The original camera
estimates used all views; the conditional holdout is not independent validation.

Primary references checked again:

- TAPNext++: https://tap-next-plus-plus.github.io/ . Recurrent online tracking
  explicitly targets re-detection after occlusion or leaving the image. Its
  published benchmark/runtime does not establish accuracy or speed here.
- CoTracker3: https://cotracker3.github.io/ and
  https://github.com/facebookresearch/co-tracker . Joint tracking and an online
  variant provide an alternative to separate model calls for each point.
- LightGlue: https://github.com/cvg/LightGlue . Adaptive network depth and
  keypoint pruning reduce matching work. It matches image pairs, not persistent
  video identities by itself. Keyframe retrieval and trigger scheduling would
  be additional engineering components.
- ORB-SLAM3: https://github.com/UZ-SLAMLab/ORB_SLAM3 . Borrow keyframe memory,
  local-map tracking and relocalization design; this is not a drop-in model for
  a hand-held rotating onion with an independently moving background.
- VGGSfM: https://vggsfm.github.io/ and
  https://github.com/facebookresearch/vggsfm/blob/main/vggsfm/models/triangulator.py .
  Tracks feed triangulation and bundle adjustment. Visibility/appearance scores
  do not replace geometric checks.
- PMVS / CMVS: https://www.di.ens.fr/pmvs/ . Patch-based expansion and filtering
  provide a reconstruction reference for the proposed region approach. The
  method uses images with camera parameters and returns oriented surface points;
  visibility information can reduce computation. It cannot establish unseen
  backside shape. No PMVS installation or execution was performed in this review.

Recommended engineering allocation: reuse image features and temporal state for
many queries; keep invisible points dormant instead of producing observations;
trigger bounded keyframe re-matching only for lost or ambiguous identities;
verify reappearances with local texture and multiple-view geometry. Bound the
keyframe bank, candidate count, search frequency, and time. This scheduling is a
proposal, not a measured speedup or a claim that all of these modules are wired.

The supplied book's extracted PDF page 56 (printed page 62) permits approximate
affine subimage matching with epsilon and describes enumerating alternative
anchor triples. That provides a local consistency concept, not a low-cost video
re-detection scheduler or a general perspective guarantee. Avoid implementing
the exhaustive triple enumeration across all video frames.
