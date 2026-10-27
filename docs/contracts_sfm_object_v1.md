# sfm-native-object-v1

Date: 2026-10-06. New bounded object-workflow profile under the full project goal.
This adds `cad-object-v1`; the frozen `t02a-v1` and `cad02a-shared-v1` stay reachable
with their existing limits/options. Historical accepted bytes remain in snapshots.

The existing photo-io-v1 input, identity, source encoding, native coordinate,
error/status, model roundtrip and incremental-evidence rules remain applicable.
The new profile accepts 2–96 distinct canonical RGB8 images, long side <=1024;
equal dimensions are required for the shared SIMPLE_RADIAL camera. Duplicates
do not increase the distinct-image count. No silent resizing or model merging.

Native parameters remain CPU, two threads, requested4096 SIFT features, seed0,
exhaustive matching, initial focal1.2*max dimension, principal point width/2 and
height/2, zero initial radial distortion. The4096 option is not a strict observed
keypoint-row upper bound. All four stages still share600 seconds. Caller must
provide the bounded Job resource supervisor and write a fresh execution plan.

Before importing NumPy, set OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, MKL_NUM_THREADS,
BLIS_NUM_THREADS, NUMEXPR_NUM_THREADS to2. `scripts/reconstruct_object.py` does this
automatically for its own process and descendants; it does not change system
settings. The new profile rejects missing/different settings before input staging
or native calls, records them, and includes the required policy in options used
by the fingerprint. Other profiles keep their existing option shape.

Runtime policy evidence is CAD_RUNTIME_DIAG/20261006T150711Z_runtime. It measures
the NumPy OpenBLAS instance, not every library's actual thread count. This profile
does not guarantee complete coverage, calibrated physical scale, or a CAD solid.
SFM component selection, invalid-track cleanup, undistortion, masking, meshing,
solid construction and independent validation remain separate workflow stages.
