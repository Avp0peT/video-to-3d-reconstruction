# Original-observation multiview diagnostic

This experiment continues `MULTI_PAIR_ABLATION.md`. It consumes frozen pair
candidates, including the intentionally cyclically mismatched right observations.
It never changes correspondences or raw image coordinates.

## Three methods and a fixed split

- A reuses the unquantized, common-y pair candidate.
- C reuses the actual AlgT′ certificate's calibrated pair candidate. Its absence
  remains explicit; no replacement or re-selection is made.
- J initializes XYZ with normalized DLT and refines only XYZ using bounded LM on
  the sum of squared original SIMPLE_RADIAL pixel residuals. Camera intrinsics
  and poses stay fixed. This is standard engineering, not another Kozlov theorem
  and not full bundle adjustment.

For each left track, sort all singleton view IDs excluding the original pair.
Require at least two additional views. Positions 0,2,4,... augment the fit;
positions 1,3,5,... are held out from this XYZ fit. Every point slot stays in the
ledger, including insufficient extra views, missing C candidates, optimizer
refusals and projection failures. Negative controls retain the wrong right
track; all additional observations belong to the left track.

The prior COLMAP calibration, bundle adjustment and track filtering used these
same images, including the held-out observations. Consequently this is a
conditional diagnostic, not independent validation or known physical accuracy.
High-overlap pair selection and image-ID parity also impose selection bias.
Repeated track IDs in different pairs remain correlated point slots.

## Objective, metrics and scope

`X_camera = R X + t`, `(u,v) = X_camera[:2]/X_camera[2]`, and
`pixel = f*(u,v)*(1+k*(u*u+v*v))+(cx,cy)`.
The analytic Jacobian is checked with finite differences. Invalid radial
domains, duplicate view IDs, degenerate triangulation and nonpositive depth are
refused. LM takes at most 20 steps with at most 12 damping trials per step.
It accepts only descent; a feasible estimate at the iteration limit reports
`converged=false`, never a claim of optimality.

Residuals are 2D Euclidean distances in original pixels. Predeclared 1/2/4 px
levels count point slots whose maximum held-out residual exceeds each level;
these are sensitivity diagnostics, not accuracy acceptance thresholds. Pair
residuals are retained too: a joint fit can dilute one wrong observation among
many correct extra observations, so a small held-out residual alone cannot
certify all original matches. Projection refusal is a separate outcome.

`paired_comparisons` restricts A/J and C/J separately to identical point slots
and held-out observations. Training residuals are reported separately and are
not advertised as validation improvement. Old COLMAP XYZ is not consumed for
fitting or scoring. Source observations and prior evidence are hashed before
and after execution; a published prior checksum manifest, when present, is
checked before its evidence is used.

## Run

```powershell
python -B -X utf8 scripts/experiment_heldout_multiview.py --prior-ablation '<prior RUN/real01>' --output '<new output>'
```

The CLI fixes numerical library threads to 2. The experiment has a shared
180-second soft deadline; use an outer 240-second timeout. It creates a new
directory, writes PLAN before calculations, then SAMPLES, RESULT and a linked
HTML report. It produces no fused point cloud, mesh or CAD artifact and never
reruns SAM3, native COLMAP/OpenMVS, or an exact paper construction.
