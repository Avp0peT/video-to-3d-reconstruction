# Original-view identifiability diagnosis

This continues the heldout experiment without another reconstruction. It selects
all prior negative C candidates with heldout maximum residual <=2 px, including
aliases whose distinct right track ID has identical measured coordinates.
Selection is post hoc: it cannot estimate general accuracy or detection rates.

`scripts/audit_mismatch_identifiability.py --prior-run <heldout RUN> --output <new>`
requires the published checksum manifest and checks input/source bindings before
and after. It writes a plan before numerical diagnostics and preserves every
previous artifact. The CLI caps numerical library threads at two; use a 120 s
outer timeout. No inference, feature extraction, triangulation optimization,
camera optimization, exact paper construction, or CAD conversion is executed.

## Independent support

For each selected slot, the reference is the original unquantized A candidate,
not COLMAP XYZ or physical truth. The comparison is the saved negative C. The
support set consists of the original left view and all recorded extra singleton
views; the original right camera is excluded from support and used only for
projection comparisons. Changing the right track never changes support pixels.

Each candidate is evaluated in all support views and separately in the original
training-extra and heldout subsets. Exact signed residuals obey
`negative_residual = reference_residual + negative_projection-reference_projection`.
The report also saves `J_reference * (X_negative-X_reference)` and its discrepancy
from the exact nonlinear projection change. A poor linear approximation is not
used as a causal explanation.

The Jacobian SVD and original measured-ray angles describe local observability
under fixed existing camera parameters. Rank uses a relative 1e-10 numerical
tolerance; rank three does not imply sufficiently strong constraints. Optional
prediction response assumes one pixel of independent noise in each supplied
coordinate. This is a hypothetical linear response, not measured noise, physical
uncertainty, a confidence interval, or an accepted length unit. Repeated poses
do not add depth rank, though an independence assumption can change noise gain.

## Full-observation diagnostic

Every prior positive and negative point slot is retained in
`FULL_OBSERVATION_DIAGNOSTICS.json`. For C candidates with a ready original split,
evaluate **pair + training-extra + heldout**, using the supplied right observation
even when it is the deliberate wrong one. No hindsight correct-right coordinate
is used for this decision.

- A missing candidate, insufficient split, or failed projection stays explicit.
- Maximum vector residual >2 px reports `OBSERVATION_INCONSISTENCY_DETECTED`.
- Otherwise it reports `WITHIN_DIAGNOSTIC_LINE_NOT_VERIFIED`, never acceptance.

This prevents a low heldout residual from masking a contradiction in another
available view. It is a diagnostic addition, not permission to delete, merge,
repair or publish geometry. The 2 px line was fixed in the earlier experiment;
near-threshold cases remain sensitivity-dependent.

## Limits

Camera calibration and tracks already used these images. The data is not an
independent validation set. A few finite points along a candidate-to-reference
segment do not prove the whole continuous segment feasible. Original A and C may
both be inaccurate. Neither low residual, exact certificate, nor a full-rank
Jacobian proves the true identity of a physical feature.
