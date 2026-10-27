# Four-center strict fits: sufficient profile v1

`four_center_fits(reference, reference_parts, source, source_parts)` accepts each
complete rational point set of at most seven points, partitioned into exactly
four nonempty parts. It does not select four representatives from a larger set.
Reference primary and the exact centers of every complete source part are
independently certified. Source centers must be in general affine position;
target centers must have rank two. Algebraic centers outside this rational
profile remain unresolved.

The center stage uses `exact-four-center-alignment-v1`: all 24 correspondences,
an exact Euclidean minimax lower bound, invertible attainment, and a sufficient
common-affine stability proof with fixed epsilon1 and a positive point-displacement
radius. The global result implies the local condition of S-2025 PDF9–10. A tied
nonparallel minimum without that proof, or an unattained infimum, does not certify
fits and does not disprove the original local fits definition.

For every invertible globally minimal correspondence, the constructor transforms
ALL points of every source part. It requires both strict disk membership under
the primary's fixed radius and strictly closer distance to the corresponding
center than every other center. Disk-boundary and Voronoi ties are rejected. It
does not stop at the first center mapping if its complete blocks fail. Source
part convex hulls need not be independent; the reference cover must be.

Success `FOUR_CENTER_FITS_SUFFICIENT_WITNESS` carries both complete inputs,
partitions, primary proof, all four source center proofs, the full alignment
ledger, the selected ledger index and exact per-point checks. The separate
`assess_four_center_fits_certificate(reference, source, certificate)` checks
these independently without center construction, fitting or search. It rejects
bool-as-int identities, changed inputs and missing/changed point checks.

The given-m verifier accepts this schema in addition to zero-error and collinear
fits. Automatic finite compatibility search invokes it only for four-part
candidates. That search's complete environment still has at most seven points;
it therefore cannot contain two disjoint four-point members. The standalone
fits API accepts separate complete reference/source inputs, each at most seven.

Each construction plus internal verification shares a cooperative work/time
budget (default 2,000,000 work and 30 seconds). Standalone verification receives
its own budget. Exhaustion is `UNKNOWN`; absence of a supported global stable
mapping with strict full-block containment is `NO_SUPPORTED_WITNESS`, never a
general nonexistence claim. A successful default example has exact maximum
center error squared 1/16. This module does not process photos or claim geometric
or CAD acceptance.
