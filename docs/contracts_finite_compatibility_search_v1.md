# Automatic finite compatibility witness search v1

`search_finite_compatibility(environment, group, max_work=2000000, seconds=30)`
accepts a complete rational labelled environment of at most seven points and
nonempty, label-disjoint group members. It supplies reference selection, reference
cover and every member's full source partition automatically. No sampling occurs.

Following S-2025 PDF12 / journal157, all members use the SAME actual reference
member and SAME mathematical primary futlyar. The descending search visits m up to
the minimum member cardinality, every actual reference and every full partition.
Only reference covers require convex-hull independence. Source partitions have
no hull-independence gate. Existing primary and exact-center implementations and
zero-error then collinear sufficient solvers supply the geometry. For m=4 the
four-center globally stable strict-fits profile is a third sufficient solver.
For m=5..7, the multi-centre circuit lower-bound/attainment/stability profile is
tried instead as the third solver. The complete environment remains <=7 points;
this does not claim testing two disjoint five-point group members in that domain.

Each visited candidate has a stable tuple id, parts and local solver records.
Interrupted pending candidates are marked UNKNOWN. A reference TRUE describes
its primary stage, not a completed all-member compatibility result.
Strict containment failure and unsupported centers remain local evidence; neither
exhaustion nor absence of a supported witness proves general incompatibility.
Current m=1 primary/fit boundary remains UNKNOWN. A single decreasing work ledger
and wall deadline apply to all children; nested child totals are charged once.
Existing Budget records a charge before rejecting an exhausted cap, so work_used
retains that final attempted charge and can exceed the requested cap; no later
candidate is attempted after the exception. Time limits are cooperative checks.
Only independently verified complete fits are cached, with exact labels and
reference/source partitions within this call. Partial results are never cached.

Every success passes the independent given-m verifier. `LOWER_BOUND` means a
complete m-compatible witness below the cardinality upper bound, never a maximum.
Only a witness attaining that bound passes the existing granularity constructor
and yields `EXACT_GRANULARITY_CERTIFIED`. `UNKNOWN` covers exhausted supported
candidates, unsupported geometry and resource limits; `UNSUPPORTED_INPUT` rejects
malformed input or strict budget values. General compatibility remains undecided.

`assess_finite_compatibility_certificate(environment, group, certificate)` invokes
only existing given-m and granularity verifiers, exact cardinality and complete
input/group identity checks; it invokes no search, producer or center solver.
Strict bool-as-int and changed complete inputs are rejected. This search does not
enumerate every etalon, decide general 2D nonzero fits, full or closed groups, or
alter the production reconstruction pipeline.
