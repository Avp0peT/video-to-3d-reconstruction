# circuit-fits-v1: multi-centre sufficient strict fits

`circuit_fits(reference, reference_parts, source, source_parts)` accepts two
complete rational planar sets, each at most seven points, and equal numbers
of nonempty complete parts between four and seven. Labels, exact coordinates,
primary conditions and full-block center proofs retain existing contracts.
It does not reduce the supplied input to selected four-point representatives.

The primary's centers and complete source block centers are passed to
`align_circuit_centers`. A successful `circuit-center-alignment-v1` proof covers
every n! correspondence by an exact affine-dependence lower bound. Its selected
invertible transform attains the bound on ALL centers and has a sufficient
common-affine stability proof, including the original point-displacement radius.
This generalizes the strong sufficient profile beyond four centers; it does not
decide general minimax/stable fits when these certificates are not tight.

The selected transform must also put ALL source block points strictly within
their corresponding primary disks and unique closest-center regions. The full
points, both partitions, all block-center certificates, primary, alignment
certificate and per-point distances are bound in the output. A failed selected
transform returns `NO_SUPPORTED_WITNESS`; this wrapper does not search every
other stable transform. An unresolved alignment or budget returns `UNKNOWN`.
Neither result proves nonexistence. Source block hull independence is not imposed.

Only `CIRCUIT_FITS_SUFFICIENT_WITNESS` is a success. Independent
`assess_circuit_fits_certificate(reference, source, certificate)` invokes only
child verifiers and complete exact point substitutions, without reconstruction
of centers or searching transforms. Certificate field sets and label types are
strict, with no bool-as-int equivalence. The alignment verifier handles its
explicit 5040-row bound without weakening the existing generic 20000-node codec.
UI downloads use explicit bounded row pieces for large ledgers.

Construction plus internal verification shares 2,000,000 cooperative work and
30 seconds by default; callers can supply smaller budgets. Public independent
verification receives its own budget, while UI shares total remaining work/time.
No budget increase is implicit. General-position seven-point construction may
exhaust this profile; a seven-point example with ten admissible circuits was
fully executed within it. Successful examples are not a universal completion
or runtime guarantee.

The given-m verifier accepts this schema, and finite compatibility search tries
it for m=5..7 after prior sufficient solvers. That search still takes one complete
environment <=7, so nontrivial separate groups of two such large members lie
outside the current group profile. Standalone fits takes two separate inputs.

Theory references: the norm-constraint form and dual lower-bound principle in
[Lobo et al.](https://stanford.edu/~boyd/papers/socp.html) and
[Boyd/Vandenberghe](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf) inform
this source-derived proof. The particular sparse affine relations, attainment
and stability tests are this project's derivation, not a quoted Kozlov or Boyd
algorithm. CVXPY/Clarabel were not installed or used to generate these exact
results. General nonlinear quantifier elimination remains a separate task.
