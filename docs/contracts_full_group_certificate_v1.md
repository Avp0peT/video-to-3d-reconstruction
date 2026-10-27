# Full group certificate v1

S-2025 PDF12 defines full as absence of a proper superset group with the same
granularity M. Group extension adds images (label subsets); it does not enlarge
each existing image. The complete external finite environment is essential.

`certify_full_group(environment, group, granularity_certificate,
extension_certificate=None, max_work=2000000, seconds=30)` accepts the existing
independently verified exact granularity certificate. A single shared work ledger
and deadline charge each complete child verification once. The existing bounded
complete rational environment (at most seven labels), nonempty disjoint-member
contract is preserved. Existing Budget retains an attempted over-cap charge.

Let r count the complement of the union of member labels. If r<M, every proper
extension adds at least one nonempty disjoint image of cardinality <=r<M, hence
the extension's granularity is <=its minimum member cardinality<M. This yields
`FULL_GROUP_CERTIFIED`, including environments with unused labels. Otherwise the
result is `UNKNOWN`; it is not a general non-full decision.

With an independently verified exact-M certificate of a proper superset group in
the SAME complete environment, `NOT_FULL_GIVEN_EXTENSION` proves non-fullness.
References and primary futlyars may differ between the two granularity witnesses.
Source geometric overlap is irrelevant to label-disjointness. Invalid/unresolved
granularity or extension returns UNKNOWN, malformed inputs UNSUPPORTED_INPUT.

`assess_full_group_certificate(environment, group, certificate, ...)` is independent:
it invokes only the original granularity verifier and label-set/cardinality checks,
never the constructor or any search. It strictly binds full environment, group,
M, unused-label complement, positive/negative proof fields and nested witnesses.
It proves neither closedness nor every etalon.
