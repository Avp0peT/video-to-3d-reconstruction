# Automatic face/mask candidate isolation

New regional surface runs automatically execute `face_mask_gate` after the
surface support diagnostic, under a 240-second / 3-GiB supervised stage. The
original dense points and raw mesh remain available. The completed run defaults
to `retained_viewer.html`, which shows only the derived candidate and is labelled
as not accepted. `comparison_viewer.html` separately preserves the original/new
comparison; the default derived viewer does not overlay the raw model.

The default v3 adapter starts again from the native original mesh, not the v2
retained mesh. It runs the two existing rules over the same selected views.
The original v1 rule examines triangle projections, including edges and interior,
and uses saved depth maps for vertex-support evidence. The added gap rule finds
mask-defined foreground/background/foreground intervals and tests whole projected
faces against that bounded background. Both foreground banks are established by
the mask; an erroneous small face entirely inside the gap can still be a candidate.
The gap rule does not first require the erroneous face's vertices to have depth or
to lie in foreground. Unbounded exterior background cannot supply a gap vote.

Each rule independently requires conflict evidence from at least two views
separated by at least 8 degrees. One v1 vote and one gap vote cannot be combined
to meet that requirement. v3 first restricts each negative vote to the same
view's edge-eligible faces, then takes the union of independently qualified
partitions, and finally vetoes every core-protected face. Repeated identical directions cannot supply
independent angular votes.

An entire face's projected enclosing circle must lie inside the known-valid
contour band to be edge eligible. The circle radius, contour distance and pixel
centre allowance (0.707px) are included. Band width is initially 2% of object
extent, clamped to 6–32px. Any view can protect a face when at least one of its
three vertices or its centroid projects deeper than twice that band into
foreground and the saved centre depth agrees within 2%. This is sampled core
protection, not proof that the entire face is correct. These thresholds are
engineering settings.
Large faces crossing far outside that narrow band remain unchanged this round;
later local clipping would be a separate intervention. No subdivision is done.

The v3 quarantined set is checked to be a subset of the same-run v2 set. Thus its
retained opaque triangle union contains v2's retained union: projected foreground
coverage cannot decrease relative to that v2 partition. Non-edge and protected
source faces are retained with original coordinates and winding. These are
subset/coordinate guarantees, not measured visibility, complete finger separation,
physical geometry, or CAD acceptance.
The 3-pixel margin, 1-percent conflict fraction and 4-pixel minimum are versioned
engineering starting values, not calibrated physical error or accuracy bounds.
The gap maximum-span fraction initially equals 20%; its measurement domain is
recorded by the gap implementation and metadata. See the
generated `policy`, `gap_policy`, `edge_policy` and ledger for exact measurement definitions.
The 3-pixel background margin remains; narrow gaps may still be unresolvable.

Foreground masks and geometry share source imagery. External occlusion, hand
deformation, camera-pose error and unavailable depth can remain unknown. The
rule is not an unconditional intersection of every view and does not establish
physical truth. Retained faces have not been proven reliable; quarantined faces
are suspect candidates. The derived mesh may have holes and is not CAD accepted.
Zero quarantined faces means no demonstrated improvement from this gate.

For an already completed regional surface, run the bounded worker using the
project's LIGHT Python and an existing process-tree supervisor:

```text
python -B -X utf8 scripts/build_region_surface.py face-mask-worker --source <completed-surface-run> --output <fresh-disjoint-gate-output>
```

The worker does not rerun segmentation, SfM, MVS or Blender. Use a new output;
never overwrite a prior gate. Products include `index.html`,
`retained_scenes.json`, `retained_viewer.html`, `comparison_viewer.html`,
`retained.ply`, `quarantined.ply`, `FACE_LEDGER.csv`, and
`per_view/*.npz` evidence (v1 semantics preserved), plus independent
`gap_per_view/*.npz` and gap-mask PNGs. The v2 ledger appends `rule` (`v1`, `v2`,
`both`, or `none`) and `gap_*` counts/witnesses; original columns remain v1 counts.
`edge_per_view/*.npz` independently preserves edge-eligibility and core-protection
flags. Existing ledger columns still refer to original v1/v2 evidence; appended
edge fields show restricted votes, veto and restored face decisions.
`VIEWS.json` retains v1 reasons and adds gap reasons/metadata. Counts distinguish
`v1_quarantined`, `gap_quarantined`, and `gap_only_quarantined`.
The v3 counters include `previous_v2_quarantined`, `restored_from_v2`,
`core_protected_faces`, `edge_eligible_faces`, `edge_v1_quarantined` and
`edge_gap_quarantined` (the last two include the core veto).
Original face IDs, view counts, angular witnesses and
decisions remain inspectable. An empty quarantined mesh is legitimate; an
undeliverable candidate or failed diagnostic cannot be advertised as filtering
success. The controller publishes candidate links only after checking status
`CANDIDATE_SURFACE_NOT_ACCEPTED`, unchanged sources, `geometry_accepted=false`,
and the required files. All produced files are bound for later drift checks.
