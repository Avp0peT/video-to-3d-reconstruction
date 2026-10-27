# Object boundaries: mask uncertainty and local surface candidates

The two stages have separate evidence and coordinate domains. A binary SAM3/temporal union is an upper-bound candidate, not a verified object silhouette. A surface projection is not proof of visibility. Neither stage establishes complete geometry, physical scale or CAD acceptance.

## Mask review

`scripts/review_mask_precision.py` prepares and executes a source-bound CPU review of a fusion run. It groups alternatives by propagation root, chooses the highest-IoU alternative per root, and uses the frozen source-IoU threshold 0.8 and boundary margin 3 px. Agreement is not ground truth. The diagnostic trimap has labels 0 background, 128 unknown and 255 source-agreed interior; validity is false on UNKNOWN. The unchanged original binary union is delivered separately.

```powershell
& $python -B -X utf8 scripts/review_mask_precision.py run --source $fusionRun --output $freshOutput --indices 86 327 328 512
```

Run the CLI's `--help` for the exact argument format. Never pass a trimap to a binary mask consumer. The report explicitly sets `published_for_reconstruction=false`. Source images/masks are not overwritten. Mask accuracy and object identity still require image review or independent annotations.

`scripts/refine_mask_boundary.py` can derive a separate image-guided boundary candidate from that review, using fixed CPU GrabCut (two iterations, one thread, seed zero):

```powershell
& $python -B -X utf8 scripts/refine_mask_boundary.py run --review $reviewOutput --output $freshCandidateOutput --indices 86 327 328 512
```

Only UNKNOWN pixels inside the original union may be removed. Definite core/background must remain unchanged, no original background or holes may become foreground, each original foreground component must survive, and changed pixels may not exceed 1% of the original foreground. These are guardrails, not accuracy proof. RGB comes from the review's explicit bound image records. The proposals are not published for reconstruction and are not consumed by the local mesh stage. Full-video precision and temporal consistency have not been validated by the four-frame experiment.

In the final implementation `default_masks` always retains the byte-identical original union, while `proposals` contains unaccepted candidates. On late failure the default is restored from a prechecked in-memory original, not from a possibly drifted source file. The CLI returns nonzero for a refused experiment. The historical four-frame experiment predates this default-file correction; its old outputs remain unchanged and `POSTRUN_DELTA.json` explains their meanings.

## Automatic mesh stage

`scripts/build_region_surface.py run` now executes `face_local` after the existing `face_mask_gate`. It runs in a supervised process with 480 s / 4 GiB limits and a 420 s internal deadline. Only a completed candidate with unchanged inputs and all required outputs becomes the preferred viewer. Failure leaves the preceding candidate viewer available and is recorded as a failed/unverified stage.

To process an existing bound native surface and its v3 result:

```powershell
& $python -B -X utf8 scripts/build_region_surface.py face-local-worker --source $surfaceRun --gate $v3Gate --output $freshOutput
```

The worker is intended to be launched through the existing supervisor in routine use. It does not rerun SAM3, SfM or OpenMVS.

1. Select only original faces with v2 multi-view gap evidence that v3 retained.
2. Apply two midpoint subdivision rounds. Shared-edge closure includes neighboring faces so there are no hanging nodes; closure faces never acquire deletion eligibility. Midpoints are interpolated, not new measurements.
3. Evaluate every child independently in all bound views. A negative requires its full projected enclosing disk to lie in a valid bounded background gap, a 3 px foreground margin, and at least four actual interior pixel centers. Missing/ambiguous evidence never votes.
4. Require two distinct views separated by at least 8 degrees. Any supported core sample in any view vetoes a new deletion. Original parent probes protect only children that contain those probes, including both sides of shared edges.
5. Preserve the existing v3 quarantine, partition the refined mesh reversibly and export the retained candidate. No smoothing, hole filling, hand-specific locations or unobserved surfaces are introduced.

`LINEAGE.npz` binds refined faces to original parent IDs and records each midpoint's endpoints. PLY mapping files refer to the refined mesh, not raw original face IDs. `FACE_LEDGER.csv` records both IDs, eligibility, protection and witness views. `per_view/` contains the actual negative certificates and pixel witnesses. `original_probe_membership` uses closed-triangle membership.

The core veto applies only to new quarantines; this stage does not restore faces already removed by v3. Fixed-view projection comparison is an internal consistency diagnostic. Subdivision can slightly change raster coverage even where the continuous surface is unchanged; report measured foreground losses as well as gap reduction. A larger face count is not improved geometric detail.

## Evidence for the current glove

`outputs/delegation/OBJECT_BOUNDARY/20261024_boundary01/` contains the frozen execution plan, joint tests, a single supervised local run, three fixed-camera projections, source-preservation checks and a native Blender save/reopen. Raw masks and previous runs remain unchanged. This is a candidate result; remaining adhesion and mask uncertainty are explicitly retained in the report.
