# Fixed-reference texture point tracking

This CPU experiment tracks one selected point inside observed object masks. It
does not admit geometry, prove physical identity, infer unseen positions, or
classify a failed match as confirmed back-side occlusion.

The first frame supplies a frozen 33 x 33 texture patch. Each new frame gets a
moving 129 x 129 search window and one affine alignment. The prediction uses
only previously accepted positions and actual timestamps; it is not an
observation. The unchanged first template prevents gradual template replacement.
Training and held-out pixels have separate photometric checks. An independent
reverse alignment must return the selected fractional point within 1.5 pixels.
The point's 3 x 3 neighborhood must have all bilinear contributors inside the
current observed binary mask. Competing texture peaks trigger rejection.

The first missing mask, unsupported center, ambiguous correspondence or failed
texture/shape/cycle check stops the track permanently. Later rows are
`NOT_RUN_AFTER_STOP`. No candidate is snapped to the mask, no point is replaced,
and no temporary mask-search support is promoted to a SAM observation.

## Reproducible entry

Run from the project root using the existing light Python environment. The
source directory must contain the bound `INPUTS.json` and `MASKS.json` produced
by the masked-point input stage. The output directory must not exist. Choose
the seed from the first frame before inspecting its tracking performance.

```powershell
$py = 'outputs/delegation/T02A/envs/20261002T190239Z_9b9b5ddb/venv/Scripts/python.exe'
& $py -B -X utf8 scripts/trace_texture_object_point.py --prepare `
  --source outputs/delegation/NEW_VIDEO/20261019_new01/masked_point_motion/real01 `
  --output outputs/delegation/NEW_VIDEO/20261019_new01/texture_point_stop/new_run `
  --seed 410 830
& $py -B -X utf8 scripts/trace_texture_object_point.py --execute `
  --output outputs/delegation/NEW_VIDEO/20261019_new01/texture_point_stop/new_run
```

Preparation validates all source bytes, decoded pixel identity, mask identity
and dimensions, freezes the point/profile/code, and produces `seed_view.png`.
Execution uses the existing Windows Job supervisor for tracking and reporting.
It refuses a second tracking entry. All limits and real process exits are in
`processes/`. No model load, SAM call, GPU work or installation occurs.

Outputs include `TRACE.json`, `RESULT.json`, `index.html`, `motion.gif`, a contact
sheet, a fixed-template zoom and all-frame panels. The GIF is deliberately
slowed. Only accepted points are green. Stopped frames show no valid location;
any rejected candidate comparison is separately labelled.

## Evidence and limitations

The current initial experiment accepts frames 555, 556 and 557, including the
manually specified first-frame seed. Frame 558 fails the existing affine shape
gate: the minimum singular value is 0.56555, below 0.65. The preserved raw
translation match score is approximately 0.7918. Later matches are not run.
This establishes a conservative stop, not successful tracking until confirmed
invisibility. Motion blur, rotation and mismatch cannot be disambiguated from
this stop alone. The point differs from the earlier root-area seed, so retention
counts are not a controlled improvement comparison with that experiment.

Book reference: `book_selected_pages.txt`, PDF page 56, discusses approximate
affine correspondence. It supplies neither these numerical thresholds nor an
occlusion detector. The implementation reuses the existing engineering
`area_alignment` routine and [OpenCV masked ECC alignment](https://docs.opencv.org/4.13.0/dc/d6b/group__video__track.html).

Verification: 26 tests passed in the dedicated verification run, including real
CPU translation/brightness, center-mask occlusion, repeated textures, missing
masks, fractional coordinates, and the reused affine alignment regression set.
Synthetic tests are distinct from the real-video result and do not certify
physical point identity or 3D accuracy.
