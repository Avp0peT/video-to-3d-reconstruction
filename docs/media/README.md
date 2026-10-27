# README demonstration media

This directory contains a small, reviewed subset of real project outputs used by the repository homepage. It is intentionally separate from the full local `outputs/` evidence tree.

## Glove video

- `glove-mask-demo.mp4` is a compressed copy of the real mask-overlay video from `HAND_VIDEO/20261022_full_masks_delivery01`.
- `glove-mask-preview.gif` is a four-times-speed, 320-pixel-wide README preview derived from that MP4.
- `glove-mask-contact-sheet.jpg` is the original review contact sheet from the same run.
- `glove-model.blend` and `glove-model.ply` come from `BOUNDARY_MODEL/20261024_model01`.
- `glove-model-preview.png` was rendered headlessly from that `.blend` with `scripts/render_readme_model.py`.

The glove surface is an experimental open reconstruction. It preserves the visible roughness, sparse regions and boundary errors and is not labelled as a closed or accepted CAD solid.

## Dice model

- `dice-model.ply` is the 2,000-face prepared mesh from `GENERAL_PIPELINE/20261013_resume_cad_real02`.
- `dice-model.blend` was generated from exactly that PLY with Blender 5.1.
- `dice-model.FCStd` and `dice-model.step` are the corresponding closed faceted-solid deliveries from the same run.
- `dice-model-preview.png` was rendered headlessly from `dice-model.blend` with `scripts/render_readme_model.py`.

The model has arbitrary scale because no physical reference length was supplied. The preview material and lighting are presentation-only and do not alter mesh geometry.

## Rebuild previews

```powershell
$blender = 'C:\Program Files\Blender Foundation\Blender 5.1\blender.exe'
& $blender -b .\docs\media\dice-model.blend `
  --python .\scripts\render_readme_model.py -- .\docs\media\dice-model-preview.png
```

Git records the exact bytes of every media file. The full run ledgers, intermediate depth maps, original video and model environments remain local and are excluded from the repository.
