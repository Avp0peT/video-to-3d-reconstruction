# object-mask-v1 to mvs-mask-v1 bridge

`scripts/prepare_mvs_masks.py` consumes an accepted COLMAP->OpenMVS bridge, the
source SFM run and one or more **object-mask-v1** manifests, and writes
undistorted-domain masks for OpenMVS 2.4.0. It fixes the file, identity and
domain layer only: no SfM, densification, meshing, CAD, SAM3 or OpenMVS call, and
no mask is ever generated or edited.

```powershell
& $PinnedPython -B scripts/prepare_mvs_masks.py `
  --bridge  outputs/delegation/MVS_B/<tag>/run_44img `
  --sfm-run outputs/delegation/T02A/CAD_FULL66/<...>/<sfm_run> `
  --mask-manifest ABS/object_mask_run/manifest.json `
  --output  outputs/delegation/MVS_MASK_B/<tag>/run_allow_partial `
  --allow-partial
```

`--mask-manifest` may be repeated. `--output` must not exist; an existing path is
refused with exit 2 and is never written into.

## Accepted manifest dialect

The dialect is the one the production adapter writes
(`integrations/sam3/sam3_adapter/runner.py`, `worker.py`), not the dialect of any
earlier test fixture:

* candidate: `candidate_id`, `mask_path`, `mask_sha256`, `foreground_pixels`,
  `score`, `score_semantics`;
* `selected_mask`: exactly `path` and `sha256`, naming the selected candidate's
  raster;
* `selection`: `method` in `unique_candidate` / `explicit_candidate`, `source`,
  and `mask_source` in `automatic_sam3` / `automatic_sam3_human_edited`;
* `status == "OK"` always carries a non-null `selected_candidate_id`,
  `selected_mask` and `selection`, and any other status carries none of them.

`selected_mask.path` must equal the selected candidate's `mask_path`, its
`sha256` must equal that candidate's `mask_sha256`, and the bytes on disk must
then match. `counts` is recomputed from the per-frame statuses and compared with
the declared block (`success`/`no_match`/`needs_selection`/`error`/`not_run`).

## What is verified before anything is written

* **Strict JSON.** Duplicate keys are rejected, and so are `NaN`, `Infinity` and
  overflowing literals such as `1e999`; JSONL is checked line by line. Integer
  fields reject `bool`.
* **Bridge.** `PLAN.json`, `RESULT.json`, `INPUT_BINDINGS.json`,
  `UNDISTORT_COMPARISON.json` and `scene.mvs` must exist; `RESULT.json` must
  declare `PASS`. `INPUT_BINDINGS.json` is verified **entry by entry** against the
  bytes on disk, every source-model file must be declared, and every staged image
  that is used must be declared too. Source model and image directories come from
  the PLAN, and `--sfm-run` must equal `PLAN.source_images`.
* **T01.** Located from `source_io.path`; `run.json` and `frames.jsonl` bytes must
  match the declared digests, the declared `run_id` must equal the run's own id,
  the run must be `photo-io-v1` with status `SUCCESS`, and that id must equal the
  SFM `upstream_io_run_id`. Directories are never found by searching for a run id.
* **Identity.** bridge `image.name` -> `frames_map.staged_image_name` ->
  `(input_index, frame_id)` -> the T01 line with the same pair -> the manifest
  frame with the same pair. Filename order, `image_id` equality, numeric filenames
  and basename guessing are never used.
* **Original versus canonical.** T01 `image_sha256` is the original input digest,
  preserved as `photo_io_image_sha256`; the original photograph is not re-read.
  It may differ from the canonical PNG. Manifest `source_image` must resolve to
  the corresponding T01 `image_output`, with domain `photo_io_canonical`.
  Its actual digest must equal `source_image_sha256`, SFM
  `canonical_image_sha256`, and the staged file digest. Canonical and staged are
  exact copies; even equal pixels with different PNG encoding are rejected.
  Canonical RGB8 PNG pixels, dimensions and the source camera size are checked.
* **Selected set.** The registered native model determines the consumed images.
  Other frames_map rows are retained as unselected. A manifest's nonselected
  frame without a selected mask may be ignored only after its T01/SFM identity,
  original reference, canonical path and available digest have been checked.
* **Paths.** Every relative path (`staged_image_name`, the T01 image path,
  `source_image`, `mask_path`) must resolve inside its own declared root; escapes
  are rejected.
* **Models.** Source and undistorted images must share names, poses within
  `rtol=atol=1e-8` (rotation, translation, centre), and the undistorted image file
  size must equal the target camera; the target must be
  `PINHOLE`/`SIMPLE_PINHOLE`. Both cameras' full models, both image ids and the
  image bytes are recorded.
* **Scene.** The existing MVSI v7 reader validates the scene. Full names must
  equal `undistorted/images/` plus the native image name and resolve to the actual
  target file inside the bridge. Scene global IDs must equal native image IDs.
  Sizes and K/R/C must agree; InterfaceCOLMAP subtracts 0.5 from the native
  principal point (K tolerance 1e-7 pixels), while R/C use rtol=atol=1e-8.
  This identity check applies no extra offset to the mask warp. Unverified edges
  produce INPUT_ERROR and prevent publication.
* **Duplicates and collisions.** The same image in two manifests is refused even
  when the bytes are identical; two manifests must share one `object_id` and one
  T01 run; a bridge image claimed twice is refused; OpenMVS output names
  (`basename` with the last extension dropped plus `.mask.png`) must not collide
  case-insensitively, across directories, or between different extensions.

## Geometry

Warping reuses `integrations/mvs_masks/mask_geometry.py`
(`colmap-center-nearest-label-v1`). The output size comes from the **real**
undistorted camera, never a hard-coded value. Source masks must be single-frame
8-bit `L`, opaque, orientation 1 or absent, strictly 0/255 and exactly the
canonical size; nothing is resized or re-encoded. Unknown, outside and non-finite
mappings are 0 in the mask **and** 0 in the separate validity raster, while valid
background is mask 0 with validity 255. A selected empty mask or a foreground
count inconsistent with its pixels is INPUT_ERROR. A valid input whose warped
foreground is empty becomes `EMPTY_AFTER_WARP` and is never a usable mask.

Written masks **and** validity rasters are read back and compared pixel by pixel
with the arrays handed to the writer; the digests must match, the labels must stay
in {0,255}, the validity raster must encode exactly `counts.unknown` unknown
pixels, and no foreground may sit on an unknown pixel.

## Failure regimes

* **Before the directory is claimed**: schema, hash, identity, path, model, pose,
  collision or an existing output directory. Exit 2 (or 4 for our own defect) with
  a message on stderr, and **no output directory is created** - so a rejection can
  never claim an output or overwrite somebody else's.
* **After the directory is claimed**: a strict coverage shortfall writes a
  manifest with per-frame `NOT_RUN`/`MISSING` states and no mask at all; a warp,
  write or read-back failure writes `FAILURE.json` with the first cause, the
  all target frames (completed, failed, and not started) and artifacts already
  produced. The failed frame carries its exception reason; the run preserves
  its first cause separately. If even that
  cannot be written the error is recorded separately and reported on stderr; a
  manifest is never faked.

Exit codes: complete conversion 0; `--allow-partial` with a real shortfall 1;
input contradiction, a strict shortfall or drift 2; our own product failure 4.

## Publication gate

`mvs_mask_arguments.available` is true **only** when the final status is
`SUCCESS`, coverage is `COMPLETE`, the inputs did not drift during the run, every
frame's mask and validity read back correctly, and no unbound manifest frame
remains - `publishable_arguments` states the same decision explicitly. A complete
run that drifted, or any error, shortfall or failed read-back, publishes nothing.
The published fragment (`--mask-path <output>/masks --ignore-mask-label 0`) is
marked `verified: false`; `actual_mvs_invocations` is always 0. The default
`--ignore-mask-label -1` disables ignoring, so nothing here claims the background
is actually excluded.

## Binding boundaries

`INPUT_BINDINGS.json` declares the bridge's own inputs, and those entries are
verified one by one. The bridge's own products (undistorted sparse model,
undistorted images, `scene.mvs`) have no upstream declaration, so they are bound
with method `local_hash_only` - their current bytes are recorded, and no
historical declaration is invented for them. Each manifest lists every verified
edge and these boundaries explicitly.

## Evidence levels and limits

Masks from a manifest marked `synthetic_interface_fixture` are recorded with that
evidence level; synthetic labels are never SAM3 predictions, human truth, or
evidence about segmentation quality or about OpenMVS excluding the background.
The pilot04 source contains one completed real SAM3 mask. The saved R3 real
conversion covers 1/44 images with PARTIAL/exit1 and no full MVS arguments; its
synthetic suite was red. The independent main closure has its own evidence under
`outputs/delegation/MVS_MASK_MAIN/20261008_closure/`; consult its RESULT for the
actual verification outcome. Source RESOURCE_LIMIT status is preserved when an
already completed OK frame is consumed with `--allow-partial`.
OpenMVS mask loading, label ignoring, dense surface, mesh, CAD and absolute scale
remain **not verified** by this tool; a `SUCCESS` here means a format conversion
and nothing more.
# 显式选择重复来源（object-mask-selection-v1）

默认行为保持：同一帧在多份 manifest 出现即拒绝，即使其中一行是
NO_MATCH。只有明确传入 `--selection-ledger ABS_JSON` 才使用下述来源选择。
账本不生成 mask、不改任何原 manifest，也不认证分割或跨视质量。

账本 JSON 的根必须恰含 `schema_version`、`manifests`、`selections`：

```json
{
  "schema_version": "object-mask-selection-v1",
  "manifests": [
    {"path": "C:/absolute/primary/manifest.json", "sha256": "<actual lowercase sha256>"},
    {"path": "C:/absolute/supplement/manifest.json", "sha256": "<actual lowercase sha256>"}
  ],
  "selections": [
    {
      "input_index": 59,
      "frame_id": "<actual bound frame_id>",
      "manifest_path": "C:/absolute/supplement/manifest.json",
      "selected_candidate_id": "<actual selected candidate>",
      "selected_mask_sha256": "<actual selected mask sha256>"
    }
  ]
}
```

示例仅展示字段；实际 selections 必须逐一列出桥接全部目标，不能只填
补充的两帧。每个目标恰一行，来源必须是已选中合法 mask 的 OK 行。
所有传入 manifest 恰一次绑定，路径必须绝对；遗漏、多余、双选、未知
来源、错误候选/hash/身份均拒绝。账本模式不接受源清单中桥接集合之外
的行；需另有显式受审的来源子集方案，不能借账本隐藏未知输入。

仍须把全部来源用重复 `--mask-manifest ABS_PATH` 传入。所有原清单完整
经过 schema、counts、T01/object/图像身份核验；包括未选行的 canonical
文件和候选 PNG 字节、尺寸、标签、前景计数。只在这些检查后使用账本
明确选择的行，原 run 的 PARTIAL/NO_MATCH 状态不被改写为 SUCCESS。

输出 `mask_selection` 绑定账本、原 `source_runs` 状态/counts、有效选择，
以及每条 `unselected_source_rows` 的原 status、来源路径/hash、未选理由
和替代来源/candidate/mask hash。被显式替换的原 NO_MATCH 不进入
`unbound_object_mask_frames`；它仍完整保留在未选记录中。账本、全部来源
manifest、所有候选 mask 及选择模块字节加入输入稳定性检查。姿态、scene、
PNG 输出读回与完整覆盖门保持；新 consumer 的 SUCCESS 仅表示域转换
完整，`quality_status/cross_view_status=NOT_EVALUATED`，原生消费仍 NOT_RUN。


## 已批准43视图证书与源59排除

object-mask-selection-v1在显式43模式额外且同时要求 `subset_certificate:{"path":"ABS_CERT","sha256":"ACTUAL_SHA"}` 与 `excluded_source_rows`。无证书时旧字段集合/桥接集合外拒绝门保持。证书必须独立重算、属于当前bridge source_sparse及同SfM/T01；所选目标必须是精确43。

excluded_source_rows恰两行，分别绑定原44及human-skull补2清单的原59 NO_MATCH；每行字段恰为 manifest_path（绝对）、manifest_sha256、input_index=59、frame_id（真实）、reason="NO_MATCH_VIEW_EXCLUDED"。省略、多余、别的身份、错误hash或任何OK/候选mask均拒绝。全部源manifest仍完整schema/count/hash核验；CLI仍把原清单交bind_frames(validate_all_sources=True)，对被排59也检查T01/frame-map/canonical字节、尺寸和图像解码，而非只在helper过滤后假装通过。

selections恰43：42原OK＋补60 OK；不改原status/counts。输出unselected_source_rows保留两条原NO_MATCH、来源path/hash、证书绑定、缺视图理由与replacement_manifest=null，不造mask。证书及其全部模型/映射/图像/实现绑定加入输入前后稳定性复核。完整转换只表示43覆盖，不升级44完整、分割真值或OpenMVS消费。
