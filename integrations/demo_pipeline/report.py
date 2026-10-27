"""Offline, evidence-bound photo/segmentation/reconstruction demo report.

Public API: render_report(run_dir: Path) -> Path.
CLI: python report.py RUN_DIR

Reads input/RESULT.json, INPUT_INVENTORY.json and committed segmentation
PRODUCT_OUTCOME.json -> manifest.json. REPORT_DATA.json overrides report_data
in pipeline.json. Report metadata accepts title, object_label, source,
frame_notes (input-index keyed), visual_review, processing_notes, limitations,
pipeline_errors, errors, stage_statuses,
sfm_masks_applied and pointcloud. The latter accepts status, summary, stats,
limitations, evidence_paths, viewer_path, ply_path. Artifact paths are relative
to RUN_DIR; absolute existing paths are also accepted for evidence links.

No inference is run here. Pillow only produces display copies and overlays.
The authoritative mask remains the exact selected, SHA-256 verified PNG.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import html
import io
import json
import os
import re
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlsplit

from PIL import Image, ImageChops, ImageFilter


def _read(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        return {} if default is None else default
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path(value: Any, base: Path) -> Path | None:
    if isinstance(value, dict):
        value = value.get("path")
    if not value or not isinstance(value, (str, os.PathLike)):
        return None
    path = Path(value)
    return (path if path.is_absolute() else base / path).resolve()


def _within(path: Path, base: Path) -> bool:
    try:
        path.resolve().relative_to(base.resolve())
        return True
    except ValueError:
        return False


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _uri(path: Path, report_dir: Path) -> str:
    try:
        return quote(Path(os.path.relpath(path, report_dir)).as_posix(), safe="/.-_")
    except ValueError:
        return path.as_uri()


def _image_data(image: Image.Image, *, png: bool = False) -> str:
    buffer = io.BytesIO()
    if png:
        image.save(buffer, format="PNG")
        kind = "png"
    else:
        image.convert("RGB").save(buffer, format="JPEG", quality=91, optimize=True)
        kind = "jpeg"
    return "data:image/" + kind + ";base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _binary_data(path: Path, mime: str) -> str:
    return "data:" + mime + ";base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def _stage_status(pipeline: dict, name: str) -> str:
    stages = pipeline.get("stage_statuses", pipeline.get("stages", {}))
    if isinstance(stages, dict):
        stage = stages.get(name, {})
        return str(stage.get("status", "PENDING") if isinstance(stage, dict) else stage)
    if isinstance(stages, list):
        for stage in stages:
            if isinstance(stage, dict) and stage.get("name", stage.get("stage")) == name:
                return str(stage.get("status", "PENDING"))
    return "PENDING"


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]
    return []


def _status_label(status: str) -> str:
    return {
        "SUCCESS": "已完成", "OK": "已完成", "NO_MODEL": "未重建成功",
        "FAILED": "未完成", "ERROR": "运行失败", "PENDING": "尚无结果",
        "NOT_RUN": "尚未运行", "SKIPPED": "未运行", "RUNNING": "处理中",
        "TIMEOUT": "已超时", "PARTIAL": "部分完成", "BLOCKED": "暂未完成",
        "PREPARATION_FAILED": "输入处理失败", "PRECONDITION_BLOCKED": "运行前检查未通过",
        "BACKEND_ERROR": "后端运行失败", "RESOURCE_LIMIT": "资源限制已触发",
        "INPUTS_PREPARED": "已整理", "NOT_EVALUATED": "未评估",
    }.get(status.upper(), "待核查")


def _pipeline_context(pipeline: dict, metadata: dict, prepared: dict) -> tuple[dict, list[str]]:
    """Retain source/cleanup errors even when individual artifact stages succeeded."""
    stages: dict[str, Any] = {}
    for records in (pipeline.get("stages", {}), metadata.get("stage_statuses", {})):
        if isinstance(records, dict):
            stages.update(records)
        elif isinstance(records, list):
            for record in records:
                if isinstance(record, dict) and record.get("name", record.get("stage")):
                    stages[str(record.get("name", record.get("stage")))] = record
    errors: list[str] = []
    def add(value: Any) -> None:
        if isinstance(value, (tuple, list)):
            for item in value:
                add(item)
        elif isinstance(value, dict):
            add(value.get("message") or value.get("error") or value.get("reason") or json.dumps(value, ensure_ascii=False))
        elif value is not None and str(value).strip() and str(value) not in errors:
            errors.append(str(value))
    for value in (pipeline.get("errors"), metadata.get("pipeline_errors"), metadata.get("errors"), prepared.get("error")):
        add(value)
    names = {"input": "输入", "segmentation": "目标分割", "sfm": "三维重建", "viewer": "点云展示", "surface": "表面重建", "blender": "Blender 导出", "report": "报告"}
    failure_statuses = {"FAILED", "ERROR", "PREPARATION_FAILED", "BACKEND_ERROR", "PRECONDITION_BLOCKED", "TIMEOUT", "RESOURCE_LIMIT", "BLOCKED"}
    for name, stage in stages.items():
        if not isinstance(stage, dict):
            stage = {"status": stage}
        label = names.get(name, name)
        if str(stage.get("status", "")).upper() in failure_statuses:
            add(f"{label}阶段：{_status_label(str(stage['status']))}（{stage['status']}）。")
        if stage.get("reclaim_confirmed") is False:
            add(f"{label}阶段：进程回收未确认，已有产物不代表整个流程正常结束。")
        for field in ("error", "errors", "log_errors"):
            add(stage.get(field))
    return {"stages": stages}, errors


def _load_frames(run: Path, prepared: dict, inventory: dict,
                 metadata: dict, warnings: list[str]) -> tuple[list[dict], dict, Path | None]:
    outcome = _read(run / "segmentation" / "PRODUCT_OUTCOME.json")
    seg_root = _path(outcome.get("run_dir"), run)
    manifest_path = _path(outcome.get("manifest_path"), run)
    if manifest_path is None and seg_root is not None:
        manifest_path = seg_root / "manifest.json"
    manifest = _read(manifest_path) if manifest_path else {}
    if manifest_path and seg_root is None:
        seg_root = manifest_path.parent
    committed: dict[int, dict] = {}
    for frame in manifest.get("frames", []):
        index = frame.get("input_index")
        if isinstance(index, int):
            if index in committed:
                warnings.append(f"分割清单中的照片索引 {index} 重复，该照片的掩膜未展示。")
                committed[index] = {"status": "INVALID_DUPLICATE"}
            else:
                committed[index] = frame
    photo_io = prepared.get("photo_io") or {}
    canonical = photo_io.get("canonical_images", [])
    identities = {item.get("input_index"): item for item in photo_io.get("identities", [])}
    source_frames = prepared.get("frames", [])
    io_run_file = _path(photo_io.get("run"), run)
    io_ledger_file = _path(photo_io.get("frames"), run)
    if io_ledger_file is None and io_run_file:
        io_ledger_file = io_run_file.parent / "frames.jsonl"
    canonical_by_index: dict[int, dict] = {}
    is_video = prepared.get("input_kind") == "video" or metadata.get("source_kind") == "video"
    if io_ledger_file and io_ledger_file.is_file() and io_run_file:
        ledger_ref = photo_io.get("frames") or {}
        ledger_sha = ledger_ref.get("sha256") if isinstance(ledger_ref, dict) else photo_io.get("frames_sha256")
        if ledger_sha and _sha(io_ledger_file) != ledger_sha:
            warnings.append("标准化照片清单的文件校验未通过，分割结果不计入已验证输出。")
            committed = {}
        ledger = [json.loads(line) for line in io_ledger_file.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        normalized = []
        source_mapping = {item.get("photo_input_index" if is_video else "source_index"): item for item in source_frames}
        source_bindings = prepared.get("source_bindings") or []
        video_source = source_bindings[0] if source_bindings else {"path": metadata.get("source")}
        seen_inputs: set[int] = set()
        for item in sorted(ledger, key=lambda row: row.get("input_index", -1)):
            index = item.get("input_index")
            if not isinstance(index, int) or item.get("status") != "OK" or not item.get("image_output"):
                continue
            if index in seen_inputs:
                warnings.append(f"标准化照片索引 {index} 重复，后续重复记录未展示。")
                committed.pop(index, None)
                continue
            seen_inputs.add(index)
            image_path = _path(item["image_output"], io_run_file.parent)
            if image_path is None or not _within(image_path, io_run_file.parent):
                warnings.append(f"标准化照片索引 {index} 的路径无效，未展示。")
                continue
            ref = canonical[index] if 0 <= index < len(canonical) else {}
            if ref and _path(ref, run) != image_path:
                warnings.append(f"标准化照片索引 {index} 的路径记录不一致，分割结果不计入已验证输出。")
                committed.pop(index, None)
            canonical_by_index[index] = {"path": str(image_path), "sha256": ref.get("sha256") or item.get("canonical_sha256") or item.get("image_sha256")}
            identities[index] = {"frame_id": item.get("frame_id")}
            original_entry = source_mapping.get(index, {})
            if is_video:
                source_name = Path(video_source.get("path") or "视频").name
                extracted_index = original_entry.get("output_index", index)
                entry = {"source_index": index, "source": video_source, "status": "PREPARED",
                         "display_filename": f"{source_name} · 抽取帧 {extracted_index + 1}",
                         "metadata": {"raw_dimensions": [item.get("raw_width"), item.get("raw_height")]}}
            else:
                entry = dict(original_entry) if original_entry else {"source_index": index, "source": {"path": item.get("input_image", item["image_output"])}}
            normalized.append(entry)
        source_frames = normalized
    if not source_frames:
        source_frames = [{"source_index": idx, "source": item, "status": "NOT_PREPARED"}
                         for idx, item in enumerate(inventory.get("files", []))]
    notes = metadata.get("frame_notes", {})
    rows: list[dict] = []
    for fallback_index, entry in enumerate(source_frames):
        index = entry.get("source_index", fallback_index)
        source = entry.get("source", {})
        source_path = _path(source, run)
        filename = entry.get("display_filename") or (source_path.name if source_path else f"照片 {index + 1}")
        canonical_ref = canonical_by_index.get(index) or (canonical[index] if isinstance(index, int) and 0 <= index < len(canonical) else entry.get("prepared", {}))
        image_path = _path(canonical_ref, run)
        is_prepared = image_path is not None
        if image_path is None:
            image_path = source_path
            canonical_ref = source
        row: dict[str, Any] = {
            "index": index, "number": len(rows) + 1, "filename": filename,
            "prepared": is_prepared, "original": None, "overlay": None, "mask": None,
            "mask_verified": False, "state": "尚无分割结果", "state_kind": "pending",
            "note": str(notes.get(str(index), notes.get(index, ""))), "coverage": None,
            "source_path": str(source_path) if source_path else None,
            "image_path": str(image_path) if image_path else None,
            "source_sha256": source.get("sha256"), "mask_path": None,
            "raw_dimensions": entry.get("metadata", {}).get("raw_dimensions"),
        }
        try:
            if image_path is None or not image_path.is_file():
                raise ValueError("照片文件缺失")
            expected = canonical_ref.get("sha256") if isinstance(canonical_ref, dict) else None
            if not expected or _sha(image_path) != expected:
                raise ValueError("照片的文件校验未通过")
            with Image.open(image_path) as opened:
                original = opened.convert("RGB")
            row["dimensions"] = list(original.size)
            row["original"] = _image_data(original)
            thumb = original.copy()
            thumb.thumbnail((180, 180), Image.Resampling.LANCZOS)
            row["thumbnail"] = _image_data(thumb)
            frame = committed.get(index, {})
            row["committed_status"] = frame.get("status", "NOT_RUN")
            if frame.get("status") != "OK":
                state = frame.get("status", "NOT_RUN")
                row["state"] = {"NO_MATCH": "未找到目标", "NEEDS_SELECTION": "待选择目标",
                                "ERROR": "分割失败", "NOT_RUN": "尚无分割结果"}.get(state, "无可展示的已提交掩膜")
                if state not in ("NOT_RUN", "PENDING"):
                    row["state_kind"] = "warning"
                rows.append(row)
                continue
            if not is_prepared:
                raise ValueError("缺少用于对齐的标准化照片")
            if _path(frame.get("source_image"), seg_root or run) != image_path:
                raise ValueError("分割记录与展示照片的路径不一致")
            if frame.get("source_image_sha256") != expected:
                raise ValueError("分割记录与展示照片的哈希不一致")
            identity = identities.get(index, {})
            if identity.get("frame_id") and frame.get("frame_id") != identity["frame_id"]:
                raise ValueError("分割记录与输入照片的身份不一致")
            selected = frame.get("selected_mask") or {}
            mask_path = _path(selected.get("path"), seg_root or run)
            if mask_path is None or not mask_path.is_file() or seg_root is None or not _within(mask_path, seg_root):
                raise ValueError("已选择的掩膜文件缺失或不在对应结果目录中")
            if not selected.get("sha256") or _sha(mask_path) != selected["sha256"]:
                raise ValueError("已选择的掩膜文件校验未通过")
            with Image.open(mask_path) as opened:
                mask = opened.convert("L")
            if mask.size != original.size:
                raise ValueError("掩膜与展示照片尺寸不一致")
            histogram = mask.histogram()
            if sum(histogram[1:255]):
                raise ValueError("选中的文件不是二值掩膜")
            foreground = histogram[255]
            if not foreground:
                raise ValueError("选中的掩膜为空")
            teal = Image.new("RGB", original.size, (27, 185, 159))
            tinted = Image.blend(original, teal, 0.38)
            overlay = Image.composite(tinted, original, mask)
            border = ImageChops.subtract(mask.filter(ImageFilter.MaxFilter(5)), mask.filter(ImageFilter.MinFilter(5)))
            overlay = Image.composite(Image.new("RGB", original.size, (49, 236, 194)), overlay, border)
            row.update(mask_verified=True, state="已输出目标掩膜", state_kind="success",
                       mask=_binary_data(mask_path, "image/png"), overlay=_image_data(overlay),
                       mask_path=str(mask_path), mask_sha256=selected["sha256"],
                       foreground_pixels=foreground, coverage=round(foreground / (mask.width * mask.height) * 100, 1))
        except (OSError, ValueError, TypeError) as exc:
            row.update(state="结果待核查", state_kind="warning", verification_error=str(exc))
            warnings.append(f"{filename}：{exc}。")
        rows.append(row)
    return rows, {"outcome": outcome, "manifest": manifest}, manifest_path


class _Links(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.urls: list[tuple[str, str]] = []
        self.data_images = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for name, value in attrs:
            if value and name in ("href", "src"):
                self.urls.append((tag, value))
                self.data_images += int(value.startswith("data:image/"))


def _viewer_is_self_contained(source: str) -> bool:
    """Check resource references without mistaking ordinary // comments for URLs."""
    links = _Links()
    links.feed(source)
    if any(not url.startswith(("data:", "#", "blob:")) for _, url in links.urls):
        return False
    # An unqualified // is a JavaScript comment. Protocol-relative URLs require
    # a quoted value or a CSS url(); absolute URLs and runtime loaders remain
    # conservative reasons to preserve the viewer's original local base URL.
    network_or_loader = re.compile(
        r"https?://|[\"'`]\s*//[^/\s]|\bfetch\s*\(|\bimport\s*\("
        r"|\b(?:import|export)\s+(?:[^;()\n]*?\s+from\s+)?[\"']",
        re.IGNORECASE,
    )
    if network_or_loader.search(source):
        return False
    for match in re.finditer(r"\burl\(\s*([\"']?)(.*?)\1\s*\)", source, re.IGNORECASE | re.DOTALL):
        if not match.group(2).strip().startswith(("data:", "#", "blob:")):
            return False
    return True


def _pointcloud(run: Path, report_dir: Path, metadata: dict, pipeline: dict,
                warnings: list[str]) -> tuple[str, dict, list[Path]]:
    cloud = metadata.get("pointcloud", {})
    if not isinstance(cloud, dict):
        cloud = {}
    status = str(cloud.get("status", _stage_status(pipeline, "sfm"))).upper()
    stats = cloud.get("stats", {}) if isinstance(cloud.get("stats", {}), dict) else {}
    registered = stats.get("registered_images", stats.get("registered", None))
    points = stats.get("points3d", stats.get("points", None))
    viewer = _path(cloud.get("viewer_path"), run)
    ply = _path(cloud.get("ply_path"), run)
    evidence = [_path(path, run) for path in cloud.get("evidence_paths", [])]
    evidence = [path for path in evidence if path and path.is_file()]
    ready = status in ("SUCCESS", "OK", "PARTIAL") and viewer is not None and viewer.is_file() and _within(viewer, run)
    if status in ("NO_MODEL", "FAILED", "ERROR", "TIMEOUT", "BLOCKED"):
        ready = False
    if ready:
        viewer_source = viewer.read_text(encoding="utf-8-sig")
        if _viewer_is_self_contained(viewer_source):
            iframe = f'<iframe title="可交互的场景稀疏点云" sandbox="allow-scripts allow-downloads" srcdoc="{_esc(viewer_source)}"></iframe>'
            embedded = True
        else:
            iframe = f'<iframe title="可交互的场景稀疏点云" sandbox="allow-scripts allow-same-origin allow-downloads" src="{_esc(_uri(viewer, report_dir))}"></iframe>'
            embedded = False
        actions = f'<a class="button secondary" href="{_esc(_uri(viewer, report_dir))}" target="_blank" rel="noopener">单独打开点云 <span>↗</span></a>'
        if ply and ply.is_file() and _within(ply, run):
            actions += f'<a class="button secondary" href="{_esc(_uri(ply, report_dir))}" download>下载 PLY 点云 <span>↓</span></a>'
        viewer_html = f'<div class="cloud-viewer">{iframe}</div><div class="cloud-actions">{actions}</div>'
    else:
        embedded = False
        if status == "NO_MODEL":
            headline = "这些照片尚未形成三维模型"
            body = "本次重建未建立可用的相机位置与三维点，因此这里没有点云可展示。分割结果仍可独立查看。"
            symbol = "—"
        elif status in ("FAILED", "ERROR", "TIMEOUT", "BLOCKED"):
            headline = "三维重建未完成"
            body = "当前没有经确认可展示的三维结果。具体运行状态见下方处理记录。"
            symbol = "—"
        else:
            headline = "尚无可展示的三维结果"
            body = "待重建完成并确认对应产物后，这里将显示可交互的场景稀疏点云。"
            symbol = "···"
        viewer_html = f'<div class="cloud-empty"><div class="empty-symbol" aria-hidden="true">{symbol}</div><h3>{headline}</h3><p>{body}</p><div class="empty-stats"><div><strong>{_esc(registered if registered is not None else "—")}</strong><span>已注册照片</span></div><div><strong>{_esc(points if points is not None else "—")}</strong><span>三维点</span></div><div><strong>见后续结果</strong><span>表面 / Blender</span></div></div></div>'
    note = cloud.get("summary", "稀疏点云用于检查多视角的几何关系，不代表完整物体表面。")
    limits = _strings(cloud.get("limitations"))
    detail_html = ''.join(f'<li>{_esc(item)}</li>' for item in limits)
    detail = f'<ul class="clean-list">{detail_html}</ul>' if detail_html else ''
    cloud_html = f'<div class="cloud-header"><div><p class="eyebrow">02 · 空间重建</p><h2>场景稀疏点云</h2></div><span class="pill {"success" if ready else "warning"}">{_esc(_status_label(status))}</span></div><p class="section-intro">{_esc(note)}</p>{viewer_html}{detail}'
    return cloud_html, {"status": status, "viewer_available": ready, "viewer_embedded": embedded,
                        "registered_images": registered, "points3d": points}, evidence


def _geometry_outputs(run: Path, report_dir: Path, metadata: dict,
                      pipeline: dict, warnings: list[str]) -> tuple[str, dict, dict]:
    """Expose only declared, hash-bound outputs; never infer success from a file."""
    sections = []
    results = {}
    for name, title in (("surface", "开放表面网格"), ("blender", "Blender 文件")):
        record = metadata.get(name, {})
        record = record if isinstance(record, dict) else {}
        record = dict(record)
        for canonical, alias in (("selected_input_indices", "selected_indices"),
                                 ("missing_input_indices", "missing_indices")):
            if canonical not in record and alias in record:
                record[canonical] = record[alias]
        status = str(record.get("status", _stage_status(pipeline, name))).upper()
        declared = record.get("artifacts", {}) if name == "surface" else {"blend": record.get("blend", {"path": record.get("path"), "sha256": record.get("sha256")})}
        if isinstance(declared, list):
            declared = {str(index): item for index, item in enumerate(declared)}
        declared = declared if isinstance(declared, dict) else {}
        reopen = record.get("reopen_validation", {})
        reopen_ok = (record.get("reopen_verified") is True or
                     isinstance(reopen, dict) and str(reopen.get("status", "")).upper() in {"PASS", "OK", "SUCCESS"})
        permitted = status == "SUCCESS" and (name != "blender" or reopen_ok)
        verified = []
        actions = []
        for label, ref in declared.items():
            if not isinstance(ref, dict):
                continue
            path = _path(ref, run)
            expected = ref.get("sha256")
            valid = False
            if path and path.is_file() and isinstance(expected, str) and re.fullmatch(r"[0-9a-fA-F]{64}", expected):
                valid = _sha(path).lower() == expected.lower()
            if name == "blender" and path and path.suffix.lower() != ".blend":
                valid = False
            verified.append({"label": label, "path": str(path) if path else None,
                             "sha256": expected, "verified": valid})
            if permitted and valid and (name == "blender" or path.suffix.lower() != ".blend"):
                actions.append(f'<a class="button secondary" href="{_esc(_uri(path, report_dir))}" download>{_esc("下载 .blend" if name == "blender" else "下载 " + str(label))} ↓</a>')
            elif not valid:
                warnings.append(f"{title}产物 {label} 的文件或 SHA-256 校验未通过，未提供下载链接。")
        available = bool(actions)
        limits = _strings(record.get("limitations"))
        if name == "surface":
            limits.insert(0, "开放表面允许孔洞；真实尺度未知，不代表封闭实体或精确测量成果。")
        else:
            limits.insert(0, "仅在导出结果通过、文件哈希一致且独立重新打开验证通过时提供 .blend 下载。")
        details = []
        for key, label in (("selected_model_id", "选中模型"), ("selected_input_indices", "选中输入索引"), ("missing_input_indices", "缺失输入索引")):
            if key in record:
                details.append(f"{label}：{record[key]}")
        details.extend(_strings(record.get("error")))
        details.extend(_strings(record.get("errors")))
        body = ''.join(f'<li>{_esc(item)}</li>' for item in details + limits)
        file_label = '文件已就绪' if available else _status_label(status)
        sections.append(f'<section class="cloud-section" id="{name}"><div class="cloud-header"><h2>{title}</h2><span class="pill {"success" if available else "warning"}">{_esc(file_label)}</span></div><p>{"已校验产物可下载。" if available else "当前没有通过全部展示条件的产物。"}</p><div class="cloud-actions">{"".join(actions)}</div><ul class="clean-list">{body}</ul></section>')
        results[name] = {"status": status, "available": available, "artifacts": verified,
                         "limitations": limits, **{key: record[key] for key in ("selected_model_id", "selected_input_indices", "missing_input_indices") if key in record}}
        if name == "blender":
            results[name]["reopen_verified"] = reopen_ok
    return ''.join(sections), results["surface"], results["blender"]


def _cad_outputs(run: Path, report_dir: Path, pipeline: dict, warnings: list[str]) -> tuple[str, dict]:
    """CAD links require the native delivery verifier, never display metadata."""
    record = pipeline.get('stages', {}).get('cad', {})
    status = record.get('status', 'NOT_RUN')
    result = {'status': status, 'available': False, 'artifacts': []}
    actions = []
    if status == 'SUCCESS':
        try:
            from integrations.demo_pipeline.cad import verify_cad_output
            output = (run / 'cad').resolve()
            if not output.is_relative_to(run) or Path(record.get('output', '')).resolve() != output:
                raise ValueError('CAD output does not belong to this run')
            checked = verify_cad_output(output, run)
            for ref in checked['artifacts']:
                path = Path(ref['path'])
                actions.append(f'<a class="button secondary" href="{_esc(_uri(path, report_dir))}" download>下载 {_esc(path.suffix)} ↓</a>')
            result.update(available=True, artifacts=checked['artifacts'], scale_status=checked['scale_status'])
        except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
            result['error'] = str(exc)
            warnings.append('CAD 产物未通过当前核验，未提供下载：' + str(exc))
    text = ('有效闭合单实体、FreeCAD 重开及 STEP 重新导入已通过。' if result['available']
            else '尚无通过检查的 CAD 实体；已通过验证的 Blender 表面可单独保留。')
    note = '真实尺度与实物精度未验证；面片化实体不包含原设计的参数化特征树。开放表面不自动补洞。'
    html_text = f'<section class="cloud-section" id="cad"><h2>CAD 实体</h2><p>{text}</p><div class="cloud-actions">{"".join(actions)}</div><p>{note}</p></section>'
    return html_text, result


def _view_geometry_outputs(run: Path, report_dir: Path, metadata: dict, pipeline: dict, warnings: list[str]) -> tuple[str, dict]:
    record=metadata.get('view_geometry',{})
    record=record if isinstance(record,dict) else {}
    status=str(record.get('status','NOT_RUN')); actions=[]
    validation={'status':status,'available':False,'geometry_acceptance':False,'scale_status':'unknown'}
    try:
        if status in ('DIAGNOSTIC_COMPLETE','PARTIAL_DIAGNOSTIC'):
            from integrations.demo_pipeline.pipeline import validate_view_geometry
            output=run/'view_geometry'; ref=record.get('result',{}); path=_path(ref,run)
            if (not _within(output,run) or path!=output/'RESULT.json' or not path.is_file()
                    or path.stat().st_size!=ref.get('bytes') or _sha(path)!=ref.get('sha256')):
                raise ValueError('几何诊断 RESULT 身份不匹配')
            payload=_read(path)
            source_run=Path(record['source_run']).resolve(strict=True)
            sfm_stage=pipeline.get('stages',{}).get('sfm',{})
            if source_run!=Path(sfm_stage['run_json']).resolve(strict=True):
                raise ValueError('几何诊断与当前 SfM 来源不匹配')
            validate_view_geometry(output,payload,source_run,record['selected_model_id'])
            if payload['status']!=status or payload['artifacts']!=record.get('artifacts') or payload['source_bindings']!=record.get('source_bindings'):
                raise ValueError('几何诊断清单不匹配')
            for item in [ref]+payload['artifacts']:
                artifact=_path(item,run)
                actions.append(f'<a class="button secondary" href="{_esc(_uri(artifact,report_dir))}">{_esc(artifact.name)}</a>')
            validation.update(available=True,result=ref,artifacts=payload['artifacts'])
    except (OSError,ValueError,KeyError,TypeError,AttributeError) as exc:
        warnings.append(str(exc));validation['verification_error']=str(exc)
    note=record.get('error') or record.get('reason') or '原生 SfM 数值诊断；不修改重建，不表示物理尺度或精度验收。'
    content=('<section class="cloud-section" id="view-geometry"><h2>自动选视图与几何诊断（非精度验收）</h2>'
             +'<p>'+_esc(status)+'；模型 '+_esc(record.get('selected_model_id','未选择'))+'；源 SfM '+_esc(record.get('source_sfm_status','未核实'))+'</p>'
             +'<p>'+_esc(note)+'</p><p>计数：'+_esc(record.get('counts',{}))+'</p><div class="cloud-actions">'+''.join(actions)+'</div></section>')
    return content,validation



def _post_review_outputs(run, report_dir, metadata, pipeline, warnings):
    detail=metadata.get('post_review') or pipeline.get('stages',{}).get('post_review',{})
    detail=detail if isinstance(detail,dict) else {}
    status=detail.get('status','NOT_RUN');links=[]
    verified={'status':status,'available':False,'geometry_acceptance':False}
    try:
        if status in ('DIAGNOSTIC_COMPLETE','INSUFFICIENT_DATA','FAILED') and detail.get('record'):
            output=(run/'post_review').resolve()
            for role,name in [('record','RESULT.json'),('report','index.html')]:
                ref=detail[role];path=Path(ref['path']).resolve(strict=True)
                if not _within(output,run) or path!=output/name or path.stat().st_size!=ref.get('bytes') or _sha(path)!=ref.get('sha256'):
                    raise ValueError('自动复核文件绑定不匹配')
                links.append(f'<a class="button secondary" href="{_esc(_uri(path,report_dir))}">{_esc(name)}</a>')
            if _read(output/'RESULT.json').get('status')!=status:raise ValueError('自动复核状态不匹配')
            verified['available']=True
    except (OSError,ValueError,KeyError,TypeError,AttributeError) as exc:
        warnings.append(str(exc));verified['verification_error']=str(exc);links=[]
    html=('<section class="cloud-section" id="post-review"><h2>自动离线复核（非建模验收）</h2><p>'
          +_esc(status)+'</p><p>复核多视图对应与外观证据；不改变表面/CAD验收，不宣称精度或对应真值。</p>'
          +'<p>'+_esc(detail.get('error') or detail.get('reason',''))+'</p><div class="cloud-actions">'+''.join(links)+'</div></section>')
    return html,verified

def _point_review_outputs(run, report_dir, metadata, pipeline, warnings):
    detail=metadata.get('point_review') or pipeline.get('stages',{}).get('point_review',{})
    detail=detail if isinstance(detail,dict) else {}
    status=detail.get('status','NOT_RUN');links=[]
    verified={'status':status,'available':False,'geometry_acceptance':False}
    try:
        if status in ('AUTO_POINT_REVIEW_COMPLETE','INSUFFICIENT_DATA','FAILED') and detail.get('record'):
            output=(run/'point_review').resolve()
            for role,name in [('record','RESULT.json'),('report','index.html')]:
                ref=detail[role];path=Path(ref['path']).resolve(strict=True)
                if not _within(output,run) or path!=output/name or path.stat().st_size!=ref.get('bytes') or _sha(path)!=ref.get('sha256'):
                    raise ValueError('自动复核文件绑定不匹配')
                links.append(f'<a class="button secondary" href="{_esc(_uri(path,report_dir))}">{_esc(name)}</a>')
            if _read(output/'RESULT.json').get('geometry_accepted') is not False:raise ValueError('点复核验收契约不匹配')
            if _read(output/'RESULT.json').get('status')!=status:raise ValueError('自动复核状态不匹配')
            verified['available']=True
    except (OSError,ValueError,KeyError,TypeError,AttributeError) as exc:
        warnings.append(str(exc));verified['verification_error']=str(exc);links=[]
    html=('<section class="cloud-section" id="point-review"><h2>自动时序点与候选复核（非建模验收）</h2><p>'
          +_esc(status)+'</p><p>复核多视图对应与外观证据；不改变表面/CAD验收，不宣称精度或对应真值。</p>'
          +'<p>'+_esc(detail.get('error') or detail.get('reason',''))+'</p><div class="cloud-actions">'+''.join(links)+'</div></section>')
    return html,verified

def _research_outputs(run: Path, report_dir: Path, metadata: dict, warnings: list[str]) -> tuple[str, dict]:
    record = metadata.get("research", {})
    record = record if isinstance(record, dict) else {}
    status = str(record.get("status", "NOT_RUN"))
    validation = {"status": status, "available": False, "artifacts": []}
    actions = []
    payload = {}
    try:
        ref = record.get("result", {})
        path = _path(ref, run)
        research_root = (run / "research").resolve()
        if status == "SUCCESS":
            if (not research_root.is_relative_to(run) or not isinstance(ref, dict) or path != research_root / "RESULT.json"
                    or not path.is_file() or _sha(path) != ref.get("sha256")):
                raise ValueError("研究 RESULT 身份校验未通过")
            payload = _read(path)
            if (payload.get("status") != "SUCCESS" or payload.get("complete") is not True
                    or str(payload.get("source_model_id")) != str(record.get("selected_model_id"))
                    or payload.get("algorithm_called") != "kozlov_lab.stereo.official.indexed_code3d.indexed_body_code"):
                raise ValueError("研究状态、选定模型或实际调用记录不匹配")
            refs = payload.get("artifacts")
            if not isinstance(refs, list) or not refs or refs != record.get("artifacts"):
                raise ValueError("研究产物清单不匹配")
            # Gate all links together; a stale point file invalidates the code card.
            for artifact in refs:
                artifact_path = _path(artifact, run)
                if (artifact_path is None or not _within(artifact_path, research_root)
                        or not artifact_path.is_file() or _sha(artifact_path) != artifact.get("sha256")):
                    raise ValueError("研究产物路径或 SHA-256 校验未通过")
            actions.append(f'<a class="button secondary" href="{_esc(_uri(path, report_dir))}" download>下载研究运行记录</a>')
            for artifact in refs:
                artifact_path = _path(artifact, run)
                actions.append(f'<a class="button secondary" href="{_esc(_uri(artifact_path, report_dir))}" download>{_esc(artifact_path.name)}</a>')
            validation.update(available=True, artifacts=refs, result=ref, native_point_ids=payload.get("native_point_ids"), counts=payload.get("counts"), controls=payload.get('controls'))
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        warnings.append(str(exc))
        validation["verification_error"] = str(exc)
    note = record.get("error") or record.get("reason") or "固定六点的完整索引体积比，精确针对已存储坐标；不表示整物体，不证明重建收益。"
    controls_html = ''
    if validation.get('available') and isinstance(payload.get('controls'), dict):
        controls = payload['controls']
        controls_html = ('<p>六点有理仿射正对照 / 非等价负对照：' + _esc(controls.get('status', 'UNKNOWN'))
                         + '；正对照 ' + _esc(controls.get('positive_supplied_mapping'))
                         + '；负对照 ' + _esc(controls.get('negative_existential'))
                         + '；独立判定器检查双射数 ' + _esc(controls.get('independent_permutations_checked'))
                         + '。这项代数对照不代表真实照片精度或重建质量收益。</p>')
    content = f'<section class="cloud-section" id="research"><div class="cloud-header"><h2>Kozlov 研究表示：固定六点子集</h2><span class="pill {"success" if validation["available"] else "warning"}">{_esc(status)}</span></div><p>{_esc(note)}</p><p>论文：1996 §2，PDF p.6–7 / 期刊325–326。K(A)=(M_A,T_A)，不同无序四元组的非零分母体积比。</p><p>point_id：{_esc(payload.get("native_point_ids", []))}；计数：{_esc(payload.get("counts", {}))}；源 SfM：{_esc(payload.get("source_sfm_status", record.get("source_sfm_status", "未核实")))}</p><p>未执行 AlgT′/AlgS0；不表示2025完整理论、真实尺度或物理精度。文件 hash 不是重编号不变量。</p><div class="cloud-actions">{"".join(actions)}</div></section>'
    return content + controls_html, validation


CSS = r"""
:root{--ink:#213e3a;--muted:#687b73;--teal:#1e7163;--mint:#c5e7d4;--paper:#f4f3ec;--white:#fffefa;--line:#dce1d6;--gold:#a47332;--warn:#8b6833;font-family:"Segoe UI","Microsoft YaHei","PingFang SC",sans-serif;color:var(--ink);background:var(--paper)}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0}a{color:inherit}button{font:inherit}button,a{-webkit-tap-highlight-color:transparent}button:focus-visible,a:focus-visible{outline:3px solid #dc9f44;outline-offset:4px}button{cursor:pointer}header,.shell{width:min(1240px,calc(100% - 80px));margin:auto}header{display:flex;justify-content:space-between;align-items:center;height:94px;border-bottom:1px solid var(--line)}.brand{display:flex;align-items:center;gap:12px;font-weight:700;font-size:14px;letter-spacing:1.3px}.brand-mark{display:grid;grid-template-columns:repeat(2,7px);gap:3px;transform:rotate(-10deg)}.brand-mark i{width:7px;height:7px;background:var(--teal);border-radius:1px}.brand-mark i:nth-child(2){opacity:.4}.brand-sub{font-size:10px;color:var(--muted);display:block;letter-spacing:2px;font-weight:500;margin-top:4px}.header-right{display:flex;gap:28px;font-size:12px;color:var(--muted)}.header-right a{text-decoration:none}.hero{display:grid;grid-template-columns:1.13fr 1fr;gap:64px;align-items:center;padding:64px 0 50px}.eyebrow{font-size:11px;font-weight:700;letter-spacing:2px;color:var(--teal);margin:0 0 22px}.hero h1{font-size:clamp(35px,4.25vw,60px);line-height:1.28;letter-spacing:-1.8px;margin:0 0 22px;font-weight:650;max-width:680px}.hero-copy{font-size:15px;line-height:1.95;color:var(--muted);max-width:515px;margin:0 0 30px}.hero-status{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:30px}.pill{display:inline-flex;align-items:center;gap:7px;padding:7px 11px;border-radius:100px;font-size:11px;font-weight:600;line-height:1.45;background:#e9ede5;color:var(--muted)}.pill:before{content:"";width:5px;height:5px;border-radius:50%;background:currentColor;flex:none}.pill.success{background:#e0efe3;color:var(--teal)}.pill.warning{background:#f2e9d7;color:var(--warn)}.button{display:inline-flex;align-items:center;justify-content:center;gap:24px;text-decoration:none;padding:13px 20px;background:var(--teal);color:white;font-size:12px;font-weight:600;border:1px solid var(--teal);border-radius:7px}.button.secondary{background:transparent;color:var(--teal);border-color:#cedacf}.hero-visual{position:relative;display:flex;align-items:center;justify-content:center;min-height:416px;background:#e6e9df;border-radius:24px;padding:20px;overflow:hidden}.hero-visual:before{position:absolute;content:"";width:330px;height:330px;border:1px solid #bacbbf;border-radius:50%;right:-90px;top:26px}.hero-photo{position:relative;width:260px;height:350px;object-fit:cover;border:9px solid #fffefa;border-radius:12px;transform:rotate(6deg);box-shadow:0 16px 36px #27433424}.hero-tag{position:absolute;left:20px;bottom:25px;background:var(--white);box-shadow:0 6px 22px #29453615;border:1px solid #e5e8df;border-radius:10px;padding:12px 16px;font-size:11px;line-height:1.8}.hero-tag strong{display:block;font-size:13px}.hero-index{position:absolute;right:21px;top:19px;font-size:10px;letter-spacing:2px;writing-mode:vertical-rl;color:#788f7f}.summary{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-bottom:70px}.summary-card{background:var(--white);padding:24px 27px;border:1px solid var(--line);border-radius:12px;position:relative}.summary-label{font-size:12px;color:var(--muted)}.summary-value{font-size:35px;line-height:1.4;margin-top:10px;letter-spacing:-1px}.summary-value small{font-size:14px;letter-spacing:0;color:var(--muted);padding-left:5px}.summary-value.text{font-size:24px;padding-top:10px;padding-bottom:4px}.summary-note{font-size:11px;line-height:1.7;color:var(--muted);margin:9px 0 0}.summary-order{position:absolute;right:24px;top:24px;font-size:10px;color:#acb9af;font-weight:600}.section-head{display:flex;justify-content:space-between;gap:24px;align-items:flex-end;margin-bottom:24px}.section-head .eyebrow,.cloud-header .eyebrow{margin-bottom:12px}h2{font-size:29px;letter-spacing:-.8px;font-weight:600;margin:0}h3{font-weight:600}.section-intro{font-size:13px;line-height:1.9;color:var(--muted);max-width:800px;margin:16px 0 24px}.section-caption{font-size:11px;line-height:1.7;color:var(--muted);max-width:390px}.gallery{background:var(--white);border:1px solid var(--line);border-radius:17px;overflow:hidden}.gallery-bar{display:flex;justify-content:space-between;align-items:center;gap:16px;padding:18px 24px;border-bottom:1px solid var(--line)}.tabs{display:flex;gap:4px;background:#edf0e9;border-radius:7px;padding:4px}.tab{border:0;border-radius:5px;padding:9px 15px;font-size:11px;color:var(--muted);background:transparent}.tab.active{background:var(--white);box-shadow:0 1px 4px #213e3a12;color:var(--teal);font-weight:650}.tab:disabled{opacity:.35;cursor:not-allowed}.gallery-number{font-size:11px;letter-spacing:1px;color:var(--muted)}.gallery-main{display:grid;grid-template-columns:1fr 280px}.image-stage{position:relative;min-height:550px;background:#e9ece4;display:flex;align-items:center;justify-content:center;padding:20px;background-image:radial-gradient(#97ab9824 .8px,transparent .8px);background-size:13px 13px}.image-stage img{width:100%;height:530px;object-fit:contain;border-radius:4px}.image-stage.mask-view{background:#dfe5dc}.image-caption{position:absolute;bottom:29px;left:30px;max-width:calc(100% - 60px);font-size:10px;background:#fffefae8;backdrop-filter:blur(6px);padding:7px 10px;border-radius:5px;color:var(--muted)}.frame-info{padding:28px 24px;border-left:1px solid var(--line)}.frame-info .eyebrow{font-size:10px;margin-bottom:18px}.frame-info h3{font-size:22px;margin:0 0 11px}.filename{font-size:11px;line-height:1.8;word-break:break-all;color:var(--muted);margin:17px 0 24px}.fact{padding:17px 0;border-top:1px solid var(--line)}.fact span{font-size:10px;color:var(--muted);display:block;margin-bottom:7px}.fact strong{font-size:13px;font-weight:550}.frame-note{font-size:12px;line-height:1.9;color:var(--muted);padding-top:8px}.legend{font-size:11px;line-height:1.85;color:var(--muted);margin-top:25px}.legend i{display:inline-block;width:8px;height:8px;background:#42b79a;border-radius:2px;margin-right:6px}.thumbnails{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;padding:20px 24px;border-top:1px solid var(--line)}.thumb{display:flex;gap:12px;align-items:center;background:transparent;border:1px solid var(--line);padding:7px;text-align:left;border-radius:9px;min-width:0;color:var(--ink)}.thumb.active{border-color:var(--teal);background:#edf4eb;box-shadow:0 0 0 1px var(--teal)}.thumb img{width:43px;height:58px;object-fit:cover;border-radius:4px}.thumb b{font-size:11px;font-weight:600;display:block;margin-bottom:6px}.thumb span{font-size:9px;color:var(--muted)}.review-note{display:flex;align-items:flex-start;gap:13px;margin:17px 0 68px;padding:16px 19px;border:1px solid #e2ddcc;border-radius:8px;font-size:12px;line-height:1.9;color:#756b50;background:#f5f0e3}.review-note .info-icon{flex:none;display:flex;justify-content:center;align-items:center;border:1px solid #b5a988;border-radius:50%;width:16px;height:16px;font-size:10px;margin-top:3px}.cloud-section{margin-bottom:66px}.cloud-header{display:flex;justify-content:space-between;align-items:flex-end;gap:25px}.cloud-empty{background:#e9ede4;border:1px solid #d9dfd1;border-radius:16px;padding:39px 30px 0;text-align:center;overflow:hidden}.empty-symbol{margin:auto auto 16px;display:grid;place-items:center;width:52px;height:52px;border:1px solid #c2ccbc;border-radius:15px;font-size:28px;font-weight:300;color:#7f927d}.cloud-empty h3{font-size:21px;margin:0 0 10px}.cloud-empty p{font-size:13px;color:var(--muted);line-height:1.9;max-width:590px;margin:0 auto 30px}.empty-stats{border-top:1px solid #d5ddcf;display:grid;grid-template-columns:repeat(3,1fr);max-width:780px;margin:auto;padding:24px 0 25px}.empty-stats>div+div{border-left:1px solid #d5ddcf}.empty-stats strong{display:block;font-size:24px;font-weight:500;line-height:1.7}.empty-stats span{font-size:11px;color:var(--muted)}.cloud-viewer{border-radius:14px;overflow:hidden;border:1px solid var(--line);background:#15221f}.cloud-viewer iframe{width:100%;height:560px;border:0;display:block}.cloud-actions{display:flex;flex-wrap:wrap;gap:12px;margin:18px 0}.clean-list{font-size:12px;line-height:1.9;color:var(--muted);padding-left:18px;margin-top:20px}.clean-list li{margin:6px 0}.next-section{background:#24493f;color:#fffefa;border-radius:17px;padding:34px 36px;margin-bottom:64px;display:grid;grid-template-columns:.78fr 1.35fr;gap:48px}.next-section .eyebrow{color:#a5cbb5;margin-bottom:14px}.next-section h2{font-size:25px}.next-section p{font-size:12px;color:#bad0c2;line-height:1.9;margin-top:16px}.next-steps{display:grid;grid-template-columns:repeat(2,1fr);gap:24px 27px}.next-step{display:flex;gap:11px}.next-step span{font-size:11px;color:#a5cbb5;padding-top:3px}.next-step b{font-size:12px;font-weight:600}.next-step p{margin:5px 0 0;font-size:11px;line-height:1.8}.notes{display:grid;grid-template-columns:1fr 1fr;gap:40px;margin-bottom:32px}.notes h3{font-size:15px;margin:0 0 14px}.notes p,.notes li{font-size:12px;color:var(--muted);line-height:1.95}.notes ul{padding-left:17px;margin:0}.evidence{border-top:1px solid var(--line);border-bottom:1px solid var(--line);margin-bottom:42px}.evidence summary{cursor:pointer;font-size:12px;padding:20px 0;list-style:none;display:flex;justify-content:space-between;align-items:center}.evidence summary:after{content:"＋";color:var(--teal);font-size:17px}.evidence[open] summary:after{content:"−"}.evidence-content{padding:0 0 22px;font-size:11px;line-height:1.9;color:var(--muted)}.evidence-links{display:flex;flex-wrap:wrap;gap:9px;margin:14px 0}.evidence-links a{display:inline-block;border:1px solid var(--line);border-radius:5px;padding:6px 10px;text-decoration:none}.evidence-content code{font-family:Consolas,monospace;font-size:10px;overflow-wrap:anywhere}.audit-table{width:100%;border-collapse:collapse;font-size:10px}.audit-table td,.audit-table th{border-top:1px solid var(--line);padding:10px;text-align:left;overflow-wrap:anywhere;vertical-align:top}.audit-table th{font-weight:600}.audit-table td:first-child{width:28%}.audit-table code{font-size:9px}.warning-list{background:#f3e8d6;padding:13px 15px;border-radius:6px;margin-top:15px;color:var(--warn)}footer{display:flex;justify-content:space-between;gap:18px;color:#839185;font-size:10px;line-height:1.8;padding-bottom:36px}.source-path{word-break:break-all}.empty-gallery{padding:55px 20px;text-align:center;font-size:13px;color:var(--muted)}[hidden]{display:none!important}
@media(max-width:980px){header,.shell{width:calc(100% - 48px)}.hero{gap:30px}.hero-visual{min-height:375px}.hero-photo{height:310px;width:235px}.gallery-main{grid-template-columns:1fr 235px}.frame-info{padding:25px 20px}.thumbnails{gap:8px}.thumb{gap:8px}.thumb span{font-size:8px}.next-section{gap:30px}.image-stage{min-height:500px}.image-stage img{height:480px}}
@media(max-width:700px){header,.shell{width:calc(100% - 32px)}header{height:77px}.header-right{gap:12px;font-size:10px}.header-right .date{display:none}.hero{grid-template-columns:1fr;padding:40px 0 28px;gap:30px}.hero h1{font-size:39px;letter-spacing:-1.2px;max-width:520px}.hero-copy{font-size:13px;margin-bottom:20px}.hero-status{margin-bottom:24px}.hero-visual{min-height:328px}.hero-photo{height:280px;width:212px}.summary{gap:8px;margin-bottom:47px}.summary-card{padding:16px 13px}.summary-label{font-size:10px}.summary-order{display:none}.summary-value{font-size:26px}.summary-value small{font-size:11px}.summary-value.text{font-size:16px;padding-top:12px}.summary-note{font-size:9px;line-height:1.8}.section-head{display:block;margin-bottom:20px}.section-caption{margin-top:12px}h2{font-size:25px}.eyebrow{font-size:10px}.gallery-bar{padding:13px 12px;gap:8px}.tabs{gap:0}.tab{padding:9px 11px;font-size:10px}.gallery-number{font-size:9px}.gallery-main{grid-template-columns:1fr}.image-stage{min-height:415px;padding:15px}.image-stage img{height:415px}.image-caption{bottom:21px;left:21px;font-size:9px}.frame-info{border-left:0;border-top:1px solid var(--line);padding:22px;display:grid;grid-template-columns:1fr 1fr;gap:0 16px}.frame-info .eyebrow{grid-column:1/-1;margin-bottom:10px}.frame-info h3{font-size:19px;align-self:center;margin:0}.frame-info>.pill{justify-self:end;align-self:center;font-size:9px}.filename{grid-column:1/-1;margin:10px 0 16px;font-size:10px}.fact{padding:13px 0}.frame-note,.legend{grid-column:1/-1;margin:0;padding:8px 0 0;font-size:11px}.legend{padding-top:13px}.thumbnails{grid-template-columns:repeat(2,1fr);padding:16px;gap:9px}.thumb{gap:10px}.thumb img{width:35px;height:46px}.thumb b{font-size:10px;margin-bottom:5px}.thumb span{font-size:9px}.review-note{font-size:11px;margin-bottom:45px}.cloud-header{align-items:center}.cloud-header .pill{font-size:10px}.section-intro{font-size:12px}.cloud-empty{padding:30px 15px 0}.cloud-empty h3{font-size:18px}.cloud-empty p{font-size:12px}.empty-stats{padding:19px 0}.empty-stats strong{font-size:21px}.empty-stats span{font-size:10px}.cloud-viewer iframe{height:430px}.cloud-section{margin-bottom:45px}.next-section{grid-template-columns:1fr;gap:16px;padding:27px 24px;margin-bottom:40px}.next-steps{gap:20px}.notes{grid-template-columns:1fr;gap:27px}.evidence-content{overflow-x:auto}.audit-table{min-width:580px}footer{display:block;font-size:9px}footer>div+div{margin-top:5px}}
@media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}}
.pipeline-alert{border:1px solid #c7a785;border-left:4px solid #9b5f37;background:#f4e8db;border-radius:10px;padding:23px 27px;margin:0 0 24px;color:#704a30}.pipeline-alert h2{font-size:19px;letter-spacing:0;margin:0 0 9px}.pipeline-alert p,.pipeline-alert li{font-size:12px;line-height:1.9;overflow-wrap:anywhere}.pipeline-alert p{margin:0 0 9px}.pipeline-alert ul{margin:0;padding-left:18px}.stage-strip{display:flex;gap:12px;flex-wrap:wrap;margin:-44px 0 52px;font-size:11px;color:var(--muted)}.stage-strip span{padding:7px 10px;background:#e9ede5;border-radius:5px}.stage-strip .stage-failed{color:#884d2c;background:#f2e5d6}@media(max-width:700px){.pipeline-alert{padding:19px;margin-bottom:20px}.pipeline-alert h2{font-size:17px}.stage-strip{margin:-27px 0 35px;gap:7px;font-size:10px}}
"""


JS = r"""
const frames=JSON.parse(document.getElementById('frame-data').textContent);
let selected=0,mode='overlay';
function pickFrame(index){selected=index;render();}
function pickMode(value){mode=value;render();}
function render(){
 const frame=frames[selected]; if(!frame)return;
 if(!frame[mode])mode='original';
 const picture=document.getElementById('main-image');picture.src=frame[mode]||'';
 picture.alt=`照片 ${frame.number} · ${frame.filename} · ${{original:'原图（展示缩放）',overlay:'目标分割叠图',mask:'二值掩膜'}[mode]}`;
 picture.hidden=!frame[mode];document.getElementById('missing-image').hidden=!!frame[mode];
 document.getElementById('image-stage').classList.toggle('mask-view',mode==='mask');
 document.getElementById('frame-title').textContent=`照片 ${String(frame.number).padStart(2,'0')}`;
 document.getElementById('gallery-number').textContent=`${String(frame.number).padStart(2,'0')} / ${String(frames.length).padStart(2,'0')}`;
 document.getElementById('filename').textContent=frame.filename;
 document.getElementById('frame-state').textContent=frame.state;
 document.getElementById('frame-state').className='pill '+frame.state_kind;
 document.getElementById('dimension').textContent=frame.dimensions?frame.dimensions.join(' × ')+' px':'—';
 document.getElementById('coverage').textContent=frame.coverage===null?'—':frame.coverage+'%';
 document.getElementById('frame-note').textContent=frame.note||'此处展示单张照片的分割结果。跨视角一致性与轮廓精度尚未经过定量评估。';
 document.getElementById('image-caption').textContent={original:'原图（展示缩放） · 保留完整画面',overlay:'绿色区域为已提交掩膜的可视化叠加',mask:'白色 = 目标区域 · 黑色 = 背景'}[mode];
 document.querySelectorAll('.tab').forEach(button=>{const active=button.dataset.mode===mode;button.classList.toggle('active',active);button.setAttribute('aria-pressed',active);button.disabled=!frame[button.dataset.mode];});
 document.querySelectorAll('.thumb').forEach((button,index)=>{const active=index===selected;button.classList.toggle('active',active);button.setAttribute('aria-pressed',active);});
}
document.querySelectorAll('.tab').forEach(button=>button.addEventListener('click',()=>pickMode(button.dataset.mode)));
document.querySelectorAll('.thumb').forEach((button,index)=>button.addEventListener('click',()=>pickFrame(index)));
render();
"""


def render_report(run_dir: Path) -> Path:
    """Render an offline report exclusively from this run's declared artifacts."""
    run = Path(run_dir).resolve()
    if not run.is_dir():
        raise FileNotFoundError(run)
    report_dir = run / "report"
    report_dir.mkdir(exist_ok=True)
    pipeline = _read(run / "pipeline.json")
    from integrations.demo_pipeline.quality import read_for_display, display_message
    object_quality = read_for_display(run, pipeline)
    quality_html = ('<section class="pipeline-alert" aria-label="物体外形验收"><h2>文件目标与物体目标</h2><p>'
                    + _esc(display_message(object_quality)) + '</p><p>文件阶段的已完成状态仅表示产物执行与格式验证通过。</p></section>')
    inherited = pipeline.get("report_data", pipeline)
    metadata = dict(inherited) if isinstance(inherited, dict) else {}
    metadata.update(_read(run / "REPORT_DATA.json"))
    prepared = _read(run / "input" / "RESULT.json")
    inventory = _read(run / "INPUT_INVENTORY.json")
    stage_context, pipeline_errors = _pipeline_context(pipeline, metadata, prepared)
    warnings: list[str] = []
    rows, segmentation, manifest_path = _load_frames(run, prepared, inventory, metadata, warnings)
    count = len(rows)
    valid_masks = sum(row["mask_verified"] for row in rows)
    valid_images = sum(bool(row.get("original")) for row in rows)
    title = metadata.get("title") or f"{count}张照片的物体分割演示"
    object_label = metadata.get("object_label") or "目标物体"
    source = metadata.get("source") or prepared.get("source_directory") or inventory.get("source") or "本次运行的照片输入"
    if isinstance(source, dict):
        source = source.get("path", source.get("label", "本次运行的照片输入"))
    cloud_html, cloud_result, extra_evidence = _pointcloud(run, report_dir, metadata, stage_context, warnings)
    cloud_status = cloud_result["status"]
    geometry_html, surface_result, blender_result = _geometry_outputs(run, report_dir, metadata, stage_context, warnings)
    geometry_html = quality_html + geometry_html
    view_geometry_html, view_geometry_result = _view_geometry_outputs(run, report_dir, metadata, pipeline, warnings)
    research_html, research_result = _research_outputs(run, report_dir, metadata, warnings)
    point_review_html, point_review_result = _point_review_outputs(run, report_dir, metadata, pipeline, warnings)
    post_review_html, post_review_result = _post_review_outputs(run, report_dir, metadata, pipeline, warnings)
    cad_html, cad_result = _cad_outputs(run, report_dir, pipeline, warnings)
    seg_metadata = metadata.get("segmentation") or {}
    seg_status = segmentation["outcome"].get("status") or (seg_metadata.get("status") if isinstance(seg_metadata, dict) else str(seg_metadata)) or _stage_status(stage_context, "segmentation")
    seg_label = f"分割输出 {valid_masks}/{count}" if count else _status_label(str(seg_status))
    cloud_label = "三维重建未成功" if cloud_status == "NO_MODEL" else f"三维重建 · {_status_label(cloud_status)}"
    default_copy = "从真实照片出发，逐张查看目标区域与二值掩膜。三维部分单独呈现本次运行的实际结果。"
    hero_image = next((row["original"] for row in rows if row.get("original")), None)
    hero_visual = f'<img class="hero-photo" src="{hero_image}" alt="{_esc(object_label)}的输入照片">' if hero_image else '<div class="hero-photo" aria-label="暂无可展示照片"></div>'
    hero_visual += f'<div class="hero-tag"><strong>{_esc(object_label)}</strong>真实输入 · {count} 张照片</div><span class="hero-index">PHOTO STUDY / {_esc(run.name)}</span>'
    thumbs = []
    for row in rows:
        thumb_image = f'<img src="{row.get("thumbnail")}" alt="">' if row.get("thumbnail") else ''
        label = "已有掩膜" if row["mask_verified"] else row["state"]
        thumbs.append(f'<button class="thumb" type="button" aria-label="查看照片 {row["number"]}：{_esc(row["filename"])}" aria-pressed="false">{thumb_image}<div><b>照片 {row["number"]:02d}</b><span>{_esc(label)}</span></div></button>')
    gallery = f'''<div class="gallery"><div class="gallery-bar"><div class="tabs" role="group" aria-label="照片显示方式"><button class="tab" data-mode="original" type="button">原图</button><button class="tab active" data-mode="overlay" type="button">分割叠图</button><button class="tab" data-mode="mask" type="button">二值掩膜</button></div><span class="gallery-number" id="gallery-number"></span></div><div class="gallery-main"><div class="image-stage" id="image-stage"><img id="main-image" alt=""/><p id="missing-image" hidden>暂无通过校验的照片</p><span class="image-caption" id="image-caption"></span></div><aside class="frame-info"><p class="eyebrow">当前照片</p><h3 id="frame-title"></h3><span id="frame-state" class="pill"></span><p class="filename" id="filename"></p><div class="fact"><span>展示照片尺寸</span><strong id="dimension"></strong></div><div class="fact"><span>目标占画面面积</span><strong id="coverage"></strong></div><p class="frame-note" id="frame-note"></p><div class="legend"><i></i>绿色叠图仅辅助查看。<br>二值掩膜保留实际输出，未做轮廓修补。</div></aside></div><div class="thumbnails">{''.join(thumbs)}</div></div>''' if rows else '<div class="gallery empty-gallery">暂无可展示的已登记输入照片。请查看处理记录中的输入阶段状态。</div>'
    visual_review = metadata.get("visual_review") or "成功输出掩膜不等于轮廓完全正确。当前未完成分割精度与跨视角一致性的定量评估。"
    mask_used = metadata.get("sfm_masks_applied") is True
    if mask_used:
        mask_statement = "先分割目标，再在掩膜内提取特征；原生数据库中的特征位置已按掩膜回查。此约束不证明分割或三维几何正确。"
    elif metadata.get("sfm_masks_requested") is True:
        mask_statement = "本次要求先分割再选点，尚无已验证的原生掩膜消费记录；未自动退回整图选点。"
    else:
        mask_statement = "目标分割与三维重建分别运行。本次分割掩膜未用于 SfM 特征筛选，重建面向完整照片中的场景。"
    input_stage_status = _stage_status(stage_context, "input")
    preparation_status = input_stage_status if input_stage_status != "PENDING" else prepared.get("status", "PENDING")
    processing = [f"输入阶段：{_status_label(str(preparation_status))}。页面中可展示 {valid_images} 张已登记图像。",
                  f"目标分割阶段：{_status_label(str(seg_status))}。已校验并展示 {valid_masks} 张目标掩膜。", mask_statement]
    if prepared.get("error"):
        warnings.append("输入处理记录：" + str(prepared["error"]))
    alerts_html = ''
    if pipeline_errors:
        alerts_html = '<section class="pipeline-alert" role="alert" aria-label="运行问题"><h2>本次运行有问题需要检查</h2><p>以下问题会影响对本次运行的判断。请结合错误记录理解已生成的结果。</p><ul>' + ''.join(f'<li>{_esc(error)}</li>' for error in pipeline_errors) + '</ul></section>'
    stage_badges = []
    for name, label in (("input", "输入"), ("segmentation", "分割"), ("sfm", "重建")):
        status = str(preparation_status) if name == "input" else _stage_status(stage_context, name)
        if status == "PENDING":
            status = str(seg_status) if name == "segmentation" else cloud_status
        failed = status not in {"SUCCESS", "OK", "INPUTS_PREPARED", "NOT_RUN", "PENDING", "RUNNING"}
        stage_badges.append(f'<span class="{"stage-failed" if failed else ""}">{label}阶段：{_esc(_status_label(status))}（{_esc(status)}）</span>')
    stage_html = '<div class="stage-strip" aria-label="处理阶段状态">' + ''.join(stage_badges) + '</div>'
    extra_processing = _strings(metadata.get("processing_notes"))
    limits = ["分割输出尚未经过人工标注对照，不能据此给出准确率。", "稀疏点云即使生成，也不等于完整物体表面、可用 CAD 或尺寸测量结果。", "本次没有可靠的物理尺度，不能从页面读取真实长度、直径或体积。"]
    limits.extend(_strings(metadata.get("limitations")))
    next_steps = metadata.get("capture_next_steps") or [
        {"title": "连续环绕拍摄", "body": "保持物体和背景静止，移动相机；相邻照片保留充分重叠，避免视角突然跳变。"},
        {"title": "补齐不同高度", "body": "增加平视和斜俯视照片，让顶面、侧面与底部边缘都出现在连续视图中。"},
        {"title": "保持清晰与柔和光线", "body": "使用均匀、漫射的光照，避免运动模糊与强反光，减少高光随视角变化。"},
        {"title": "为测量补充已知尺度", "body": "如果后续需要尺寸，拍摄清晰可见、尺寸已知的标尺，并单独验证测量精度。"},
    ]
    steps_html = ''.join(f'<div class="next-step"><span>{index + 1:02d}</span><div><b>{_esc(item.get("title", ""))}</b><p>{_esc(item.get("body", ""))}</p></div></div>' for index, item in enumerate(next_steps) if isinstance(item, dict))
    next_html = f'<section class="next-section"><div><p class="eyebrow">下一步 · 改善输入</p><h2>让下一次拍摄<br>更有利于重建</h2><p>这些是一般拍摄建议，并不保证重建成功。先补充连续、清晰且有重叠的视角，再检查实际结果。</p></div><div class="next-steps">{steps_html}</div></section>'
    evidence_paths = [("输入文件登记", run / "INPUT_INVENTORY.json"), ("输入处理结果", run / "input" / "RESULT.json"),
                      ("分割运行结果", run / "segmentation" / "PRODUCT_OUTCOME.json"), ("分割正式清单", manifest_path),
                      ("演示运行记录", run / "pipeline.json"), ("报告数据", run / "REPORT_DATA.json")]
    evidence_paths.extend(("重建证据 " + str(idx + 1), path) for idx, path in enumerate(extra_evidence))
    evidence_links = ''.join(f'<a href="{_esc(_uri(path, report_dir))}">{_esc(label)} ↗</a>' for label, path in evidence_paths if path and path.is_file())
    processing_details = '<ul>' + ''.join(f'<li>{_esc(item)}</li>' for item in extra_processing) + '</ul>' if extra_processing else ''
    audit_rows = ''.join(f'<tr><td>{_esc(row["filename"])}</td><td>{_esc(row.get("committed_status", "NOT_RUN"))}<br>{_esc(row["state"])}</td><td><code>{_esc(row.get("mask_sha256") or "—")}</code></td></tr>' for row in rows)
    warning_html = '<div class="warning-list">' + '<br>'.join(_esc(item) for item in warnings) + '</div>' if warnings else ''
    now = datetime.now().astimezone()
    display_date = now.strftime("%Y.%m.%d")
    payload = [{key: value for key, value in row.items() if key not in ("source_path", "image_path", "source_sha256", "mask_path", "mask_sha256", "thumbnail") } for row in rows]
    frame_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    photo_note = "原图（展示缩放）与掩膜保持同一画面坐标。网页照片副本经过 JPEG 压缩，原文件与正式掩膜不受影响。"
    if valid_masks == count and count:
        mask_note = "逐张掩膜已校验；轮廓质量仍需评估"
    else:
        mask_note = "仅展示已提交且通过文件校验的掩膜"
    cloud_card_note = "本次没有可展示的三维几何" if not cloud_result["viewer_available"] else "场景稀疏点云；不代表完整物体"
    page = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="color-scheme" content="light"><title>{_esc(title)} · 视觉处理报告</title><style>{CSS}</style></head><body>
<header><div class="brand"><span class="brand-mark" aria-hidden="true"><i></i><i></i><i></i><i></i></span><div>KOZLOV VISUAL LAB<span class="brand-sub">视觉处理演示</span></div></div><nav class="header-right" aria-label="页面导航"><a href="#segmentation">照片与分割</a><a href="#reconstruction">三维结果</a><span class="date">{display_date}</span></nav></header>
<main class="shell"><section class="hero"><div><p class="eyebrow">REAL PHOTOS · LOCAL DEMO</p><h1>{_esc(title)}</h1><p class="hero-copy">{_esc(metadata.get("description") or default_copy)}</p><div class="hero-status"><span class="pill {"success" if valid_masks else "warning"}">{_esc(seg_label)}</span><span class="pill {"success" if cloud_result["viewer_available"] else "warning"}">{_esc(cloud_label)}</span></div><a class="button" href="#segmentation">查看照片与分割 <span>↓</span></a></div><div class="hero-visual">{hero_visual}</div></section>
{alerts_html}<section class="summary" aria-label="本次演示概览"><div class="summary-card"><span class="summary-order">01</span><div class="summary-label">真实输入照片</div><div class="summary-value">{count}<small>张</small></div><p class="summary-note">{valid_images} 张可展示 · 保留原始文件名</p></div><div class="summary-card"><span class="summary-order">02</span><div class="summary-label">目标分割输出</div><div class="summary-value">{valid_masks}<small>/ {count} 张</small></div><p class="summary-note">{mask_note}</p></div><div class="summary-card"><span class="summary-order">03</span><div class="summary-label">稀疏重建结果</div><div class="summary-value text">{_esc(_status_label(cloud_status))}</div><p class="summary-note">{cloud_card_note}</p></div></section>{stage_html}
<section id="segmentation"><div class="section-head"><div><p class="eyebrow">01 · 照片与分割</p><h2>逐张查看，切换结果</h2></div><div class="section-caption">选择下方照片，再切换原图、绿色叠图或黑白掩膜。<br>目标区域来自本次实际输出。</div></div>{gallery}<div class="review-note"><span class="info-icon" aria-hidden="true">i</span><div>{_esc(visual_review)}</div></div></section>
<section id="reconstruction" class="cloud-section">{cloud_html}</section>{geometry_html}{cad_html}{view_geometry_html}{research_html}{point_review_html}{post_review_html}{next_html}
<section class="notes"><div><h3>这次做了什么</h3><ul>{''.join(f'<li>{_esc(item)}</li>' for item in processing)}</ul></div><div><h3>如何理解这些结果</h3><ul>{''.join(f'<li>{_esc(item)}</li>' for item in limits)}</ul></div></section>
<details class="evidence"><summary>查看来源与处理记录</summary><div class="evidence-content"><p>照片来源：<span class="source-path">{_esc(source)}</span><br>报告生成时间：{_esc(now.isoformat(timespec="seconds"))}<br>演示目录：<code>{_esc(run.name)}</code></p><p>{photo_note} 分割叠图由标准化照片与正式清单中选定的二值掩膜生成，仅作显示。只有状态为 OK、图像身份匹配、掩膜哈希匹配且尺寸一致的记录才计入分割输出，目录中的孤立文件不计入结果。</p>{processing_details}<p>文件链接用于在完整结果目录中核查；报告中的照片已嵌入页面，可离线查看。</p><div class="evidence-links">{evidence_links}</div><table class="audit-table"><thead><tr><th>原始文件名</th><th>正式状态 / 展示状态</th><th>选定掩膜 SHA-256</th></tr></thead><tbody>{audit_rows}</tbody></table>{warning_html}</div></details>
<footer><div>KOZLOV VISUAL LAB · 真实输入的处理记录</div><div>本页展示视觉处理结果，不提供 CAD、尺寸或精度保证。</div></footer></main><script id="frame-data" type="application/json">{frame_json}</script><script>{JS}</script></body></html>'''
    output = report_dir / "index.html"
    output.write_text(page, encoding="utf-8")
    parsed = _Links()
    parsed.feed(page)
    broken: list[str] = []
    for _, value in parsed.urls:
        if value.startswith(("data:", "#", "blob:")):
            continue
        url = urlsplit(value)
        if url.scheme:
            continue
        if not (report_dir / unquote(url.path)).resolve().is_file():
            broken.append(value)
    validation = {
        "generated_utc": now.astimezone(timezone.utc).isoformat(), "report": str(output),
        "input_frames": count, "display_images": valid_images,
        "verified_committed_masks": valid_masks, "segmentation_status": str(seg_status),
        "input_status": str(preparation_status), "pipeline_errors": pipeline_errors,
        "stage_statuses": {name: _stage_status(stage_context, name) for name in stage_context["stages"]},
        "pointcloud": cloud_result, "surface": surface_result, "blender": blender_result,
        "research": research_result,
        "post_review": post_review_result,
        "point_review": point_review_result,
        "view_geometry": view_geometry_result,
        "cad": cad_result,
        "object_quality": object_quality,
        "broken_links": broken, "warnings": warnings,
        "embedded_image_elements": parsed.data_images,
        "mask_bindings": [{key: row.get(key) for key in ("index", "filename", "committed_status", "mask_verified", "mask_path", "mask_sha256", "foreground_pixels", "coverage", "verification_error")} for row in rows],
        "html_bytes": output.stat().st_size,
    }
    (report_dir / "REPORT_VALIDATION.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if broken:
        raise ValueError("Generated report contains missing local links: " + ", ".join(broken))
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Render an offline photo processing report from a completed or partial demo run.")
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    print(render_report(args.run_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
