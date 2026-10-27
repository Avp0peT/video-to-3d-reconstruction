"""Read-only six-native-point representation; never solves SfM or photographs.

S-1996, PDF pp.6-7 / journal pp.325-326: K(A)=(M_A,T_A), indexed
nonnegative tetrahedron-volume ratios, distinct unordered quadruples and
nonzero denominator. Arithmetic is exact for supplied stored binary64 values.
"""
from __future__ import annotations
from fractions import Fraction
import hashlib
import html
import json
import math
from pathlib import Path
import struct
import time

ROOT = Path(__file__).resolve().parents[2]
MAX_BINARY = 32 * 1024**2
MAX_DESCRIPTION = 64 * 1024**2
LIMITATIONS = [
    "仅表示按原生 point_id 排序的固定六点子集，不是整个物体或完整点云。",
    "精确针对已存储 binary64 坐标，不证明照片恢复精确、真实尺度或物理精度。",
    "论文依据：1996 §2，PDF p.6–7 / 期刊325–326；未运行 AlgT′ / AlgS0。",
    "完整带索引体积比代码不等于2025三类形象/ostov/futlyar，也不证明研究重建收益。",
    "JSON SHA-256 仅绑定编号呈现和字节，不是重编号不变量或类别识别。",
]


def binding(path):
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def read_json(path, maximum):
    path = Path(path)
    if path.stat().st_size > maximum:
        raise ValueError("JSON input exceeds explicit byte budget")
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    return json.loads(path.read_text(encoding="utf-8-sig"), object_pairs_hook=pairs)


def read_native_six(path):
    """COLMAP little-endian points3D.bin, bounded full scan; retain first six IDs.

    Record layout: uint64 id, 3*float64 xyz, 3*uint8 RGB, float64 error,
    uint64 track length, track entries int32 image_id/int32 point2D_idx.
    No pycolmap/native library is imported. Full scan rejects duplicate IDs,
    nonfinite records, malformed track lengths, trailing bytes and truncation.
    """
    path = Path(path)
    if path.stat().st_size > MAX_BINARY:
        raise ValueError("points3D.bin exceeds 32 MiB byte budget")
    data = path.read_bytes()
    def unpack(fmt, offset):
        size = struct.calcsize(fmt)
        if offset + size > len(data):
            raise ValueError("truncated points3D.bin")
        return struct.unpack_from(fmt, data, offset), offset + size
    (count,), offset = unpack("<Q", 0)
    if count > 100000:
        raise ValueError("native point count exceeds 100000 scan budget")
    selected, ids = [], set()
    for _ in range(count):
        record, offset = unpack("<QdddBBBd", offset)
        ident, xyz, error = record[0], record[1:4], record[-1]
        if not 0 < ident < 2**63 or ident in ids or not all(math.isfinite(v) for v in (*xyz, error)):
            raise ValueError("invalid, duplicate or nonfinite native point")
        ids.add(ident)
        (track_length,), offset = unpack("<Q", offset)
        if track_length > (len(data) - offset) // 8:
            raise ValueError("invalid native track length")
        offset += track_length * 8
        selected.append((ident, xyz))
        selected.sort(key=lambda row: row[0])
        del selected[6:]
    if offset != len(data):
        raise ValueError("trailing bytes in points3D.bin")
    return count, selected


def _write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")


def export_native_code6(sfm_run, model_id, output, *, max_entries=10000, controls=True, max_control_permutations=720):
    """API returns RESULT dict, retaining honest rejection/undefined evidence.

    Accepts explicit successful native sfm-native-object-v1 run and model_id;
    no component selection heuristic, reconstruction, retry or coordinate change.
    """
    from kozlov_lab.stereo.official.indexed_code3d import indexed_body_code, CodeBudgetError
    output = Path(output).resolve()
    if not output.is_relative_to((ROOT / "outputs").resolve()) or output == (ROOT / "outputs").resolve():
        raise ValueError("new output must be inside project outputs")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    result = {"schema": "native-six-point-research-v1", "status": "REJECTED",
              "source_model_id": str(model_id), "selection_rule": "six smallest native point_id; no reselection",
              "arithmetic": "exact_rational_of_stored_binary64", "limitations": LIMITATIONS,
              "source_bindings": [], "artifacts": [], "max_entries": max_entries,
              "source_pages": {"source": "S-1996", "pdf": [6, 7], "journal": [325, 326]},
              "controls": {"status": "NOT_RUN"},
              "reconstruction_performed": False, "research_gain_evaluated": False}
    try:
        if type(max_entries) is not int or not 1 <= max_entries <= 10000:
            raise ValueError("max_entries must be explicitly 1..10000; no automatic increase")
        sfm_run = Path(sfm_run).resolve(strict=True)
        run_path = sfm_run / "run.json"
        run_binding = binding(run_path)
        run = read_json(run_path, 4 * 1024**2)
        if run.get("contract_version") not in ("sfm-native-object-v1", "sfm-native-object-pinhole-v1") or run.get("status") not in ("SUCCESS", "PARTIAL"):
            raise ValueError("requires successful or partial native SfM run with a verified usable model")
        result["source_sfm_status"] = run["status"]
        candidates = [row for row in run.get("models", []) if str(row.get("model_id")) == str(model_id)]
        if len(candidates) != 1:
            raise ValueError("explicit model_id must identify exactly one model")
        model = candidates[0]
        if model.get("usable") is not True or model.get("roundtrip_verified") is not True:
            raise ValueError("model usability/native roundtrip not verified")
        binary = (sfm_run / model["binary_path"] / "points3D.bin").resolve(strict=True)
        description = (sfm_run / model["descriptions_path"]).resolve(strict=True)
        if not binary.is_relative_to(sfm_run) or not description.is_relative_to(sfm_run):
            raise ValueError("model artifacts escape selected SfM run")
        result["source_bindings"] = [run_binding, binding(binary), binding(description)]
        count, selected = read_native_six(binary)
        result.update(native_point_count=count, native_point_ids=[row[0] for row in selected], input_origin=run.get("data_kind"))
        described = read_json(description, MAX_DESCRIPTION)["source"]
        if described.get("num_points3D") != count or model.get("num_points3D") != count:
            raise ValueError("native count differs from source/model description")
        for ident, xyz in selected:
            expected = described["points3D"][str(ident)]["xyz"]
            if len(expected) != 3 or any(type(v) not in (int, float) or not math.isfinite(v) for v in expected):
                raise ValueError("invalid source description coordinates")
            if tuple(float(v).hex() for v in expected) != tuple(v.hex() for v in xyz):
                raise ValueError("native/source binary64 identity differs")
        points = [{"native_point_id": ident, "xyz_hex": [v.hex() for v in xyz],
                   "xyz_rational": [{"numerator_hex": hex(v.as_integer_ratio()[0]),
                                     "denominator_hex": hex(v.as_integer_ratio()[1])} for v in xyz]}
                  for ident, xyz in selected]
        _write(output / "SOURCE_POINTS.json", {"selection_rule": result["selection_rule"], "points": points})
        if len(selected) != 6:
            result["status"] = "NOT_APPLICABLE_INSUFFICIENT_POINTS"
        else:
            coordinates = [[Fraction.from_float(v) for v in xyz] for _, xyz in selected]
            code = indexed_body_code(coordinates, labels=[ident for ident, _ in selected], max_entries=max_entries)
            payload = code.to_dict()
            _write(output / "INDEXED_CODE3D.json", payload)
            result.update(status="SUCCESS" if code.complete else "UNDEFINED_PLANAR", code_status=code.status,
                          complete=code.complete, counts=payload["counts"], theorem_domain=payload["theorem_domain"],
                          algorithm_called="kozlov_lab.stereo.official.indexed_code3d.indexed_body_code")
            if controls and code.complete:
                try:
                    from integrations.demo_pipeline.research_controls import affine_controls
                    checked, affine, negative = affine_controls(code, max_permutations=max_control_permutations)
                    _write(output / "AFFINE_CODE3D.json", affine)
                    _write(output / "NEGATIVE_CODE3D.json", negative)
                    _write(output / "CONTROLS.json", checked)
                    result["controls"] = {"status": checked["status"], "result": binding(output / "CONTROLS.json"),
                        "positive_supplied_mapping": checked["positive_supplied_mapping"]["status"],
                        "negative_existential": checked["negative_existential_comparison"]["status"],
                        "independent_permutations_checked": checked["independent_permutations_checked"],
                        "max_permutations": max_control_permutations}
                except Exception as exc:
                    checked = {"status": "FAILED", "error": f"{type(exc).__name__}: {exc}",
                               "max_permutations": max_control_permutations, "no_reselection": True}
                    _write(output / "CONTROLS.json", checked)
                    result["controls"] = {**checked, "result": binding(output / "CONTROLS.json")}
        if any(binding(ref["path"]) != ref for ref in result["source_bindings"]):
            raise ValueError("input changed during research representation")
    except CodeBudgetError as exc:
        result.update(status="BUDGET_EXCEEDED", error=str(exc), complete=False)
    except Exception as exc:
        result.update(status="REJECTED", error=f"{type(exc).__name__}: {exc}", complete=False)
    for name in ("SOURCE_POINTS.json", "INDEXED_CODE3D.json", "AFFINE_CODE3D.json", "NEGATIVE_CODE3D.json", "CONTROLS.json"):
        if (output / name).is_file():
            result["artifacts"].append(binding(output / name))
    result["implementation_bindings"] = [binding(__file__), binding(ROOT / "src/kozlov_lab/stereo/official/indexed_code3d.py")]
    if controls:
        result["implementation_bindings"].append(binding(Path(__file__).with_name("research_controls.py")))
    result["wall_seconds"] = time.monotonic() - started
    _write(output / "RESULT.json", result)
    esc = lambda v: html.escape(str(v), quote=True)
    links = ''.join(f'<li><a href="{esc(Path(ref["path"]).name)}">{esc(Path(ref["path"]).name)}</a> SHA-256 {esc(ref["sha256"])}</li>' for ref in result["artifacts"])
    page = f'<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Kozlov六点研究表示</title><body><h1>Kozlov研究表示：真实来源固定六点子集</h1><p>运行状态：{esc(result["status"])}；模型 {esc(model_id)}；point_id：{esc(result.get("native_point_ids", []))}</p><p>计数：{esc(result.get("counts", {}))}</p><p>定理域：{esc(result.get("theorem_domain", {}))}</p><h2>已知有理仿射正对照与固定负对照</h2><p>对照状态：{esc(result["controls"])}</p><p>仅已构造的六点对应；不表示未知照片对应恢复。对照 UNKNOWN/FAILED 不冒称通过，也不抹去已构建的表示。</p><p>{esc(result.get("error", ""))}</p><ul>{"".join("<li>"+esc(item)+"</li>" for item in LIMITATIONS)}</ul><h2>来源与研究记录</h2><ul>{links}</ul><a href="RESULT.json">完整运行状态与输入哈希</a></body></html>'
    (output / "index.html").write_text(page, encoding="utf-8")
    return result
