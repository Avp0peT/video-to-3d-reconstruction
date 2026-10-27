"""Two-stage supervised Blender save and independent readback."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import struct

from integrations.viewer.export import load_scenes, PROFILE_LIMITS
from integrations.mvs_surface.supervision import run_supervised, strict_directory_size

ROOT = Path(__file__).resolve().parents[2]
MAX_OUTPUT = 128 * 1024**2


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def find_blender(explicit=None):
    if explicit is not None:
        candidates = [Path(explicit)]
    else:
        candidates = [Path(r"C:\Program Files\Blender Foundation\Blender 5.1\blender.exe")]
        found = shutil.which("blender")
        if found:
            candidates.append(Path(found))
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError("Blender executable not found; no installation or download attempted")


def _write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)


def _stage_ok(result):
    return (result.get("status") == "completed" and result.get("root_exit_code") == 0
            and result.get("reclaim_confirmed") is True and not result.get("errors")
            and not result.get("log_errors"))


def _float32_geometry(scene):
    converted = [[struct.unpack("f", struct.pack("f", value))[0] for value in vertex]
                 for vertex in scene["vertices"]]
    def zero_area(vertices, face):
        a, b, c = [vertices[index] for index in face]
        u = [b[i]-a[i] for i in range(3)]
        v = [c[i]-a[i] for i in range(3)]
        return all(value == 0 for value in (u[1]*v[2]-u[2]*v[1],
                                          u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0]))
    newly_degenerate = sum(not zero_area(scene["vertices"], face) and zero_area(converted, face)
                           for face in scene["faces"])
    stats = {"new_zero_area_triangles": newly_degenerate,
             "additional_duplicate_points": len(set(map(tuple, scene["vertices"]))) - len(set(map(tuple, converted)))}
    if newly_degenerate:
        raise ValueError(f"float32 conversion introduces {newly_degenerate} zero-area triangles")
    return stats


def export_blend(spec_path, output, *, blender_exe=None, profile='standard'):
    """Export to a new directory under repository outputs; retain partial evidence."""
    output = Path(output).absolute()
    if not output.resolve().is_relative_to((ROOT / "outputs").resolve()):
        raise ValueError("output must be a new directory under repository outputs")
    if output.exists():
        raise ValueError("refusing to overwrite output directory")
    scenes, spec_sha = load_scenes(Path(spec_path), profile=profile)
    # Blender mesh coordinates use float32, including for double-precision PLY.
    conversion = [_float32_geometry(scene) for scene in scenes]
    executable = find_blender(blender_exe)
    output.mkdir(parents=True, exist_ok=False)
    payload = output / "payload.json"
    _write(payload, {"scenes": scenes, "spec_sha256": spec_sha})
    payload_sha = digest(payload)
    sources = {s["source_path"]: s["sha256"] for s in scenes}
    sources.update({e["path"]: e["sha256"] for s in scenes for e in s["evidence"]})
    sources[str(Path(spec_path).resolve())] = spec_sha
    report = {"schema": "ply-blender-export-v1", "status": "PARTIAL", "profile": profile,
              "geometry_limits": PROFILE_LIMITS[profile], "spec_sha256": spec_sha,
              "payload_sha256": payload_sha, "source_hashes_before": sources,
              "float32_conversion": conversion,
              "report_path": str(output / "RESULT.json"),
              "limits": {"stage_wall_seconds": 90, "memory_bytes": (2 if profile == "standard" else 4)*1024**3,
                         "output_bytes": MAX_OUTPUT, "maximum_native_launches": 2}, "stages": []}
    worker = Path(__file__).with_name("worker.py")
    report["worker_sha256"] = digest(worker)
    report["blender_exe"] = str(executable)
    blend = output / "model.blend"
    try:
        for mode in ("create", "verify"):
            stage = output / mode
            stage.mkdir()
            argv = [str(executable), "--background", "--factory-startup", "--disable-autoexec", "--offline-mode", "--threads", "2",
                    "--python-exit-code", "7", "--python", str(worker), "--", mode,
                    str(payload), payload_sha, str(blend), str(stage / "result.json")]
            result = run_supervised(argv, cwd=str(output), output=str(stage), wall_seconds=90,
                                    memory_bytes=report["limits"]["memory_bytes"], output_bytes=MAX_OUTPUT,
                                    label="blender_" + mode)
            report["stages"].append({"mode": mode, "supervision": result})
            _write(stage / "supervision.json", result)
            if not _stage_ok(result):
                raise RuntimeError(mode + " stage failed or reclamation unconfirmed")
            data = json.loads((stage / "result.json").read_text(encoding="utf-8"))
            report["stages"][-1]["data"] = data
            if data.get("status") != "PASS" or data.get("payload_sha256") != payload_sha:
                raise RuntimeError(mode + " data verification failed")
            if mode == "create":
                report["blend_sha256_before_readback"] = digest(blend)
            elif data.get("scenes") != report["stages"][0]["data"].get("scenes"):
                raise RuntimeError("readback scene evidence changed")
        report["blend_sha256"] = digest(blend)
        report["source_hashes_after"] = {path: digest(path) for path in sources}
        if report["source_hashes_after"] != sources or digest(payload) != payload_sha:
            raise RuntimeError("source, evidence, spec or payload changed")
        if report["blend_sha256"] != report["blend_sha256_before_readback"]:
            raise RuntimeError("blend file changed during independent readback")
        report["blend"] = {"path": str(blend), "sha256": report["blend_sha256"], "bytes": blend.stat().st_size}
        report["scenes"] = report["stages"][-1]["data"]["scenes"]
        final_report_bytes = len(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2).encode("utf-8"))
        if strict_directory_size(output) + 2*final_report_bytes > MAX_OUTPUT:
            raise RuntimeError("output budget including final reports exceeded")
        report["status"] = "SUCCESS"
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    report["report_path"] = str(output / "RESULT.json")
    _write(output / "RESULT.json", report)
    _write(output / "report.json", report)
    return report
