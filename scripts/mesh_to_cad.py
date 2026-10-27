#!/usr/bin/env python3
"""Convert a prevalidated closed mesh with FreeCAD's bundled Python.

This standalone entry point intentionally does not import kozlov_lab.  Passing
this converter proves numerical CAD conversion, not fidelity to photographs.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

RTOL = 1e-7
ATOL = 1e-9
SEWING_TOLERANCE = 1e-7
MAX_INPUT_BYTES = 64 * 1024 * 1024


class Rejected(Exception):
    def __init__(self, reason: str, message: str, exit_code: int = 2):
        super().__init__(message)
        self.reason = reason
        self.exit_code = exit_code


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--mesh", type=Path, required=True, help="Existing .ply/.stl/.obj triangle mesh")
    result.add_argument("--output", type=Path, required=True, help="New directory; any existing path is refused")
    result.add_argument("--source-run", type=Path, required=True, help="Existing provenance file or run directory")
    result.add_argument("--scale-status", choices=("unknown", "reference_scaled"), required=True)
    result.add_argument("--length-unit", choices=("arbitrary", "mm"), required=True)
    result.add_argument("--scale-reference", type=Path, help="Existing reference evidence file, required for reference_scaled")
    result.add_argument("--max-faces", type=positive_int, default=10000)
    result.add_argument("--synthetic", action="store_true", help="Label an explicitly synthetic input; never a photo result")
    return result


def configure_runtime(output: Path) -> dict:
    """Keep FreeCAD's own config/temp files and process-only settings local."""
    if os.environ.get("PYTHONPATH") or os.environ.get("PYTHONHOME"):
        raise Rejected("INHERITED_PYTHON_ENV", "Unset PYTHONPATH and PYTHONHOME before launching bundled Python.", 3)
    if sys.version_info[:2] != (3, 11):
        raise Rejected("WRONG_PYTHON", "Use FreeCAD 1.1.4 portable bin/python.exe (Python 3.11).", 3)
    if not sys.flags.isolated:
        raise Rejected("PYTHON_NOT_ISOLATED", "Launch bundled Python with -I -B before the script path.", 3)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["PYTHONNOUSERSITE"] = "1"
    sys.dont_write_bytecode = True
    settings = {}
    for key, name in (
        ("FREECAD_USER_HOME", "config"), ("FREECAD_USER_DATA", "data"),
        ("FREECAD_USER_TEMP", "temp"), ("TEMP", "temp"), ("TMP", "temp"),
        ("APPDATA", "appdata"), ("LOCALAPPDATA", "localappdata"),
    ):
        folder = output / "runtime" / name
        folder.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(folder)
        settings[key] = str(folder)
    for key in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
        "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS",
    ):
        os.environ[key] = "2"
        settings[key] = "2"
    # OCCT BOP checks explicitly request parallelism. Bound this process only;
    # environment thread variables alone do not constrain that native scheduler.
    if os.name == "nt":
        import ctypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        kernel.GetProcessAffinityMask.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t))
        kernel.SetProcessAffinityMask.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
        process = kernel.GetCurrentProcess()
        allowed, system = ctypes.c_size_t(), ctypes.c_size_t()
        if not kernel.GetProcessAffinityMask(process, ctypes.byref(allowed), ctypes.byref(system)):
            raise OSError(ctypes.get_last_error(), "GetProcessAffinityMask failed")
        bits = [1 << index for index in range(allowed.value.bit_length()) if allowed.value & (1 << index)]
        mask = sum(bits[:2])
        if not mask or not kernel.SetProcessAffinityMask(process, mask):
            raise OSError(ctypes.get_last_error(), "SetProcessAffinityMask failed")
        settings["own_process_affinity_mask"] = mask
    return settings


def topology(mesh) -> dict:
    points, facets = mesh.Topology
    coordinates = [(float(p.x), float(p.y), float(p.z)) for p in points]
    finite = all(math.isfinite(value) for point in coordinates for value in point)
    edges = defaultdict(list)
    links = defaultdict(lambda: defaultdict(set))
    seen = set()
    duplicate_faces = degenerate_faces = invalid_faces = 0
    volume_terms = []
    for face in facets:
        if len(face) != 3 or any(index < 0 or index >= len(points) for index in face):
            invalid_faces += 1
            continue
        a, b, c = map(int, face)
        key = tuple(sorted((a, b, c)))
        duplicate_faces += key in seen
        seen.add(key)
        if len(set(face)) != 3:
            degenerate_faces += 1
        elif finite:
            p, q, r = coordinates[a], coordinates[b], coordinates[c]
            u, v = [q[i]-p[i] for i in range(3)], [r[i]-p[i] for i in range(3)]
            cross = (u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0])
            if not all(math.isfinite(x) for x in cross) or all(x == 0 for x in cross):
                degenerate_faces += 1
            origin = coordinates[0]
            translated = [p[i]-origin[i] for i in range(3)]
            volume_terms.append(sum(translated[i]*cross[i] for i in range(3)) / 6.0)
        for left, right in ((a, b), (b, c), (c, a)):
            edges[tuple(sorted((left, right)))].append(1 if left < right else -1)
        for center, left, right in ((a, b, c), (b, c, a), (c, a, b)):
            links[center][left].add(right)
            links[center][right].add(left)
    bad_vertex_links = 0
    for neighbors in links.values():
        if any(len(adjacent) != 2 for adjacent in neighbors.values()):
            bad_vertex_links += 1
            continue
        reached, pending = set(), [next(iter(neighbors))]
        while pending:
            vertex = pending.pop()
            if vertex not in reached:
                reached.add(vertex)
                pending.extend(neighbors[vertex] - reached)
        bad_vertex_links += len(reached) != len(neighbors)
    bbox = None
    if coordinates and finite:
        bbox = [min(p[axis] for p in coordinates) for axis in range(3)] + [max(p[axis] for p in coordinates) for axis in range(3)]
    signed_volume = None
    if finite and not invalid_faces and not degenerate_faces and all(math.isfinite(x) for x in volume_terms):
        try:
            value = math.fsum(volume_terms)
            signed_volume = value if math.isfinite(value) else None
        except OverflowError:
            pass
    return {
        "vertices": len(points), "faces": len(facets), "finite_coordinates": finite,
        "edges": len(edges), "boundary_edges": sum(len(v) == 1 for v in edges.values()),
        "nonmanifold_edges": sum(len(v) != 2 for v in edges.values()),
        "inconsistently_oriented_edges": sum(len(v) == 2 and sum(v) != 0 for v in edges.values()),
        "nonmanifold_vertex_links": bad_vertex_links, "duplicate_faces": duplicate_faces,
        "degenerate_faces": degenerate_faces, "invalid_faces": invalid_faces, "bbox": bbox,
        "signed_volume": signed_volume, "volume_method": "Oriented triangle sum about the first input vertex, accumulated with math.fsum",
    }


def require_manual_mesh(stats: dict) -> None:
    if stats["boundary_edges"]:
        raise Rejected("OPEN_MESH", "Input has boundary edges; no filling, capping, or convex-hull fallback is performed.")
    counts = ("nonmanifold_edges", "inconsistently_oriented_edges", "nonmanifold_vertex_links", "duplicate_faces", "degenerate_faces", "invalid_faces")
    if not stats["finite_coordinates"] or any(stats[key] for key in counts):
        raise Rejected("INVALID_MESH_TOPOLOGY", "Manual finite, nondegenerate, oriented-manifold gates failed before native analysis.")
    if stats["signed_volume"] is None or stats["signed_volume"] <= 0:
        raise Rejected("MESH_VOLUME_ORIENTATION", "Input must have finite positive oriented volume; normals are not auto-reversed.")


def require_mesh(stats: dict) -> None:
    if stats["boundary_edges"] or not stats["native_is_solid"]:
        raise Rejected("OPEN_MESH", "Input is not closed; no filling, capping, or convex-hull fallback is performed.")
    if stats["native_component_count"] != 1:
        raise Rejected("MULTIPLE_COMPONENTS", "Exactly one edge-connected component is required.")
    counts = ("nonmanifold_edges", "inconsistently_oriented_edges", "nonmanifold_vertex_links", "duplicate_faces", "degenerate_faces", "invalid_faces")
    flags = ("native_has_nonmanifolds", "native_has_self_intersections", "native_has_nonuniform_orientation", "native_has_invalid_neighbourhood")
    if not stats["finite_coordinates"] or any(stats[key] for key in counts + flags):
        raise Rejected("INVALID_MESH_TOPOLOGY", "Mesh has nonfinite, degenerate, duplicate, nonmanifold, unoriented, or intersecting geometry.")


def shape_metrics(shape) -> dict:
    if shape.isNull():
        raise Rejected("NULL_SHAPE", "FreeCAD returned a null shape.", 4)
    box, volume = shape.BoundBox, float(shape.Volume)
    bbox = [float(box.XMin), float(box.YMin), float(box.ZMin), float(box.XMax), float(box.YMax), float(box.ZMax)]
    finite = math.isfinite(volume) and all(math.isfinite(value) for value in bbox)
    return {"finite": finite, "is_valid": bool(shape.isValid()), "is_closed": bool(shape.isClosed()),
            "solid_count": len(shape.Solids), "shell_count": len(shape.Shells), "face_count": len(shape.Faces),
            "volume": volume if math.isfinite(volume) else None,
            "bbox": [value if math.isfinite(value) else None for value in bbox]}


def require_shape(shape) -> dict:
    values = shape_metrics(shape)
    if not (values["finite"] and values["is_valid"] and values["is_closed"] and values["solid_count"] == 1 and values["volume"] > 0):
        raise Rejected("INVALID_SOLID", f"Valid closed single solid with finite positive volume required: {values}", 4)
    try:
        result = shape.check(True)
    except Exception as exc:
        raise Rejected("BREP_INTERFERENCE", f"OCCT detailed/BOP validation rejected the shape: {exc}", 4) from exc
    if result is not None:
        raise Rejected("UNEXPECTED_BREP_CHECK", "Unsupported shape.check(True) return contract.", 4)
    values["occt_bop_check_passed"] = True
    return values


def compare(source: dict, target: dict) -> dict:
    """Compare CAD roundtrips without allowing the world origin to set tolerance.

    Absolute coordinates can be large even for a small object. The source's
    longest bbox extent sets the relative linear tolerance; volume retains its
    separate fixed relative/absolute rule. These are conversion tolerances,
    never physical measurement accuracy.
    """
    for label, metrics in (("source", source), ("target", target)):
        box = metrics.get("bbox") if isinstance(metrics, dict) else None
        volume = metrics.get("volume") if isinstance(metrics, dict) else None
        if not isinstance(box, (list, tuple)) or len(box) != 6:
            raise Rejected("INVALID_NUMERIC_METRICS", f"{label} bbox must have six coordinates.", 4)
        values = [*box, volume]
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in values):
            raise Rejected("INVALID_NUMERIC_METRICS", f"{label} metrics must be finite numbers.", 4)
        if volume <= 0 or any(box[axis] > box[axis + 3] for axis in range(3)):
            raise Rejected("INVALID_NUMERIC_METRICS", f"{label} requires ordered bounds and positive volume.", 4)
    extents = [source["bbox"][axis + 3] - source["bbox"][axis] for axis in range(3)]
    reference_length = max(extents)
    if not math.isfinite(reference_length) or min(extents) <= 0:
        raise Rejected("INVALID_NUMERIC_METRICS", "Source solid requires finite positive bbox extents.", 4)
    linear_tolerance = max(ATOL, RTOL * reference_length)
    bbox_deltas = [abs(a-b) for a, b in zip(source["bbox"], target["bbox"])]
    volume_delta = abs(source["volume"] - target["volume"])
    if not all(math.isfinite(delta) for delta in bbox_deltas):
        raise Rejected("INVALID_NUMERIC_METRICS", "Coordinate differences exceed finite range.", 4)
    passed = math.isclose(source["volume"], target["volume"], rel_tol=RTOL, abs_tol=ATOL) and all(
        delta <= linear_tolerance for delta in bbox_deltas)
    return {"pass": passed, "volume_absolute_difference": volume_delta, "bbox_absolute_differences": bbox_deltas,
            "comparison_policy": "source_extent_v1", "linear_reference_length": reference_length,
            "linear_tolerance": linear_tolerance,
            "volume_tolerance": max(ATOL, RTOL * max(source["volume"], target["volume"]))}


def convert(args, output: Path, report: dict, save, log) -> None:
    try:
        mesh_path, source_run = args.mesh.resolve(strict=True), args.source_run.resolve(strict=True)
    except OSError as exc:
        raise Rejected("INPUT_NOT_FOUND", f"Mesh and source-run must both exist: {exc}") from exc
    if not mesh_path.is_file() or mesh_path.suffix.lower() not in (".ply", ".stl", ".obj"):
        raise Rejected("INPUT_FORMAT", "Input must be an existing .ply/.stl/.obj file.")
    if mesh_path.stat().st_size > MAX_INPUT_BYTES:
        raise Rejected("INPUT_TOO_LARGE", "Input exceeds the 64 MiB loading limit.")
    if (args.scale_status, args.length_unit) not in (("unknown", "arbitrary"), ("reference_scaled", "mm")):
        raise Rejected("SCALE_CONFLICT", "Use unknown/arbitrary or reference_scaled/mm. No automatic rescaling is performed.")
    reference = None
    if args.scale_status == "reference_scaled":
        if args.scale_reference is None or not args.scale_reference.is_file():
            raise Rejected("MISSING_SCALE_REFERENCE", "reference_scaled/mm requires --scale-reference with an existing evidence file.")
        reference = {"path": str(args.scale_reference.resolve()), "sha256": sha256(args.scale_reference)}
    elif args.scale_reference is not None:
        raise Rejected("SCALE_CONFLICT", "A scale reference is only accepted with reference_scaled/mm.")
    report["input"] = {"mesh": str(mesh_path), "mesh_bytes": mesh_path.stat().st_size, "mesh_sha256": sha256(mesh_path),
                       "source_run": str(source_run), "source_run_sha256": sha256(source_run) if source_run.is_file() else None}
    report["scale"] = {"source_scale_status": args.scale_status, "declared_length_unit": args.length_unit,
                       "source_length_unit": None if args.scale_status == "unknown" else "mm",
                       "cad_file_length_unit": "mm", "applied_coordinate_factor": 1.0,
                       "scale_reference": reference, "physical_accuracy_status": "NOT_EVALUATED",
                       "notice": "Arbitrary coordinates preserved; mm is only the CAD file convention, not physical millimetres." if args.scale_status == "unknown" else "Caller declares coordinates already reference-scaled in mm; converter does not verify reference measurements or physical accuracy."}
    report["runtime_environment"] = configure_runtime(output)
    save()
    import FreeCAD as App
    import Mesh
    import Part
    report["backend"] = {"freecad_version": list(App.Version()), "python_version": sys.version,
                         "python_executable": sys.executable, "module_path": getattr(App, "__file__", None),
                         "freecad_user_data": App.getUserAppDataDir()}
    if report["backend"]["freecad_version"][:3] != ["1", "1", "4"]:
        raise Rejected("BACKEND_VERSION", "This converter is pinned to the validated FreeCAD 1.1.4 API.", 3)
    log("Loading input mesh with FreeCAD Mesh.Mesh")
    mesh = Mesh.Mesh(str(mesh_path))
    report["loaded_face_count"] = int(mesh.CountFacets)
    if mesh.CountFacets < 4 or mesh.CountFacets > args.max_faces:
        # An open three-face tetrahedron is still classified specifically below.
        if mesh.CountFacets > args.max_faces:
            raise Rejected("FACE_LIMIT", f"Loaded {mesh.CountFacets} faces; limit is {args.max_faces}.")
        if mesh.CountFacets == 0:
            raise Rejected("EMPTY_MESH", "Input has no triangular faces.")
    stats = topology(mesh)
    report["mesh_topology"] = stats
    save()
    require_manual_mesh(stats)
    stats.update({
        "native_is_solid": bool(mesh.isSolid()), "native_component_count": int(mesh.countComponents()),
        "native_has_nonmanifolds": bool(mesh.hasNonManifolds()),
        "native_has_self_intersections": bool(mesh.hasSelfIntersections()),
        "native_has_nonuniform_orientation": bool(mesh.hasNonUniformOrientedFacets()),
        "native_has_invalid_neighbourhood": bool(mesh.hasInvalidNeighbourhood()),
        "self_intersection_method": "FreeCAD Mesh detector followed by OCCT BOP self-interference checks on converted and reloaded BReps; native tolerances",
    })
    save()
    require_mesh(stats)
    log("Closed single-component mesh gates passed; constructing BRep")
    shape = Part.Shape()
    shape.makeShapeFromMesh(mesh.Topology, SEWING_TOLERANCE)
    if len(shape.Shells) != 1 or not shape.Shells[0].isClosed():
        raise Rejected("SHELL_CONVERSION", "Mesh conversion did not produce exactly one closed shell.", 4)
    shell = shape.Shells[0]
    if len(shape.Faces) != len(shell.Faces) or len(shape.Edges) != len(shell.Edges) or len(shape.Vertexes) != len(shell.Vertexes) or len(shell.Faces) != stats["faces"]:
        raise Rejected("SHELL_GEOMETRY_LOSS", "Conversion has loose geometry or changed the number of triangular faces.", 4)
    solid = Part.Solid(shell)
    original = require_shape(solid)
    mesh_numeric = compare({"bbox": stats["bbox"], "volume": stats["signed_volume"]}, original)
    report["checks"]["mesh_to_brep_numeric"] = mesh_numeric
    if not mesh_numeric["pass"]:
        raise Rejected("MESH_BREP_NUMERIC", "Mesh-to-BRep bbox or signed volume changed beyond the fixed tolerance.", 4)
    report["checks"]["source_solid"] = original
    save()
    doc = App.newDocument("MeshToCAD")
    obj = doc.addObject("Part::Feature", "FacetedApproximation")
    obj.Label = ("SYNTHETIC " if args.synthetic else "") + "Faceted approximation - " + args.length_unit
    obj.Shape = solid
    for name, value in (("SourceMeshSHA256", report["input"]["mesh_sha256"]), ("ScaleStatus", args.scale_status), ("ScaleNotice", report["scale"]["notice"]), ("SourceRun", str(source_run))):
        obj.addProperty("App::PropertyString", name, "Provenance")
        setattr(obj, name, value)
    doc.recompute()
    fcstd, step = output / "model.FCStd", output / "model.step"
    doc.saveAs(str(fcstd))
    solid.exportStep(str(step))
    App.closeDocument(doc.Name)
    reopened = App.openDocument(str(fcstd))
    objects = [obj for obj in reopened.Objects if hasattr(obj, "Shape")]
    if len(objects) != 1:
        raise Rejected("FCSTD_OBJECT_COUNT", "Reopened FCStd must contain exactly one shape object.", 4)
    fcstd_values = require_shape(objects[0].Shape)
    report["checks"]["fcstd_reopened"] = fcstd_values
    report["checks"]["fcstd_numeric"] = compare(original, fcstd_values)
    App.closeDocument(reopened.Name)
    imported_doc = App.newDocument("STEPReimport")
    imported = imported_doc.addObject("Part::Feature", "STEPReimport")
    imported.Shape = Part.read(str(step))
    imported_doc.recompute()
    step_values = require_shape(imported.Shape)
    report["checks"]["step_reimport"] = step_values
    report["checks"]["step_numeric"] = compare(original, step_values)
    App.closeDocument(imported_doc.Name)
    if not report["checks"]["fcstd_numeric"]["pass"] or not report["checks"]["step_numeric"]["pass"]:
        raise Rejected("ROUNDTRIP_NUMERIC", "Reopened CAD geometry exceeds fixed bbox/volume tolerances.", 4)
    if sha256(mesh_path) != report["input"]["mesh_sha256"]:
        raise Rejected("INPUT_CHANGED", "Source mesh changed during conversion.", 4)
    report["artifacts"] = [{"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)} for path in (fcstd, step)]
    log("Native FCStd and STEP reimport gates passed")


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    output = args.output.resolve()
    try:
        output.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        print(f"OUTPUT_REFUSED: new output directory required: {exc}", file=sys.stderr)
        return 2
    started = time.monotonic()
    report = {"status": "STARTED", "synthetic_input": args.synthetic, "evidence_scope": "Geometric CAD conversion only; photo fidelity, GUI operation, physical accuracy and parametric design recovery are not evaluated.",
              "started_utc": datetime.now(timezone.utc).isoformat(), "argv": [sys.executable, *sys.argv], "cwd": str(Path.cwd()),
              "converter_sha256": sha256(Path(__file__)), "max_faces": args.max_faces,
              "tolerances": {"relative": RTOL, "absolute": ATOL, "mesh_sewing": SEWING_TOLERANCE,
                             "bbox_policy": "source_extent_v1",
                             "bbox_rule": "max(absolute, relative * longest source bbox extent)",
                             "volume_rule": "math.isclose with separate relative and absolute limits"}, "checks": {}}

    def save():
        report["elapsed_seconds"] = time.monotonic() - started
        (output / "conversion_report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")

    def log(message):
        with (output / "conversion.log").open("a", encoding="utf-8") as stream:
            stream.write(datetime.now(timezone.utc).isoformat() + " " + message + "\n")

    code = 3
    try:
        save()
        convert(args, output, report, save, log)
        report["status"] = "PASS_SYNTHETIC_CAD_CONVERSION" if args.synthetic else "PASS_GEOMETRIC_CAD_CONVERSION"
        code = 0
    except Rejected as exc:
        report["status"] = "REJECTED"
        report["reason"] = exc.reason
        report["error"] = str(exc)
        code = exc.exit_code
        (output / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
    except Exception as exc:
        report["status"] = "BACKEND_ERROR"
        report["reason"] = type(exc).__name__
        report["error"] = str(exc)
        (output / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
    finally:
        report["script_exit_code"] = code
        report["completed_utc"] = datetime.now(timezone.utc).isoformat()
        save()
    print(json.dumps({"status": report["status"], "report": str(output / "conversion_report.json"), "script_exit_code": code}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
