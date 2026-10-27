"""Bounded, evidence-bearing COLMAP undistortion and OpenMVS format bridge.

This does not reconstruct, densify, repair, merge, or infer object completeness.
The MVS roundtrip preserves calibrated geometry and view memberships, not the
original feature residuals: InterfaceCOLMAP reprojects its float32 vertices.
Point IDs may be renumbered and geometries may repeat; verification therefore
requires a perfect bijection, not a unique correspondence.
"""
from __future__ import annotations

import os

THREAD_ENV = {name: "2" for name in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
)}
os.environ.update(THREAD_ENV)
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import argparse
from datetime import datetime, timezone
import enum
import hashlib
import itertools
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import traceback


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def plain(value):
    """Strict recursive JSON normalization; no arbitrary-object string fallback."""
    if isinstance(value, enum.Enum):
        return {"enum_type": f"{type(value).__module__}.{type(value).__qualname__}", "name": value.name, "value": plain(value.value)}
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        require(math.isfinite(value), "non-finite number cannot be evidence")
        return value
    if isinstance(value, dict):
        require(all(isinstance(k, str) for k in value), "JSON evidence keys must be strings")
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    # pybind11 enums in pycolmap 4.2.1 are not Python enum.Enum instances.
    # Require module, symbolic member and integral value, then verify the member
    # exists on that exact native type; unrelated native objects are rejected.
    if type(value).__module__ == "pycolmap._core":
        name, number = getattr(value, "name", None), getattr(value, "value", None)
        if isinstance(name, str) and isinstance(number, int) and getattr(type(value), name, None) == value:
            return {"enum_type": f"{type(value).__module__}.{type(value).__qualname__}", "name": name, "value": number}
    raise TypeError(f"unsupported evidence type: {type(value).__module__}.{type(value).__qualname__}")


def save(path, value):
    Path(path).write_text(json.dumps(plain(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def structure(model):
    """Reject duplicate views, broken reciprocity and invalid referenced points."""
    ids = set(model.reg_image_ids())
    require(len(ids) >= 2 and len(model.points3D) > 0, "empty/insufficient sparse geometry")
    require(len({model.images[i].name for i in ids}) == len(ids), "duplicate image names")
    observations = 0
    for pid, point in model.points3D.items():
        seen = set()
        require(len(point.track.elements) >= 2, f"point {pid} has fewer than two views")
        for e in point.track.elements:
            require(e.image_id in ids and e.image_id not in seen, f"point {pid} has unregistered or duplicate view")
            seen.add(e.image_id)
            image = model.images[e.image_id]
            require(0 <= e.point2D_idx < len(image.points2D), f"point {pid} index outside image")
            require(image.points2D[e.point2D_idx].point3D_id == pid, f"point {pid} broken track reciprocity")
            observations += 1
    reverse = 0
    for iid in ids:
        for idx, p in enumerate(model.images[iid].points2D):
            if p.has_point3D():
                require(p.point3D_id in model.points3D, "image references absent 3D point")
                require(any(e.image_id == iid and e.point2D_idx == idx for e in model.points3D[p.point3D_id].track.elements), "image observation missing reverse track")
                reverse += 1
    require(reverse == observations, "observation counts disagree")
    return {"images": len(ids), "points": len(model.points3D), "observations": observations}


def image_files(model, image_root):
    root = image_root.resolve()
    result = []
    for iid in sorted(model.reg_image_ids()):
        name = model.images[iid].name
        path = (root / name).resolve()
        require(path.is_relative_to(root) and path.is_file(), f"unsafe or missing image {name}")
        result.append({"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size})
    return result


def numeric_comparison(a, b, *, same_ids):
    import numpy as np
    left = {a.images[i].name: a.images[i] for i in a.reg_image_ids()}
    right = {b.images[i].name: b.images[i] for i in b.reg_image_ids()}
    require(set(left) == set(right), "registered image names changed")
    stats = {"rotation_max_abs": 0.0, "translation_max_abs": 0.0, "center_max_abs": 0.0}
    for name, ia in left.items():
        ib = right[name]
        if same_ids:
            require(ia.image_id == ib.image_id, "image identity changed during undistortion")
        for field, va, vb in (
            ("rotation_max_abs", ia.cam_from_world().rotation.matrix(), ib.cam_from_world().rotation.matrix()),
            ("translation_max_abs", ia.cam_from_world().translation, ib.cam_from_world().translation),
            ("center_max_abs", ia.projection_center(), ib.projection_center()),
        ):
            delta = float(np.max(np.abs(np.asarray(va) - np.asarray(vb))))
            require(np.isfinite(delta) and np.allclose(va, vb, rtol=1e-8, atol=1e-8), f"pose mismatch {name}: {field} {delta}")
            stats[field] = max(stats[field], delta)
    return left, right, stats


def undistort(args):
    import numpy as np
    import pycolmap
    require(pycolmap.__version__ == "4.2.1", "this bridge was validated only with pycolmap 4.2.1")
    root = Path(args.output)
    source = pycolmap.Reconstruction(args.sparse)
    initial_structure = structure(source)
    bindings = image_files(source, Path(args.image_root)) + [
        {"path": str(p.resolve()), "sha256": sha(p), "bytes": p.stat().st_size}
        for p in sorted(Path(args.sparse).iterdir()) if p.is_file()
    ]
    save(root / "INPUT_BINDINGS.json", bindings)
    options = pycolmap.UndistortCameraOptions()
    options.max_image_size = 1024
    save(root / "UNDISTORT_OPTIONS.json", {"options": options.todict(), "copy_policy": "copy", "num_threads": 2, "pycolmap": pycolmap.__version__})
    pycolmap.undistort_images(
        output_path=str(root / "undistorted"), input_path=args.sparse,
        image_path=args.image_root, output_type="COLMAP",
        copy_policy=pycolmap.FileCopyType.copy,
        undistort_options=options, num_threads=2,
    )
    target = pycolmap.Reconstruction(str(root / "undistorted" / "sparse"))
    require(structure(target) == initial_structure, "undistortion changed geometry/observation counts")
    require(set(source.points3D) == set(target.points3D), "undistortion changed point identities")
    point_max = 0.0
    for pid, point in source.points3D.items():
        delta = float(np.max(np.abs(point.xyz - target.points3D[pid].xyz)))
        require(delta <= 1e-12, f"undistortion changed point {pid}")
        point_max = max(point_max, delta)
    left, right, stats = numeric_comparison(source, target, same_ids=True)
    ray_max = 0.0
    ray_count = 0
    for name, ia in left.items():
        ib = right[name]
        ca, cb = source.cameras[ia.camera_id], target.cameras[ib.camera_id]
        require(cb.model_name in ("PINHOLE", "SIMPLE_PINHOLE"), "undistortion is not pinhole")
        require(len(ia.points2D) == len(ib.points2D), "undistortion changed keypoint indexing")
        # Check all features, including untriangulated ones, to catch a coordinate
        # convention error without hiding it behind reconstruction residuals.
        xy_a = np.array([p.xy for p in ia.points2D])
        xy_b = np.array([p.xy for p in ib.points2D])
        ra, rb = ca.cam_from_img(xy_a), cb.cam_from_img(xy_b)
        delta = float(np.max(np.abs(ra - rb)))
        require(np.isfinite(delta) and delta <= 1e-5, f"normalized ray mismatch {name}: {delta}")
        ray_max = max(ray_max, delta)
        ray_count += len(xy_a)
    save(root / "UNDISTORT_COMPARISON.json", {"status": "PASS", "structure": initial_structure, "point_xyz_max_abs": point_max, **stats, "normalized_ray_max_abs": ray_max, "ray_count": ray_count, "normalized_ray_tolerance": 1e-5, "tolerance_basis": "Numerical inverse-distortion tolerance in dimensionless camera coordinates; not a physical accuracy threshold. At focal 2500 px, 1e-5 is 0.025 px."})


# --- Point correspondence search -------------------------------------------------
# The functions below are pure data transformations: they take plain records and
# return plain records. No native process, no pycolmap and no file system access is
# involved, so the acceptance rules can be tested without running InterfaceCOLMAP.
# Native dependencies are imported inside the function that needs them.

def point_records(model):
    """Identity-bearing point records; duplicated geometry is preserved.

    RGB and the complete set of viewing image names belong to a point's identity.
    Nothing is deduplicated and no point is dropped.
    """
    return [
        {
            "id": int(pid),
            "xyz": [float(v) for v in model.points3D[pid].xyz],
            "rgb": [int(v) for v in model.points3D[pid].color],
            "views": sorted(model.images[e.image_id].name for e in model.points3D[pid].track.elements),
        }
        for pid in sorted(model.points3D)
    ]


def same_color_and_views(a, b):
    """Same-XYZ points stay distinguishable: color and full membership must match."""
    return a["rgb"] == b["rgb"] and a["views"] == b["views"]


def compatibility_edges(src, dst, tolerance):
    """Candidate edges for the correspondence search, keyed by source index.

    A spatial grid only narrows candidates; it never decides a correspondence.
    An edge requires the XYZ agreement and the full identity of color and view
    membership, so two points sharing XYZ but differing in color or membership are
    never interchangeable. Candidate lists are sorted for determinism.
    """
    import numpy as np
    require(tolerance > 0.0 and math.isfinite(tolerance), "tolerance must be finite and positive")
    dst_xyz, cells = [], {}
    for j, record in enumerate(dst):
        xyz = np.asarray(record["xyz"], dtype=float)
        require(xyz.shape == (3,) and np.all(np.isfinite(xyz)), f"roundtrip point {record['id']} has non-finite XYZ")
        dst_xyz.append(xyz)
        cells.setdefault(tuple(int(v) for v in np.floor(xyz / tolerance)), []).append(j)
    adjacency = {}
    for i, record in enumerate(src):
        xyz = np.asarray(record["xyz"], dtype=float)
        require(xyz.shape == (3,) and np.all(np.isfinite(xyz)), f"source point {record['id']} has non-finite XYZ")
        key = np.floor(xyz / tolerance).astype(np.int64)
        found = set()
        for offset in itertools.product((-1, 0, 1), repeat=3):
            for j in cells.get(tuple(int(k + d) for k, d in zip(key, offset)), []):
                if same_color_and_views(record, dst[j]) and float(np.max(np.abs(xyz - dst_xyz[j]))) <= tolerance:
                    found.add(j)
        adjacency[i] = sorted(found)
    return adjacency


def _augment(u, adjacency, match_right, used):
    """Kuhn augmenting step; re-assignment of earlier choices is expected."""
    for v in adjacency[u]:
        if used[v]:
            continue
        used[v] = True
        if match_right[v] == -1 or _augment(match_right[v], adjacency, match_right, used):
            match_right[v] = u
            return True
    return False


def _allowed_edges(adjacency, match_left, match_right, n_right):
    """Edges that occur in at least one perfect matching.

    With a perfect matching present, direct matched edges right->left and unmatched
    edges left->right; an unmatched edge lies on an alternating cycle (hence in
    another perfect matching) exactly when both endpoints share a strongly
    connected component. Edges on no perfect matching are excluded, which is how
    ambiguity is detected instead of being assumed.
    """
    n_left = len(adjacency)
    total = n_left + n_right
    graph = [[] for _ in range(total)]
    for i, neighbours in adjacency.items():
        for j in neighbours:
            if match_right[j] == i:
                graph[n_left + j].append(i)
            else:
                graph[i].append(n_left + j)
    order, seen = [], [False] * total
    for start in range(total):
        if seen[start]:
            continue
        seen[start] = True
        stack = [(start, iter(graph[start]))]
        while stack:
            node, it = stack[-1]
            nxt = next(it, None)
            if nxt is None:
                order.append(node)
                stack.pop()
            elif not seen[nxt]:
                seen[nxt] = True
                stack.append((nxt, iter(graph[nxt])))
    reverse = [[] for _ in range(total)]
    for node, targets in enumerate(graph):
        for t in targets:
            reverse[t].append(node)
    component, next_id = [-1] * total, 0
    for start in reversed(order):
        if component[start] != -1:
            continue
        component[start] = next_id
        stack = [start]
        while stack:
            for t in reverse[stack.pop()]:
                if component[t] == -1:
                    component[t] = next_id
                    stack.append(t)
        next_id += 1
    return {
        i: sorted(j for j in neighbours if match_right[j] == i or component[i] == component[n_left + j])
        for i, neighbours in adjacency.items()
    }


def correspondence(src, dst, tolerance):
    """Existence-of-perfect-bijection check with a deterministic witness.

    A perfect bijection over the whole point set is required (so point
    multiplicities must be complete), and every matched pair must satisfy the
    numerical tolerance, identical RGB and an identical view-member set. A unique
    one-to-one correspondence is *not* required: duplicated geometry is legitimate
    and is reported as an ambiguity group. The witness does not claim that the
    historical point identities were recovered.
    """
    require(len(src) > 0 and len(dst) > 0, "empty point set")
    require(len(src) == len(dst), f"point multiplicity changed: source {len(src)}, roundtrip {len(dst)}")
    adjacency = compatibility_edges(src, dst, tolerance)
    limit = 4 * (len(src) + len(dst)) + 4096
    if sys.getrecursionlimit() < limit:
        sys.setrecursionlimit(limit)
    match_right = [-1] * len(dst)
    for i in sorted(adjacency):
        _augment(i, adjacency, match_right, [False] * len(dst))
    match_left, matched = [-1] * len(src), 0
    for j, i in enumerate(match_right):
        if i != -1:
            match_left[i] = j
            matched += 1
    require(matched == len(src), f"no perfect bijection: {matched} of {len(src)} points can be matched")
    allowed = _allowed_edges(adjacency, match_left, match_right, len(dst))
    holders = {}
    for i, js in allowed.items():
        for j in js:
            holders.setdefault(j, []).append(i)
    # A source point has a determined partner exactly when only one of its edges can
    # appear in a perfect matching: then every perfect bijection uses that edge.
    ambiguous = sorted(i for i, js in allowed.items() if len(js) != 1)
    ambiguous_set = set(ambiguous)
    groups, claimed = [], set()
    for i in ambiguous:
        if i in claimed:
            continue
        members, frontier = {i}, [i]
        while frontier:
            current = frontier.pop()
            for j in allowed[current]:
                for other in holders[j]:
                    if other in ambiguous_set and other not in members:
                        members.add(other)
                        frontier.append(other)
        claimed |= members
        ordered = sorted(members)
        groups.append({
            "source_ids": [src[k]["id"] for k in ordered],
            "roundtrip_ids": sorted({dst[j]["id"] for k in ordered for j in allowed[k]}),
            "size": len(ordered),
        })
    witnessed = {str(src[i]["id"]): dst[match_left[i]]["id"] for i in sorted(adjacency)}
    return {
        "witness": witnessed,
        "witness_by_source_index": {str(i): match_left[i] for i in sorted(adjacency)},
        "ambiguity_groups": groups,
        "ambiguous_source_count": len(ambiguous),
        "uniquely_determined_source_count": len(src) - len(ambiguous),
        "candidate_edge_count": sum(len(v) for v in adjacency.values()),
        "max_candidate_degree": max((len(v) for v in adjacency.values()), default=0),
        "allowed_edge_count": sum(len(v) for v in allowed.values()),
        "unique_correspondence": not ambiguous,
    }


def verify_mvs(args):
    import numpy as np
    import pycolmap
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    base = Path(args.source_dir).resolve() if args.source_dir else root
    a = pycolmap.Reconstruction(str(base / "undistorted" / "sparse"))
    b = pycolmap.Reconstruction(str(base / "roundtrip" / "sparse"))
    sa, sb = structure(a), structure(b)
    require(sa == sb, f"MVS changed image/point/observation counts: {sa} -> {sb}")
    left, right, stats = numeric_comparison(a, b, same_ids=False)
    calibration_max = 0.0
    for name, ia in left.items():
        ca, cb = a.cameras[ia.camera_id], b.cameras[right[name].camera_id]
        require(ca.width == cb.width and ca.height == cb.height, "image dimensions changed")
        delta = float(np.max(np.abs(ca.calibration_matrix() - cb.calibration_matrix())))
        require(np.isfinite(delta) and delta <= 1e-7, f"calibration mismatch {name}: {delta}")
        calibration_max = max(calibration_max, delta)
    # OpenMVS stores vertices as float32 and renumbers them. Point IDs may change;
    # what must hold is the existence of a perfect bijection over the whole point
    # set satisfying the tolerance, the RGB equality and the complete view-member
    # equality. Duplicated geometry is legitimate and is reported as an ambiguity
    # group rather than used to reject the conversion.
    extent = max(1.0, float(np.max(np.abs([p.xyz for p in a.points3D.values()]))))
    tolerance = extent * 1e-6
    src, dst = point_records(a), point_records(b)
    found = correspondence(src, dst, tolerance)
    by_id = {record["id"]: record for record in dst}
    # Independent re-assertion of the witness: one-to-one, complete, and every
    # matched pair satisfies all compatibility conditions. Nothing is derived from
    # the search internals here.
    seen = set()
    point_max, reprojection_max = 0.0, 0.0
    source_by_id = {record["id"]: record for record in src}
    for pid, qid in sorted(found["witness"].items(), key=lambda item: int(item[0])):
        p = source_by_id[int(pid)]
        q = by_id[int(qid)]
        require(int(qid) not in seen, f"witness is not injective at {pid}")
        seen.add(int(qid))
        require(same_color_and_views(p, q), f"point identity mismatch {pid}")
        delta = float(np.max(np.abs(np.asarray(p["xyz"]) - np.asarray(q["xyz"]))))
        require(np.isfinite(delta) and delta <= tolerance, f"point XYZ outside tolerance {pid}: {delta}")
        point_max = max(point_max, delta)
        pe = a.points3D[int(pid)]
        qe = b.points3D[int(qid)]
        for e in qe.track.elements:
            im = b.images[e.image_id]
            cam = b.cameras[im.camera_id]
            expected = cam.img_from_cam(im.cam_from_world() * qe.xyz)
            pixel = float(np.max(np.abs(expected - im.points2D[e.point2D_idx].xy)))
            require(np.isfinite(pixel) and pixel <= 1e-4, "MVS reprojected pixel convention mismatch")
            reprojection_max = max(reprojection_max, pixel)
        require(len(pe.track.elements) == len(qe.track.elements), f"observation multiplicity changed {pid}")
    require(len(seen) == len(dst) == len(src), "witness does not cover every point")
    save(root / "MVS_COMPARISON.json", {"status": "PASS", "structure": sa, **stats, "calibration_max_abs_pixels": calibration_max, "calibration_tolerance_pixels": 1e-7, "point_xyz_max_abs": point_max, "point_xyz_tolerance": tolerance, "point_tolerance_basis": "1e-6 times max(1, max abs source coordinate), conservatively above float32 rounding, not physical accuracy", "reprojected_pixel_max_abs": reprojection_max, "reprojected_pixel_tolerance": 1e-4, "point_identity_map": found["witness"], "point_identity_map_note": "Deterministic witness of an existence-of-perfect-bijection check; point IDs are allowed to change and this witness does not claim the historical point identity was recovered.", "ambiguity_groups": found["ambiguity_groups"], "ambiguous_source_count": found["ambiguous_source_count"], "uniquely_determined_source_count": found["uniquely_determined_source_count"], "unique_correspondence": found["unique_correspondence"], "candidate_edge_count": found["candidate_edge_count"], "allowed_edge_count": found["allowed_edge_count"], "max_candidate_degree": found["max_candidate_degree"], "point_multiplicity_complete": True, "verification_input_dir": str(base), "verification_output_dir": str(root), "verifier": "maximum-bipartite-matching bijection with SCC allowed-edge ambiguity analysis", "image_identity_map": {name: [int(left[name].image_id), int(right[name].image_id)] for name in left}, "source_observations_preserved": False, "observation_note": "InterfaceCOLMAP reprojects float32 vertices and resets error to zero; original SIFT coordinates/residuals must be read from the immutable native input, not this roundtrip.", "half_pixel_handling": "InterfaceCOLMAP internally subtracts/adds 0.5; this script applies no additional offset."})


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sparse", required=True)
    p.add_argument("--image-root", required=True)
    p.add_argument("--interface-colmap", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--source-dir", help=argparse.SUPPRESS)
    p.add_argument("--wall-seconds", type=float, default=120)
    p.add_argument("--max-output-mib", type=float, default=200)
    p.add_argument("--internal", choices=("undistort", "verify"), help=argparse.SUPPRESS)
    return p


def main():
    args = parser().parse_args()
    if args.internal:
        (undistort if args.internal == "undistort" else verify_mvs)(args)
        return 0
    require(args.wall_seconds > 0 and args.max_output_mib > 0, "budgets must be positive")
    root = Path(args.output).resolve()
    require(not root.exists(), "output already exists; use a new directory")
    sparse, image_root, executable = (Path(x).resolve() for x in (args.sparse, args.image_root, args.interface_colmap))
    require(sparse.is_dir() and image_root.is_dir() and executable.is_file(), "input directory or InterfaceCOLMAP executable missing")
    root.mkdir(parents=True)
    started = time.monotonic()
    deadline = started + args.wall_seconds
    base = [sys.executable, "-B", str(Path(__file__).resolve()), "--sparse", str(sparse), "--image-root", str(image_root), "--interface-colmap", str(executable), "--output", str(root)]
    stages = [
        ("undistort", base + ["--internal", "undistort"]),
        ("mvs_import", [str(executable), "-i", str(root / "undistorted"), "-o", str(root / "scene.mvs"), "--image-folder", str(root / "undistorted" / "images"), "--max-threads", "2", "--normalize", "0"]),
        ("mvs_export", [str(executable), "-i", str(root / "scene.mvs"), "-o", str(root / "roundtrip") + os.sep, "--image-folder", str(root / "undistorted" / "images"), "--max-threads", "2", "--binary", "1", "--force-points", "1"]),
        ("verify", base + ["--internal", "verify"]),
    ]
    plan = {"created_utc": datetime.now(timezone.utc).isoformat(), "script_sha256": sha(__file__), "tool_sha256": sha(executable), "source_sparse": str(sparse), "source_images": str(image_root), "single_batch": True, "native_and_tool_wall_seconds": args.wall_seconds, "cleanup_seconds": 5, "output_limit_bytes": int(args.max_output_mib * 1024 * 1024), "runtime_env_overrides": THREAD_ENV, "threads": 2, "stages": [{"stage": name, "argv": argv, "cwd": str(root)} for name, argv in stages], "tolerances": {"normalized_rays": 1e-5, "poses_rtol_atol": 1e-8, "pinhole_K_pixels": 1e-7, "xyz": "1e-6 * max(1,max_abs_source_coordinate)", "exported_reprojection_pixels": 1e-4}, "correspondence_rule": "A perfect bijection over the whole point set is required. Point IDs may be renumbered and point geometries may repeat; a unique match is not required, but a one-to-one assignment satisfying XYZ tolerance, identical RGB and identical view-member sets must exist. Ambiguity groups are reported.", "scope": "Format compatibility only. No dense reconstruction, mesh, CAD, absolute scale or accuracy conclusion."}
    save(root / "PLAN.json", plan)
    steps = []
    error = None
    try:
        for name, argv in stages:
            require(time.monotonic() < deadline, "shared deadline exhausted before stage")
            record = {"stage": name, "argv": argv, "cwd": str(root), "started_utc": datetime.now(timezone.utc).isoformat(), "started_monotonic": time.monotonic(), "pid": None, "returncode": None}
            steps.append(record)
            save(root / "STEPS.json", steps)
            with (root / f"{name}.stdout.txt").open("wb") as out, (root / f"{name}.stderr.txt").open("wb") as err:
                process = subprocess.Popen(argv, cwd=root, stdout=out, stderr=err)
                record["pid"] = process.pid
                save(root / "STEPS.json", steps)
                try:
                    while process.poll() is None:
                        require(time.monotonic() < deadline, "shared deadline exceeded")
                        require(sum(p.stat().st_size for p in root.rglob("*") if p.is_file()) <= plan["output_limit_bytes"], "output size exceeded")
                        time.sleep(0.1)
                except BaseException:
                    # The Popen handle is still live and identity-bound here.
                    if os.name == "nt":
                        subprocess.run(["taskkill.exe", "/PID", str(process.pid), "/T", "/F"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=3)
                    else:
                        process.kill()
                    process.wait(timeout=2)
                    raise
                finally:
                    record["returncode"] = process.poll()
                    record["ended_monotonic"] = time.monotonic()
                    record["ended_utc"] = datetime.now(timezone.utc).isoformat()
                    save(root / "STEPS.json", steps)
            require(record["returncode"] == 0, f"stage {name} failed with exit {record['returncode']}")
        require((root / "scene.mvs").is_file(), "MVS scene missing")
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        (root / "ERROR.txt").write_text(traceback.format_exc(), encoding="utf-8")
    input_bindings = root / "INPUT_BINDINGS.json"
    unchanged = None
    if input_bindings.exists():
        unchanged = all(Path(item["path"]).is_file() and sha(item["path"]) == item["sha256"] for item in json.loads(input_bindings.read_text(encoding="utf-8")))
        if not unchanged:
            error = (error + "; " if error else "") + "input bindings changed"
    output_bytes = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    if output_bytes > plan["output_limit_bytes"]:
        error = (error + "; " if error else "") + "output size exceeded"
    save(root / "RESULT.json", {"status": "PASS" if error is None else "FAIL", "error": error, "wall_seconds": time.monotonic() - started, "output_bytes_before_result": output_bytes, "input_bindings_unchanged": unchanged, "native_stages_started": len(steps), "real_reconstruction": "NOT_RUN", "dense_mesh_CAD": "NOT_RUN", "scale_status": "unknown", "physical_length_unit": None})
    return 0 if error is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
