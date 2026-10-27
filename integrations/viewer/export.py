"""Export bounded PLY scenes to one offline HTML file; no geometry is inferred."""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

from integrations.viewer.ply_data import read_ply

ASSETS = Path(__file__).with_name("assets")
PROFILE_LIMITS = {
    "standard": {"max_bytes": 16 * 1024 * 1024, "max_vertices": 100000, "max_faces": 200000},
    "surface400k": {"max_bytes": 16 * 1024 * 1024, "max_vertices": 200000, "max_faces": 400000},
}


def _object(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate JSON key: {key}")
        out[key] = value
    return out


def _text(value, name, limit=1600):
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(f"invalid {name}")
    return value


def load_scenes(spec_path: Path, *, profile: str = "standard"):
    if profile not in PROFILE_LIMITS:
        raise ValueError("unsupported viewer profile")
    limits = PROFILE_LIMITS[profile]
    spec_path = spec_path.resolve()
    if spec_path.stat().st_size > 256 * 1024:
        raise ValueError("scene specification exceeds 256 KiB")
    raw = spec_path.read_bytes()
    spec = json.loads(raw, object_pairs_hook=_object,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    if not isinstance(spec, dict) or set(spec) != {"schema", "scenes"} or spec["schema"] != "ply-viewer-scenes-v1":
        raise ValueError("expected ply-viewer-scenes-v1 specification")
    entries = spec["scenes"]
    if not isinstance(entries, list) or not 1 <= len(entries) <= 8:
        raise ValueError("one to eight scenes required")
    scenes = []
    total_vertices = total_faces = 0
    for entry in entries:
        allowed = {"ply", "sha256", "title", "data_kind", "declared_unit", "note", "evidence"}
        required = {"ply", "sha256", "title", "data_kind", "declared_unit", "note"}
        if not isinstance(entry, dict) or not required <= entry.keys() <= allowed:
            raise ValueError("invalid scene fields")
        path = (spec_path.parent / _text(entry["ply"], "ply path")).resolve()
        expected = _text(entry["sha256"], "source SHA256", 64)
        if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
            raise ValueError("source SHA256 must be 64 lowercase hex digits")
        if entry["data_kind"] not in ("real_photos", "synthetic", "unspecified"):
            raise ValueError("invalid declared data_kind")
        if entry["declared_unit"] not in (None, "mm", "cm", "m", "in"):
            raise ValueError("invalid declared_unit")
        mesh = read_ply(path, **limits)
        if mesh["sha256"] != expected:
            raise ValueError(f"source hash mismatch: {path.name}")
        vertices = mesh["vertices"]
        if not vertices:
            raise ValueError("empty scene")
        total_vertices += len(vertices)
        total_faces += len(mesh["faces"])
        if total_vertices > limits["max_vertices"] or total_faces > limits["max_faces"]:
            raise ValueError("aggregate viewer geometry budget exceeded")
        low = [min(p[a] for p in vertices) for a in range(3)]
        high = [max(p[a] for p in vertices) for a in range(3)]
        extent = [b-a for a,b in zip(low,high)]
        if not all(math.isfinite(x) for x in extent):
            raise ValueError("display extent overflows")
        span = max(extent)
        center = [a+(b-a)*0.5 for a,b in zip(low,high)]
        # These values only set the camera. Original doubles remain in vertices.
        view_scale = 2/span if span else 1.0
        if not math.isfinite(view_scale) or view_scale <= 0:
            raise ValueError("display normalization is not representable")
        evidence = []
        refs = entry.get("evidence", [])
        if not isinstance(refs, list) or len(refs) > 8:
            raise ValueError("invalid evidence references")
        for ref in refs:
            ep = (spec_path.parent / _text(ref, "evidence path")).resolve()
            if ep.stat().st_size > 16*1024*1024:
                raise ValueError("evidence file too large")
            evidence.append({"path": str(ep), "sha256": hashlib.sha256(ep.read_bytes()).hexdigest()})
        scenes.append({**mesh, "source_path": str(path),
                       "title": _text(entry["title"], "title", 160),
                       "data_kind": entry["data_kind"], "declared_unit": entry["declared_unit"],
                       "note": _text(entry["note"], "note"), "evidence": evidence,
                       "bbox": [low, high], "center": center, "view_scale": view_scale})
    return scenes, hashlib.sha256(raw).hexdigest()


def export_html(spec_path: Path, destination: Path, *, profile: str = "standard"):
    if destination.exists():
        raise ValueError("refusing to overwrite viewer")
    scenes, spec_sha = load_scenes(spec_path, profile=profile)
    payload = json.dumps(scenes, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    # JSON is data in a script element, never markup or executable interpolation.
    payload = payload.replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    html = (ASSETS/"viewer.html").read_text(encoding="utf-8")
    replacements = {"/*VIEWER_CSS*/": (ASSETS/"viewer.css").read_text(encoding="utf-8"),
                    "/*SCENE_DATA*/": payload,
                    "/*VIEWER_JS*/": (ASSETS/"viewer.js").read_text(encoding="utf-8")}
    if any(html.count(token) != 1 for token in replacements):
        raise ValueError("invalid viewer template")
    # Substitute the original template once; user text may itself contain tokens.
    html = re.sub(r"/\*(?:VIEWER_CSS|SCENE_DATA|VIEWER_JS)\*/", lambda m: replacements[m[0]], html)
    encoded = html.encode("utf-8")
    if len(encoded) > 32*1024*1024:
        raise ValueError("HTML exceeds 32 MiB")
    with destination.open("xb") as out:
        out.write(encoded)
    return {"schema": "ply-viewer-export-v1", "status": "EXPORTED_NOT_UI_VERIFIED",
            "profile": profile, "geometry_limits": dict(PROFILE_LIMITS[profile]),
            "spec_sha256": spec_sha, "html_sha256": hashlib.sha256(encoded).hexdigest(),
            "html_bytes": len(encoded), "network_dependencies": 0,
            "scenes": [{k: s[k] for k in ("title", "source_path", "sha256", "bytes", "format", "data_kind", "declared_unit", "evidence")} |
                       {"vertices": len(s["vertices"]), "faces": len(s["faces"]), "has_rgb": s["colors"] is not None} for s in scenes],
            "boundary": "Source labels and units are declarations. Evidence files are hashed, not semantically revalidated. No reconstruction, metrology, CAD import or geometry repair is performed."}
