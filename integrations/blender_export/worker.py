"""Blender-only worker. No geometry repair or object-space normalization."""
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

import bpy
from mathutils import Matrix, Vector


def boundary(faces):
    counts = {}
    for face in faces:
        for a, b in zip(face, face[1:] + face[:1]):
            key = tuple(sorted((a, b)))
            counts[key] = counts.get(key, 0) + 1
    return sum(count == 1 for count in counts.values())


def metadata(scene):
    return {"title": scene["title"], "source_path": scene["source_path"], "source_sha256": scene["sha256"],
            "data_kind": scene["data_kind"], "declared_unit": scene["declared_unit"] or "unknown",
            "scale_status": "declared_unverified" if scene["declared_unit"] else "unknown",
            "note": scene["note"], "evidence": json.dumps(scene["evidence"], ensure_ascii=False),
            "geometry_kind": "surface" if scene["faces"] else "pointcloud",
            "appearance": "solid_color_no_photo_texture"}


def check(scenes):
    if len(bpy.context.scene.objects) != len(scenes):
        raise ValueError("unexpected scene objects")
    result = []
    for index, source in enumerate(scenes):
        obj = bpy.data.objects.get(f"PLY_{index + 1:02d}")
        if obj is None or obj.type != "MESH" or obj.modifiers or obj.matrix_world != Matrix.Identity(4):
            raise ValueError("object missing, transformed or modified")
        if any(obj.get(key) != value for key, value in metadata(source).items()):
            raise ValueError("source or scale metadata changed")
        mesh = obj.data
        if len(mesh.vertices) != len(source["vertices"]):
            raise ValueError("vertex count changed")
        error = 0.0
        for vertex, original in zip(mesh.vertices, source["vertices"]):
            for actual, value in zip(vertex.co, original):
                represented = struct.unpack("f", struct.pack("f", value))[0]
                if not math.isfinite(actual) or actual != represented:
                    raise ValueError("coordinate differs from source float32 representation")
                error = max(error, abs(actual - value))
        faces = [list(p.vertices) for p in mesh.polygons]
        if faces != source["faces"]:
            raise ValueError("face order, indices or winding changed")
        edge_boundary = boundary(faces)
        if edge_boundary != boundary(source["faces"]):
            raise ValueError("boundary changed")
        result.append({"title": source["title"], "vertices": len(mesh.vertices), "faces": len(faces),
                       "kind": obj["geometry_kind"], "boundary_edges": edge_boundary,
                       "max_absolute_float32_error": error, "source_sha256": source["sha256"]})
    return result


def main():
    mode, payload_path, expected, blend_path, report_path = sys.argv[sys.argv.index("--") + 1:]
    raw = Path(payload_path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("payload hash changed")
    scenes = json.loads(raw)["scenes"]
    if mode == "create":
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete(use_global=False)
        for index, source in enumerate(scenes):
            name = f"PLY_{index + 1:02d}"
            mesh = bpy.data.meshes.new(name)
            mesh.from_pydata(source["vertices"], [], source["faces"])
            mesh.update()
            obj = bpy.data.objects.new(name, mesh)
            bpy.context.collection.objects.link(obj)
            for key, value in metadata(source).items():
                obj[key] = value
            material = bpy.data.materials.new(name + "_solid")
            material.diffuse_color = (0.32, 0.62, 0.78, 1)
            material.use_nodes = True
            principled = next((node for node in material.node_tree.nodes if node.type == "BSDF_PRINCIPLED"), None)
            if principled is None:
                raise ValueError("solid material has no Principled BSDF")
            principled.inputs["Base Color"].default_value = material.diffuse_color
            obj.data.materials.append(material)
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj
        lows = [min(s["bbox"][0][axis] for s in scenes) for axis in range(3)]
        highs = [max(s["bbox"][1][axis] for s in scenes) for axis in range(3)]
        center = Vector([a + (b-a)*0.5 for a, b in zip(lows, highs)])
        distance = max(max(b-a for a, b in zip(lows, highs))*2, 0.01)
        for screen in bpy.data.screens:
            for area in screen.areas:
                if area.type == "VIEW_3D":
                    area.spaces.active.region_3d.view_location = center
                    area.spaces.active.region_3d.view_distance = distance
                    area.spaces.active.clip_end = max(distance*100, 1000)
        checks = check(scenes)
        bpy.ops.wm.save_as_mainfile(filepath=blend_path, check_existing=True)
    elif mode == "verify":
        bpy.ops.wm.open_mainfile(filepath=blend_path, use_scripts=False)
        checks = check(scenes)
    else:
        raise ValueError("unknown worker mode")
    with Path(report_path).open("x", encoding="utf-8") as stream:
        json.dump({"status": "PASS", "payload_sha256": expected, "scenes": checks,
                   "blender_version": bpy.app.version_string,
                   "precision": "float32 coordinate representation; not double lossless"}, stream,
                  ensure_ascii=False, allow_nan=False, indent=2)


if __name__ == "__main__":
    main()
