"""Render a compact README preview from the currently opened Blender scene.

Usage::

    blender -b model.blend --python scripts/render_readme_model.py -- output.png

The source ``.blend`` is never saved.  The script only adds temporary camera,
lighting and presentation materials in Blender's in-memory copy.
"""

from __future__ import annotations

import sys
from pathlib import Path

import bpy
from mathutils import Vector


def _look_at(obj: bpy.types.Object, target: Vector) -> None:
    obj.rotation_euler = (target - obj.location).to_track_quat("-Z", "Y").to_euler()


def main() -> None:
    if "--" not in sys.argv or len(sys.argv[sys.argv.index("--") + 1 :]) != 1:
        raise SystemExit("expected one output PNG after --")
    output = Path(sys.argv[sys.argv.index("--") + 1]).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not meshes:
        raise RuntimeError("scene contains no mesh")

    corners = [obj.matrix_world @ Vector(corner) for obj in meshes for corner in obj.bound_box]
    low = Vector(tuple(min(point[i] for point in corners) for i in range(3)))
    high = Vector(tuple(max(point[i] for point in corners) for i in range(3)))
    center = (low + high) / 2
    extent = high - low
    # The bounding sphere, unlike the longest edge alone, also fits diagonal
    # views of nearly cubic objects.
    radius = max(extent.length / 2, 1.0e-6)

    material = bpy.data.materials.new("README cobalt")
    material.use_nodes = True
    shader = material.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = (0.018, 0.12, 0.46, 1.0)
    shader.inputs["Metallic"].default_value = 0.12
    shader.inputs["Roughness"].default_value = 0.34
    for obj in meshes:
        obj.data.materials.clear()
        obj.data.materials.append(material)
        for polygon in obj.data.polygons:
            polygon.use_smooth = True

    camera_data = bpy.data.cameras.new("README camera")
    camera = bpy.data.objects.new("README camera", camera_data)
    bpy.context.scene.collection.objects.link(camera)
    camera.location = center + Vector((1.35, -1.75, 1.05)).normalized() * radius * 3.8
    camera_data.lens = 58
    _look_at(camera, center)
    bpy.context.scene.camera = camera

    def add_area(name: str, direction: tuple[float, float, float], energy: float, size: float) -> None:
        data = bpy.data.lights.new(name, "AREA")
        data.energy = energy
        data.shape = "DISK"
        data.size = size
        light = bpy.data.objects.new(name, data)
        bpy.context.scene.collection.objects.link(light)
        light.location = center + Vector(direction).normalized() * radius * 3.0
        _look_at(light, center)

    add_area("Key", (-1.2, -1.4, 2.0), 550 * radius, radius * 1.8)
    add_area("Fill", (1.7, -0.4, 0.8), 260 * radius, radius * 2.2)
    add_area("Rim", (0.4, 1.5, 1.6), 420 * radius, radius * 1.5)

    world = bpy.context.scene.world or bpy.data.worlds.new("README world")
    bpy.context.scene.world = world
    world.color = (0.012, 0.018, 0.035)

    scene = bpy.context.scene
    # Blender has used both identifiers across releases/builds.  Probe the
    # current enum instead of tying README generation to one executable.
    try:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    except TypeError:
        scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = 960
    scene.render.resolution_y = 720
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = False
    scene.render.filepath = str(output)
    scene.render.image_settings.color_depth = "8"
    bpy.ops.render.render(write_still=True)
    print(f"README_RENDER={output}")


if __name__ == "__main__":
    main()
