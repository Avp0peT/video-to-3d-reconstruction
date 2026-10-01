"""Synthetic dataset generation (README section 13).

Public entry points:

* :func:`~kozlov_lab.datasets.shapes.build_shape` / ``SHAPES`` — the shape
  registry used by the GUI's "generate an example" buttons;
* :func:`~kozlov_lab.datasets.shapes.random_points` and friends — raw
  generators;
* :mod:`kozlov_lab.datasets.corpus` — the *corpus* layer that turns shapes
  into the labelled experiment sets of Experiments A–E (added in Phase 8).
"""

from __future__ import annotations

from .shapes import (
    SHAPES,
    asymmetric_shape,
    build_shape,
    circle,
    collinear_chain,
    cube_3d,
    grid,
    letter_l,
    near_collinear_triangle,
    parallelogram,
    pentagon,
    pyramid_3d,
    random_points,
    rectangle,
    shape_names,
    square,
    star,
    triangle,
    triangle_demo,
)

__all__ = [
    "SHAPES",
    "shape_names",
    "build_shape",
    "triangle_demo",
    "triangle",
    "square",
    "rectangle",
    "parallelogram",
    "pentagon",
    "star",
    "letter_l",
    "grid",
    "circle",
    "random_points",
    "asymmetric_shape",
    "collinear_chain",
    "near_collinear_triangle",
    "cube_3d",
    "pyramid_3d",
]
