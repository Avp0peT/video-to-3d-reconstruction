"""Shared fixtures for the kozlov_lab test suite."""

from __future__ import annotations

import math

import numpy as np
import pytest

from kozlov_lab.datasets import shapes
from kozlov_lab.geometry import AffineFrame, AffineTransform, PointSet2D


# --------------------------------------------------------------------------- #
# point sets
# --------------------------------------------------------------------------- #
@pytest.fixture
def demo_set() -> PointSet2D:
    """The 4-point example from README section 4."""
    return shapes.triangle_demo()


@pytest.fixture
def triangle_set() -> PointSet2D:
    return shapes.triangle()


@pytest.fixture
def square_set() -> PointSet2D:
    """Square with 3 samples per side -> contains collinear triples and parallel pairs.

    Chosen on purpose: a 4-corner square has *no* collinear triple, so the
    collinearity row of the invariant table would be vacuous on it.
    """
    return shapes.square(2.0, n_per_side=3)


@pytest.fixture
def grid_set() -> PointSet2D:
    return shapes.grid(4, 4)


@pytest.fixture
def star_set() -> PointSet2D:
    return shapes.star(5, 1.0, 0.4)


@pytest.fixture
def collinear_set() -> PointSet2D:
    return shapes.collinear_chain(5)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20261001)


# --------------------------------------------------------------------------- #
# transforms
# --------------------------------------------------------------------------- #
@pytest.fixture
def transform_bank() -> dict[str, AffineTransform]:
    """One representative per transformation family (README section 5)."""
    return {
        "identity": AffineTransform.identity(),
        "translation": AffineTransform.translation_of(3.5, -2.25),
        "rotation": AffineTransform.rotation(37.0),
        "rotation_about_center": AffineTransform.rotation(37.0, center=[1.0, 1.0]),
        "reflection": AffineTransform(
            np.array([[math.cos(2 * 0.7), math.sin(2 * 0.7)], [math.sin(2 * 0.7), -math.cos(2 * 0.7)]]),
            np.zeros(2),
        ),
        "similarity": AffineTransform.similarity(2.5, 41.0),
        "scale_nonuniform": AffineTransform.scaling(2.0, 0.5),
        "shear": AffineTransform.shear(0.6, 0.0),
        "shear_both": AffineTransform.shear(0.5, -0.3),
        "affine_general": AffineTransform(
            np.array([[1.3, 0.45], [-0.2, 0.9]]), np.array([-1.0, 2.0])
        ),
        "singular": AffineTransform(np.array([[1.0, 2.0], [2.0, 4.0]]), np.zeros(2)),
    }


@pytest.fixture
def well_conditioned_frame(square_set: PointSet2D) -> AffineFrame:
    """A frame whose condition number is small, so 1e-9-level claims make sense."""
    frame = AffineFrame.from_indices(square_set.as_array(), (0, 3, 5))
    assert frame.is_valid()
    assert frame.condition_number < 10.0, "fixture must stay well conditioned"
    return frame
