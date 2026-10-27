"""One atomic round of conforming midpoint subdivision, without mesh filtering.

Only ``refine`` chooses edges to split. Every incident face uses those same
midpoints, including ineligible closure faces. Children inherit their input
face's eligibility and root parent ID; no observation or deletion vote is
inferred. Midpoints and their RGB values are interpolated, not measured.
"""
from __future__ import annotations

import numpy as np


def _positive_limit(value: int, name: str) -> int:
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer)) or value < 1):
        raise ValueError(name + ' must be a positive integer')
    return int(value)


def subdivide_conforming(
    vertices: np.ndarray,
    faces: np.ndarray,
    parent_ids: np.ndarray,
    eligible: np.ndarray,
    refine: np.ndarray,
    *,
    colors: np.ndarray | None = None,
    max_faces: int = 400_000,
    max_vertices: int = 200_000,
) -> dict:
    """Return a deterministic conforming 0/1/2/3-edge subdivision round.

    Output faces follow input-face order and preserve winding. All input
    vertices remain in order; new IDs follow lexicographically sorted endpoint
    pairs. ``midpoint_lineage`` has columns [new_id, endpoint_0, endpoint_1],
    referring to this round's input vertices. ``source_face_indices`` identifies
    this round's input faces, whereas ``parent_ids`` preserves caller root IDs.

    RGB midpoint channels use integer nearest rounding, with .5 rounded up.
    ``refine`` and ``eligible`` are independent: even an explicitly refined
    ineligible face stays ineligible. Rebuild the current edge map on each call,
    so green children can be refined in later rounds without hanging nodes.
    Output budgets include closure faces and are checked before output creation.
    Inputs are never modified or returned as writable aliases. This operation
    does not certify manifoldness or geometric quality of the input mesh.
    """
    max_faces = _positive_limit(max_faces, 'max_faces')
    max_vertices = _positive_limit(max_vertices, 'max_vertices')
    raw_v = np.asarray(vertices)
    f = np.asarray(faces)
    parents = np.asarray(parent_ids)
    allowed = np.asarray(eligible)
    selected = np.asarray(refine)
    if (raw_v.ndim != 2 or raw_v.shape[1:] != (3,)
            or not np.issubdtype(raw_v.dtype, np.number)
            or np.issubdtype(raw_v.dtype, np.complexfloating)):
        raise ValueError('finite real Nx3 vertices required')
    with np.errstate(over='ignore', invalid='ignore'):
        v = raw_v.astype(np.float64, copy=False)
    if not np.isfinite(v).all():
        raise ValueError('finite real Nx3 vertices required')
    if (f.ndim != 2 or f.shape[1:] != (3,)
            or not np.issubdtype(f.dtype, np.integer)
            or (f.size and (f.min() < 0 or f.max() >= len(v)))):
        raise ValueError('valid integer Mx3 face indices required')
    if np.any((f[:, 0] == f[:, 1]) | (f[:, 1] == f[:, 2]) | (f[:, 2] == f[:, 0])):
        raise ValueError('face vertex indices must be distinct')
    f = f.astype(np.int64, copy=False)
    n, m = len(v), len(f)
    if (parents.shape != (m,) or not np.issubdtype(parents.dtype, np.integer)
            or (parents.size and (parents.min() < 0 or parents.max() > np.iinfo(np.int64).max))):
        raise ValueError('one nonnegative int64-compatible parent ID per face required')
    parents = parents.astype(np.int64, copy=False)
    if allowed.dtype != np.bool_ or allowed.shape != (m,):
        raise ValueError('one boolean eligible value per face required')
    if selected.dtype != np.bool_ or selected.shape != (m,):
        raise ValueError('one boolean refine value per face required')
    rgb = None if colors is None else np.asarray(colors)
    if rgb is not None and (rgb.dtype != np.uint8 or rgb.shape != (n, 3)):
        raise ValueError('optional colors must be uint8 Nx3 RGB')
    if n > max_vertices or m > max_faces:
        raise ValueError('subdivision budget exceeded by input mesh')

    # The inverse map includes every face, not only selected/eligible faces.
    # Thus all incidents of a shared edge (also nonmanifold ones) split together.
    if selected.any():
        face_edges = np.sort(f[:, ((0, 1), (1, 2), (2, 0))], axis=2)
        edges, inverse = np.unique(face_edges.reshape(-1, 2), axis=0, return_inverse=True)
        inverse = inverse.reshape(m, 3)
        split_ids = np.unique(inverse[selected])
        split_edges = edges[split_ids]
        edge_midpoints = np.full(len(edges), -1, dtype=np.int64)
        edge_midpoints[split_ids] = n + np.arange(len(split_ids), dtype=np.int64)
        mids = edge_midpoints[inverse]
    else:
        split_edges = np.empty((0, 2), dtype=np.int64)
        mids = np.full((m, 3), -1, dtype=np.int64)
    split_counts = np.count_nonzero(mids >= 0, axis=1)
    child_counts = split_counts + 1
    output_face_count = int(child_counts.sum())
    output_vertex_count = n + len(split_edges)
    if output_face_count > max_faces or output_vertex_count > max_vertices:
        raise ValueError(
            f'subdivision budget exceeded: faces={output_face_count}/{max_faces}, '
            f'vertices={output_vertex_count}/{max_vertices}'
        )

    # Outputs are allocated only after the complete closure budget is known.
    new_v = np.empty((output_vertex_count, 3), dtype=np.float64)
    new_v[:n] = v
    new_v[n:] = v[split_edges[:, 0]] * .5 + v[split_edges[:, 1]] * .5
    new_rgb = None
    if rgb is not None:
        new_rgb = np.empty((output_vertex_count, 3), dtype=np.uint8)
        new_rgb[:n] = rgb
        sums = rgb[split_edges[:, 0]].astype(np.uint16) + rgb[split_edges[:, 1]].astype(np.uint16)
        new_rgb[n:] = ((sums + 1) // 2).astype(np.uint8)
    new_f = np.empty((output_face_count, 3), dtype=np.int64)
    cursor = 0
    for face, mid, count in zip(f, mids, split_counts):
        a, b, c = face
        ab, bc, ca = mid
        if count == 0:
            children = ((a, b, c),)
        elif count == 1:
            j = int(np.flatnonzero(mid >= 0)[0])
            a, b, c = face[j], face[(j + 1) % 3], face[(j + 2) % 3]
            ab = mid[j]
            children = ((a, ab, c), (ab, b, c))
        elif count == 2:
            # Rotate so b--c is the unsplit edge. The remaining quadrilateral
            # always uses the fixed diagonal ab--c; all rotations keep winding.
            j = int(np.flatnonzero(mid < 0)[0])
            a, b, c = face[(j + 2) % 3], face[j], face[(j + 1) % 3]
            ab, ca = mid[(j + 2) % 3], mid[(j + 1) % 3]
            children = ((a, ab, ca), (ab, b, c), (ab, c, ca))
        else:
            children = ((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca))
        new_f[cursor:cursor + len(children)] = children
        cursor += len(children)

    return dict(
        vertices=new_v,
        faces=new_f,
        colors=new_rgb,
        parent_ids=np.repeat(parents, child_counts),
        eligible=np.repeat(allowed, child_counts),
        source_face_indices=np.repeat(np.arange(m, dtype=np.int64), child_counts),
        midpoint_lineage=np.column_stack((
            np.arange(n, output_vertex_count, dtype=np.int64), split_edges,
        )),
        midpoint_kind='INTERPOLATED_NOT_MEASURED',
        geometry_accepted=False,
    )
