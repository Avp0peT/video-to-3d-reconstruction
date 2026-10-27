"""Conservative local linear interpolation; faces do not observe interior detail.

No hole filling is performed. Only bounded Delaunay candidates in one fixed
anchor image per spatial component are considered, then checked in every common
view. Input points are never removed or modified.
"""
from collections import Counter
import numpy as np


def _delaunay(xy):
    """Deterministic bounded Bowyer-Watson (caller limits input to 1000)."""
    n = len(xy)
    if n < 3:
        return []
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    span = float(np.max(hi - lo))
    if span <= 0:
        return []
    # Normalize so determinant tolerances do not depend on pixel magnitude.
    p = (xy - (lo + hi) / 2) / span
    p = np.vstack((p, [[-32., -16.], [32., -16.], [0., 32.]]))
    triangles = [(n, n + 1, n + 2)]
    for i in range(n):
        bad = []
        for t in triangles:
            q = p[list(t)] - p[i]
            det = np.linalg.det(np.column_stack((q, np.sum(q*q, axis=1))))
            a, b, c = p[list(t)]
            orient = np.linalg.det(np.stack((b-a, c-a)))
            if det * np.sign(orient) > 1e-12:
                bad.append(t)
        edges = Counter(tuple(sorted(e)) for t in bad
                        for e in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])))
        bad_set = set(bad)
        triangles = [t for t in triangles if t not in bad_set]
        for (a, b), count in sorted(edges.items()):
            if count == 1:
                cross = np.linalg.det(np.stack((p[b]-p[a], p[i]-p[a])))
                if abs(cross) > 1e-12:
                    triangles.append((a, b, i) if cross > 0 else (b, a, i))
    return sorted({tuple(sorted(t)) for t in triangles if max(t) < n})


def build_faces(points, max_edge_px=60., max_edge_factor=3.):
    """Return ``(faces, face_support, diagnostics)`` using original point indices.

    Components larger than 1000 points are refused intact, never split into
    arbitrary chunks. Invalid input raises ValueError before candidate generation.
    """
    if not (np.isfinite(max_edge_px) and max_edge_px > 0 and
            np.isfinite(max_edge_factor) and max_edge_factor > 0):
        raise ValueError('edge limits must be finite and positive')
    xyz, observations, ids = [], [], set()
    for point in points:
        pid = point['point_id']
        if type(pid) is not int or pid in ids:
            raise ValueError('point_id must be a unique plain int')
        ids.add(pid)
        q = np.asarray(point['xyz'], dtype=float)
        if q.shape != (3,) or not np.isfinite(q).all():
            raise ValueError('xyz must be finite 3-vectors')
        xyz.append(q)
        obs = {}
        for observation in point['observations']:
            image_id = observation['image_id']
            if type(image_id) is not int:
                raise ValueError('image_id must be a plain int')
            pixels = np.asarray(observation['raw_pixels'], dtype=float)
            if pixels.shape != (2,) or not np.isfinite(pixels).all():
                raise ValueError('raw_pixels must be finite 2-vectors')
            if image_id in obs:
                raise ValueError('duplicate image observation')
            obs[image_id] = pixels
        observations.append(obs)
    n = len(xyz)
    refused = Counter()
    diag = dict(method='LOCAL_LINEAR_INTERPOLATION', interior_detail_observed=False,
                input_point_count=n, refused_reasons=refused, boundary_edges=[],
                components=[], max_edge_3d=None)
    if n < 3:
        return [], [], diag
    xyz = np.asarray(xyz)
    # Row-wise distances avoid an unbounded N x N allocation.
    nearest = np.array([np.min(np.linalg.norm(np.delete(xyz, i, axis=0)-q, axis=1))
                        for i, q in enumerate(xyz)])
    limit = float(np.median(nearest) * max_edge_factor)
    diag['max_edge_3d'] = limit
    if limit <= 0:
        refused['zero_nearest_neighbor_scale'] += 1
        return [], [], diag
    remaining = set(range(n))
    faces, support, edge_counts, seen = [], [], Counter(), set()
    while remaining:
        seed = min(remaining)
        remaining.remove(seed)
        component, queue = [seed], [seed]
        while queue:
            i = queue.pop()
            neighbors = [j for j in sorted(remaining)
                         if np.linalg.norm(xyz[i]-xyz[j]) <= limit]
            remaining.difference_update(neighbors)
            component.extend(neighbors)
            queue.extend(neighbors)
        component.sort()
        record = dict(point_indices=component, anchor_image_id=None)
        diag['components'].append(record)
        if len(component) > 1000:
            refused['component_over_1000'] += 1
            continue
        counts = Counter(k for i in component for k in observations[i])
        if not counts:
            refused['no_anchor'] += 1
            continue
        anchor = sorted(counts, key=lambda k: (-counts[k], str(k)))[0]
        record['anchor_image_id'] = anchor
        visible = [i for i in component if anchor in observations[i]]
        record['anchor_visible_point_indices'] = visible
        xy = np.array([observations[i][anchor] for i in visible])
        if len(visible) < 3:
            continue
        # Coincident projections cannot safely select a depth layer.
        unique, multiplicity = np.unique(xy, axis=0, return_counts=True)
        duplicate = {tuple(q) for q, c in zip(unique, multiplicity) if c > 1}
        candidates = _delaunay(xy)
        for candidate in candidates:
            a, b, c = xy[list(candidate)]
            if np.linalg.det(np.stack((b-a, c-a))) < 0:
                candidate = (candidate[0], candidate[2], candidate[1])
            face = tuple(visible[i] for i in candidate)
            if any(tuple(xy[i]) in duplicate for i in candidate):
                refused['coincident_anchor_projection'] += 1
                continue
            common = set.intersection(*(set(observations[i]) for i in face))
            if len(common) < 3:
                refused['fewer_than_three_common_views'] += 1
                continue
            edges = [tuple(sorted((face[a], face[b]))) for a, b in ((0,1),(1,2),(2,0))]
            if any(np.linalg.norm(xyz[a]-xyz[b]) > limit for a,b in edges):
                refused['long_3d_edge'] += 1
                continue
            valid = True
            for image_id in common:
                q = np.array([observations[i][image_id] for i in face])
                if any(np.linalg.norm(q[a]-q[b]) > max_edge_px for a,b in ((0,1),(1,2),(2,0))):
                    refused['long_image_edge'] += 1
                    valid = False
                    break
                if abs(np.linalg.det(np.stack((q[1]-q[0], q[2]-q[0])))) <= 1e-8:
                    refused['degenerate_image_triangle'] += 1
                    valid = False
                    break
            if not valid:
                continue
            if np.linalg.norm(np.cross(xyz[face[1]]-xyz[face[0]], xyz[face[2]]-xyz[face[0]])) <= 1e-12 * limit**2:
                refused['degenerate_3d_triangle'] += 1
                continue
            key = tuple(sorted(face))
            if key in seen or any(edge_counts[e] >= 2 for e in edges):
                refused['duplicate_or_nonmanifold'] += 1
                continue
            seen.add(key)
            edge_counts.update(edges)
            faces.append(list(face))
            support.append(sorted(common, key=str))
    diag['boundary_edges'] = [list(e) for e,c in sorted(edge_counts.items()) if c == 1]
    diag['boundary_edge_count'] = len(diag['boundary_edges'])
    diag['refused_reasons'] = dict(refused)
    return faces, support, diag
