"""Bounded, read-only mesh/dense support diagnostics; never an acceptance gate.

All nearest searches use the complete stored dense cloud. Only fixed face/edge
samples are queried. Proximity cannot establish visibility or physical identity.
"""
from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from integrations.viewer import ply_data

SCHEMA = 'region-surface-support-v1'
LIMITS = dict(max_bytes=16 << 20, max_vertices=200000, max_faces=400000)
BARYCENTRIC = np.array([[1/3, 1/3, 1/3], [.6, .2, .2], [.2, .6, .2], [.2, .2, .6]])
LIMITATION = ('Fixed samples only; ratios 3/5/10 are descriptive probes, not calibrated error, '
              'removal or acceptance thresholds. Dense points are estimates, not ground truth. '
              'No visibility, physical identity, self-intersection, dimensional accuracy or '
              'complete-surface acceptance. No vertices or faces are changed.')


def _need(ok, message):
    if not ok:
        raise ValueError(message)


def _xyz(value, name):
    if not isinstance(value, np.ndarray):
        _need(not any(isinstance(x, (bool, np.bool_)) for row in value for x in row), name+' contains bool')
    array = np.asarray(value)
    if array.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    _need(array.ndim == 2 and array.shape[1] == 3 and array.dtype.kind in 'iuf', name+' requires numeric Nx3')
    array = np.asarray(array, dtype=np.float64)
    _need(len(array) <= LIMITS['max_vertices'] and np.isfinite(array).all(), name+' exceeds count budget or is nonfinite')
    # Avoid overflow in squared Euclidean distances while retaining arbitrary scale.
    _need(np.max(np.abs(array), initial=0) < 1e100, name+' exceeds numeric range')
    return array


def _arrays(vertices, faces, dense):
    v, d = _xyz(vertices, 'vertices'), _xyz(dense, 'dense')
    f = np.asarray(faces)
    if f.size == 0:
        f = np.empty((0, 3), dtype=np.int64)
    _need(f.ndim == 2 and f.shape[1] == 3 and f.dtype.kind in 'iu', 'faces require integer Nx3')
    _need(len(f) <= LIMITS['max_faces'], 'face count budget exceeded')
    _need(not len(f) or (f.min() >= 0 and f.max() < len(v)), 'face index outside vertices')
    return v, np.asarray(f, dtype=np.int64), d


def _nearest(queries, dense, k=1):
    """Exact distances, at most 8 query rows per temporary block; no ANN/GPU."""
    values, indices = [], []
    for start in range(0, len(queries), 8):
        delta = queries[start:start+8, None, :] - dense[None, :, :]
        square = np.einsum('qpi,qpi->qp', delta, delta)
        if k == 1:
            ix = square.argmin(axis=1)
            values.extend(np.sqrt(square[np.arange(len(ix)), ix])); indices.extend(ix)
        else:
            ix = np.argpartition(square, k-1, axis=1)[:, :k]
            distance = np.take_along_axis(square, ix, axis=1)
            order = np.argsort(distance, axis=1)
            values.extend(np.sqrt(np.take_along_axis(distance, order, axis=1)))
            indices.extend(np.take_along_axis(ix, order, axis=1))
    return np.asarray(values), np.asarray(indices, dtype=np.int64)


def _query_support(queries, dense, duplicate_count):
    if not len(queries):
        return []
    if not len(dense):
        return [dict(distance=None, local_spacing=None, ratio=None, reason='NO_DENSE_POINTS') for _ in queries]
    distance, nearest = _nearest(queries, dense)
    reason = ('DUPLICATE_DENSE_POINTS' if duplicate_count else
              'INSUFFICIENT_DENSE_POINTS' if len(dense) < 7 else None)
    if reason:
        return [dict(distance=float(x), local_spacing=None, ratio=None, reason=reason) for x in distance]
    unique, reverse = np.unique(nearest, return_inverse=True)
    neighborhood, _ = _nearest(dense[unique], dense, k=7)
    spacing = np.median(neighborhood[:, 1:], axis=1)[reverse]
    return [dict(distance=float(x), local_spacing=float(s) if s > 0 else None,
                 ratio=float(x/s) if s > 0 and np.isfinite(x/s) else None,
                 reason=None if s > 0 and np.isfinite(x/s) else 'ZERO_LOCAL_SPACING')
            for x, s in zip(distance, spacing)]


def _quantiles(values):
    return None if not len(values) else {str(p): float(x) for p, x in
        zip((0, 50, 95, 99, 100), np.percentile(values, (0, 50, 95, 99, 100)))}


def _summary(records):
    samples = [s for record in records for s in record['samples']]
    ratios = [s['ratio'] for s in samples if s['ratio'] is not None]
    return dict(samples=len(samples), valid_ratios=len(ratios), unknown_ratios=len(samples)-len(ratios),
                ratio_quantiles=_quantiles(ratios),
                descriptive_counts_above={str(t): sum(x > t for x in ratios) for t in (3, 5, 10)})


def sampled_support(vertices, faces, dense):
    """Return deterministic diagnostics without changing arrays or launching tools."""
    v, f, d = _arrays(vertices, faces, dense)
    tri = v[f]
    areas = .5*np.linalg.norm(np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]), axis=1)
    edges = np.unique(np.sort(np.concatenate((f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]])), axis=1), axis=0)
    lengths = np.linalg.norm(v[edges[:, 1]]-v[edges[:, 0]], axis=1)
    order = np.argsort(-areas, kind='stable')
    n = min(32, len(f))
    largest = order[:n]
    control = np.arange(n, dtype=np.int64)*len(f)//n if n else np.array([], dtype=np.int64)
    selected = np.concatenate((largest, control))
    edge_ids = np.argsort(-lengths, kind='stable')[:32]
    duplicate_count = len(d)-len(np.unique(d, axis=0))
    points = (BARYCENTRIC[None, :, :] @ tri[selected]).reshape(-1, 3)
    mids = v[edges[edge_ids]].mean(axis=1)
    support = _query_support(np.concatenate((points, mids)), d, duplicate_count)
    sampled_faces = [dict(group='largest' if j < n else 'control', face_index=int(fi),
        vertex_indices=f[fi].tolist(), area=float(areas[fi]),
        samples=support[j*4:j*4+4]) for j, fi in enumerate(selected)]
    sampled_edges = [dict(edge_index=int(ei), vertex_indices=edges[ei].tolist(), length=float(lengths[ei]),
        samples=[support[len(points)+j]]) for j, ei in enumerate(edge_ids)]
    top_count = (len(f)+99)//100
    total = float(areas.sum())
    return dict(schema=SCHEMA, status='DIAGNOSTIC_ONLY', geometry_accepted=False,
        counts=dict(vertices=len(v), faces=len(f), dense_points=len(d), unique_edges=len(edges),
                    duplicate_dense_points=duplicate_count, degenerate_faces=int((areas == 0).sum())),
        sampling=dict(largest_faces=n, control_faces=n, longest_edges=len(edge_ids), samples_per_face=4,
                      barycentric=BARYCENTRIC.tolist(), control_rule='floor(i*N/min(32,N)); groups can overlap',
                      nearest_search='Complete stored dense cloud; exact float64 distance; blocks of 8 queries',
                      local_spacing='Median of 6 nearest neighbors of the nearest dense sample, self excluded'),
        area_quantiles=_quantiles(areas), edge_length_quantiles=_quantiles(lengths),
        largest_one_percent=dict(count=top_count, area_fraction=float(areas[order[:top_count]].sum()/total) if total > 0 else None,
                                 reason=None if total > 0 else 'ZERO_TOTAL_AREA'),
        sampled_faces=sampled_faces, sampled_edges=sampled_edges,
        summaries=dict(largest=_summary(sampled_faces[:n]), control=_summary(sampled_faces[n:]), edges=_summary(sampled_edges)),
        limitations=LIMITATION, scale='unknown; arbitrary reconstruction units')


def _binding(path):
    path = Path(path).resolve()
    _need(path.is_file() and path.stat().st_size <= LIMITS['max_bytes'], 'missing source or byte budget exceeded: '+str(path))
    data = path.read_bytes()
    return dict(path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def _json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            _need(key not in result, 'duplicate JSON field')
            result[key] = value
        return result
    return json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def _preview(path, vertices, faces, dense, diagnostic):
    image = Image.new('RGB', (1500, 600), 'white'); draw = ImageDraw.Draw(image)
    draw.text((10, 10), 'DIAGNOSTIC ONLY | largest faces RED; uniform controls BLUE | not confirmed errors', fill='black')
    v, f, d = _arrays(vertices, faces, dense)
    stride = max(1, (len(d)+29999)//30000)
    if len(v):
        for panel, (a, b, name) in enumerate(((0, 1, 'XY'), (0, 2, 'XZ'), (1, 2, 'YZ'))):
            low, high = v[:, [a, b]].min(axis=0), v[:, [a, b]].max(axis=0)
            span = float(np.max(high-low)); scale = 450/span if span > 0 else 1
            def project(x):
                return (x[:, [a, b]]-(low+high)/2)*scale*[1, -1]+[panel*500+250, 310]
            for x, y in project(d[::stride]):
                if np.isfinite(x+y) and panel*500 <= x < (panel+1)*500 and 45 <= y < 570:
                    draw.point((float(x), float(y)), fill=(185, 185, 185))
            for group, color in (('control', (20, 90, 230)), ('largest', (220, 30, 30))):
                for item in diagnostic['sampled_faces']:
                    if item['group'] == group:
                        xy = project(v[f[item['face_index']]])
                        draw.line([tuple(p) for p in xy]+[tuple(xy[0])], fill=color, width=2)
            draw.text((panel*500+15, 35), name+' projection; no visibility test', fill='black')
    draw.text((10, 575), 'Gray: sampled dense points. Projection overlap does not prove physical contact.', fill='black')
    with path.open('xb') as stream:
        image.save(stream, format='PNG')


def run(surface_run, output):
    """Bind raw mesh and dense from one completed producer; publish a new report."""
    source = Path(surface_run).resolve()
    producer = source if source.is_file() else source/'surface/result.json'
    root = producer.parent.parent if source.is_file() else source
    output = Path(output).resolve()
    _need(not output.exists() and output != root and not root.is_relative_to(output), 'output must be a fresh directory')
    _need(not output.is_relative_to(producer.parent), 'output must not be inside native input directory')
    refs = [_binding(producer), _binding(__file__), _binding(ply_data.__file__)]
    document = _json(producer)
    _need(document.get('contract_version') == 'mvs-surface-run-v1' and
          document.get('status') == 'SURFACE_GENERATED_NOT_ACCEPTED', 'completed surface producer required')
    found = {}
    for role, stage_name in (('dense', 'dense'), ('pre_cleanup_surface', 'mesh')):
        rows = [r for s in document.get('stages', []) if s.get('name') == stage_name
                and s.get('status') == 'GEOMETRY_READBACK_PASSED'
                for r in s.get('outputs', {}).get('geometry', []) if r.get('role') == role]
        _need(len(rows) == 1, 'unique completed producer role required: '+role)
        rec = rows[0]; path = Path(rec['path'])
        _need(path.is_absolute() and not path.resolve().is_relative_to(output), 'invalid input/output path')
        actual = _binding(path)
        _need(actual['bytes'] == rec.get('bytes') and actual['sha256'] == rec.get('sha256'), 'source binding mismatch')
        data = ply_data.read_ply(path, **LIMITS)
        _need(data['sha256'] == actual['sha256'] and len(data['vertices']) == rec.get('vertices')
              and len(data['faces']) == rec.get('triangles'), 'producer count or byte mismatch')
        refs.append(actual); found[role] = data
    _need(not found['dense']['faces'], 'dense source must be a point cloud')
    mesh, dense = found['pre_cleanup_surface'], found['dense']
    diagnostic = sampled_support(mesh['vertices'], mesh['faces'], dense['vertices'])
    _need(all(_binding(r['path']) == r for r in refs), 'input drift during diagnostic')
    output.mkdir(parents=True, exist_ok=False)
    diagnostic.update(source_run=str(root), producer_record=str(producer), source_bindings=refs,
                      source_unchanged=True, native_launches=0, gpu_launches=0)
    _preview(output/'candidates.png', mesh['vertices'], mesh['faces'], dense['vertices'], diagnostic)
    _need(all(_binding(r['path']) == r for r in refs), 'input drift during report generation')
    diagnostic['artifacts'] = dict(image=_binding(output/'candidates.png'), html=str(output/'index.html'))
    content = ('<!doctype html><meta charset="utf-8"><title>Surface support diagnostic</title>'
        '<h1>DIAGNOSTIC_ONLY — 表面支持疑点</h1><p>红色：最多32个最大面；蓝色：最多32个均匀对照面。'
        '每面4个内部样本，另检查最多32条长边中点。最近距离查询使用完整稠密点集。</p>'
        '<p>阈值仅作描述，不是验收标准；疑点不等于错误。没有删除、平滑或修复网格。</p>'
        '<p>'+html.escape(LIMITATION)+'</p><img style="max-width:100%" src="candidates.png" alt="候选面投影，不含遮挡判断">'
        '<h2>统计（空值表示未知）</h2><pre>'+html.escape(json.dumps(diagnostic['summaries'], ensure_ascii=False, indent=2))+
        '</pre><p><a href="RESULT.json">完整结果、空值原因和字节绑定</a></p>')
    with (output/'index.html').open('x', encoding='utf-8') as stream:
        stream.write(content)
    with (output/'RESULT.json').open('x', encoding='utf-8') as stream:
        json.dump(diagnostic, stream, ensure_ascii=False, allow_nan=False, indent=2)
    return diagnostic
