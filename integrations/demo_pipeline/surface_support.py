"""Bounded sample-to-dense distances; no geometry acceptance or modification."""
import hashlib
import html
import json
from pathlib import Path
import numpy as np
from integrations.meshprep.ply_io import read_ply as read_mesh
from integrations.viewer.ply_data import read_ply as read_cloud

MAX_OPERATIONS = 200_000_000
LIMITS = ['Distances use unknown native world units, not metres.',
          'Only deterministic sampled face centers and edge midpoints are measured.',
          'Nearest dense distance is not visibility, physical truth or a face deletion gate.',
          'Top one percent is descriptive, not an acceptance threshold. No geometry PASS.']


def _xyz(value, name, limit):
    def has_bool(v):
        if isinstance(v, (bool, np.bool_)):
            return True
        if isinstance(v, (list, tuple)):
            return any(has_bool(item) for item in v)
        return False
    raw = np.asarray(value)
    if has_bool(value) or raw.dtype.kind == 'b' or (raw.dtype.kind == 'O' and any(isinstance(v, (bool, np.bool_)) for v in raw.flat)):
        raise ValueError(name + ' bool coordinates forbidden')
    a = np.asarray(value, dtype=np.float64)
    if a.size == 0:
        a = a.reshape((0, 3))
    if a.ndim != 2 or a.shape[1] != 3 or len(a) > limit or not np.isfinite(a).all():
        raise ValueError(name + ' invalid shape, nonfinite or over count budget')
    return a


def _sample(n, count):
    return np.linspace(0, n - 1, min(n, count), dtype=np.int64) if n else np.empty(0, dtype=np.int64)


def _summary(values):
    a = np.asarray(values).ravel()
    if not len(a):
        return dict(count=0, min=None, p50=None, p90=None, p95=None, p99=None, max=None)
    if not np.isfinite(a).all():
        raise ValueError('nonfinite derived metric')
    return dict(count=len(a), min=float(a.min()), p50=float(np.percentile(a, 50)),
                p90=float(np.percentile(a, 90)), p95=float(np.percentile(a, 95)),
                p99=float(np.percentile(a, 99)), max=float(a.max()))


def _nearest(query, dense, excluded=None):
    # Difference arithmetic avoids cancellation of tiny local distances.
    result = []
    for start in range(0, len(query), 32):
        delta = query[start:start + 32, None, :] - dense[None, :, :]
        squared = np.einsum('ijk,ijk->ij', delta, delta)
        if excluded is not None:
            squared[np.arange(len(delta)), excluded[start:start + len(delta)]] = np.inf
        result.extend(np.sqrt(squared.min(axis=1)).tolist())
    return np.asarray(result)


def diagnose_arrays(dense, vertices, faces):
    dense = _xyz(dense, 'dense', 100000)
    vertices = _xyz(vertices, 'vertices', 200000)
    f = np.asarray(faces)
    if f.size == 0:
        f = np.empty((0, 3), dtype=np.int64)
    if f.ndim != 2 or f.shape[1] != 3 or len(f) > 200000 or f.dtype.kind not in 'iu':
        raise ValueError('faces invalid integer shape or over count budget')
    if f.size and (f.min() < 0 or f.max() >= len(vertices)):
        raise ValueError('face index out of range')
    ordinary = _sample(len(f), 256)
    dense_ids = _sample(len(dense), 256) if len(dense) >= 2 else np.empty(0, dtype=np.int64)
    top_count = (len(f) + 99) // 100
    query_count = 4 * (len(ordinary) + min(top_count, 128)) + len(dense_ids)
    operations = query_count * len(dense)
    if operations > MAX_OPERATIONS:
        raise ValueError('nearest-distance operation budget exceeded')
    tri = vertices[f]
    edges = np.stack([np.linalg.norm(tri[:, i] - tri[:, j], axis=1) for i, j in ((0, 1), (1, 2), (2, 0))], axis=1)
    areas = .5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    longest = edges.max(axis=1) if len(f) else np.empty(0)
    # Exact ceil 1% cardinality, descending edge length then ascending face ID.
    top = np.lexsort((np.arange(len(f)), -longest))[:top_count]
    top_sample = top[_sample(len(top), 128)]
    spacing = _nearest(dense[dense_ids], dense, dense_ids) if len(dense_ids) else np.empty(0)
    spacing_summary = _summary(spacing)
    median = spacing_summary['p50']
    samples = {}
    for name, ids in [('ordinary', ordinary), ('top_one_percent', top_sample)]:
        chosen = tri[ids]
        points = np.stack([chosen.mean(axis=1), (chosen[:, 0] + chosen[:, 1]) / 2,
                           (chosen[:, 1] + chosen[:, 2]) / 2, (chosen[:, 2] + chosen[:, 0]) / 2], axis=1).reshape((-1, 3))
        distances = _nearest(points, dense) if len(dense) else np.empty(0)
        samples[name] = {'face_ids': ids.tolist(), 'sample_order': ['centroid', 'edge01_midpoint', 'edge12_midpoint', 'edge20_midpoint'],
                         'status': 'MEASURED' if len(dense) else 'NO_DENSE_SUPPORT_REFERENCE',
                         'nearest_dense_distances': distances.reshape((-1, 4)).tolist() if len(dense) else None,
                         'distance_summary': _summary(distances),
                         'distance_over_median_dense_spacing_summary': _summary(distances / median) if median is not None and median > 0 else None}
    return {'schema': 'offline-surface-support-v1', 'status': 'DIAGNOSTIC_ONLY_NOT_ACCEPTED',
            'dense_count': len(dense), 'vertex_count': len(vertices), 'face_count': len(f),
            'nearest_distance_operation_count': operations, 'operation_budget': MAX_OPERATIONS,
            'edge_lengths': _summary(edges), 'face_longest_edges': _summary(longest), 'face_areas': _summary(areas),
            'top_one_percent': {'face_ids': top.tolist(), 'count': len(top), 'area_fraction': float(areas[top].sum() / areas.sum()) if areas.sum() > 0 else None},
            'dense_spacing': {'point_ids': dense_ids.tolist(), 'nearest_other_distances': spacing.tolist(), 'summary': spacing_summary},
            'samples': samples, 'limitations': LIMITS, 'geometry_modified': False, 'geometry_acceptance': False}


def _binding(path):
    path = Path(path).resolve(strict=True)
    size = path.stat().st_size
    if size > 64 << 20:
        raise ValueError('source exceeds 64 MiB byte budget')
    with path.open('rb') as stream:
        data = stream.read((64 << 20) + 1)
    if len(data) > 64 << 20 or len(data) != size:
        raise ValueError('source byte budget or size drift')
    return dict(path=str(path), bytes=size, sha256=hashlib.sha256(data).hexdigest())


def discover_surface_sources(discovery, refs):
    """Locate same-producer mesh/dense by recorded roles; never guess paths."""
    surface = discovery.get('stages', {}).get('surface', {})
    artifacts = surface.get('artifacts', [])
    if not artifacts or not surface.get('evidence_paths'):
        return None
    if len(artifacts) != 1:
        raise ValueError('surface discovery requires one selected mesh')
    artifact = artifacts[0]
    def authenticated(record):
        if not all(k in record for k in ('path', 'bytes', 'sha256')):
            raise ValueError('producer binding incomplete')
        actual = _binding(record['path'])
        if actual != {k: record[k] for k in ('path', 'bytes', 'sha256')}:
            # Normalize path spelling, but require the same resolved file.
            if (actual['path'] != str(Path(record['path']).resolve(strict=True))
                    or actual['bytes'] != record['bytes'] or actual['sha256'] != record['sha256']):
                raise ValueError('surface producer binding drift')
        return actual
    selected = authenticated(artifact)
    found = []
    for evidence in surface['evidence_paths']:
        path = Path(evidence['path'] if isinstance(evidence, dict) else evidence)
        if not path.is_file():
            continue
        producer_binding = _binding(path)
        if isinstance(evidence, dict):
            authenticated(evidence)
        with path.open('rb') as stream:
            producer = json.loads(stream.read((64 << 20) + 1))
        if producer.get('contract_version') != 'mvs-surface-run-v1':
            continue
        roles = {}
        for stage in producer.get('stages', []):
            for geometry in stage.get('outputs', {}).get('geometry', []):
                key = (stage.get('name'), geometry.get('role'))
                if key in (('mesh', 'pre_cleanup_surface'), ('dense', 'dense')):
                    if key in roles:
                        raise ValueError('duplicate producer geometry role')
                    roles[key] = geometry
        if ('mesh', 'pre_cleanup_surface') not in roles or ('dense', 'dense') not in roles:
            continue
        mesh = authenticated(roles[('mesh', 'pre_cleanup_surface')])
        dense = authenticated(roles[('dense', 'dense')])
        if mesh != selected:
            raise ValueError('selected mesh differs from same-run producer')
        if _binding(path) != producer_binding:
            raise ValueError('producer record drift')
        found.append((producer_binding, mesh, dense))
    if not found:
        return None
    if len(found) != 1:
        raise ValueError('multiple surface producers are ambiguous')
    producer_binding, mesh, dense = found[0]
    refs.extend([producer_binding, mesh, dense])
    return {'mesh_path': mesh['path'], 'dense_path': dense['path'],
            'producer_record': producer_binding['path'], 'source_bindings': [producer_binding, mesh, dense]}


def review_surface_sources(mesh_path, dense_path, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError('surface diagnostic output must be new')
    mesh_path, dense_path = Path(mesh_path).resolve(strict=True), Path(dense_path).resolve(strict=True)
    if output.resolve().is_relative_to(mesh_path.parent) or output.resolve().is_relative_to(dense_path.parent):
        raise ValueError('diagnostic output must be outside source directories')
    refs = [_binding(mesh_path), _binding(dense_path)]
    vertices, faces, _ = read_mesh(mesh_path, max_vertices=200000, max_faces=200000)
    cloud = read_cloud(dense_path, max_bytes=64 << 20, max_vertices=100000, max_faces=200000)
    result = diagnose_arrays(cloud['vertices'], vertices, faces)
    if refs != [_binding(mesh_path), _binding(dense_path)]:
        raise ValueError('source binding drift')
    result['source_bindings'] = refs
    output.mkdir(parents=True, exist_ok=False)
    record = output / 'RESULT.json'
    record.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2), encoding='utf-8')
    page = output / 'index.html'
    def display(value):
        return '不可用' if value is None else format(value, '.6g') if isinstance(value, float) else str(value)
    table_rows = [('稠密点数', result['dense_count']), ('网格顶点数', result['vertex_count']),
                  ('网格面数', result['face_count']), ('最长边前1%面数', result['top_one_percent']['count']),
                  ('前1%面面积占比', result['top_one_percent']['area_fraction'])]
    for key, label in [('ordinary', '普通面样本'), ('top_one_percent', '长边面样本')]:
        sample = result['samples'][key]
        ratio = sample['distance_over_median_dense_spacing_summary']
        table_rows.extend([(label + '到dense距离中位数（未知单位）', sample['distance_summary']['p50']),
                           (label + '距离 / dense间距中位数', ratio['p50'] if ratio else None)])
    table = '<table>' + ''.join('<tr><th>' + html.escape(label) + '</th><td>' + html.escape(display(value)) + '</td></tr>' for label, value in table_rows) + '</table>'
    page.write_text('<!doctype html><meta charset="utf-8"><title>表面采样支持诊断</title><style>body{font:16px system-ui;max-width:1000px;margin:24px auto}th,td{padding:9px;text-align:left;border-bottom:1px solid #ddd}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style><h1>表面采样支持诊断</h1><p>仅为样本到稠密点的距离描述；不是可见性、物理真值、几何验收或删面门。长度为未知世界单位。面积占比为0–1比例；零间距时倍数不可用。</p>' + table + '<details><summary>完整采样ID、距离与输入绑定 JSON</summary><pre>' + html.escape(json.dumps(result, ensure_ascii=False, indent=2)) + '</pre></details>', encoding='utf-8')
    return {'status': result['status'], 'result': _binding(record), 'report': _binding(page), 'geometry_acceptance': False}
