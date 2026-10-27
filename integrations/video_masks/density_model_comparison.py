"""Read existing surfaces for side-by-side metrics; no alignment or execution."""
from pathlib import Path
import html
import json
import math
import os
import numpy as np

from integrations.demo_pipeline.balanced_point_run import binding, check_bindings
from integrations.viewer.ply_data import read_ply


def mesh_metrics(vertices, faces):
    """Scale-invariant area concentration and edge incidence, not accuracy."""
    v = np.asarray(vertices, dtype=float).reshape(-1, 3)
    f = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if not np.isfinite(v).all() or (len(f) and (f.min() < 0 or f.max() >= len(v))):
        raise ValueError('invalid finite indexed triangle mesh')
    if len(f) > 2_000_000 or len(v) > 2_000_000:
        raise ValueError('mesh metrics exceed bounded size')
    if not len(f):
        return dict(vertices=len(v), faces=0, largest_one_percent_face_area_fraction=None,
                    degenerate_faces=0, boundary_edges=0, nonmanifold_edges=0)
    tri = v[f]
    area = np.linalg.norm(np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]), axis=1)/2
    if not np.isfinite(area).all():
        raise ValueError('nonfinite computed triangle areas')
    total = float(area.sum())
    take = max(1, math.ceil(len(f)*.01))
    edges = np.sort(np.concatenate((f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]])), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return dict(vertices=len(v), faces=len(f),
                largest_one_percent_face_area_fraction=float(np.sort(area)[-take:].sum()/total) if total > 0 else None,
                degenerate_faces=int((area == 0).sum()), boundary_edges=int((counts == 1).sum()),
                nonmanifold_edges=int((counts > 2).sum()),
                edge_count_interpretation='Triangle edge incidence only; no self-intersection or CAD acceptance')


def compare(old_surface_run, new_surface_run, output):
    """Write a fresh report without changing either existing surface run."""
    roots = [Path(p).resolve(strict=True) for p in (old_surface_run, new_surface_run)]
    out = Path(output).resolve()
    if out.exists() or any(out.is_relative_to(r) or r.is_relative_to(out) for r in roots):
        raise ValueError('fresh disjoint comparison output required')
    refs = []
    def read(path):
        refs.append(binding(path))
        return json.loads(path.read_text(encoding='utf-8-sig'))
    rows = []
    for label, root in zip(('old', 'new'), roots):
        result = read(root/'RESULT.json')
        export = read(root/'export/EXPORT.json')
        surface = read(root/'surface/result.json')
        item = dict(label=label, surface_run=str(root), status=result.get('status'),
                    surface_status=surface.get('status'),
                    original_source_geometry_counts=result.get('before'),
                    derived_geometry_counts=result.get('derived_counts'),
                    exported_registered_views=export.get('registered_views'),
                    exported_sparse_points=export.get('points'),
                    dense_points=None, raw_mesh=None, links={}, errors=[])
        for kind, filename in (('dense', '__mvs_dense.ply'), ('mesh', '__mvs_surface_raw.ply')):
            path = root/'surface/work'/filename
            if not path.is_file():
                item['errors'].append(kind+' PLY missing; metric unknown')
                continue
            try:
                refs.append(binding(path))
                data = read_ply(path, max_bytes=128<<20, max_vertices=2_000_000, max_faces=2_000_000)
                if kind == 'dense':
                    item['dense_points'] = len(data['vertices'])
                else:
                    item['raw_mesh'] = mesh_metrics(data['vertices'], data['faces'])
            except (ValueError, OSError) as exc:
                item['errors'].append(kind+' metrics unknown: '+str(exc))
        for name, relative in (('mesh_viewer', 'mesh_viewer.html'), ('dense_viewer', 'dense_viewer.html'),
                               ('blender_model', 'blender/model.blend')):
            path = root/relative
            if path.is_file():
                refs.append(binding(path)); item['links'][name] = str(path)
        density_path = root/'point_density/RESULT.json'
        item['density_report'] = None
        if density_path.is_file():
            density = read(density_path)
            item['density_report'] = dict(status=density.get('status'),
                mean_frame_met_area_fraction=density.get('mean_frame_met_area_fraction'),
                dense_visibility_coverage=density.get('dense_visibility_coverage'))
            item['links']['density_report'] = str(root/'point_density/index.html')
        rows.append(item)
    if check_bindings(refs):
        raise ValueError('comparison input drift')
    report = dict(schema='density-model-comparison-v1', status='COMPARED_NOT_QUALITY_ACCEPTED',
                  runs=rows, source_bindings=refs, source_unchanged=True, geometry_accepted=False,
                  limitations=['Independent SfM scales and poses: side-by-side qualitative comparison only.',
                               'No alignment, dimensional accuracy, or density improvement claim.',
                               'Face area concentration measures triangle size distribution, not visible 3D coverage.',
                               'Missing or unreadable metrics remain unknown.'])
    out.mkdir(parents=True)
    (out/'RESULT.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    columns = []
    for item in rows:
        links = ''.join('<li><a href="'+html.escape(Path(os.path.relpath(p, out)).as_posix(), quote=True)+'">'
                        +html.escape(name)+'</a></li>' for name, p in item['links'].items())
        summary = {k: value for k, value in item.items() if k != 'links'}
        columns.append('<section><h2>'+item['label']+'</h2><ul>'+links+'</ul><pre>'
                       +html.escape(json.dumps(summary, ensure_ascii=False, indent=2))+'</pre></section>')
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Density model comparison</title>'
        '<style>body{font:16px system-ui;margin:30px}main{display:flex;gap:30px}section{width:48%}pre{white-space:pre-wrap}</style>'
        '<h1>Existing model comparison</h1><p>Independent scales and poses. No alignment or accuracy claim.</p>'
        '<a href="RESULT.json">Source bindings and complete metrics</a><main>'+''.join(columns)+'</main>', encoding='utf-8')
    return report
