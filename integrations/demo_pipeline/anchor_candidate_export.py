"""Automatic candidate-only PLY delivery; never modifies native reconstruction."""
from collections import defaultdict
import hashlib
import html
import json
from pathlib import Path

from .anchor_candidate_gate import classify_original_candidates, PROFILE as CANDIDATE_PROFILE
from integrations.viewer.ply_data import read_ply
from integrations.viewer.export import export_html

TIERS = ('STRICT_CONDITIONAL', 'RELAXED_CONDITIONAL', 'TWO_VIEW_PROVISIONAL')
BRANCHES = ('CENTRAL_BRANCH_CONDITIONAL', 'MONOTONIC_RADIAL_CONDITIONAL')
COLORS = dict(STRICT_CONDITIONAL=(70, 190, 125), RELAXED_CONDITIONAL=(240, 170, 65),
              TWO_VIEW_PROVISIONAL=(95, 155, 235))
FILE_TIER = dict(STRICT_CONDITIONAL='strict', RELAXED_CONDITIONAL='relaxed', TWO_VIEW_PROVISIONAL='two')
FILE_BRANCH = dict(CENTRAL_BRANCH_CONDITIONAL='center', MONOTONIC_RADIAL_CONDITIONAL='mono')
RIGHTS = dict(geometry_accepted=False, native_injected=False, identity_verified=False,
              scale_status='unknown', surface_generation_allowed=False)


def binding(path):
    path = Path(path).resolve(strict=True)
    data = path.read_bytes()
    return dict(path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def _write_json(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def export_candidate_clouds(diagnostics, output, diagnostic_binding, budget=lambda: None,
                            verify_sources=lambda: None):
    """Recheck metrics once, then export each model/tier/branch independently.

    Caller has authenticated original image/mask/track/camera inputs and checked
    their closing hashes. The bound diagnostic must equal this in-memory value.
    Readback and final binding checks are required before publishing a manifest.
    """
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('fresh candidate output required')
    source = Path(diagnostic_binding['path'])
    if binding(source) != diagnostic_binding:
        raise ValueError('diagnostic binding drift')
    if json.loads(source.read_text(encoding='utf-8')) != diagnostics:
        raise ValueError('diagnostic bytes differ from in-memory source')
    budget()
    classification = classify_original_candidates(diagnostics, budget=budget)
    grouped = defaultdict(list)
    for row in classification['rows']:
        if row['tier'] in TIERS:
            if row['branch_scope'] not in BRANCHES or any(row[k] is not False for k in
                    ('geometry_accepted', 'native_injected', 'identity_verified')):
                raise ValueError('invalid candidate export scope')
            grouped[(row['model_key'], row['tier'], row['branch_scope'])].append(row)
    if sum(map(len, grouped.values())) != classification['counts']['exportable']:
        raise ValueError('candidate count mismatch')
    output.mkdir(parents=True, exist_ok=False)
    artifacts = []
    try:
        _write_json(output / 'CLASSIFICATION.json', classification)
        classification_binding = binding(output / 'CLASSIFICATION.json')
        models = {key: number for number, key in enumerate(sorted({key[0] for key in grouped}))}
        for (model, tier, branch), rows in sorted(grouped.items()):
            budget()
            folder = output / f'model_{models[model]:03d}'
            folder.mkdir(exist_ok=True)
            path = folder / f'{FILE_TIER[tier]}_{FILE_BRANCH[branch]}.ply'
            color = COLORS[tier]
            header = ['ply', 'format ascii 1.0', 'comment conditional candidate points only',
                      'comment unknown world units; diagnostic RGB is not measured color',
                      f'comment {tier} {branch}', f'element vertex {len(rows)}',
                      'property double x', 'property double y', 'property double z',
                      'property uchar red', 'property uchar green', 'property uchar blue', 'end_header']
            body = [' '.join([*(format(v, '.17g') for v in row['xyz']), *(str(c) for c in color)])
                    for row in rows]
            with path.open('x', encoding='ascii', newline='\n') as stream:
                stream.write('\n'.join(header + body) + '\n')
            # Reader budgets must be positive; the separate faces assertion
            # below enforces the stricter point-only output contract.
            parsed = read_ply(path, max_bytes=1 << 20, max_vertices=CANDIDATE_PROFILE['max_groups'], max_faces=1)
            if (parsed['vertices'] != [row['xyz'] for row in rows] or parsed['faces'] or
                    parsed['colors'] != [list(color)] * len(rows)):
                raise ValueError('candidate PLY readback differs')
            artifacts.append(dict(model_key=model, tier=tier, branch_scope=branch,
                coordinate_frame='unaltered saved model world frame; no inter-model alignment',
                color_semantics='diagnostic tier, not measured texture', path=path.relative_to(output).as_posix(),
                binding=binding(path), vertices=len(rows), faces=0, readback=True,
                vertex_ledger=[dict(vertex_index=i, group_id=row['group_id'], track_id=row['track_id'],
                    identity_segment=row['identity_segment'], xyz=row['xyz'],
                    source_indices=row['source_indices'], frame_ids=row['frame_ids']) for i, row in enumerate(rows)],
                **RIGHTS))
        viewers = []
        for begin in range(0, len(artifacts), 8):
            budget()
            batch = artifacts[begin:begin+8]
            spec = output / f'scenes_{begin // 8:03d}.json'
            _write_json(spec, dict(schema='ply-viewer-scenes-v1', scenes=[dict(
                ply=a['path'], sha256=a['binding']['sha256'], title=f"模型 {a['model_key']} · {a['tier']}",
                data_kind='real_photos', declared_unit=None,
                note=f"条件候选点；{a['branch_scope']}；不同模型坐标不对齐。颜色为分类标签，不代表纹理。未通过物理几何、尺度或CAD验收。",
                evidence=['CLASSIFICATION.json']) for a in batch]))
            destination = output / f'viewer_{begin // 8:03d}.html'
            viewer = export_html(spec, destination)
            viewers.append(dict(path=destination.name, binding=binding(destination), specification=binding(spec),
                                status=viewer['status'], scenes=viewer['scenes']))
        budget()
        if binding(source) != diagnostic_binding or binding(output / 'CLASSIFICATION.json') != classification_binding:
            raise ValueError('candidate source or classification drift')
        for artifact in artifacts:
            if binding(artifact['binding']['path']) != artifact['binding']:
                raise ValueError('candidate PLY drift')
        # The caller's original image/mask/camera/code closure belongs inside the
        # transaction: failure here leaves no successful child publication.
        verify_sources()
        result = dict(status='CANDIDATE_CLOUDS_EXPORTED' if artifacts else 'NO_EXPORTABLE_CANDIDATES',
            counts=classification['counts'], diagnostic_binding=diagnostic_binding,
            classification_binding=classification_binding, artifacts=artifacts, viewers=viewers,
            new_native_model_points=0, source_unchanged=True,
            source_closure_scope='bound bytes observed before publication; later edits invalidate bindings', **RIGHTS)
        links = ''.join(f'<li><a href="{html.escape(v["path"], quote=True)}">交互点云 {i+1}</a>（各模型独立）</li>'
                        for i, v in enumerate(viewers))
        files = ''.join(f'<li>模型 {html.escape(a["model_key"])} · {a["tier"]} · {a["vertices"]}点：'
                        f'<a href="{html.escape(a["path"], quote=True)}">PLY</a> · {a["branch_scope"]}</li>' for a in artifacts)
        page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>自动候选点云</title>
<style>body{{font:16px/1.8 system-ui;max-width:1000px;margin:32px auto;padding:20px;color:#24334a}}pre{{white-space:pre-wrap}}aside{{background:#fff1d6;padding:18px}}</style>
<h1>自动候选点云</h1><aside>{result['status']}。这是通过条件性内部一致性检查的点层，尚不是完整物体。双视点没有留一验证；负径向相机依赖中心分支假设。不同模型未对齐，不应直接合并。</aside>
<p>分组、重新计算误差、分级、PLY导出与数值回读均由建模入口自动执行。未改原生模型或相机，拒绝与未评估组保留在账本。</p>
<pre>{html.escape(json.dumps(result['counts'], ensure_ascii=False, indent=2))}</pre><ul>{links}</ul><ul>{files}</ul>
<p><a href="CLASSIFICATION.json">全部分组及拒绝原因</a> · <a href="MANIFEST.json">文件与顶点来源</a> · <a href="../index.html">完整自动复核</a></p></html>'''
        with (output / 'index.html').open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(page)
        _write_json(output / 'MANIFEST.json', result)
        return dict(status=result['status'], counts=result['counts'],
                    record=binding(output / 'MANIFEST.json'), report=binding(output / 'index.html'), **RIGHTS)
    except Exception as exc:
        try:
            _write_json(output / 'FAILURE.json', dict(status='CANDIDATE_EXPORT_FAILED',
                reason=f'{type(exc).__name__}: {exc}', completed_artifacts=artifacts,
                partial_files_not_publishable=True, **RIGHTS))
        except OSError:
            pass
        raise
