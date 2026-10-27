"""Bound regional mesh -> reversible mask-conflict face quarantine.

This is an engineering candidate filter, not an estimator of true visible
surfaces. Unknown observations never vote for deletion or geometry acceptance.
"""
from __future__ import annotations

import csv
from dataclasses import asdict
import hashlib
import html
import json
from pathlib import Path
import shutil
import struct
import time

import numpy as np
from PIL import Image, ImageDraw

from integrations.viewer.ply_data import read_ply
from integrations.mvs_surface.scene import read_scene
from integrations.openmvs_depth.dmap import read_dmap
from .face_mask_gate import Policy, evaluate_view, aggregate_views, UNKNOWN, CONFLICT
from .face_gap_gate import GapPolicy, evaluate_gap_view
from .face_edge_gate import EdgePolicy, evaluate_edge_view

ROOT = Path(__file__).resolve().parents[2]
LIMITS = dict(max_bytes=16 << 20, max_vertices=200000, max_faces=400000)
LIMITATIONS = [
    'Candidate quarantine, not physical error labels or accepted geometry.',
    'Depth supports endpoints only; it does not establish whole-face visibility.',
    'External occlusion, nonrigid motion, mask and pose correctness remain unverified.',
    'Angular separation is not statistical independence; thresholds are engineering starting values.',
    'Unknown, subpixel and out-of-view evidence does not vote; missing depth blocks v1 only, not mask-derived gap evidence.',
    'Kept faces may still be wrong; no hole filling, vertex displacement, remeshing or CAD acceptance.',
    'v3 only quarantines whole faces inside a bounded contour band; large crossing faces remain for later local clipping.',
]


def _need(value, message):
    if not value:
        raise ValueError(message)


def _read(path):
    def pairs(items):
        obj = {}
        for k, v in items:
            _need(k not in obj, 'duplicate JSON key: '+k)
            obj[k] = v
        return obj
    return json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=pairs,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def _save(path, value):
    with Path(path).open('w', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, allow_nan=False, indent=2)


def bind(path, expected=None):
    p = Path(path).resolve(strict=True)
    _need(p.is_file() and p.stat().st_size <= 128 << 20, 'input size/type refused: '+str(p))
    stat = p.stat()
    digest = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            digest.update(chunk)
    _need(p.stat().st_size == stat.st_size and p.stat().st_mtime_ns == stat.st_mtime_ns,
          'input changed while hashing')
    result = dict(path=str(p), bytes=stat.st_size, sha256=digest.hexdigest())
    if expected is not None:
        _need(result['sha256'] == expected.get('sha256') and result['bytes'] == expected.get('bytes'),
              'declared input hash/size mismatch: '+str(p))
    return result


def _inside(path, root):
    p = Path(path).resolve(strict=True)
    _need(p.is_relative_to(root), 'bound file outside its source run: '+str(p))
    return p


def _binary(record, root, size, bindings):
    p = _inside(record['path'], root)
    bindings.append(bind(p, record))
    with Image.open(p) as image:
        _need(image.mode == 'L' and image.size == size, 'binary mask mode/size mismatch')
        array = np.asarray(image).copy()
    _need(np.isin(array, [0, 255]).all(), 'nonbinary mask/validity')
    _need((record.get('width'), record.get('height')) == size, 'declared mask size mismatch')
    return array == 255


def verify_depth(d, image):
    """No filename-only or resized-K join. Native pixel-centre K is unchanged."""
    size = (image['width'], image['height'])
    _need(d.view_ids and d.view_ids[0] == image['global_id'], 'depth global image identity mismatch')
    _need(Path(d.image_name_utf8.replace('\\', '/')).name == Path(image['name']).name,
          'depth image name mismatch')
    _need(d.image_size == d.depth_size == size, 'unsupported scaled depth domain')
    for name in ('K', 'R', 'C'):
        _need(np.allclose(getattr(d, name), np.asarray(image[name]), rtol=0, atol=1e-7),
              'native scene/depth '+name+' mismatch')


def write_partition(path, vertices, faces, colors, original_face_ids):
    """Exact coordinate subset; stable source face/vertex mapping and readback."""
    face_ids = np.asarray(original_face_ids, dtype=np.int64)
    _need(face_ids.ndim == 1 and len(np.unique(face_ids)) == len(face_ids), 'duplicate partition face IDs')
    _need(not len(face_ids) or (face_ids.min() >= 0 and face_ids.max() < len(faces)), 'bad partition face ID')
    selected = faces[face_ids]
    vertex_ids = np.unique(selected) if len(selected) else np.empty(0, dtype=np.int64)
    remap = np.full(len(vertices), -1, dtype=np.int64)
    remap[vertex_ids] = np.arange(len(vertex_ids))
    local = remap[selected]
    header = ('ply\nformat binary_little_endian 1.0\ncomment face-mask candidate partition; not accepted\n'
              f'element vertex {len(vertex_ids)}\nproperty double x\nproperty double y\nproperty double z\n')
    if colors is not None:
        header += 'property uchar red\nproperty uchar green\nproperty uchar blue\n'
    header += f'element face {len(local)}\nproperty list uchar int vertex_indices\nend_header\n'
    with Path(path).open('xb') as f:
        f.write(header.encode('ascii'))
        for vi in vertex_ids:
            f.write(struct.pack('<3d', *vertices[vi]))
            if colors is not None:
                f.write(bytes(colors[vi]))
        for face in local:
            f.write(struct.pack('<B3i', 3, *face))
    loaded = read_ply(path, **LIMITS)
    _need(np.array_equal(np.asarray(loaded['vertices']).reshape(-1, 3), vertices[vertex_ids]),
          'partition vertex readback differs')
    _need(np.array_equal(np.asarray(loaded['faces']).reshape(-1, 3), local), 'partition face readback differs')
    if colors is not None:
        _need(loaded['colors'] == colors[vertex_ids].tolist(), 'partition color readback differs')
    np.savez_compressed(Path(path).with_suffix('.mapping.npz'), source_face_ids=face_ids, source_vertex_ids=vertex_ids)
    return dict(**bind(path), vertices=len(vertex_ids), faces=len(face_ids), exact_readback=True)


def _topology(faces):
    if not len(faces):
        return dict(faces=0, edges=0, boundary_edges=0, nonmanifold_edges=0)
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return dict(faces=len(faces), edges=len(counts), boundary_edges=int((counts == 1).sum()),
                nonmanifold_edges=int((counts > 2).sum()))


def aggregate_rules(vertices, faces, depth_rows, gap_rows, cameras, policy, gap_policy, check=lambda: None):
    """Aggregate rules separately before OR: one vote per rule cannot form a pair."""
    depth = aggregate_views(vertices, faces, depth_rows, cameras, policy, check=check)
    gap = aggregate_views(vertices, faces, gap_rows, cameras,
                          Policy(min_angle_degrees=gap_policy.min_angle_degrees), check=check)
    gap['quarantine'] = np.asarray(gap['quarantine'], bool) & (gap['conflict_views'] >= gap_policy.min_views)
    gap['quarantine_face_indices'] = np.flatnonzero(gap['quarantine'])
    gap['kept_face_indices'] = np.flatnonzero(~gap['quarantine'])
    combined = np.asarray(depth['quarantine'], bool) | np.asarray(gap['quarantine'], bool)
    return depth, gap, combined


def aggregate_edge_rules(vertices, faces, depth_rows, gap_rows, edge_rows, cameras, policy, gap_policy, check=lambda: None):
    """Restrict same-view negative votes, then veto all positive core protection."""
    depth_rows, gap_rows, edge_rows = list(depth_rows), list(gap_rows), list(edge_rows)
    _need(len(depth_rows)==len(gap_rows)==len(edge_rows), 'one edge evaluation per rule view required')
    edge_by_id={r['view_id']:r for r in edge_rows}
    _need(len(edge_by_id)==len(edge_rows), 'duplicate edge view identity')
    expected={r['view_id'] for r in depth_rows}
    _need(expected=={r['view_id'] for r in gap_rows}==set(edge_by_id), 'rule/edge view identities differ')
    eligible_any=np.zeros(len(faces),bool); protected_any=np.zeros(len(faces),bool)
    for row in edge_rows:
        for key in ('edge_eligible','core_protected'):
            value=np.asarray(row[key])
            _need(value.dtype==np.bool_ and value.shape==(len(faces),), 'one bool edge/core flag per original face required')
        eligible_any |= row['edge_eligible']; protected_any |= row['core_protected']
    def filtered(rows):
        output=[]
        for row in rows:
            edge=edge_by_id[row['view_id']]['edge_eligible']; item=dict(row)
            item['status']=np.where((np.asarray(row['status'])==CONFLICT)&~edge,UNKNOWN,row['status'])
            output.append(item)
        return output
    depth,gap,previous=aggregate_rules(vertices,faces,depth_rows,gap_rows,cameras,policy,gap_policy,check)
    edge_depth,edge_gap,limited=aggregate_rules(vertices,faces,filtered(depth_rows),filtered(gap_rows),cameras,policy,gap_policy,check)
    final=limited & ~protected_any
    _need(not np.any(final & ~previous), 'edge candidate is not a subset of previous v2 quarantine')
    return dict(depth=depth,gap=gap,edge_depth=edge_depth,edge_gap=edge_gap,
                previous_v2_quarantine=previous,protected_any=protected_any,
                edge_eligible_any=eligible_any,quarantine=final,subset_assert=True)


def _overlay(path, image_path, labels, vertices, faces, ids, camera, title):
    from integrations.demo_pipeline.temporal_review import load_cv
    cv = load_cv([])
    with Image.open(image_path) as source:
        im = source.convert('RGB')
    _need(im.size == (camera['width'], camera['height']), 'overlay image dimensions mismatch')
    rgba = im.convert('RGBA'); layer = Image.new('RGBA', im.size, (0, 0, 0, 0)); draw = ImageDraw.Draw(layer)
    contours, _ = cv.findContours(labels.astype('uint8'), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        xy = [tuple(map(int, x)) for x in contour[:, 0]]
        if len(xy) > 1:
            draw.line(xy+[xy[0]], fill=(255, 220, 0, 255), width=2)
    cam = (vertices-np.asarray(camera['C']))@np.asarray(camera['R']).T
    homogeneous = cam@np.asarray(camera['K']).T
    with np.errstate(invalid='ignore', divide='ignore'):
        uv = homogeneous[:, :2]/homogeneous[:, 2:]
    for face_id in ids:
        pts = [tuple(map(float, p)) for p in uv[faces[face_id]]]
        draw.polygon(pts, fill=(255, 35, 60, 90), outline=(255, 35, 60, 255))
    rgba = Image.alpha_composite(rgba, layer).convert('RGB')
    canvas = Image.new('RGB', (im.width, im.height+74), (20, 26, 40)); canvas.paste(rgba, (0, 74))
    ImageDraw.Draw(canvas).text((8, 8), title+'\nYellow: mask / Red: candidate faces\nProjection only; no physical visibility claim', fill='white')
    canvas.save(path)


def _render(out, result):
    c = result.get('counts', {})
    images = ''.join('<figure><img style="width:280px" src="'+html.escape(p)+'"><figcaption>'+html.escape(p)+'</figcaption></figure>'
                     for p in result.get('overlays', []))
    links = ''.join('<li><a href="'+html.escape(name)+'">'+html.escape(label)+'</a></li>'
                    for name, label in [('retained_viewer.html', '默认查看：单独修正版候选（未验收）'),
                    ('comparison_viewer.html', '原始 / 候选隔离后三维对照'),
                    ('quarantined_viewer.html', '只查看被隔离面'), ('original.ply', '原始网格'),
                    ('retained.ply', '候选隔离后网格（保留孔洞）'), ('quarantined.ply', '隔离面'),
                    ('FACE_LEDGER.csv', '逐面判定记录'), ('RESULT.json', '完整结果'), ('PLAN.json', '固定规则')]
                    if (out/name).is_file())
    text = '<!doctype html><meta charset="utf-8"><title>面片跨掩膜检查</title><style>body{font:17px system-ui;max-width:1100px;margin:35px auto;background:#f5f7fa;color:#172438}figure{display:inline-block;margin:10px}a{color:#1260aa}pre{white-space:pre-wrap}</style>'
    text += '<h1>自动隔离跨掩膜空隙的候选面片</h1><p>'+html.escape(result['status'])+'</p>'
    text += '<p>原始面 '+str(c.get('original_faces', '?'))+'；候选隔离 '+str(c.get('quarantined_faces', '?'))+'；保留 '+str(c.get('retained_faces', '?'))+'。</p>'
    text += '<p>v3从原始网格重新评估：v1深度背景与v2掩膜间隙各自仅在同视图整面位于边缘带时投负票；各自满足多视角角差后取并集，任一视图核心支持可否决隔离。大跨缝面本轮保留，需后续局部裁剪。不移动点、不补洞；保留面仍可能有错，遮挡、变形、掩膜与姿态正确性未验收。</p>'
    text += '<ul>'+links+'</ul>'+images+'<pre>'+html.escape(result.get('error', ''))+'</pre>'
    (out/'index.html').write_text(text, encoding='utf-8')


def run(surface_run, output):
    source, out = Path(surface_run).resolve(strict=True), Path(output).resolve()
    _need(out.is_relative_to(ROOT/'outputs') and not out.exists(), 'fresh output under repository outputs required')
    # A controller may place this stage within its own run, but never over any inputs.
    _need(not source.is_relative_to(out) and out != source, 'output overlaps source ancestor')
    for protected in ('surface', 'bridge', 'masks', 'export'):
        _need(not out.is_relative_to(source/protected), 'output overlaps protected source subtree')
    out.mkdir(parents=True); (out/'per_view').mkdir(); (out/'gap_per_view').mkdir(); (out/'edge_per_view').mkdir()
    started = time.monotonic(); bindings = []; results = []; gap_results = []; edge_results = []; cameras = {}; frames = {}; views = []
    policy, gap_policy, edge_policy = Policy(), GapPolicy(), EdgePolicy()
    result = dict(schema='face-mask-quarantine-v3', status='RUNNING', geometry_accepted=False,
                  quality_status='NOT_EVALUATED', scale_status='unknown', source_unchanged=False,
                  source_run=str(source), policy=asdict(policy), gap_policy=asdict(gap_policy), edge_policy=asdict(edge_policy),limitations=LIMITATIONS,
                  new_sam_executions=0, new_sfm_executions=0, new_mvs_executions=0, overlays=[])
    def check():
        if time.monotonic()-started > 210:
            raise TimeoutError('face-mask stage cooperative 210 second budget exhausted')
    def add(path, expected=None):
        value = bind(path, expected); bindings.append(value); return value
    def process(image, d=None):
        check(); name = Path(image['name']).name; frame = frames[name]; view_id = str(image['global_id'])
        mask = _binary(frame['mask'], source, (image['width'], image['height']), bindings)
        valid = _binary(frame['validity'], source, (image['width'], image['height']), bindings)
        _need(not np.any(mask & ~valid), 'foreground outside known mask domain')
        row = evaluate_view(vertices, faces, K=np.asarray(image['K']), R=np.asarray(image['R']),
                            C=np.asarray(image['C']), mask=mask, validity=valid,
                            depth=d.depth if d is not None else None, view_id=view_id, policy=policy, check=check)
        arrays = {key: value for key, value in row.items() if isinstance(value, np.ndarray)}
        np.savez_compressed(out/'per_view'/('view_'+view_id+'.npz'), **arrays)
        gap_row = evaluate_gap_view(vertices, faces, K=np.asarray(image['K']), R=np.asarray(image['R']),
                                   C=np.asarray(image['C']), mask=mask, validity=valid,
                                   view_id=view_id, policy=gap_policy, check=check)
        gap_arrays = {key: value for key, value in gap_row.items() if isinstance(value, np.ndarray)}
        np.savez_compressed(out/'gap_per_view'/('view_'+view_id+'.npz'), **gap_arrays)
        gap_image = 'gap_per_view/view_'+view_id+'_gap.png'
        Image.fromarray(gap_row['gap_mask'].astype(np.uint8)*255).save(out/gap_image)
        edge_row=evaluate_edge_view(vertices,faces,K=np.asarray(image['K']),R=np.asarray(image['R']),
                                   C=np.asarray(image['C']),mask=mask,validity=valid,
                                   depth=d.depth if d is not None else None,view_id=view_id,policy=edge_policy,check=check)
        edge_arrays={key:value for key,value in edge_row.items() if isinstance(value,np.ndarray)}
        np.savez_compressed(out/'edge_per_view'/('view_'+view_id+'.npz'),**edge_arrays)
        summary = dict(view_id=view_id, input_index=frame['input_index'], image_name=name,
                       depth_available=d is not None, camera=image,
                       counts={str(key): int((row['reason'] == int(key)).sum()) for key in row['reason_names']},
                       reason_names=row['reason_names'], conflicts=int((row['status'] == CONFLICT).sum()),
                       gap_counts={str(key): int((gap_row['reason'] == int(key)).sum()) for key in gap_row['reason_names']},
                       gap_reason_names=gap_row['reason_names'], gap_metadata=gap_row['gap_metadata'],
                       gap_mask=gap_image, gap_conflicts=int((gap_row['status'] == CONFLICT).sum()),
                       edge_eligible_faces=int(edge_row['edge_eligible'].sum()),core_protected_faces=int(edge_row['core_protected'].sum()),
                       edge_band_pixels=float(edge_row['edge_band_pixels']),core_margin_pixels=float(edge_row['core_margin_pixels']))
        views.append(summary); cameras[view_id] = image['C']; results.append(row); gap_results.append(gap_row); edge_results.append(edge_row)
        _save(out/'PROGRESS.json', dict(processed_views=len(views), expected_views=len(frames), last_view=summary,
                                      elapsed_seconds=time.monotonic()-started))
        print('FACE_MASK_VIEW '+view_id+' conflicts='+str(summary['conflicts']), flush=True)
    try:
        code_root = Path(__file__).resolve().parents[2]
        for path in (Path(__file__), Path(__file__).with_name('face_mask_gate.py'),
                     Path(__file__).with_name('face_gap_gate.py'),
                     Path(__file__).with_name('face_edge_gate.py'),
                     code_root/'integrations/openmvs_depth/dmap.py', code_root/'integrations/mvs_surface/scene.py',
                     code_root/'integrations/viewer/ply_data.py', code_root/'integrations/viewer/export.py'):
            add(path)
        surface_record = source/'surface/result.json'; masks_record = source/'masks/RESULT.json'
        add(surface_record); add(masks_record)
        native, masks = _read(surface_record), _read(masks_record)
        _need(native.get('status') == 'SURFACE_GENERATED_NOT_ACCEPTED' and not native.get('input_drift'), 'native surface record refused')
        _need(masks.get('coverage_status') == 'COMPLETE' and masks.get('adapter_source_chain') == 'VERIFIED'
              and not masks.get('input_drift') and not masks.get('output_drift'), 'mask source chain refused')
        mesh_records = [g for stage in native['stages'] for g in stage.get('outputs', {}).get('geometry', [])
                        if g.get('role') == 'pre_cleanup_surface']
        _need(len(mesh_records) == 1, 'unique native raw mesh required')
        raw = _inside(mesh_records[0]['path'], source); raw_ref = add(raw, mesh_records[0])
        mesh = read_ply(raw, **LIMITS)
        vertices, faces = np.asarray(mesh['vertices'], float), np.asarray(mesh['faces'], dtype=np.int64)
        _need(vertices.ndim == 2 and faces.ndim == 2 and len(faces) > 0, 'nonempty triangle mesh required')
        colors = np.asarray(mesh['colors'], dtype=np.uint8) if mesh['colors'] is not None else None
        scene_path = source/'bridge/scene.mvs'; add(scene_path, masks['scene_binding']); scene = read_scene(scene_path)
        _need(np.array_equal(scene['transform'], np.eye(4)), 'nonidentity scene transform unsupported')
        _need(1 <= len(scene['images']) <= 256, 'view count budget refused')
        frames = {f['exported_name']: f for f in masks['frames']}
        _need(len(frames) == len(masks['frames']) == len(scene['images']), 'duplicate/missing mask view')
        images = {Path(i['name']).name: i for i in scene['images']}
        _need(len(images) == len(scene['images']) and set(images) == set(frames), 'mask/scene image set mismatch')
        _need(len({i['global_id'] for i in images.values()}) == len(images), 'duplicate scene global ID')
        for name, image in images.items():
            frame = frames[name]
            _need(frame.get('status') == 'OK' and frame.get('readback_verified') is True
                  and frame['target_image_id'] == image['global_id'], 'mask image identity/status mismatch')
        _save(out/'PLAN.json', dict(schema='face-mask-quarantine-plan-v3', policy=asdict(policy),
              gap_policy=asdict(gap_policy),edge_policy=asdict(edge_policy),
              rule_combination='SAME_VIEW_EDGE_ELIGIBILITY_THEN_INDEPENDENT_QUORUM_OR_THEN_CORE_VETO', source=str(source),
              source_mesh=raw_ref, views=len(images), check_all_faces=True, max_seconds=210, threads=2,
              coordinates='Native MVSI and dmap K/R/C; integer pixel centres; no extra half-pixel shift',
              physical_acceptance=False, limitations=LIMITATIONS))
        depth_paths = sorted((source/'surface/work').glob('depth*.dmap')); processed = set()
        _need(len(depth_paths) <= len(images), 'depth count exceeds source views')
        for path in depth_paths:
            check(); d = read_dmap(path)
            name = Path(d.image_name_utf8.replace('\\', '/')).name
            _need(name in images and name not in processed, 'unknown/duplicate saved depth identity')
            verify_depth(d, images[name]); add(path, dict(sha256=d.sha256, bytes=d.file_bytes))
            process(images[name], d); processed.add(name); del d
        for name in sorted(set(images)-processed):
            process(images[name])
        _need(len(results) == len(images), 'not all views evaluated')
        combined=aggregate_edge_rules(vertices,faces,results,gap_results,edge_results,cameras,policy,gap_policy,check=check)
        aggregated,gap_aggregated,quarantine=combined['depth'],combined['gap'],combined['quarantine']
        edge_depth,edge_gap=combined['edge_depth'],combined['edge_gap']
        previous_v2,protected_any,eligible_any=combined['previous_v2_quarantine'],combined['protected_any'],combined['edge_eligible_any']
        v1_quarantine = np.asarray(aggregated['quarantine'], bool)
        gap_quarantine = np.asarray(gap_aggregated['quarantine'], bool)
        _need(quarantine.shape == (len(faces),) and not quarantine.all(), 'invalid/empty retained partition')
        retained_ids, quarantine_ids = np.flatnonzero(~quarantine), np.flatnonzero(quarantine)
        _need(len(retained_ids)+len(quarantine_ids) == len(faces), 'partition incomplete')
        check()
        outputs = {}
        for label, ids in [('retained', retained_ids), ('quarantined', quarantine_ids)]:
            outputs[label] = write_partition(out/(label+'.ply'), vertices, faces, colors, ids)
        shutil.copyfile(raw, out/'original.ply'); outputs['original'] = bind(out/'original.ply', raw_ref)
        with (out/'FACE_LEDGER.csv').open('x', newline='', encoding='utf-8') as f:
            writer = csv.writer(f); writer.writerow(['source_face_id','decision','conflict_views','no_conflict_views',
                'unknown_views','angle_pair_evaluated','maximum_conflict_angle_degrees','witness_view_1','witness_view_2',
                'rule','gap_conflict_views','gap_no_conflict_views','gap_unknown_views','gap_angle_pair_evaluated',
                'gap_maximum_conflict_angle_degrees','gap_witness_view_1','gap_witness_view_2',
                'previous_v2_quarantined','edge_eligible_any','core_protected_any','edge_rule',
                'edge_v1_conflict_views','edge_v1_maximum_conflict_angle_degrees','edge_v1_witness_view_1','edge_v1_witness_view_2',
                'edge_gap_conflict_views','edge_gap_maximum_conflict_angle_degrees','edge_gap_witness_view_1','edge_gap_witness_view_2'])
            for i in range(len(faces)):
                pair = aggregated['witness_view_ids'][i] or (None, None)
                gap_pair = gap_aggregated['witness_view_ids'][i] or (None, None)
                rule = 'both' if v1_quarantine[i] and gap_quarantine[i] else 'v1' if v1_quarantine[i] else 'v2' if gap_quarantine[i] else 'none'
                edge_pair=edge_depth['witness_view_ids'][i] or (None,None)
                edge_gap_pair=edge_gap['witness_view_ids'][i] or (None,None)
                edge_rule='core_veto' if protected_any[i] else 'both' if edge_depth['quarantine'][i] and edge_gap['quarantine'][i] else 'v1' if edge_depth['quarantine'][i] else 'v2' if edge_gap['quarantine'][i] else 'none'
                writer.writerow([i, 'QUARANTINED_CANDIDATE' if quarantine[i] else 'RETAINED_NOT_ACCEPTED',
                    int(aggregated['conflict_views'][i]), int(aggregated['no_conflict_views'][i]),
                    int(aggregated['unknown_views'][i]), bool(aggregated['angle_pair_evaluated'][i]),
                    float(aggregated['max_conflict_angle_degrees'][i]) if aggregated['angle_pair_evaluated'][i] else None, *pair,
                    rule,int(gap_aggregated['conflict_views'][i]),int(gap_aggregated['no_conflict_views'][i]),
                    int(gap_aggregated['unknown_views'][i]),bool(gap_aggregated['angle_pair_evaluated'][i]),
                    float(gap_aggregated['max_conflict_angle_degrees'][i]) if gap_aggregated['angle_pair_evaluated'][i] else None,*gap_pair,
                    bool(previous_v2[i]),bool(eligible_any[i]),bool(protected_any[i]),edge_rule,
                    int(edge_depth['conflict_views'][i]),float(edge_depth['max_conflict_angle_degrees'][i]) if edge_depth['angle_pair_evaluated'][i] else None,*edge_pair,
                    int(edge_gap['conflict_views'][i]),float(edge_gap['max_conflict_angle_degrees'][i]) if edge_gap['angle_pair_evaluated'][i] else None,*edge_gap_pair])
        _save(out/'VIEWS.json', views)
        # A few deterministic counterexample projections, not a whole-object quality score.
        display_rows = []
        for row, gap_row, edge_row in zip(results, gap_results, edge_results):
            display = dict(row)
            display['status'] = np.where(((row['status'] == CONFLICT) | (gap_row['status'] == CONFLICT)) & edge_row['edge_eligible'], CONFLICT, UNKNOWN)
            display['background_pixels'] = np.maximum(row['background_pixels'], gap_row['background_pixels'])
            display_rows.append(display)
        ordered = sorted(zip(views, display_rows), key=lambda x: (-int(np.count_nonzero((x[1]['status'] == CONFLICT)&quarantine)), int(x[0]['view_id'])))
        for view, row in ordered[:4]:
            ids = np.flatnonzero((row['status'] == CONFLICT)&quarantine)
            if not len(ids):
                continue
            ids = ids[np.argsort(-row['background_pixels'][ids], kind='stable')[:64]]
            image = view['camera']; frame = frames[view['image_name']]
            image_path = _inside(source/'bridge'/image['name'], source); add(image_path)
            with Image.open(frame['mask']['path']) as m:
                labels = np.asarray(m) == 255
            name = 'candidate_view_'+str(frame['input_index'])+'.png'
            _overlay(out/name, image_path, labels, vertices, faces, ids, image,
                     'Frame '+str(frame['input_index'])+' / largest background evidence, '+str(len(ids))+' faces')
            result['overlays'].append(name)
        drift = [b['path'] for b in bindings if bind(b['path']) != b]
        _need(not drift, 'source changed during face gate: '+str(drift))
        result.update(status='CANDIDATE_SURFACE_NOT_ACCEPTED', source_unchanged=True, input_drift=[],
            counts=dict(original_faces=len(faces), quarantined_faces=len(quarantine_ids), retained_faces=len(retained_ids),
                        v1_quarantined=int(v1_quarantine.sum()),gap_quarantined=int(gap_quarantine.sum()),
                        gap_only_quarantined=int((gap_quarantine & ~v1_quarantine).sum()),
                        previous_v2_quarantined=int(previous_v2.sum()),restored_from_v2=int((previous_v2 & ~quarantine).sum()),
                        core_protected_faces=int(protected_any.sum()),edge_eligible_faces=int(eligible_any.sum()),
                        edge_v1_quarantined=int((edge_depth['quarantine'] & ~protected_any).sum()),
                        edge_gap_quarantined=int((edge_gap['quarantine'] & ~protected_any).sum()),
                        original_vertices=len(vertices), views=len(views), depth_views=len(processed),
                        missing_depth_views=len(views)-len(processed),
                        faces_with_any_conflict=int((aggregated['conflict_views'] > 0).sum()),
                        faces_with_no_evaluable_view=int((aggregated['unknown_views'] == len(views)).sum()),
                        gap_faces_with_any_conflict=int((gap_aggregated['conflict_views'] > 0).sum()),
                        gap_faces_with_no_evaluable_view=int((gap_aggregated['unknown_views'] == len(views)).sum())),
            subset_of_previous_v2_assert=True,non_edge_faces_unchanged_assert=not bool(np.any(quarantine & ~eligible_any)),
            core_faces_retained_assert=not bool(np.any(quarantine & protected_any)),
            projection_scope='Opaque projection coverage cannot decrease relative to the same-run v2 partition because v3 retained faces are its superset; not physical visibility or 3D accuracy.',
            topology=dict(original=_topology(faces), retained=_topology(faces[retained_ids])),
            outputs=outputs, source_bindings=bindings, elapsed_seconds=time.monotonic()-started)
        # Frozen evidence stays unchanged while HTML and Blender consume it.
        _save(out/'EVIDENCE.json', result)
        from integrations.viewer.export import export_html
        def spec_entry(label, title):
            return dict(ply=str(out/(label+'.ply')), sha256=outputs[label]['sha256'], title=title,
                        data_kind='real_photos', declared_unit=None,
                        note='Candidate mask-conflict partition. Geometry/visibility not accepted; holes preserved.',
                        evidence=[str(out/'EVIDENCE.json'), str(out/'PLAN.json')])
        _save(out/'comparison_scenes.json', dict(schema='ply-viewer-scenes-v1', scenes=[
            spec_entry('original', '原始表面'), spec_entry('retained', '候选隔离后 / 未验收')]))
        export_html(out/'comparison_scenes.json', out/'comparison_viewer.html', profile='surface400k')
        _save(out/'retained_scenes.json', dict(schema='ply-viewer-scenes-v1', scenes=[
            spec_entry('retained', '单独修正版候选 / 未验收')]))
        export_html(out/'retained_scenes.json', out/'retained_viewer.html', profile='surface400k')
        if len(quarantine_ids):
            _save(out/'quarantined_scenes.json', dict(schema='ply-viewer-scenes-v1', scenes=[spec_entry('quarantined', '隔离面 / 可逆候选')]))
            export_html(out/'quarantined_scenes.json', out/'quarantined_viewer.html', profile='surface400k')
        check()
        _need(all(bind(b['path']) == b for b in bindings), 'source drift during final export')
    except Exception as exc:
        result.update(status='FAILED', error=type(exc).__name__+': '+str(exc), source_unchanged=False,
                      processed_views=len(views), elapsed_seconds=time.monotonic()-started)
    _save(out/'RESULT.json', result); _render(out, result)
    return result
