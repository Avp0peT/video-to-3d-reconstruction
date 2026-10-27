"""Bounded candidate regions anchored by existing world points, never anatomy."""
from pathlib import Path
import heapq
import hashlib
import html
import json
import sys
import time
import shutil
import numpy as np
from PIL import Image
from integrations.video_masks.region_surface_export import _raw_binary_structure
from integrations.video_masks.boundary_geometry_binding import _bind, _read, _need


def world_groups(ids, xyz):
    ids, xyz = np.asarray(ids), np.asarray(xyz, float)
    _need(ids.ndim == 1 and len(ids) >= 3 and len(set(ids.tolist())) == len(ids) and xyz.shape == (len(ids), 3) and np.isfinite(xyz).all(), 'unique world points required')
    order = np.argsort(ids); ids, xyz = ids[order], xyz[order]
    anchors = [int(np.argmin(np.sum((xyz-np.median(xyz, axis=0))**2, axis=1)))]
    for _ in range(2):
        distance = np.min(np.stack([np.sum((xyz-xyz[j])**2, axis=1) for j in anchors]), axis=0)
        distance[anchors] = -1
        anchors.append(int(np.argmax(distance)))
    assigned = np.argmin(np.stack([np.sum((xyz-xyz[j])**2, axis=1) for j in anchors]), axis=0)
    return [dict(anchor_id=int(ids[j]), point_ids=ids[assigned == g].tolist()) for g, j in enumerate(anchors)]


def candidate_domains(rgb, mask, valid, seeds, check=lambda: None):
    """Conservative all-pixel coarse cells; no unobserved seed propagation."""
    _need(mask.dtype == valid.dtype == np.bool_ and mask.shape == valid.shape and rgb.shape == (*mask.shape, 3), 'same-grid RGB/mask/valid required')
    h, w = mask.shape; hh, ww = h//4, w//4
    allowed = (mask[:hh*4, :ww*4] & valid[:hh*4, :ww*4]).reshape(hh, 4, ww, 4).all(axis=(1, 3))
    colors = rgb[:hh*4, :ww*4].reshape(hh, 4, ww, 4, 3).mean(axis=(1, 3))
    distances = np.full((3, hh, ww), np.inf)
    usable = []
    for g, points in enumerate(seeds):
        queue = []; used = 0
        for x, y in points:
            xx, yy = int(np.floor(x/4)), int(np.floor(y/4))
            if 0 <= yy < hh and 0 <= xx < ww and allowed[yy, xx] and distances[g, yy, xx] != 0:
                distances[g, yy, xx] = 0; heapq.heappush(queue, (0., yy, xx)); used += 1
        usable.append(used); pops = 0
        while queue:
            if pops % 1024 == 0: check()
            pops += 1; cost, y, x = heapq.heappop(queue)
            if cost != distances[g, y, x]: continue
            for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                yy, xx = y+dy, x+dx
                if not (0 <= yy < hh and 0 <= xx < ww and allowed[yy, xx]): continue
                d = cost + 4*(1+4*np.abs(colors[y, x]-colors[yy, xx]).mean()/255)
                if d <= 96 and d < distances[g, yy, xx]:
                    distances[g, yy, xx] = d; heapq.heappush(queue, (d, yy, xx))
    best = distances.min(axis=0); result = []
    for g in range(3):
        coarse = np.isfinite(distances[g]) & (distances[g] <= best+8) & (distances[g] <= 96)
        full = np.zeros((h, w), bool); full[:hh*4, :ww*4] = coarse.repeat(4, axis=0).repeat(4, axis=1)
        result.append(full & mask & valid)
    return result, usable


def prepare(subset_root, plan_path, output):
    subset_root, plan_path = Path(subset_root).resolve(strict=True), Path(plan_path).resolve(strict=True)
    output = Path(output).resolve()
    _need(not output.exists() and not output.is_relative_to(subset_root) and not subset_root.is_relative_to(output) and not output.is_relative_to(plan_path.parent), 'fresh disjoint output required')
    output.mkdir(parents=True); started = time.monotonic(); refs = []
    def save(name, data): (output/name).write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')
    save('COMMAND.json', dict(argv=sys.argv, executable=sys.executable, threads=2, native_execution=False))
    def check():
        if time.monotonic()-started > 300: raise TimeoutError('300 second domain deadline')
    try:
        plan = _read(plan_path, refs); selected = _read(subset_root/'SELECTED_VIEWS.json', refs)
        _need(plan['contract_version'] == 'mvs-surface-run-v1' and Path(plan['scene']['path']).resolve() == subset_root/'scene.mvs', 'bound subset plan required')
        _bind(subset_root/'scene.mvs', refs, plan['scene'])
        _read(subset_root/'DERIVATION.json', refs)
        for name in ('images.bin', 'points3D.bin', 'cameras.bin', 'frames.bin', 'rigs.bin'): _bind(subset_root/'undistorted/sparse'/name, refs)
        for path in (__file__, Path(__file__).resolve().parents[2]/'scripts/prepare_overlapping_mvs_domains.py', Path(__file__).resolve().parents[1]/'video_masks/region_surface_export.py'): _bind(path, refs)
        model = _raw_binary_structure(subset_root/'undistorted/sparse')
        frames = {Path(f['image']['name']).name: f for f in plan['frames']}
        _need(len(frames) == len(plan['frames']) == len(selected['rows']) == len(model['images']) == 12, '12-view identity required')
        loaded = {}; observations = {}
        for r in selected['rows']:
            check(); iid = r['export_image_id']; f = frames[r['exported_name']]
            _need(model['images'][iid]['name'] == r['exported_name'], 'native image identity differs')
            paths = {k: _bind(f[k]['path'], refs, f[k]) for k in ('image_file', 'mask', 'validity')}
            rgb = np.array(Image.open(paths['image_file']).convert('RGB'))
            m = np.array(Image.open(paths['mask'])); v = np.array(Image.open(paths['validity']))
            _need(m.shape == v.shape == rgb.shape[:2] and np.isin(m, [0, 255]).all() and np.isin(v, [0, 255]).all(), 'binary native rasters required')
            loaded[iid] = (rgb, m != 0, v != 0, paths, r)
        eligible = []
        for pid, p in sorted(model['points'].items()):
            obs = []
            for iid, j in p['track']:
                xy = np.array(model['images'][iid]['rows'][j][:2])-.5
                x, y = np.floor(xy+.5).astype(int); _, m, v, _, _ = loaded[iid]
                if 0 <= y < m.shape[0] and 0 <= x < m.shape[1] and m[y, x] and v[y, x]: obs.append((iid, xy.tolist()))
            if len({o[0] for o in obs}) >= 3:
                eligible.append(pid); observations[pid] = obs
        groups = world_groups(eligible, [model['points'][pid]['xyz'] for pid in eligible])
        for g in groups:
            g['candidate_group_id'] = hashlib.sha256(json.dumps(g, sort_keys=True).encode()).hexdigest()
        save('GROUPS.json', dict(groups=groups, eligible_points=len(eligible), anatomical_identity='NOT_ESTABLISHED', coordinate_convention='native observation xy minus 0.5 for MVS pixel-centre comparison only'))
        rows = []; cards = []; colors = np.array([[255, 80, 80], [80, 255, 80], [80, 160, 255]])
        for iid, (rgb, m, v, paths, r) in loaded.items():
            check(); seeds = [[xy for pid in g['point_ids'] for i, xy in observations[pid] if i == iid] for g in groups]
            domains, usable = candidate_domains(rgb, m, v, seeds, check)
            control = output/'control/masks'/Path(paths['mask']).name; control.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(paths['mask'], control)
            _bind(control, [], next(b for b in refs if b['path'] == str(paths['mask'])))
            union = np.logical_or.reduce(domains); unknown = m & v & ~union
            row = dict(decode_index=r['decode_index'], export_image_id=iid, foreground_pixels=int(m.sum()), unknown_pixels=int(unknown.sum()), union_pixels=int(union.sum()), overlap_pixels=int((np.stack(domains).sum(axis=0) > 1).sum()), groups=[])
            h, w = m.shape; canvas = Image.new('RGB', (w*3, h))
            for g, domain in enumerate(domains):
                folder = output/f'group_{g}/masks'; folder.mkdir(parents=True, exist_ok=True)
                path = folder/Path(paths['mask']).name; Image.fromarray(domain.astype(np.uint8)*255).save(path)
                additions = int((domain & (~m | ~v)).sum()); _need(additions == 0, 'forbidden pixels added')
                row['groups'].append(dict(group=g, mask=str(path), actual_seed_observations=len(seeds[g]), usable_seed_cells=usable[g], covered_pixels=int(domain.sum()), foreground_coverage=float(domain.sum()/m.sum()) if m.any() else None, forbidden_added_pixels=additions, status='CANDIDATE_NOT_PART_IDENTITY' if domain.any() else 'UNKNOWN_NO_USABLE_SEED'))
                image = rgb.copy(); image[domain] = (.55*image[domain]+.45*colors[g]).astype(np.uint8); canvas.paste(Image.fromarray(image), (g*w, 0))
            png = f'frame_{r["decode_index"]:06d}.png'; canvas.save(output/png)
            Image.fromarray(unknown.astype(np.uint8)*255).save(output/f'unknown_{r["decode_index"]:06d}.png')
            cards.append((r['decode_index'], png, row)); rows.append(row)
        save('SEEDS.json', {str(pid): dict(xyz=model['points'][pid]['xyz'], observations=observations[pid]) for pid in eligible})
        post = []
        for b in refs: check(); _bind(b['path'], post, b)
        save('INPUT_BINDINGS.json', refs); save('INPUT_BINDINGS_POST.json', post)
        active = {str(g): [r['decode_index'] for r in rows if r['groups'][g]['covered_pixels'] > 0] for g in range(3)}
        result = dict(status='OVERLAPPING_DOMAINS_PREPARED_NOT_NATIVE_EXECUTED', native_executions=0, original_rgb_modified=False, original_masks_changed=False, eligible_points=len(eligible), rows=rows, active_view_sets=active, current_runner_all_views_usable={g: len(ids) == len(rows) for g, ids in active.items()}, mask_consumption='NOT_VERIFIED', elapsed_seconds=time.monotonic()-started, policy=dict(groups=3, grid_scale=4, edge_weight=4, overlap_cost=8, maximum_cost=96, seconds=300, threads=2), limitations=['Spatial groups are candidates, not physical parts or verified identity.', 'No-seed and unreachable regions remain unknown, including any missing thumb anchors.', 'Original full-object RGB and target occluders remain untouched; only reference candidate masks are produced.', 'Zero masks are legal preparation artifacts but do not establish current runner compliance.', 'Native mask consumption must be verified before interpreting a partition experiment.'])
        save('RESULT.json', result)
        body = '<!doctype html><meta charset="utf-8"><title>重叠候选域准备</title><h1>世界坐标锚点的重叠候选域</h1><p>仅准备，未运行原生程序；不是解剖部位分类。青绿红色为候选域，原RGB与遮挡物未修改。空域为未知。目标840置首。</p>'
        for i, png, row in sorted(cards, key=lambda x: (x[0] != 840, x[0])):
            body += f'<h2>帧{i}</h2><p>前景{row["foreground_pixels"]}，并集{row["union_pixels"]}，未知{row["unknown_pixels"]}，重叠{row["overlap_pixels"]}</p><pre>{html.escape(json.dumps(row["groups"], ensure_ascii=False, indent=2))}</pre><img style="max-width:100%" src="{png}">'
        (output/'index.html').write_text(body, encoding='utf-8')
        save('EXIT.json', dict(returncode=0, native_executions=0)); return result
    except Exception as exc:
        save('FAILURE.json', dict(error=repr(exc), status='FAILED_NOT_NATIVE_EXECUTED')); save('EXIT.json', dict(returncode=1, native_executions=0)); raise
