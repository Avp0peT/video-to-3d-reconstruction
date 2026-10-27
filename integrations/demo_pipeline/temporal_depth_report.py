"""Bound temporal-window diagnostic of existing depth evidence; no native run."""
from fractions import Fraction
from pathlib import Path
import html
import json
import sys
import time
import numpy as np
from PIL import Image, ImageDraw
from .region_depth_support import project_pixels, region_raster
from .region_depth_tiers import validate_ledger
from .shape_support_audit import context
from .silhouette_review import project, raster
from .temporal_depth_consistency import analyze_windows
from integrations.video_masks.boundary_geometry_binding import _read, _bind, _need


def timeline_join(frame_rows, mask_rows, timeline_rows, view_ids):
    """Join through actual canonical mask identity, never positional/input_index."""
    def indexed(rows):
        result = {}
        for row in rows:
            k = row['decode_index']
            _need(type(k) is int and k >= 0 and k not in result, 'duplicate/invalid decode index')
            result[k] = row
        return result
    frames, masks, timeline = map(indexed, (frame_rows, mask_rows, timeline_rows))
    times, evidence = [], []
    for k in view_ids:
        _need(k in frames and k in masks and k in timeline, 'missing identity join')
        f, m, t = frames[k], masks[k], timeline[k]
        _need(Path(f['canonical_mask_path']).resolve() == Path(m['path']).resolve() and
              f['canonical_mask_sha256'] == m['sha256'], 'canonical mask identity mismatch')
        _need(m['pts'] == t['pts'] and m['time_base'] == t['time_base'], 'PTS identity mismatch')
        pts, base = t['pts'], t['time_base']
        _need(type(pts) is int and len(base) == 2 and all(type(x) is int and x > 0 for x in base), 'exact PTS integers required')
        sec = pts * Fraction(*base)
        times.append(sec)
        evidence.append(dict(decode_index=k, pts=pts, time_base=base,
                             seconds_fraction=[sec.numerator, sec.denominator],
                             canonical_mask_path=m['path'], canonical_mask_sha256=m['sha256']))
    _need(len(set(times)) == len(times) and all(a < b for a, b in zip(times, times[1:])), 'timeline must increase')
    return times, evidence


def run(source, ledger_run, region_json, mask_run, output):
    source, ledger_run, region_json, mask_run, output = map(lambda p: Path(p).resolve(), (source, ledger_run, region_json, mask_run, output))
    _need(not output.exists() and not any(output.is_relative_to(p) or p.is_relative_to(output) for p in (source, ledger_run, mask_run)), 'fresh disjoint output required')
    output.mkdir(parents=True)
    refs = []; started = time.monotonic()
    def save(name, obj):
        (output / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    def check():
        if time.monotonic() - started > 600:
            raise TimeoutError('temporal diagnostic 600 second limit')
    save('COMMAND.json', dict(argv=sys.argv, executable=sys.executable, native_executions=0, model_executions=0))
    try:
        root = Path(__file__).resolve().parents[2]
        for p in [Path(__file__), root/'integrations/demo_pipeline/temporal_depth_consistency.py', root/'scripts/audit_temporal_depth.py']:
            _bind(p, refs)
        old = _read(ledger_run/'RESULT.json', refs)
        _need(old['status'] == 'DEPTH_SUPPORTED_CANDIDATES_NOT_ACCEPTED' and old['source_unchanged'] is True, 'unrecognized ledger')
        pre = _read(ledger_run/'BINDINGS_PRE.json', refs)
        _need(pre == _read(ledger_run/'BINDINGS_POST.json', refs), 'ledger binding drift')
        for r in pre:
            check(); _bind(r['path'], refs, r)
        _, raw, _, _, views = context(source, None, refs)
        expected = [dict(decode_index=v['decode_index'], camera_name=v['camera']['name']) for v in views]
        spec = _read(region_json, refs)
        _need(any(Path(r['path']).resolve() == region_json for r in pre), 'unbound ROI')
        ids = [v['decode_index'] for v in views]
        target_index = ids.index(spec['decode_index']); target = views[target_index]
        region = region_raster(spec, target)
        ledger = _bind(ledger_run/'CANDIDATE_LEDGER.npz', refs)
        with np.load(ledger, allow_pickle=False) as a:
            points, states, origins, selected, other = [a[k].copy() for k in ('points', 'states', 'source_view', 'selected', 'other_agreeing')]
        validate_ledger(points, states, origins, selected, old['views'], expected, old)
        agree = states == 2; agree[:, target_index] = False; agree[np.arange(len(points)), origins] = False
        _need(other.dtype == np.bool_ and np.array_equal(other, agree), 'support ledger identity mismatch')
        frames = _read(source/'export/FRAME_MAP.json', refs)['frames']
        masks = _read(mask_run/'MANIFEST.json', refs)['rows']
        plan = _read(mask_run/'PLAN.json', refs)
        timeline_path = _bind(plan['timeline_path'], refs, dict(sha256=plan['timeline_sha256']))
        timeline = _read(timeline_path, refs)
        times, join = timeline_join(frames, masks, timeline['frames'], ids)
        for r in join:
            _bind(r['canonical_mask_path'], refs, dict(sha256=r['canonical_mask_sha256']))
        save('TIMELINE_JOIN.json', dict(policy='decode_index + canonical-mask path/hash + exact PTS', timeline=str(timeline_path), rows=join))
        save('INTENT.json', dict(windows_seconds=[1,2,4], candidate_universe=len(points), target_decode_index=ids[target_index],
             other_agree_min=3, pairwise_angle_degrees=8, original_relative_depth_tolerance=.02,
             originating_view_must_be_local=True, native_executions=0,
             interpretation='Exploratory evidence analysis, not an independent local reconstruction or causal test.'))
        c = target['camera']; check()
        baseline = raster(project(raw['vertices'], c['R'], c['C'], c['K']), raw['faces'], c['width'], c['height'], check=check)
        rows, layer_masks = analyze_windows(points, states, origins, [v['camera'] for v in views], target_index, times, region, baseline, selected, view_ids=ids, check=check)
        save('BINDINGS_PRE.json', refs)
        uv, _, _ = project_pixels(points, c)
        with Image.open(target['image']) as im:
            rgb = np.array(im.convert('RGB'))
        panel_rows = []; names = ('SUPPORT_ONLY','NO_LOCAL_FRONT_CONFLICT','NO_ALL_VIEW_FRONT_CONFLICT')
        for seconds in (1,2,4):
            cells = []
            for label in names:
                take = layer_masks[str(seconds)][label]
                hits = np.zeros(region.shape, bool); ii = np.flatnonzero(take); hits[uv[ii,1],uv[ii,0]] = True
                image = rgb.copy(); missing = region & ~baseline
                image[missing] = (.55*image[missing] + .45*np.array([255,150,0])).astype(np.uint8)
                image[hits] = [0,255,255]
                fn = f'window{seconds}_{label}.png'; Image.fromarray(image).save(output/fn)
                # Full-frame render deliberately preserves object/ROI context.
                tile = Image.new('RGB', (c['width'], c['height']+58), '#141b27')
                tile.paste(Image.fromarray(image),(0,58)); d=ImageDraw.Draw(tile)
                q=rows[(1,2,4).index(seconds)]['layers'][label]
                d.text((8,8),f'+/-{seconds}s  {label}',fill='white')
                d.text((8,30),f'samples={q["samples"]}  missing-pixels={q["raw_missing_unique_pixels"]}',fill='white')
                cells.append(tile)
            strip=Image.new('RGB',(sum(x.width for x in cells),cells[0].height))
            for j, cell in enumerate(cells): strip.paste(cell,(j*cell.width,0))
            strip.save(output/f'window{seconds}_comparison.jpg',quality=92)
            panel_rows.append(strip)
        near_ids=rows[1]['near_view_ids']+[ids[target_index]]
        tiles=[]
        for view in views:
            if view['decode_index'] not in near_ids: continue
            with Image.open(view['image']) as im:
                tile=im.convert('RGB');tile.thumbnail((188,342))
            card=Image.new('RGB',(188,366),'#141b27');card.paste(tile,(0,24))
            ImageDraw.Draw(card).text((5,5),f'frame {view["decode_index"]} / {float(times[ids.index(view["decode_index"])]):.3f}s',fill='white');tiles.append(card)
        contact=Image.new('RGB',(188*6,366*((len(tiles)+5)//6)),'#141b27')
        for j,tile in enumerate(tiles):contact.paste(tile,((j%6)*188,(j//6)*366))
        contact.save(output/'local_views.jpg',quality=91)
        packed={f'window{s}_{label}':mask for s, layers in layer_masks.items() for label,mask in layers.items()}
        np.savez_compressed(output/'WINDOW_MASKS.npz',**packed)
        post=[]
        for r in refs: check();_bind(r['path'],post,r)
        save('BINDINGS_POST.json',post)
        result=dict(status='TEMPORAL_DEPTH_DIAGNOSTIC_COMPLETE_NOT_ADOPTED', source_unchanged=True,
            target_decode_index=ids[target_index], roi_pixels=int(region.sum()), raw_missing_pixels=int((region & ~baseline).sum()),
            frozen_candidates=len(points), rows=rows, native_reconstruction_executions=0,
            thumb_repaired=False, finger_separation_accepted=False, default_model_replaced=False,
            elapsed_seconds=time.monotonic()-started,
            limitations=['Same global MVS depths/cameras in every window; no local reconstruction has run.',
            'Support cohort is selected by near-view agreement; near/far rates are conditioned and not causal.',
            'Unknown and behind depth are reported separately, never supporting votes.',
            'Projected missing-pixel coverage is not confirmed physical surface or metric accuracy.'])
        save('RESULT.json',result)
        body=''
        for row in rows:
            sec=row['window_seconds']['numerator']
            body+=f'<h2>±{sec} 秒</h2><table><tr><th>层</th><th>候选样本</th><th>目标投影像素</th><th>覆盖原缺失像素</th></tr>'
            for label,q in row['layers'].items():body+=f'<tr><td>{label}</td><td>{q["samples"]}</td><td>{q["target_unique_pixels"]}</td><td>{q["raw_missing_unique_pixels"]}</td></tr>'
            body+=f'</table><a href="window{sec}_comparison.jpg"><img src="window{sec}_comparison.jpg"></a>'
        (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>短时段深度一致性</title><style>body{background:#111923;color:#e9eef5;font:16px/1.6 system-ui;max-width:1300px;margin:auto;padding:24px}a{color:#8bd9ff}img{width:100%}table{border-collapse:collapse}td,th{border:1px solid #556;padding:8px}</style><h1>拇指区域：短时段深度一致性</h1><p>状态：诊断完成，未补入模型。橙色＝原网格未覆盖区域；青色＝现有深度候选投影。并非新测量或已确认表面。</p><p>所有窗口使用相同50000个冻结候选；生成视角必须在窗内，至少3个其它视角支持且方向两两相差8°，来源及目标不投票。比较均在同一窗内支持集合上进行。</p><p>注意：近处支持筛选自身带来选择偏差。局部通过而远处冲突，不能单独证明物体变形、相机错误，或忽略远处证据就能修复。</p><h2>±2秒源图（只展示，不代表已选择最佳窗口）</h2><img src="local_views.jpg">'+body+'<p><a href="RESULT.json">机器结果</a> · <a href="TIMELINE_JOIN.json">时间与身份链</a></p>',encoding='utf-8')
    except Exception as exc:
        save('FAILURE.json',dict(status='FAILED_NOT_ADOPTED',error=repr(exc),bindings=refs))
        raise
