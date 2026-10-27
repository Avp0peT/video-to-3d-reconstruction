"""Fuse committed same-frame SAM3 image/tracker evidence, without model execution."""
from __future__ import annotations

from collections import Counter
import copy
import html
import os
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image, ImageDraw

from integrations.demo_pipeline.pipeline import read, save, sha
from .full_pipeline import _mask, gather_predictions, _attach_boundary_review, _automatic_boundary_html
from .hole_fusion import repair_holes, union_masks


def combine_rows(run: Path, plan: dict, rows: list, pool: dict, mode: str = 'union') -> tuple:
    """Write derived masks and an audit. All input masks remain byte-for-byte intact."""
    if mode not in ('union', 'supported-holes'):
        raise ValueError('unknown fusion mode')
    folder = run / 'fusion_masks'
    folder.mkdir(exist_ok=False)
    new_rows, records = [], []
    begun = time.monotonic()
    for row in rows:
        if time.monotonic() - begun > 240:
            raise RuntimeError('CPU fusion exceeded 240 seconds')
        index = row['decode_index']
        if not row.get('mask_path'):
            raise ValueError('fusion requires a committed base mask for every frame')
        base = _mask(row, plan['width'], plan['height'])
        evidence, supports, seen = [], [], set()
        for item in pool.get(index, []):
            key = (item['mask_sha256'], item['root_seed_decode_index'])
            if key in seen:
                continue
            seen.add(key)
            supports.append(dict(mask=_mask(item, plan['width'], plan['height']),
                                 root_seed_decode_index=item['root_seed_decode_index']))
            evidence.append({k: item.get(k) for k in ('mask_path', 'mask_sha256', 'direction',
                'root_seed_decode_index', 'chain_depth', 'task_id', 'evidence_origin')})
        if mode == 'union':
            pixels, audit = union_masks(base, supports)
        else:
            pixels, audit = repair_holes(base, supports, row['selected_root_seed_decode_index'])
        added = int(np.count_nonzero((base == 0) & (pixels == 255)))
        removed = int(np.count_nonzero((base == 255) & (pixels == 0)))
        if removed:
            raise ValueError('fusion removed foreground')
        record = dict(decode_index=index, mode=mode, base_mask_path=row['mask_path'],
            base_mask_sha256=row['mask_sha256'], source_status=row['status'],
            original_sam3_status=row['original_status'],
            has_original_image_segmentation=row['status'] == 'ORIGINAL_RETAINED',
            support_evidence=evidence, audit=audit, added_pixels=added, removed_pixels=removed,
            added_fraction_of_base=added / max(1, int(np.count_nonzero(base))),
            interpretation='union_is_not_agreement_or_ground_truth' if mode == 'union' else 'supported_holes_are_not_ground_truth')
        result_row = copy.deepcopy(row)
        result_row['fusion'] = record
        result_row['pre_fusion_status'] = row['status']
        result_row['pre_fusion_motion'] = {key: result_row.pop(key, None) for key in
            ('adjacent_area_ratio','adjacent_centroid_distance')}
        result_row['warning_flags'] = [flag for flag in result_row.get('warning_flags', []) if flag not in
            ('area_change_gt_1_5x_review_only','centroid_change_gt_0_15_review_only')]
        result_row['foreground_pixels'] = int(np.count_nonzero(pixels))
        ys, xs = np.where(pixels != 0)
        result_row['centroid_normalized'] = ([float(xs.mean()/plan['width']), float(ys.mean()/plan['height'])]
                                             if len(xs) else None)
        if added:
            target = folder / f'{index:06d}.png'
            Image.fromarray(pixels).save(target)
            with Image.open(target) as image:
                if image.mode != 'L' or not np.array_equal(np.asarray(image), pixels):
                    raise ValueError('fusion PNG readback mismatch')
            result_row.update(mask_path=str(target), mask_sha256=sha(target),
                status='FUSED_UNION_CANDIDATE' if mode == 'union' else 'SUPPORTED_HOLE_CANDIDATE',
                selection_rule=f'{mode}; base plus bound same-frame evidence',
                foreground_pixels=int(np.count_nonzero(pixels)))
            if record['added_fraction_of_base'] > 0.05:
                result_row['warning_flags'].append('fusion_added_over_5_percent_review')
        new_rows.append(result_row)
        records.append(record)
    for previous,current in zip(new_rows,new_rows[1:]):
        if previous['foreground_pixels'] and current['foreground_pixels']:
            ratio=current['foreground_pixels']/previous['foreground_pixels']
            current['adjacent_area_ratio']=ratio
            if ratio>1.5 or ratio<2/3:current['warning_flags'].append('area_change_gt_1_5x_review_only')
        if previous['centroid_normalized'] and current['centroid_normalized']:
            distance=sum((a-b)**2 for a,b in zip(previous['centroid_normalized'],current['centroid_normalized']))**0.5
            current['adjacent_centroid_distance']=distance
            if distance>0.15:current['warning_flags'].append('centroid_change_gt_0_15_review_only')
    summary = dict(mode=mode, frames=len(rows), changed_frames=sum(r['added_pixels'] > 0 for r in records),
        added_pixels=sum(r['added_pixels'] for r in records), removed_pixels=0,
        original_image_frames=sum(r['has_original_image_segmentation'] for r in records),
        holes_before=sum(r['audit']['holes_before'] for r in records),
        holes_after=sum(r['audit']['holes_after'] for r in records),
        hole_pixels_before=sum(r['audit']['hole_pixels_before'] for r in records),
        hole_pixels_after=sum(r['audit']['hole_pixels_after'] for r in records),
        source_masks_modified=False, model_entries=0, quality_status='NOT_EVALUATED')
    save(run/'FUSION.json', dict(summary=summary, rows=records))
    save(run/'ASSEMBLY.json', dict(rows=new_rows, source_masks_modified=False,
        counts=dict(Counter(r['status'] for r in new_rows)), quality_status='NOT_EVALUATED',
        published_for_reconstruction=False, fusion=summary))
    return new_rows, summary


def make_comparisons(run: Path, rows: list) -> list:
    folder = run/'comparisons'; folder.mkdir(exist_ok=False)
    changed = sorted((r for r in rows if r['fusion']['added_pixels']),
                     key=lambda r: (-r['fusion']['added_pixels'], r['decode_index']))
    selected = {r['decode_index']: r for r in changed[:10]}
    selected[0] = rows[0]
    items = []
    for index, row in sorted(selected.items()):
        before = np.array(Image.open(row['fusion']['base_mask_path']))
        after = np.array(Image.open(row['mask_path']))
        image = Image.open(run/'frames'/f'{index:06d}.jpg').convert('RGB')
        panels = []
        for pixels in (before, after):
            panel = image.copy()
            panel.paste(Image.blend(panel, Image.new('RGB', panel.size, (0,255,0)), .35), mask=Image.fromarray(pixels))
            panels.append(panel)
        diff = image.copy()
        added = Image.fromarray(np.uint8((before == 0) & (after == 255))*255)
        diff.paste(Image.new('RGB', image.size, (0,200,255)), mask=added)
        panels.append(diff)
        w,h = image.size
        canvas = Image.new('RGB', (w*3,h+36), (18,27,35))
        draw = ImageDraw.Draw(canvas)
        for j,(panel,label) in enumerate(zip(panels,('BEFORE','AFTER','ADDED PIXELS (CYAN)'))):
            canvas.paste(panel,(j*w,36));draw.text((j*w+10,10),f'{index:06d} {label}',fill='white')
        path = folder/f'{index:06d}.jpg';canvas.save(path,quality=93)
        items.append(dict(decode_index=index,path=str(path),added_pixels=row['fusion']['added_pixels']))
    save(run/'COMPARISONS.json',dict(rows=items))
    return items


def report(run: Path, result: dict, rows: list) -> None:
    comparisons = []
    comparison_notice = None
    manifest = run/'COMPARISONS.json'
    if manifest.is_file():
        try:
            comparisons = read(manifest)['rows']
            if not isinstance(comparisons,list):raise ValueError('comparison rows must be a list')
        except (OSError,ValueError,KeyError,TypeError) as exc:
            comparisons = []
            comparison_notice = 'COMPARISON_EVIDENCE_UNAVAILABLE: invalid COMPARISONS.json: '+str(exc)
    elif rows and not (run/'comparisons').exists():
        comparisons = make_comparisons(run, rows)
    else:
        comparison_notice = 'COMPARISON_EVIDENCE_UNAVAILABLE: no COMPARISONS.json; existing comparison files are not inferred as evidence.'
    if not comparisons and comparison_notice is None:
        comparison_notice = 'COMPARISON_EVIDENCE_UNAVAILABLE: the comparison manifest has no rows.'
    page = '<!doctype html><meta charset="utf-8"><title>SAM3 掩膜融合</title><style>body{font:16px system-ui;background:#15232d;color:#eef6fa;margin:32px}a{color:#9edfff}video{height:65vh;max-width:100%}img{max-width:100%}pre{white-space:pre-wrap}</style>'
    page += '<h1>SAM3 图像识别 + 视频跟踪：掩膜融合</h1><p>并集保留任一输入的前景，可补漏分，也会保留误分。原始识别、正反传播和新增像素都可追溯；没有把融合结果改称原始识别成功。</p>'
    page += f'<p>有原始掩膜的{result.get("fusion",{}).get("original_image_frames","待统计")}个锚帧：图像识别与视频预测合并。其余帧：使用已有正反跟踪预测；同一根种子不算两份独立识别证据。单向尾段没有凭空补出另一种识别。</p>'
    page += f'<p>状态：{html.escape(result["status"])}；轮廓质量未验收，未进入几何重建。</p><pre>{html.escape(str(result.get("fusion",{})))}</pre>'
    if (run/'overlay.mp4').exists():
        page += '<p><a href="overlay.mp4">融合叠加视频</a> · <a href="masks_preview.mp4">黑白预览</a> · <a href="masks_lossless.mkv">无损视频</a> · <a href="FUSION.json">逐帧融合证据</a></p><video controls src="overlay.mp4"></video>'
    page += '<h2>新增面积最多的帧：左原结果，中融合，右青色为新增像素</h2>'
    if comparison_notice:page += '<p>'+html.escape(comparison_notice)+'</p>'
    for item in comparisons:
        relative = f'comparisons/{item["decode_index"]:06d}.jpg'
        page += f'<p>帧 {item["decode_index"]}，新增 {item["added_pixels"]} 像素</p>'
        if (run/relative).is_file():
            page += '<img loading="lazy" src="'+relative+'">'
        else:
            page += '<p>'+html.escape('COMPARISON_IMAGE_MISSING: '+relative)+'</p>'
    page += _automatic_boundary_html(run,result)
    (run/'index.html').write_text(page,encoding='utf-8')


def execute(source: Path, output: Path, mode: str = 'union') -> dict:
    root = Path(__file__).resolve().parents[2]
    source, output = Path(source).resolve(strict=True), Path(output).resolve()
    if output.exists():
        raise ValueError('output must be new')
    if mode not in ('union','supported-holes'):
        raise ValueError('unknown fusion mode')
    source_result = read(source/'RESULT.json')
    if source_result.get('inference_source_run'):
        source = Path(source_result['inference_source_run']).resolve(strict=True)
        source_result = read(source/'RESULT.json')
    if source_result['status'] != 'FULL_TIMELINE_EXPORTED' or source_result['input_drift'] or source_result['code_drift']:
        raise ValueError('source execution was incomplete or drifted')
    plan = read(source/'PLAN.json'); rows = read(source/'ASSEMBLY.json')['rows']
    count = plan['frame_count']
    if not 0 < count <= 4096 or [r['decode_index'] for r in rows] != list(range(count)):
        raise ValueError('source frame set incomplete')
    if plan['width']*plan['height'] > 4194304:
        raise ValueError('frame exceeds CPU policy')
    pool = gather_predictions(source,plan)
    bindings = dict(read(source/'START_BINDINGS.json')['input_files'])
    for name in ('PLAN.json','RESULT.json','ASSEMBLY.json','DECODE.json','START_BINDINGS.json','worker/events.jsonl'):
        bindings[str(source/name)] = sha(source/name)
    images = read(source/'DECODE.json')['frames']
    if [r['decode_index'] for r in images] != list(range(count)):
        raise ValueError('canonical image sequence incomplete')
    for item in images:
        bindings[item['path']] = item['sha256']
    for item in rows + [r for entries in pool.values() for r in entries]:
        bindings[item['mask_path']] = item['mask_sha256']
    def changed(items):
        return [p for p,d in items.items() if not Path(p).is_file() or sha(Path(p)) != d]
    if changed(bindings):
        raise ValueError('source binding mismatch before fusion')
    code_paths = ['integrations/video_masks/fusion.py','integrations/video_masks/hole_fusion.py',
        'integrations/video_masks/full_pipeline.py','integrations/video_masks/video_export.py',
        'scripts/run_photo_demo.py','scripts/propagate_video_masks.py','scripts/fuse_video_masks.py']
    codes = {str(root/p):sha(root/p) for p in code_paths}
    output.mkdir(parents=True); (output/'frames').mkdir()
    save(output/'PLAN.json',plan);save(output/'SOURCE_ASSEMBLY.json',dict(rows=rows))
    save(output/'START_BINDINGS.json',dict(input_files=bindings,code=codes,mode=mode,source_run=str(source)))
    result = dict(status='FUSING',mode=mode,model_entries=0,source_run=str(source),
        quality_status='NOT_EVALUATED',published_for_reconstruction=False)
    save(output/'RESULT.json',result)
    fused = []
    try:
        for item in images:
            dst=output/'frames'/f'{item["decode_index"]:06d}.jpg'
            try:os.link(item['path'],dst)
            except OSError:shutil.copyfile(item['path'],dst)
        fused, result['fusion'] = combine_rows(output,plan,rows,pool,mode)
        source_plan = read(Path(plan['pipeline_plan_path']))
        from .video_export import export_video
        result['export'] = export_video(root,output,plan,fused,source_plan['video_tools']['ffmpeg'],source_plan['video_tools']['ffprobe'])
        result['status'] = 'FULL_TIMELINE_FUSED_EXPORTED'
    except Exception as exc:
        result.update(status='FAILED',error=f'{type(exc).__name__}: {exc}')
    finally:
        result['input_drift'] = changed(bindings);result['code_drift'] = changed(codes)
        if result['input_drift'] or result['code_drift']:result['status']='FAILED'
        save(output/'RESULT.json',result);report(output,result,fused)
    return _attach_boundary_review(output,result,fused,report)
