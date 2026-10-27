"""Object-neutral, bounded motion-mask windows with inspectable provenance.

This stage proposes image-motion domains. It neither repairs SAM masks nor
exports physical part identity, depth, or mesh acceptance.
"""
from pathlib import Path
import hashlib
import html
import json
import math
import shutil
import sys
import time

import cv2
import numpy as np
from PIL import Image

from .motion_parts import propose_motion_parts


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def binding(path):
    path = Path(path).resolve(strict=True)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))


def strict_json(path):
    def pairs(rows):
        result = {}
        for k, v in rows:
            if k in result:
                raise ValueError('duplicate JSON key: ' + k)
            result[k] = v
        return result
    def reject(value):
        raise ValueError('nonfinite JSON: ' + value)
    return json.loads(Path(path).read_text(encoding='utf-8'),
                      object_pairs_hook=pairs, parse_constant=reject)


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                     allow_nan=False), encoding='utf-8')


def _bound_file(value, base):
    if not isinstance(value, dict) or set(value) != {'path', 'sha256'}:
        raise ValueError('file binding requires exactly path and sha256')
    if not isinstance(value['path'], str) or not value['path']:
        raise ValueError('nonempty path required')
    path = Path(value['path'])
    path = (path if path.is_absolute() else base / path).resolve(strict=True)
    b = binding(path)
    if b['sha256'] != value['sha256']:
        raise ValueError('input hash drift: ' + str(path))
    return b


def validate_manifest(manifest, max_frames=120):
    path = Path(manifest).resolve(strict=True)
    frozen = binding(path)
    doc = strict_json(path)
    if binding(path) != frozen:
        raise ValueError('manifest changed while parsing')
    if not isinstance(doc, dict) or doc.get('schema') != 'motion-part-input-v1':
        raise ValueError('motion-part-input-v1 required')
    if not isinstance(doc.get('object_id'), str) or not doc['object_id'].strip():
        raise ValueError('explicit object instance ID required (not a category)')
    rows = doc.get('frames')
    if not isinstance(rows, list) or not 1 <= len(rows) <= max_frames:
        raise ValueError('frame count exceeds declared resource limit or is empty')
    seen = set(); previous = -math.inf; normalized = []
    refs = [frozen]
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError('frame must be an object')
        fid = row.get('frame_id'); t = row.get('time_seconds')
        if not isinstance(fid, str) or not fid or fid in seen:
            raise ValueError('unique nonempty frame_id required')
        if type(t) not in (int, float) or not math.isfinite(t) or t < 0 or t <= previous:
            raise ValueError('finite strictly increasing nonnegative times required')
        seen.add(fid); previous = t
        item = dict(frame_id=fid, time_seconds=t)
        for key in ('image', 'mask', 'validity'):
            if key == 'validity' and key not in row:
                continue
            item[key] = _bound_file(row.get(key), path.parent)
            refs.append(item[key])
        normalized.append(item)
    return doc['object_id'], normalized, refs


def _load(row):
    with Image.open(row['image']['path']) as image:
        if image.width * image.height > 4194304:
            raise ValueError('image exceeds 4194304 pixel processing cap; no silent resize')
        rgb = np.asarray(image.convert('RGB')).copy()
    with Image.open(row['mask']['path']) as image:
        raw = np.asarray(image).copy()
    if raw.ndim != 2 or raw.dtype != np.uint8 or not np.isin(raw, [0, 255]).all():
        raise ValueError('mask must be uint8 single-channel 0/255')
    if rgb.shape[:2] != raw.shape:
        raise ValueError('image/mask grid differs; no implicit resampling')
    if 'validity' in row:
        with Image.open(row['validity']['path']) as image:
            valid = np.asarray(image).copy()
        if valid.shape != raw.shape or valid.dtype != np.uint8 or not np.isin(valid, [0, 255]).all():
            raise ValueError('validity must match binary grid')
        if ((raw != 0) & (valid == 0)).any():
            raise ValueError('foreground on invalid domain; input not silently trimmed')
    return rgb, raw != 0


def windows(count, size):
    if type(count) is not int or count < 1 or type(size) is not int or not 3 <= size <= 6:
        raise ValueError('positive count and window size 3..6 required')
    start = 0
    while True:
        end = min(count, start + size)
        yield list(range(start, end))
        if end == count:
            break
        start = end - 2


def _write_mask(path, array):
    Image.fromarray(array.astype(np.uint8) * 255).save(path)
    with Image.open(path) as image:
        back = np.asarray(image)
    if not np.array_equal(back, array.astype(np.uint8) * 255):
        raise ValueError('mask pixel readback differs')
    return binding(path)


def run(input_manifest, output, *, window_size=6, max_frames=120,
        seconds=300, output_bytes=256 << 20, policy=None, coherence=True):
    started = time.monotonic()
    if type(coherence) is not bool:
        raise ValueError('coherence must be boolean')
    for value, name, lower, upper in ((max_frames, 'max_frames', 1, 1200),
            (seconds, 'seconds', 1, 3600), (output_bytes, 'output_bytes', 1024, 1 << 30)):
        if type(value) is not int or not lower <= value <= upper:
            raise ValueError('invalid resource limit: ' + name)
    source = Path(input_manifest).resolve(strict=True)
    out = Path(output).resolve()
    if out.exists() or out == source.parent or source.is_relative_to(out):
        raise ValueError('fresh output disjoint from input files required')
    def check():
        if time.monotonic() - started > seconds:
            raise TimeoutError('motion review time budget exceeded')
        if out.exists() and sum(p.stat().st_size for p in out.rglob('*') if p.is_file()) > output_bytes:
            raise RuntimeError('motion review output limit exceeded')
    object_id, rows, refs = validate_manifest(source, max_frames)
    check()
    for ref in refs:
        if Path(ref['path']).is_relative_to(out):
            raise ValueError('output contains an input')
    indices = list(windows(len(rows), window_size))
    # Validate every raster before creating any output. Only one frame resident.
    shape = None
    for row in rows:
        check()
        rgb, mask = _load(row)
        if shape is not None and rgb.shape != shape:
            raise ValueError('all frames must share one image grid')
        shape = rgb.shape
    code = [Path(__file__), Path(__file__).with_name('motion_parts.py'),
            Path(__file__).with_name('motion_part_coherence.py'),
            Path(__file__).with_name('motion_window_linking.py'),
            Path(__file__).with_name('motion_window_review.py'),
            Path(__file__).resolve().parents[2] / 'scripts/prepare_motion_part_masks.py',
            Path(__file__).resolve().parents[2] / 'scripts/run_region_mask_review.py']
    refs.extend(binding(p) for p in code)
    out.mkdir(parents=True)
    cv2.setNumThreads(2)
    result = dict(schema='motion-part-review-v1', status='RUNNING', object_id=object_id,
        frame_count=len(rows), window_count=len(indices), windows=[], native_executions=0,
        neural_inferences=0, geometry_accepted=False, mask_quality='NOT_EVALUATED',
        default_model_replaced=False, part_identity='WINDOW_LOCAL_MOTION_CANDIDATES',
        cross_window_identity='NOT_ESTABLISHED', whole_mask_policy='PRESERVE_INPUT_BYTES',
        coherence_refinement=coherence,
        boundary_semantics='TEMPORAL_MASK_OVERLAP_NOT_RGB_CONTOUR',
        source_mask_accuracy='NOT_ESTABLISHED', resources=dict(seconds=seconds,
        output_bytes=output_bytes, threads=2, hard_memory_limit=False,
        max_image_pixels=4194304,
        limit_mode='COOPERATIVE_CHECKS_NOT_HARD_PROCESS_DEADLINES'),
        limitations=['Image motion can arise from camera perspective, parallax, deformation or mask error.',
        'Co-moving physical parts cannot be separated solely by this evidence.',
        'Finite propagated domains are not precise part contours; original whole mask is retained.',
        'Window IDs are local; cross-window candidate links require explicit overlap and coordinate evidence.',
        'Low texture and occlusion remain unknown; no synthetic depth or hole filling.'])
    save(out/'INPUT_BINDINGS.json', refs)
    save(out/'COMMAND.json', dict(argv=sys.argv, executable=sys.executable,
        source_manifest=str(source), window_size=window_size, policy_overrides=policy,
        coherence_refinement=coherence))
    try:
        for number, selection in enumerate(indices):
            check()
            folder = out / ('window_%03d' % number); folder.mkdir()
            loaded = [_load(rows[i]) for i in selection]
            images, masks = [x[0] for x in loaded], [x[1] for x in loaded]
            raw_candidate = propose_motion_parts(images, masks, policy=policy)
            check()
            save(folder/'RAW_MOTION_EVIDENCE.json', raw_candidate['ledger'])
            if coherence:
                from .motion_part_coherence import refine_motion_parts
                candidate = refine_motion_parts(images, masks, raw_candidate)
                save(folder/'COHERENCE_EVIDENCE.json', candidate['refinement_ledger'])
            else:
                candidate = raw_candidate
            check()
            save(folder/'MOTION_EVIDENCE.json', candidate['ledger'])
            record = dict(window_id=number, status=candidate['status'],
                part_ids=list(candidate['part_masks']), raw_part_ids=list(raw_candidate['part_masks']),
                frames=[], source_positions=selection)
            colors = [(255, 144, 30), (0, 190, 255), (245, 70, 160), (180, 240, 50)]
            for f, position in enumerate(selection):
                check(); row = rows[position]; rgb, mask = loaded[f]
                prefix = '%06d' % position
                whole = folder / (prefix+'_whole.png')
                shutil.copyfile(row['mask']['path'], whole)
                if sha(whole) != row['mask']['sha256']:
                    raise ValueError('whole mask byte preservation failed')
                union = np.zeros(mask.shape, bool); parts = {}
                overlay = rgb.copy()
                overlay[mask] = (0.72 * overlay[mask] + .28 * np.array([30, 210, 70])).astype(np.uint8)
                raw_overlay=overlay.copy(); raw_union=np.zeros(mask.shape,bool); raw_parts={}
                for j,(name,frames) in enumerate(raw_candidate['part_masks'].items()):
                    raw_part=frames[f]; raw_union |= raw_part
                    raw_parts[name]=_write_mask(folder/(prefix+'_raw_'+name+'.png'),raw_part)
                    raw_overlay[raw_part]=(.40*rgb[raw_part]+.60*np.array(colors[j%len(colors)])).astype(np.uint8)
                for j, (name, frames) in enumerate(candidate['part_masks'].items()):
                    part = frames[f]
                    if (part & ~mask).any() or (part & union).any() or (part & ~raw_union).any():
                        raise ValueError('part outside input mask or ambiguous ownership')
                    if coherence and cv2.connectedComponents(part.astype(np.uint8), connectivity=4)[0] > 2:
                        raise ValueError('refined candidate contains disconnected regions')
                    union |= part
                    parts[name] = _write_mask(folder/(prefix+'_'+name+'.png'), part)
                    hue=np.uint8([[[int((j*.61803398875)%1*179),200,255]]])
                    color=cv2.cvtColor(hue,cv2.COLOR_HSV2RGB)[0,0]
                    overlay[part] = (.40 * rgb[part] + .60 * color).astype(np.uint8)
                fallback = candidate['fallback_masks'][f]
                if not np.array_equal(union | fallback, mask) or (union & fallback).any():
                    raise ValueError('whole object support lost or multiply assigned')
                fallback_binding = _write_mask(folder/(prefix+'_fallback.png'), fallback)
                unknown_binding = _write_mask(folder/(prefix+'_unknown.png'), candidate['unknown_masks'][f])
                canvas = Image.new('RGB', (rgb.shape[1]*3, rgb.shape[0]))
                canvas.paste(Image.fromarray(rgb), (0, 0))
                canvas.paste(Image.fromarray(raw_overlay), (rgb.shape[1],0))
                canvas.paste(Image.fromarray(overlay), (rgb.shape[1]*2,0))
                preview = prefix+'_preview.jpg'; canvas.save(folder/preview, quality=88)
                record['frames'].append(dict(frame_id=row['frame_id'], time_seconds=row['time_seconds'],
                    input_position=position, whole=binding(whole), parts=parts, raw_parts=raw_parts,
                    fallback=fallback_binding, motion_unknown=unknown_binding,
                    metrics=candidate['metrics'][f], coverage_loss_pixels=0,
                    raw_candidate_pixels=int(raw_union.sum()),
                    refined_candidate_pixels=int(union.sum()),
                    candidate_returned_to_global_pixels=int((raw_union&~union).sum()),
                    candidate_retention_fraction=float(union.sum()/raw_union.sum()) if raw_union.any() else None,
                    added_background_pixels=0, preview=preview))
            result['windows'].append(record)
            save(folder/'RESULT.json', record)
            save(out/'RESULT.json', result)
        if coherence:
            from .motion_window_review import build_review
            links = build_review(result, rows, out, out/'window_links', check=check)
            result['cross_window_association'] = dict(status=links['status'],
                accepted_link_count=links['accepted_link_count'],
                report=str(out/'window_links/index.html'),
                evidence=binding(out/'window_links/RESULT.json'))
            if links['accepted_link_count']:
                result['cross_window_identity'] = 'CANDIDATE_LINKS_ONLY_NOT_PHYSICAL_IDENTITY'
        else:
            result['cross_window_association'] = dict(status='NOT_RUN_RAW_CONTROL', accepted_link_count=0)
        changed = [b['path'] for b in refs if binding(b['path']) != b]
        if changed:
            raise ValueError('input/code changed during run: '+repr(changed))
        result.update(status='MOTION_MASK_WINDOWS_GENERATED_NOT_PHYSICAL_PARTS',
            input_drift=[], elapsed_seconds=time.monotonic()-started)
        check()
    except Exception as exc:
        result.update(status='FAILED', error=type(exc).__name__+': '+str(exc),
                      elapsed_seconds=time.monotonic()-started)
        save(out/'RESULT.json', result)
        raise
    save(out/'RESULT.json', result)
    body = ['<!doctype html><meta charset="utf-8"><title>通用运动候选掩膜</title>',
        '<style>body{max-width:1200px;margin:35px auto;font:16px system-ui;background:#101925;color:#e5edf4}img{max-width:100%}a{color:#77c9ff}summary{padding:12px;background:#203247;cursor:pointer}table{border-collapse:collapse}td,th{padding:7px;border:1px solid #465569}</style>',
        '<h1>空间连通与时序一致性对照</h1><p>左：输入图像；中：原运动候选；右：连通与时序检查后。绿色为保留的整体掩膜，其他颜色表示局部候选。颜色不表示真实物理部件。</p>',
        '<p>不指定物体类别、不指定部位数量或特殊帧。整体掩膜逐字节保留，未分区处保留整体支持。当前不修正原掩膜边界，也不输出新三维模型。</p>',
        '<p>细化只保留通过检查的原连通块，不加像素、不填缝。局部候选面积减少会单独报告，不把删去候选当作形状恢复或精度提升。</p>',
        '<p>相同编号不是跨窗身份。自动用重叠真实帧及持续点对应检查候选链接；歧义或证据不足时不合并。透视、深度差与遮挡可能形成假运动分组。</p>',
        '<p><a href="RESULT.json">完整结果</a> · <a href="INPUT_BINDINGS.json">输入绑定</a></p>']
    if coherence:
        body.append('<p><a href="window_links/index.html">跨窗口对应与同帧图像对照</a> · '+
                    html.escape(result['cross_window_association']['status'])+'</p>')
    for record in result['windows']:
        prefix = 'window_%03d/' % record['window_id']
        body.append('<h2>窗口 '+str(record['window_id'])+'：'+html.escape(record['status'])+'</h2>')
        body.append('<p>候选域数量 '+str(len(record['part_ids']))+'；本页编号仅在窗口内有效，跨窗链接另见对应报告。</p>')
        for f, row in enumerate(record['frames']):
            metrics = row['metrics']
            body.append('<details'+(' open' if f==0 else '')+'><summary>'+html.escape(row['frame_id'])+
                ' · '+str(row['time_seconds'])+'s · 整体像素 '+str(metrics['global_pixels'])+
                ' · 原候选 '+str(row['raw_candidate_pixels'])+' → 保留 '+str(row['refined_candidate_pixels'])+
                ' · 返回整体 '+str(row['candidate_returned_to_global_pixels'])+'</summary>')
            body.append('<img src="'+prefix+row['preview']+'"><p>')
            for key in ('whole', 'fallback', 'motion_unknown'):
                body.append('<a href="'+prefix+Path(row[key]['path']).name+'">'+key+'</a> · ')
            for name, b in row['parts'].items():
                body.append('<a href="'+prefix+Path(b['path']).name+'">'+html.escape(name)+'</a> · ')
            body.append('</p></details>')
    (out/'index.html').write_text(''.join(body), encoding='utf-8')
    try:
        check()
    except Exception as exc:
        result.update(status='FAILED',error=type(exc).__name__+': '+str(exc))
        save(out/'RESULT.json',result)
        raise
    return result
