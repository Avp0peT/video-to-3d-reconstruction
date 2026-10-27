"""Render candidate reassociations without creating missing observations."""
from pathlib import Path
import hashlib
import html
import json
import math

import numpy as np
from PIL import Image


def render_reassociation(output: Path, recovery: dict, assessment: dict, frames: list, grays: list) -> dict:
    """Render at most six uniformly selected endpoint pairs from frozen gray arrays.

    Canonical keypoint coordinates have their first pixel center at (0.5, 0.5).
    PNG crops preserve original gray pixel values and never interpolate a gap.
    Array hashes describe the supplied arrays, not independent source-file binding.
    """
    out = Path(output).resolve()
    if out.exists():
        raise ValueError('fresh output directory required')
    if len(frames) != len(grays):
        raise ValueError('frames/gray array coverage differs')
    by_index = {}
    array_bindings = []
    original_indices = set()
    for sequence_index, (frame, gray) in enumerate(zip(frames, grays)):
        index = frame['input_index']
        if index in original_indices:
            raise ValueError('duplicate input index')
        original_indices.add(index)
        if not isinstance(gray, np.ndarray) or gray.dtype != np.uint8 or gray.ndim != 2 or not all(gray.shape):
            raise ValueError('nonempty uint8 gray array required')
        by_index[sequence_index] = (frame, gray)
        array_bindings.append(dict(frame_index=sequence_index, input_index=index, shape=list(gray.shape), array_sha256=hashlib.sha256(gray.tobytes()).hexdigest()))
    links = recovery.get('links', [])
    if not isinstance(links, list):
        raise ValueError('candidate links must be a list')
    # Validate producer sequence indices before sorting, indexing, or publishing.
    # bool is deliberately excluded even though it subclasses int in Python.
    for link in links:
        endpoints = [link.get('from_frame'), link.get('to_frame')]
        if any(type(i) is not int or not 0 <= i < len(frames) for i in endpoints):
            raise ValueError('endpoint frame index must be a bounded plain integer')
        start, end = endpoints
        missing = link.get('missing_frame_indices')
        if (not isinstance(missing, list)
                or any(type(i) is not int or not 0 <= i < len(frames) for i in missing)
                or len(set(missing)) != len(missing)):
            raise ValueError('missing frame indices must be unique bounded plain integers')
        expected = list(range(start + (1 if end > start else -1), end, 1 if end > start else -1))
        if missing != expected:
            raise ValueError('missing frame indices differ from ordered open endpoint interval')
    ordered = sorted(enumerate(links), key=lambda pair: (pair[1]['from_frame'], pair[1]['to_frame'], str(pair[1]['from_track_id']), str(pair[1]['to_track_id']), pair[0]))
    n = len(ordered)
    positions = sorted(set(round(i * (n-1) / 5) for i in range(6))) if n > 6 else list(range(n))
    for _, link in ordered:
        if link.get('status') != 'CANDIDATE_NOT_IDENTITY_VERIFIED':
            raise ValueError('candidate-only link status required')
        for endpoint in ('from', 'to'):
            index = link[endpoint + '_frame']
            if index not in by_index:
                raise ValueError('endpoint frame absent')
            gray = by_index[index][1]
            xy = link['xy_' + endpoint]
            if len(xy) != 2 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in xy):
                raise ValueError('finite observed endpoint coordinates required')
            if not (.5 <= xy[0] <= gray.shape[1]-.5 and .5 <= xy[1] <= gray.shape[0]-.5):
                raise ValueError('endpoint outside pixel-center domain')
    out.mkdir(parents=True, exist_ok=False)
    rows, cards = [], []
    for panel, position in enumerate(positions):
        original_index, link = ordered[position]
        crops, endpoints = [], []
        for endpoint in ('from', 'to'):
            index = link[endpoint + '_frame']
            frame, gray = by_index[index]
            xy = link['xy_' + endpoint]
            # Convert canonical center coordinates into zero-based pixel coordinates.
            px, py = xy[0]-.5, xy[1]-.5
            cx, cy = math.floor(px+.5), math.floor(py+.5)
            x0, y0 = max(0, cx-40), max(0, cy-40)
            x1, y1 = min(gray.shape[1], cx+41), min(gray.shape[0], cy+41)
            data = np.ascontiguousarray(gray[y0:y1, x0:x1])
            crops.append(Image.fromarray(data))
            endpoints.append(dict(endpoint=endpoint, frame_index=index, input_index=frame['input_index'], track_id=link[endpoint+'_track_id'],
                                  xy_canonical=list(xy), xy_zero_based_pixel=[px, py], crop_box_xyxy=[x0,y0,x1,y1],
                                  crop_pixel_sha256=hashlib.sha256(data.tobytes()).hexdigest(),
                                  time_seconds=frame.get('time_seconds'), model_ids=frame.get('model_ids'),
                                  original_mask_status=frame.get('original_mask_status')))
        width = sum(c.width for c in crops)+12
        height = max(c.height for c in crops)
        pair = Image.new('L', (width, height), 0)
        pair.paste(crops[0], (0,0)); pair.paste(crops[1], (crops[0].width+12,0))
        filename = 'endpoints_%02d.png' % panel
        pair.save(out/filename)
        with Image.open(out/filename) as check:
            check.load()
            if check.mode != 'L' or check.size != pair.size or check.tobytes() != pair.tobytes():
                raise ValueError('endpoint PNG pixel readback failed')
        row = dict(link_index=original_index, file=filename, endpoints=endpoints,
                   direction=link.get('direction'), missing_frame_indices=link.get('missing_frame_indices', []),
                   missing_input_indices=[frames[i]['input_index'] for i in link.get('missing_frame_indices', [])],
                   missing_observations_status='UNKNOWN_NOT_FILLED', metrics=link.get('metrics', {}),
                   status=link['status'], png_sha256=hashlib.sha256((out/filename).read_bytes()).hexdigest(),
                   png_pixel_sha256=hashlib.sha256(pair.tobytes()).hexdigest())
        rows.append(row)
        describe = html.escape(json.dumps(row, ensure_ascii=False, allow_nan=False))
        cards.append('<article><h2>候选帧序号 %s → %s · %s</h2><img src="%s" alt="两个实际观测端点的原始灰度裁剪"><p>缺失输入 index：%s；UNKNOWN，未生成缺失帧图像或坐标。</p><details><summary>真实端点坐标、裁剪框与像素哈希</summary><pre>%s</pre></details></article>' % (html.escape(str(link['from_frame'])), html.escape(str(link['to_frame'])), html.escape(str(link.get('direction',''))), filename, html.escape(str(row['missing_input_indices'])), describe))
    # Recheck supplied arrays to reject mutation during rendering.
    for receipt, gray in zip(array_bindings, grays):
        if receipt['array_sha256'] != hashlib.sha256(gray.tobytes()).hexdigest():
            raise ValueError('supplied gray array changed')
    result = dict(schema='candidate-reassociation-report-v1', status='CANDIDATE_NOT_IDENTITY_VERIFIED' if links else 'NO_CANDIDATES',
                  candidate_count=len(links), displayed_count=len(rows), selection_rule='ascending endpoint identities then uniform index spread; no metric-based selection',
                  counts=recovery.get('counts', {}), rejection_counts=recovery.get('rejection_counts', {}),
                  profile=recovery.get('profile', {}), proposal_summary=recovery.get('proposal_summary', {}),
                  patch_comparisons=recovery.get('patch_comparisons'),
                  assessment=assessment, baseline_continuity=recovery.get('baseline_continuity', recovery.get('baseline', {})),
                  joined_track_count=len(recovery.get('joined_tracks', [])), panels=rows, array_bindings=array_bindings,
                  array_binding_scope='Only supplied frozen uint8 arrays; source-file bindings belong to caller.',
                  missing_frames_filled=False, sfm_injected=False, geometry_acceptance=False,
                  joined_track_interpretation='Proposed joining preserves original observations and source IDs; it does not create observations or establish physical identity.')
    empty = '<p>没有可展示的候选（NO_CANDIDATES）。这是一项有效诊断结果，不能据此声称跟踪或重识别成功。</p>' if not links else ''
    escaped_assessment = html.escape(json.dumps(assessment, ensure_ascii=False, indent=2, allow_nan=False))
    escaped_baseline = html.escape(json.dumps(result['baseline_continuity'], ensure_ascii=False, indent=2, allow_nan=False))
    diagnostic_data = {key: result[key] for key in ('counts', 'rejection_counts', 'profile', 'proposal_summary', 'patch_comparisons')}
    diagnostics = '<h2>提议次数与拒绝原因</h2><p>rejection_counts 按端点 / 图对尝试累计，不是互斥的物理点计数。同一端点可能涉及多次尝试；这些数字不能解释为独立物理点数量。</p>'
    if result['patch_comparisons'] == 0:
        diagnostics += '<p>patch_comparisons=0：纹理检查未到达，不能据此判定纹理检查通过或失败。</p>'
    diagnostics += '<pre>' + html.escape(json.dumps(diagnostic_data, ensure_ascii=False, indent=2, allow_nan=False)) + '</pre>'
    document = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>临时缺失与候选重新关联</title><style>body{font:16px system-ui;background:#15191f;color:#eee;margin:28px;line-height:1.7}article{border:1px solid #555;padding:16px;margin:18px 0}img{image-rendering:pixelated;max-width:100%%;min-width:280px}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style><h1>视角变化：点出现、临时缺失与候选重新关联</h1><p>视角变化时可检测点会新增或暂时失去观测。跨缺失段的重新关联只是候选，不能证明同一物理点，也不能把缺失解释为遮挡或不可见。</p><p>全部关联均为 CANDIDATE_NOT_IDENTITY_VERIFIED；未注入 SfM，未生成缺失点坐标、图像或几何结果。下图只展示两个真实端点的原始灰度像素；黑色间隔只是版面分隔。坐标按 canonical xy − 0.5 转为零起点像素；裁剪未插值。</p><p>基线连续性表示原有观测链；提议重新关联表示跨段候选连接，原始观测与 source IDs 保留。两者不能作为几何真值或重识别验收。</p><p>共有 %s 个候选，按端点身份排序后均匀展示 %s 个，不按质量挑选。</p><h2>基线连续性</h2><pre>%s</pre><h2>候选重新关联评估数据</h2><pre>%s</pre>%s%s<p><a href="REPORT.json">REPORT.json：端点坐标、缺失 index、指标与输入数组哈希</a></p></html>' % (len(links),len(rows),escaped_baseline,escaped_assessment,empty,''.join(cards))
    document = document.replace('<h2>基线连续性</h2>', diagnostics + '<h2>基线连续性</h2>', 1)
    (out/'index.html').write_text(document,encoding='utf-8')
    (out/'REPORT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return result
