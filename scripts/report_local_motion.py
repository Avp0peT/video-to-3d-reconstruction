"""Render saved local motion evidence without fitting, tracking or selecting masks."""
import argparse
from collections import Counter
import hashlib
import html
import json
import os
from pathlib import Path
import sys
from urllib.parse import quote

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from review_mask_association import read, save, bind, verify


def link(path, output):
    return quote(os.path.relpath(path, output).replace('\\', '/'), safe='/:')


def stats(values):
    return dict(count=len(values), mean_px=float(np.mean(values)) if values else None,
                p90_px=float(np.percentile(values, 90)) if values else None)


def num(value):
    return '—' if value is None else f'{value:.2f}'


def predictions(row, profile, side):
    """All computed local predictions, including rejected cells and bad points."""
    local = row['local_motion']
    ids = row['track_ids']
    lookup = {v: i for i, v in enumerate(ids)}
    source = np.asarray(row['points'][side], float)
    query = np.asarray(row['points']['query'], float)
    old_matrix = np.asarray(row['global_comparison'][profile]['references'][side]['saved_matrix'], float)
    values = []
    for cell in local['profiles'][profile]['references'][side]['cells']:
        if cell['matrix'] is None:
            continue
        matrix = np.asarray(cell['matrix'], float)
        for tid, expected in zip(cell['holdout_track_ids'], cell['holdout_residuals_px']):
            i = lookup[tid]
            predicted = source[i] @ matrix[:, :2].T + matrix[:, 2]
            error = float(np.linalg.norm(predicted-query[i]))
            if not np.isclose(error, expected, rtol=1e-7, atol=1e-6):
                raise ValueError('saved residual differs from actual coordinates')
            global_error = float(np.linalg.norm(source[i] @ old_matrix[:, :2].T + old_matrix[:, 2]-query[i]))
            values.append(dict(track_id=tid, cell_id=cell['cell_id'], query_xy=query[i].tolist(),
                               predicted_xy=predicted.tolist(), local_error_px=error,
                               global_error_px=global_error, cell_passed=cell['passed']))
    if len({v['track_id'] for v in values}) != len(values):
        raise ValueError('heldout counted more than once per side')
    return values


def build(source, output):
    source = source.resolve(strict=True)
    output = output.resolve()
    if output.exists() or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('fresh nonoverlapping output required')
    files = [source/n for n in ('PLAN.json', 'RESULT.json', 'DECISIONS.json')]
    refs = [bind(p) for p in files]
    plan, result, decisions = map(read, files)
    if result['frames'] != decisions or result['selected_candidates'] != 0 or result['publishable_for_geometry']:
        raise ValueError('source state mismatch')
    for ref in plan['source_bindings'] + plan['implementation'] + plan.get('runtime_bindings', []):
        verify(ref)
    frame_map = {f['input_index']: f for f in plan['frames']}
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 17)
    output.mkdir(parents=True)
    pages, table, metrics, pngs = [], [], [], []
    totals = {p: dict(before=[], after=[], accepted=0, eligible=0) for p in ('local_similarity', 'local_affine')}
    for row in decisions:
        ix = row['input_index']
        if not row.get('local_motion'):
            table.append(f'<tr><td>{ix}</td><td>{html.escape(row["original_status"])}</td><td colspan="3">{html.escape(row["status"])}</td></tr>')
            continue
        local = row['local_motion']
        held_count = len(local['holdout_track_ids'])
        per_profile, all_vectors = {}, {}
        for profile, evidence in local['profiles'].items():
            totals[profile]['accepted'] += len(evidence['common_supported_track_ids'])
            totals[profile]['eligible'] += held_count
            side_metrics = {}
            for side in ('before', 'after'):
                values = predictions(row, profile, side)
                all_vectors[(profile, side)] = values
                totals[profile][side].extend(values)
                cells = evidence['references'][side]['cells']
                side_metrics[side] = dict(predicted=len(values), eligible=held_count,
                    global_same_ids=stats([v['global_error_px'] for v in values]),
                    local_all_fitted_cells=stats([v['local_error_px'] for v in values]),
                    cells_passed=sum(c['passed'] for c in cells), cells_total=len(cells),
                    reasons=dict(Counter(reason for c in cells for reason in c['reasons'])))
            per_profile[profile] = dict(common_supported=len(evidence['common_supported_track_ids']),
                                       eligible=held_count, sides=side_metrics)
        metrics.append(dict(input_index=ix, profiles=per_profile))
        summary = []
        for p in ('local_similarity', 'local_affine'):
            value = per_profile[p]
            summary.append(f'{value["common_supported"]}/{held_count}')
        table.append(f'<tr><td><a href="#f{ix}">{ix}</a></td><td>{html.escape(row["original_status"])}</td><td>{held_count}</td><td>{summary[0]}</td><td>{summary[1]}</td></tr>')
        details = '<h3>同一批可预测留出点：包含失败区域和超限点</h3><table><tr><th>模型 / 侧</th><th>有预测 / 全部留出</th><th>旧整体 P90</th><th>局部 P90</th><th>区域通过</th></tr>'
        for p, value in per_profile.items():
            for side, m in value['sides'].items():
                details += '<tr>'+''.join(f'<td>{html.escape(str(v))}</td>' for v in (
                    p+' / '+side, str(m['predicted'])+'/'+str(held_count), num(m['global_same_ids']['p90_px']),
                    num(m['local_all_fitted_cells']['p90_px']), str(m['cells_passed'])+'/9'))+'</tr>'
        details += '</table>'
        if ix in (51, 79, 81, 82, 84, 85, 86):
            frame = frame_map[ix]['original_frame']
            image_path = frame['path']
            image_hash = frame['sha256']
            verify(dict(path=image_path, sha256=image_hash))
            with Image.open(image_path) as image:
                picture = image.convert('RGB')
            board = Image.new('RGB', (picture.width*2, (picture.height+100)*2), '#14252e')
            id_lookup = {tid: j for j, tid in enumerate(row['track_ids'])}
            for n, ((profile, side), values) in enumerate(all_vectors.items()):
                tile = Image.new('RGB', (picture.width, picture.height+100), '#14252e')
                tile.paste(picture, (0, 100))
                d = ImageDraw.Draw(tile)
                d.text((8, 6), f'{ix}  {profile} / {side}', font=font, fill='white')
                d.text((8, 33), '青：区域通过且误差≤4；橙：有预测但未通过', font=font, fill='white')
                d.text((8, 60), '灰：证据不足；线段为预测到实测的位置差', font=font, fill='white')
                for tid in local['holdout_track_ids']:
                    x, y = row['points']['query'][id_lookup[tid]]
                    d.ellipse((x-2, y+98, x+2, y+102), fill='#b5b5b5')
                for value in values:
                    x, y = value['query_xy']; u, v = value['predicted_xy']
                    color = '#00e2e0' if value['cell_passed'] and value['local_error_px'] <= 4 else '#ff9250'
                    d.line((u, v+100, x, y+100), fill=color, width=2)
                    d.ellipse((x-3, y+97, x+3, y+103), outline=color, width=2)
                board.paste(tile, ((n % 2)*picture.width, (n//2)*(picture.height+100)))
            name = f'frame_{ix}.png'; board.save(output/name)
            pngs.append(bind(output/name))
            details += f'<a href="{name}"><img src="{name}" alt="第{ix}帧两种局部模型的实测留出点误差"></a>'
        details += '<details><summary>区域拒绝原因与完整指标</summary><pre>'+html.escape(json.dumps(per_profile, ensure_ascii=False, indent=2))+'</pre></details>'
        pages.append(f'<section id="f{ix}"><h2>帧 {ix}</h2>'+details+'</section>')
    aggregates = {}
    for p, values in totals.items():
        aggregates[p] = dict(bidirectional_supported_observations=values['accepted'],
            eligible_heldout_observations=values['eligible'], references={})
        for side in ('before', 'after'):
            rows = values[side]
            aggregates[p]['references'][side] = dict(predicted_observations=len(rows),
                global_same_ids=stats([v['global_error_px'] for v in rows]),
                local_all_fitted_cells=stats([v['local_error_px'] for v in rows]))
    title = '局部运动验证：只确认有实测支持的部分'
    body = '<h1>'+title+'</h1><p class="note">候选选择仍为 0。这里只验证已观测点的二维局部运动，不生成轮廓、不合并身份、不恢复遮挡点、不更新三维模型。</p>'
    body += '<p>区域由前后原始物体掩膜的固定 3×3 网格决定。拟合窗口可重叠，但留出点从不进入任何拟合，且每侧最多归属一个核心区域。相似与仿射是预先固定的两种假设，分别展示，不以结果选择“获胜模型”。</p>'
    body += '<p>表中为“前后两侧均通过的留出点 / 全部可用留出点”，按逐帧观测计数，同一物理点可跨帧重复。点数不是掩膜面积覆盖率。</p>'
    body += '<table><tr><th>帧</th><th>原始 SAM</th><th>留出点</th><th>局部相似 双侧</th><th>局部仿射 双侧</th></tr>'+''.join(table)+'</table>'
    body += '<p class="note">误差比较使用同一组有局部预测的留出 ID，包含区域失败与大误差点；证据不足导致没有预测的点单独计数。它仍不覆盖未观测表面，不能据此推断全物体精度。只看“通过点”的误差会有筛选偏差。</p>'
    body += ''.join(pages)
    body += '<h2>方法与范围</h2><p>书中提出几何假设后用额外实际点核验的思路，用于本轮局部诊断。网格、RANSAC、像素门槛是工程近似；并非对透视视频、遮挡或手部共同运动的定理保证。没有将局部变换外推到整张 SAM 掩膜。</p>'
    body += '<p>没有重新运行 SAM、光流、SfM、MVS 或 CAD。两侧轨迹相关，检验点也非统计独立实验。掩膜真值、物体身份、跨视与三维质量未验收。</p>'
    body += '<p><a href="'+link(source/'RESULT.json', output)+'">完整逐区域证据</a> · <a href="'+link(source/'PLAN.json', output)+'">冻结输入</a> · <a href="../SOURCE_REVIEW.md">书籍与官方资料</a></p>'
    document = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title><style>body{background:#10212b;color:#e8f0f4;font:16px/1.8 system-ui;max-width:1250px;margin:30px auto;padding:0 22px}a{color:#80d1ff}img{width:100%;height:auto}table{width:100%;border-collapse:collapse}th,td{padding:7px;border-bottom:1px solid #34505e;text-align:left}pre{white-space:pre-wrap;overflow-wrap:anywhere}.note{border-left:4px solid #ffd080;background:#26333b;padding:14px}section{margin-top:36px}</style>'+body+'</html>'
    (output/'index.html').write_text(document, encoding='utf-8')
    for ref in refs:
        verify(ref)
    result = dict(status='READBACK_REPORT_ONLY', source_bindings=refs, source_unchanged=True,
        aggregates=aggregates, frames=metrics, pngs=pngs, selected_candidates=0,
        refits=0, model_calls=0, quality_status='NOT_EVALUATED', browser_interaction='NOT_VERIFIED',
        script=bind(Path(__file__)), report=bind(output/'index.html'))
    save(output/'RESULT.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = build(args.source, args.output)
    print(json.dumps(dict(status=result['status'], aggregates=result['aggregates']), ensure_ascii=False))
