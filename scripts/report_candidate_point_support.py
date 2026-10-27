"""Display consumed saved point annotations, retaining all unknown states."""
from __future__ import annotations

import argparse
from collections import Counter
import html
import os
from pathlib import Path
import sys
from urllib.parse import quote

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from review_mask_association import read, save, bind, verify
from review_local_motion import fresh_output


def link(path, root):
    return quote(os.path.relpath(path, root).replace('\\', '/'), safe='/:')


def build(source: Path, output: Path) -> dict:
    source = source.resolve(strict=True)
    output = fresh_output(output, [source])
    inputs = [bind(source / name) for name in ('PLAN.json', 'RESULT.json', 'DECISIONS.json',
                                             'POINT_SUPPORT_LEDGER.json', 'MASK_SELECTION_LEDGER.json')]
    plan, result, decisions, ledger, masks = [read(r['path']) for r in inputs]
    if masks or result['new_masks'] != 0 or result['publishable_for_geometry'] is not False:
        raise ValueError('this reporter accepts only point-scoped unaccepted masks')
    for ref in plan['source_bindings'] + plan['implementation']:
        verify(ref)
    annotations = {}
    for name, ref in result['annotation_files'].items():
        inputs.append(verify(ref))
        annotations[name] = read(ref['path'])
    frames = {f['input_index']: f for f in plan['frames']}
    output.mkdir(parents=True)
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 18)
    small = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 14)
    pictures, sections = [], []
    for item in ledger:
        index = item['input_index']
        verify(item['image_binding'])
        verify(item['candidate_binding'])
        with Image.open(item['image_binding']['path']) as image:
            raw = image.convert('RGB')
        with Image.open(item['candidate_binding']['path']) as image:
            mask = np.array(image)
        overlay = Image.composite(Image.new('RGB', raw.size, '#55aadd'), raw,
                                  Image.fromarray(((mask > 0) * 35).astype(np.uint8)))
        board = Image.new('RGB', (raw.width * 2, raw.height + 110), '#172f3b')
        board.paste(raw, (0, 110))
        board.paste(overlay, (raw.width, 110))
        draw = ImageDraw.Draw(board)
        draw.text((12, 8), f'帧 {index}：原始 SAM = NEEDS_SELECTION', font=font, fill='white')
        draw.text((12, 40), '原图与实际点位置；原状态未改写', font=font, fill='#d1dbe0')
        draw.text((raw.width + 12, 8), f'{item["provisional_candidate_id"]}：仅对已有点提供条件性支持', font=font, fill='#55d8ff')
        draw.text((raw.width + 12, 40), f'{len(item["supported_observations"])} 个观测；整张掩膜未验收', font=font, fill='white')
        draw.text((raw.width + 12, 72), '浅蓝轮廓仅作参考，不允许用它新增点或补表面', font=small, fill='#e5c87c')
        for point in item['supported_observations']:
            x, y = point['query_point_xy']
            for shift, color in ((0, '#ffc65e'), (raw.width, '#35dbff')):
                draw.ellipse((x + shift - 5, y + 105, x + shift + 5, y + 115), outline=color, width=3)
                draw.text((x + shift + 8, y + 105), str(point['track_id']), font=small, fill=color)
        path = output / f'frame_{index}.png'
        board.save(path)
        pictures.append(bind(path))
        ids = ', '.join(str(p['track_id']) for p in item['supported_observations'])
        sections.append(f'<section><h2>帧 {index} · {len(item["supported_observations"])} 个实测点</h2>'
            f'<p>轨迹 ID：{ids}。裁剪未知边界 {item["boundary_unknown"]["clipped_px"]} px。</p>'
            f'<a href="{path.name}"><img src="{path.name}" alt="原图与条件性点支持对照"></a></section>')

    timeline = []
    for name, annotation in annotations.items():
        lo, hi = map(int, name.split('_'))
        supported = {r['track_id'] for r in annotation['rows'] if r['derived_point_support'] != 'UNCHANGED'}
        by_track = {}
        for row in annotation['rows']:
            by_track.setdefault(row['track_id'], {})[row['frame_index'] + lo] = row
        for tid in sorted(supported):
            cells = []
            for index in range(lo, hi + 1):
                row = by_track[tid].get(index)
                if row is None:
                    label, cls, title = '—', 'missing', '无实际观测'
                elif row['derived_point_support'] != 'UNCHANGED':
                    label, cls = '点支持', 'provisional'
                    title = f'原标签 {row["source_label"]}；原身份段 {row["source_identity_segment"]}；几何仍未通过'
                else:
                    label = row['source_label']
                    cls = 'original' if row['source_original_foreground'] else 'unknown'
                    title = f'原标签 {label}；原身份段 {row["source_identity_segment"]}'
                cells.append(f'<td class="{cls}" title="{html.escape(title)}">{html.escape(label)}</td>')
            timeline.append('<tr><td>' + name + ' / ' + str(tid) + '</td>' + ''.join(cells) + '</tr>')
    headers = ''.join(f'<th>{i}</th>' for i in range(77, 91))
    # The table currently contains the one affected 77..90 clip; do not imply ID sharing across clips.
    affected = [name for name, a in annotations.items() if a['counts']['provisional']]
    if affected not in ([], ['77_90']):
        raise ValueError('report timeline requires an explicit column mapping for the affected clips')
    status_rows = ''.join('<tr>' + ''.join('<td>' + html.escape(str(value)) + '</td>' for value in (
        d['input_index'], d['original_status'], d['status'], d['provisional_candidate_id'] or '—',
        len(d['supported_observations']))) + '</tr>' for d in decisions)
    title = '候选规则已接入：只接回有证据的实测点'
    body = f'<h1>{title}</h1><p class="note">自动处理了 {result["provisional_frames"]} 个原始歧义帧，'
    body += f'产生 {result["provisional_observations"]} 个条件性点支持记录，涉及 {result["unique_clip_track_ids"]} 条原轨迹。'
    body += '这是对已有坐标的附加证据，不是新测量、整张掩膜通过或三维重建成功。</p>'
    body += '<p>漏掉可信正点的候选会被排除；任何仍兼容正点的候选遇到参考外点冲突，整帧暂停自动判断。'
    body += '多个候选都兼容时保持歧义。面积、SAM分数与边缘强度不用于打破平局。</p>'
    body += f'<p>原始状态：{html.escape(str(result["original_status_counts"]))}。'
    body += f'获支持观测原标签：{html.escape(str(result["provisional_source_labels"]))}。原标签和身份段保持原样。</p>'
    body += '<h2>轨迹时间线</h2><p>浅蓝为本次点级支持；绿色为原始前景，黄色为其他原标签，灰色为没有实际观测。'
    body += '同一行不表示跨身份段已连接；悬停可查看原身份段。不填补空格。</p>'
    body += '<div class="scroll"><table><tr><th>片段 / 轨迹</th>' + headers + '</tr>' + ''.join(timeline) + '</table></div>'
    body += ''.join(sections)
    body += '<h2>全28帧状态</h2><table><tr><th>帧</th><th>原SAM</th><th>派生处理</th><th>候选</th><th>点观测</th></tr>' + status_rows + '</table>'
    body += '<h2>三维接入仍需什么</h2><p>现有几何入口要求已核对的原始前景、相机和身份连续性。本次没有将候选填进原mask，'
    body += '没有改变target_verified，也没有越过几何检查。下一步用明确的点级输入连接相机，检验同身份段、视差与重投影；'
    body += '缺少条件的部分继续保留孔洞。</p>'
    body += '<p>局部点支持来自有限观测，不能证明整物体刚性、未包含手部或焦距变化是唯一原因。'
    body += '只消费保存的真实记录；没有重新运行光流、SAM、SfM、MVS或CAD。</p>'
    body += '<p><a href="' + link(source / 'RESULT.json', output) + '">机器结果</a> · <a href="' + link(source / 'POINT_SUPPORT_LEDGER.json', output) + '">点级账本</a>'
    body += ' · <a href="../SOURCE_AND_PLAN.md">书籍、官方来源与规则</a> · <a href="../HANDOFF.md">交接</a></p>'
    css = 'body{background:#10232e;color:#e6eff5;font:16px/1.8 system-ui,"Microsoft YaHei";max-width:1280px;margin:25px auto;padding:0 22px}a{color:#7dd4ff}img{width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:7px;border:1px solid #35505e}.scroll{overflow:auto}.provisional{background:#075c75}.original{background:#1d513b}.unknown{background:#614f22}.missing{background:#303b42}.note{border-left:4px solid #58cbf4;background:#1a3948;padding:16px}section{margin-top:36px}'
    (output / 'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>' + title + '</title><style>' + css + '</style>' + body + '</html>', encoding='utf-8')
    for ref in inputs:
        verify(ref)
    receipt = dict(status='SAVED_POINT_ANNOTATION_REPORT', source_bindings=inputs, pngs=pictures,
        report=bind(output / 'index.html'), script=bind(Path(__file__)), source_unchanged=True,
        browser_interaction='NOT_VERIFIED', quality_status='NOT_EVALUATED')
    save(output / 'RESULT.json', receipt)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    build(args.source, args.output)
