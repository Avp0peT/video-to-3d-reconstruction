"""Render saved candidate geometry diagnostics; never recompute geometry."""
import argparse
from collections import Counter
import html
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from review_mask_association import read, save, bind, verify
from PIL import Image, ImageDraw, ImageFont


def esc(value):
    return html.escape(str(value))


def number(value):
    return '—' if value is None else f'{value:.3f}'


def report(source, output):
    source = source.resolve(strict=True)
    if output.exists():
        raise ValueError('fresh report directory required')
    result, groups = read(source / 'RESULT.json'), read(source / 'GROUPS.json')
    plan = read(source / 'PLAN.json')
    for ref in read(source / 'FINAL_INPUT_BINDINGS.json'):
        verify(ref)
    roots = [r for r in plan['source_bindings'] if Path(r['path']).name == 'INPUTS.json'
             and Path(r['path']).parent.name == 'coverage' and 'incremental_tracks_20261016' in r['path']]
    if not roots or len({r['sha256'] for r in roots}) != 1:
        raise ValueError('unique original registry binding required')
    ref = verify(roots[0])
    registry = {f['input_index']: f for f in read(ref['path'])['frames']}
    output.mkdir(parents=True)
    report_refs = [bind(source / (n + '.json')) for n in ('PLAN', 'RESULT', 'GROUPS', 'CAMERAS', 'FINAL_INPUT_BINDINGS')]
    report_refs.append(ref)
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 17)
    small = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 14)
    by_query = {}
    for group in groups:
        for query in group['queries']:
            by_query.setdefault(query['input_index'], []).append((query, group))
    cards = []
    panels = []
    for index, points in sorted(by_query.items()):
        frame = registry[index]
        report_refs.append(verify(dict(path=frame['path'], sha256=frame['sha256'])))
        with Image.open(frame['path']) as im:
            im = im.convert('RGB')
        draw = ImageDraw.Draw(im)
        for query, group in points:
            # Display measured query only. COLMAP center -> zero-based center.
            x, y = [v - .5 for v in query['source_xy']]
            draw.ellipse((x-7, y-7, x+7, y+7), outline=(239, 182, 255), width=3)
            label = str(group['track_id'])
            width = draw.textbbox((0, 0), label, font=font)[2]
            tx, ty = min(max(0, x+9), im.width-width-5), min(max(0, y-18), im.height-25)
            draw.rectangle((tx-2, ty-2, tx+width+2, ty+21), fill=(22, 17, 35))
            draw.text((tx, ty), label, font=font, fill=(255, 255, 255))
        im.save(output / f'query_{index}.png')
        preview = im.resize((288, 512))
        panel = Image.new('RGB', (308, 580), '#111827')
        panel.paste(preview, (10, 58))
        pd = ImageDraw.Draw(panel)
        pd.text((12, 10), f'Frame {index} | {len(points)} measured queries', font=small, fill='white')
        pd.text((12, 31), 'NO SAVED CAMERA | NOT PROJECTED', font=small, fill='#efb6ff')
        panels.append(panel)
        cards.append(f'<figure><a href="query_{index}.png"><img src="query_{index}.png" alt="第{index}帧实测候选点"/></a>'
                     f'<figcaption>第 {index} 帧 · {len(points)} 个实测候选点<br>无已存相机，未计算这些点的重投影</figcaption></figure>')
    contact = Image.new('RGB', (sum(i.width for i in panels), 580), '#111827')
    offset = 0
    for panel in panels:
        contact.paste(panel, (offset, 0)); offset += panel.width
    contact.save(output / 'query_contact.png')
    labels = {'CONDITIONAL_ANCHOR_SUPPORTED': '三视图条件支持', 'TWO_VIEW_PROVISIONAL': '两视图临时结果',
              'INSUFFICIENT_ORIGINAL_ANCHORS': '可靠锚点不足', 'STRICT_GATE_REJECTED': '严格几何检查未过'}
    table, details = [], []
    for g in groups:
        geo = g['geometry']
        raw = geo['raw_helper_result'] or {}
        metric = raw.get('metrics', {})
        errors = metric.get('reprojection_errors_px', [])
        strict = raw.get('strict_decision') or {}
        loo = [o['error_px'] for o in strict.get('loo', []) if o['error_px'] is not None]
        table.append(f'<tr><td>{g["track_id"]} / 段{g["identity_segment"]}</td><td>{", ".join(str(a["input_index"]) for a in g["anchors"])}</td>'
                     f'<td>{esc(labels.get(geo["status"], geo["status"]))}</td><td>{number(max(errors) if errors else None)}</td>'
                     f'<td>{number(metric.get("measurement_ray_angle_deg"))}</td><td>{number(max(loo) if loo else None)}</td></tr>')
        rows = ''.join(f'<tr><td>{a["input_index"]}</td><td>{esc(a["status"])}</td><td>{esc(", ".join(a["reasons"]) or "原始前景、可靠流、相机与掩膜像素均满足")}</td></tr>' for a in g['assessments'])
        details.append(f'<details><summary>轨迹 {g["track_id"]} · 段 {g["identity_segment"]} 的逐帧依据</summary>'
                       f'<p>模型 {esc(g["model_key"])}；排除其他身份段的帧：{esc(g["excluded_other_segment_frames"])}。没有自动重连身份段。</p>'
                       f'<table><tr><th>帧</th><th>状态</th><th>理由</th></tr>{rows}</table></details>')
    suites = list(ET.parse(source.parent / 'junit.xml').getroot().iter('testsuite'))
    tests = sum(int(s.attrib['tests']) for s in suites)
    counts = result['anchor_status_counts']
    document = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>候选点 → 相机 → 几何：本轮检查</title><style>
body{{font-family:system-ui,"Microsoft YaHei",sans-serif;background:#0c1220;color:#e4eaf4;max-width:1280px;margin:auto;padding:32px;line-height:1.75}}h1{{font-size:30px;line-height:1.3}}h2{{margin-top:36px;font-size:22px}}p{{max-width:1050px}}a{{color:#8ecbff}}.muted{{color:#a8b7cd}}.notice{{border-left:4px solid #e7b05c;padding:14px 20px;background:#211d18}}.stats{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}}.stat{{padding:18px;background:#182337;border-radius:10px}}.stat b{{display:block;font-size:32px}}.gallery{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}}figure{{margin:0}}img{{width:100%;border-radius:8px}}figcaption{{font-size:13px;color:#c0cadd}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:10px 12px;border-bottom:1px solid #2d3a50;text-align:left}}th{{color:#9dc7f3}}details{{background:#141e2e;margin:10px 0;padding:12px}}summary{{cursor:pointer}}code{{color:#e9c3ff}}@media(max-width:800px){{.gallery,.stats{{grid-template-columns:repeat(2,1fr)}}body{{padding:16px}}table{{font-size:12px}}}}
</style><body><p class="muted">PPP 视频 / 已保存数据诊断 / {esc(result['version'])}</p>
<h1>点的二维连续性已接入几何检查<br>下一处缺口是候选帧的相机位姿</h1>
<div class="notice">81、84、85、86 帧的 21 个实测候选观测均没有已保存相机。本轮算出的锚点坐标只用于诊断，新增模型点 <b>0</b>；未宣称已恢复这些帧的三维结构。</div>
<div class="stats"><div class="stat"><b>{counts.get('CONDITIONAL_ANCHOR_SUPPORTED',0)}</b>三视图条件支持</div><div class="stat"><b>{counts.get('TWO_VIEW_PROVISIONAL',0)}</b>两视图临时结果</div><div class="stat"><b>{counts.get('INSUFFICIENT_ORIGINAL_ANCHORS',0)}</b>可靠锚点不足</div><div class="stat"><b>{result['queries_without_saved_camera']}</b>无相机的候选观测</div></div>
<h2>同一视频点，分开检查两个问题</h2><p>“这个候选点是否沿视频稳定移动”与“它在三维中位于哪里”分别需要二维对应证据和相机证据。本轮只用同一身份段的原始已确认前景观测建锚点；没有把未知掩膜内的点改写成前景，也没有跨遮挡段拼点。</p>
<h2>候选帧仍原样保留</h2><p class="muted">紫色圈为既有的实测候选位置，不是三维重投影，也不是新检测的点。数字为该短片中的 track ID。点击可查看原尺寸标注。</p><div class="gallery">{''.join(cards)}</div>
<h2>9 条轨迹的实际数值结果</h2><table><thead><tr><th>轨迹 / 身份段</th><th>原始可靠锚帧</th><th>结论</th><th>最大重投影 px</th><th>视差角 °</th><th>最大留一误差 px</th></tr></thead><tbody>{''.join(table)}</tbody></table>
<p class="muted">两视图：≤4 px、≥1°，没有留一检查。三视图及以上：≤2 px、≥2°、固定相机留一误差≤3 px。所有结果还要求正深度与数值条件检查。误差小表示当前相机与观测内部一致，不能代替实物精度真值。</p>
<h2>为什么3条没有被升级</h2><p>轨迹735只有80帧，1064与1089只有83帧可作可靠锚点。种子观测的 <code>strong=false</code> 状态保留；query没有相机；不同身份段不自动合并。单视图不足以确定三维深度，因此继续保留未知。</p>
{''.join(details)}
<h2>下一步：补候选帧与已建局部模型的对应</h2><p>先在同一模型内收集更多已确认的3D–2D锚点，检查数量、空间分布与独立验证观测，再对缺相机的帧做有约束注册。当前81帧仅关联2个可计算锚点，84/85/86各4个，后者均只有两视图临时支持；这些不足以直接宣布可靠注册。新匹配先保持候选，无法确认的区域留孔洞。</p>
<p>这一步是透视相机几何诊断。书中的精确相容点集构造不被当成遮挡身份、透视标定或真实精度的自动保证。</p>
<h2>证据与执行边界</h2><p>{tests} 个定向测试通过；实际保存数据诊断1次，内部耗时 {result['duration_seconds']:.3f} 秒。只读4个原生模型、39个注册相机；6组数值三角化。相机拟合、SAM3、光流重跑、GPU、MVS、CAD均为0。</p>
<p><a href="../real01/geometry/RESULT.json">结果JSON</a> · <a href="../real01/geometry/GROUPS.json">逐轨迹与所有误差</a> · <a href="../real01/geometry/CAMERAS.json">原生相机复核</a> · <a href="../SOURCE_AND_PLAN.md">书籍与公开资料依据 / 冻结计划</a> · <a href="../INDEPENDENT_AUDIT.json">独立复核</a> · <a href="../HANDOFF.md">交接说明</a></p>
<p class="muted">状态 ANCHOR_DIAGNOSTIC_ONLY · 物理身份仍UNVERIFIED · 尺度unknown · 独立质量真值未评估 · 浏览器交互未验</p></body></html>'''
    (output / 'index.html').write_text(document, encoding='utf-8')
    save(output / 'REPORT_BINDINGS.json', report_refs + [bind(Path(__file__))])
    for ref in report_refs:
        verify(ref)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report(args.source, args.output)
