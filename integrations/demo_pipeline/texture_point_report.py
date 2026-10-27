"""Render saved texture-point evidence; never performs tracking or seed selection."""
from pathlib import Path
import html
import json
import math
from PIL import Image, ImageDraw, ImageFont


def render(out: Path):
    out = Path(out)
    marker = out / 'TEXTURE_REPORT.json'
    names = ['TEXTURE_REPORT.json', 'index.html', 'motion.gif', 'contact_sheet.png', 'frame_panels', 'template_reference.png']
    if any((out / name).exists() for name in names):
        raise FileExistsError('Report outputs already exist; fresh report required')
    inputs = json.loads((out / 'INPUTS.json').read_text(encoding='utf-8-sig'))
    trace = json.loads((out / 'TRACE.json').read_text(encoding='utf-8-sig'))
    sources = inputs['frames']
    rows = {r['decode_index']: r for r in trace['frames']}
    seed = inputs['seed']['xy']
    fontpath = 'C:/Windows/Fonts/msyh.ttc'
    def font(size):
        return ImageFont.truetype(fontpath, size)
    def xy(value):
        if isinstance(value, (list, tuple)) and len(value) == 2:
            try:
                p = tuple(float(v) for v in value)
                return p if all(math.isfinite(v) for v in p) else None
            except (TypeError, ValueError):
                pass
        return None
    def crop(im, center, size):
        x, y = center
        left, top = round(x)-size//2, round(y)-size//2
        return im.crop((left, top, left+size, top+size))
    def fields(value, needle, prefix=''):
        result = []
        if isinstance(value, dict):
            for k, v in value.items():
                key = prefix + str(k)
                if needle in str(k).lower():
                    result.append((key, v))
                if isinstance(v, dict):
                    result.extend(fields(v, needle, key+'.'))
        return result
    def polygon(row, point):
        for container in (row, row.get('forward', {})):
            if not isinstance(container, dict):
                continue
            for key in ('mapped_polygon', 'mapped_patch_polygon', 'patch_polygon', 'template_polygon', 'polygon'):
                value = container.get(key)
                if isinstance(value, list) and len(value) >= 4 and all(xy(p) for p in value):
                    return [xy(p) for p in value]
            evidence = container.get('pixel_evidence', {})
            if isinstance(evidence, dict):
                for key in ('mapped_polygon', 'mapped_patch_polygon', 'polygon'):
                    value = evidence.get(key)
                    if isinstance(value, list) and len(value) >= 4 and all(xy(p) for p in value):
                        return [xy(p) for p in value]
        x,y = point
        return [(x-16,y-16),(x+16,y-16),(x+16,y+16),(x-16,y+16)]
    # All rendering is downstream of the frozen trace.
    first_path = Path(sources[0]['path'])
    if not first_path.is_absolute(): first_path = out / first_path
    with Image.open(first_path) as opened:
        template = crop(opened.convert('RGB'), seed, 33)
    paneldir = out / 'frame_panels'
    paneldir.mkdir()
    template.resize((264,264), Image.Resampling.NEAREST).save(out/'template_reference.png')
    panels, captions, ledger = [], [], []
    lastgood = tuple(seed)
    stopped = False
    firststop = None
    lastaccepted = None
    for n, source in enumerate(sources):
        row = rows.get(source['decode_index'], {})
        point = xy(row.get('accepted_xy'))
        status = str(row.get('status', 'TRACE_MISSING'))
        valid = point is not None and not stopped
        if not valid:
            stopped = True
            if firststop is None: firststop = n
        else:
            lastgood = point
            lastaccepted = n
        path = Path(source['path'])
        if not path.is_absolute(): path = out / path
        with Image.open(path) as opened: original = opened.convert('RGB')
        annotated = original.copy()
        draw = ImageDraw.Draw(annotated)
        if valid:
            x,y=point
            poly=polygon(row, point)
            draw.line(poly+[poly[0]], fill='#00ef60', width=2)
            draw.line((x-8,y,x+8,y), fill='#00ef60',width=2)
            draw.line((x,y-8,x,y+8), fill='#00ef60',width=2)
        # Candidates are shown only in a distinct rejected comparison crop.
        canvas = Image.new('RGB',(1040,1160),'#111820')
        canvas.paste(annotated,(0,70))
        d=ImageDraw.Draw(canvas)
        d.text((12,8),f"帧 {source['decode_index']} | {source.get('time_seconds', row.get('time_seconds','?'))} 秒",font=font(22),fill='white')
        d.text((12,38),f"{'已接受' if valid else '已停止 / 不作为有效点'} · {status}",font=font(19),fill='#64ef99' if valid else '#ffbf75')
        d.text((600,80),'首帧所选源点 / 固定 33×33 模板',font=font(17),fill='white')
        canvas.paste(template.resize((264,264),Image.Resampling.NEAREST),(600,112))
        d.text((600,390),'129×129 当前点或最后有效位置上下文',font=font(16),fill='white')
        canvas.paste(crop(original,lastgood,129).resize((258,258)),(600,420))
        candidate=xy(row.get('candidate_xy'))
        forward = row.get('forward') or {}
        evidence = forward.get('pixel_evidence') or {}
        warped = evidence.get('warped_search_gray')
        if valid and warped:
            import numpy as np
            patch = Image.fromarray(np.rint(np.clip(warped, 0, 255)).astype(np.uint8))
            d.text((600,690),'当前纹理仿射对齐后（灰度，未更新模板）',font=font(16),fill='white')
            canvas.paste(patch.resize((264,264),Image.Resampling.NEAREST).convert('RGB'),(600,720))
        elif not valid and candidate:
            d.text((600,690),'REJECTED 候选比较（不表示对应成立）',font=font(16),fill='#ffbf75')
            canvas.paste(crop(original,candidate,129).resize((258,258)),(600,720))
        reason=json.dumps(row.get('reasons',[]),ensure_ascii=False)
        for line,textline in enumerate([f'cycle_error_px: {row.get("cycle_error_px", "未提供")}', '绿色仅表示停止前 accepted_xy', '首帧模板坐标: '+str(seed)]):
            d.text((600,1000+line*29),textline,font=font(16),fill='white')
        filename=f'{n:03d}_{source["decode_index"]}.jpg'
        canvas.save(paneldir/filename,quality=92)
        panels.append(canvas)
        captions.append(f'frame_panels/{filename}')
        ncc=fields(row,'ncc')
        holdout=[(k,v) for k,v in ncc if 'hold' in k.lower()]
        ledger.append({'decode_index':source['decode_index'],'status':status,'rendered_accepted':valid,'holdout_ncc':holdout,'ncc_fields':ncc,'reasons':row.get('reasons',[]),'cycle_error_px':row.get('cycle_error_px')})
    if not panels: raise ValueError('No source frames')
    stopend=min(len(panels),firststop+6) if firststop is not None else len(panels)
    gif=[p.resize((520,580)).convert('P',palette=Image.Palette.ADAPTIVE) for p in panels[:stopend]]
    gif[0].save(out/'motion.gif',save_all=True,append_images=gif[1:],duration=700,loop=0)
    chosen=list(dict.fromkeys([0]+([lastaccepted] if lastaccepted is not None else [])+([firststop] if firststop is not None else [])))
    sheet=Image.new('RGB',(520*len(chosen),580),'#111820')
    for j,i in enumerate(chosen): sheet.paste(panels[i].resize((520,580)),(520*j,0))
    sheet.save(out/'contact_sheet.png')
    esc=lambda v: html.escape(str(v))
    table=''.join('<tr><td>'+esc(r['decode_index'])+'</td><td>'+esc(r['status'])+'</td><td>'+esc(r['holdout_ncc'] or '未提供')+'</td><td>'+esc(r['cycle_error_px'])+'</td><td>'+esc(r['reasons'])+'</td></tr>' for r in ledger)
    links=' '.join(f'<a href="{p}">{esc(s["decode_index"])}</a>' for p,s in zip(captions,sources))
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>固定纹理点证据</title><style>body{background:#111820;color:#eee;font:16px sans-serif;max-width:1100px;margin:auto;padding:20px}a{color:#8ccaff}img{max-width:100%}table{border-collapse:collapse}td,th{border:1px solid #68717a;padding:6px}input{width:100%}pre{white-space:pre-wrap}</style><h1>固定纹理点逐帧证据</h1><p>首帧源点由冻结输入指定；固定模板为该位置 33×33 像素。绿色十字与模板范围只绘制于停止前已接受记录。停止后不恢复，不将候选或预测当作有效点。</p><p>物理点身份与遮挡原因尚未核实；外观匹配失败不等于确定转到背面；本流程不进行重现恢复。无映射多边形字段时，绿色范围仅为中心周围固定 33×33 参考框。</p><h2>全部源帧</h2><input id="slider" type="range" min="0" max="MAX" value="0"><p id="label"></p><a id="full" href=""><img id="frame"></a><p>LINKS</p><h2>慢速过程：种子 → 已接受 → 首次停止及后五帧</h2><img src="motion.gif"><h2>首帧 / 最后接受 / 首次停止</h2><img src="contact_sheet.png"><h2>逐帧 holdout NCC 与停止原因</h2><table><tr><th>解码帧</th><th>状态</th><th>holdout NCC（原字段）</th><th>循环误差 px</th><th>原因</th></tr>TABLE</table><h2>跟踪配置原文</h2><pre>PROFILE</pre><script>const paths=PATHS;const ids=IDS;let s=document.getElementById('slider');function show(){let i=+s.value;document.getElementById('frame').src=paths[i];document.getElementById('full').href=paths[i];document.getElementById('label').textContent=`${i+1} / ${paths.length} · 解码帧 ${ids[i]}`;}s.oninput=show;show();</script></html>'''
    page=page.replace('MAX',str(len(panels)-1)).replace('LINKS',links).replace('TABLE',table).replace('PROFILE',esc(json.dumps(trace.get('profile',{}),ensure_ascii=False,indent=2))).replace('PATHS',json.dumps(captions)).replace('IDS',json.dumps([s['decode_index'] for s in sources]))
    (out/'index.html').write_text(page,encoding='utf-8')
    summary={'schema':'texture-point-report-v1','frames':len(panels),'first_stop_position':firststop,'last_accepted_position':lastaccepted,'frames_rendered':ledger,'physical_identity':'UNVERIFIED','occlusion_cause':'UNVERIFIED','reappearance_recovery':False}
    marker.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    return summary
