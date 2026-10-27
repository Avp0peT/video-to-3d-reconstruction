"""Read-only report renderer for separate native candidate point clouds."""
from collections import Counter
import hashlib
import html
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw


def render(native_dir, output_dir):
    native_dir, output_dir = Path(native_dir).resolve(), Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError('Fresh output directory required: ' + str(output_dir))
    cycle_path = native_dir.parent.parent / 'reverse12801' / 'CYCLE_ASSESSMENT.json'
    inputs = [native_dir/'RESULT.json', native_dir/'GEOMETRY_EVIDENCE.json', cycle_path]
    def binding(path):
        data = path.read_bytes()
        return {'path': str(path), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
    before = [binding(path) for path in inputs]
    result, evidence, cycle = [json.loads(path.read_text(encoding='utf-8-sig')) for path in inputs]
    by_id = {str(c['model_id']): c for c in evidence['components']}
    models = []
    for model in result['models']:
        records = model['point_records']
        if any(len(p['xyz']) != 3 or not all(math.isfinite(v) for v in p['xyz']) for p in records):
            raise ValueError('Non-finite cloud coordinates')
        ev = by_id[str(model['model_id'])]
        counts = Counter(p['classification'] for p in ev['points'])
        identities = {str(obs['candidate_track_id']) for p in records for obs in p['track']}
        models.append({'id': model['model_id'], 'views': model['registered_views'],
                       'points': records, 'classifications': dict(counts),
                       'query_ids': len(identities), 'ply': Path(model['ply_path']).as_uri(),
                       'geometry_pass': sum(p['pass_frozen_geometry'] is True for p in ev['points'])})
    main = max(models, key=lambda m: (m['views'], len(m['points'])))
    # Bound native exports too; they are linked, never modified or merged.
    ply_paths = [Path(m['ply_path']) for m in result['models']]
    before += [binding(path) for path in ply_paths]
    output_dir.mkdir(parents=True)
    preview = Image.new('RGB', (1200, 360*len(models)), '#111820')
    draw = ImageDraw.Draw(preview)
    for row, model in enumerate(models):
        records = model['points']
        for col, (a, b, name) in enumerate(((0, 1, 'XY'), (0, 2, 'XZ'), (1, 2, 'YZ'))):
            left, top = col*400, row*360
            draw.text((left+12, top+12), 'Model %s | %s | %s points' % (model['id'], name, len(records)), fill='white')
            draw.text((left+12, top+30), 'NOT_QUALITY_ACCEPTED | unknown scale', fill='#ffbf75')
            if not records:
                continue
            low = [min(p['xyz'][axis] for p in records) for axis in (a,b)]
            high = [max(p['xyz'][axis] for p in records) for axis in (a,b)]
            scale = min(360/max(high[0]-low[0], 1e-9), 280/max(high[1]-low[1], 1e-9))
            for point in records:
                x = left+200+(point['xyz'][a]-(low[0]+high[0])/2)*scale
                y = top+205-(point['xyz'][b]-(low[1]+high[1])/2)*scale
                draw.ellipse((x-2,y-2,x+2,y+2), fill=tuple(point['color']))
    preview.save(output_dir/'projections.png')
    esc = lambda value: html.escape(str(value), quote=True)
    rows = ''.join('<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td><a href="%s">原生 PLY</a></td></tr>' % (
        m['id'], m['views'], len(m['points']), m['query_ids'], m['geometry_pass'], esc(m['classifications']), esc(m['ply'])) for m in models)
    options = ''.join('<option value="%s" %s>Model %s · %s views · %s points</option>' % (i, 'selected' if m is main else '', m['id'], m['views'], len(m['points'])) for i,m in enumerate(models))
    source = (native_dir.parent.parent/'tapnextpp12801'/'index.html').as_uri()
    payload = json.dumps(models, ensure_ascii=True).replace('<', '\\u003c')
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>原生候选稀疏几何</title><style>
body{background:#111820;color:#eee;font:16px sans-serif;max-width:1250px;margin:auto;padding:22px}a{color:#9cf}canvas{width:100%;height:600px;background:#090d12;touch-action:none}img{max-width:100%}table{border-collapse:collapse}td,th{border:1px solid #657281;padding:7px}pre{white-space:pre-wrap}select,button{padding:8px}</style>
<h1>原生候选稀疏点云：NOT_QUALITY_ACCEPTED</h1><p>各模型独立展示，未合并；尺度未知。主模型 ''' + esc(main['id']) + ' 有 ' + str(main['views']) + ' 视图、' + str(len(main['points'])) + ' 个三维点，来自 ' + str(main['query_ids']) + ''' 个查询 ID。三维点数多于查询 ID，表明身份链发生碎片化，不能将这些点当成同等数量的独立物理点。</p>
<p>这是一组候选稀疏重建，尚未验证整颗洋葱覆盖、物理同点身份、真实形状或尺寸；不构成整物体模型、网格或 CAD 验收。几何筛选通过仍不代表质量验收。</p>
<p><a href="''' + esc(source) + '''">查看源二维跟踪证据</a> · <a href="REPORT_BINDINGS.json">输入哈希与报告器绑定</a></p>
<h2>独立模型浏览</h2><p>拖动旋转，滚轮缩放；只显示当前模型。颜色来自原生 point_records。</p><select id="model">''' + options + '''</select> <button id="reset">重置视角</button><span id="info"></span><canvas id="cloud" width="1200" height="600"></canvas>
<h2>模型记录</h2><table><tr><th>Model</th><th>视图</th><th>3D 点</th><th>查询 ID</th><th>几何筛选通过</th><th>证据分类计数</th><th>下载</th></tr>''' + rows + '''</table>
<h2>逆向循环诊断</h2><p>共 ''' + str(cycle['all_queries']['count']) + ' 次可比较记录：' + str(cycle['comparison_status_counts']['PASS_HEURISTIC']) + ' 次 ≤4 px，' + str(cycle['comparison_status_counts']['FAIL_HEURISTIC']) + ' 次 >4 px；另有 ' + str(cycle['comparison_status_counts']['UNKNOWN']) + ''' 次未知。4 px 是声明的启发式阈值，不是真值；小循环误差不证明物理身份或几何正确。</p>
<h2>原始 XYZ 三平面投影</h2><p>每一行属于一个独立模型，投影按各自范围显示，禁止跨模型比较尺度；仅绘制实际点，无补点或表面。</p><img src="projections.png">
<script>const models=''' + payload + ''';
const canvas=document.getElementById('cloud'),ctx=canvas.getContext('2d'),selector=document.getElementById('model');let ax=-.3,ay=.4,zoom=1,drag=null;
function render(){const model=models[+selector.value],p=model.points;ctx.clearRect(0,0,1200,600);if(!p.length)return;const lo=[0,1,2].map(a=>Math.min(...p.map(v=>v.xyz[a]))),hi=[0,1,2].map(a=>Math.max(...p.map(v=>v.xyz[a]))),center=lo.map((v,a)=>(v+hi[a])/2),span=Math.max(...lo.map((v,a)=>hi[a]-v),1e-9),scale=460/span*zoom;const dots=p.map(v=>{let [x,y,z]=v.xyz.map((w,a)=>w-center[a]);let xx=x*Math.cos(ay)+z*Math.sin(ay),zz=-x*Math.sin(ay)+z*Math.cos(ay),yy=y*Math.cos(ax)-zz*Math.sin(ax);return{x:600+xx*scale,y:300-yy*scale,z:y*Math.sin(ax)+zz*Math.cos(ax),color:v.color}}).sort((a,b)=>a.z-b.z);for(const v of dots){ctx.beginPath();ctx.arc(v.x,v.y,2.6,0,Math.PI*2);ctx.fillStyle='rgb('+v.color.join(',')+')';ctx.fill()}document.getElementById('info').textContent=' Model '+model.id+' | '+p.length+' points | NOT_QUALITY_ACCEPTED';}
canvas.onpointerdown=e=>{drag=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId)};canvas.onpointermove=e=>{if(!drag)return;ay+=(e.clientX-drag[0])*.008;ax+=(e.clientY-drag[1])*.008;drag=[e.clientX,e.clientY];render()};canvas.onpointerup=()=>drag=null;canvas.onpointercancel=()=>drag=null;canvas.addEventListener('wheel',e=>{e.preventDefault();zoom=Math.max(.1,Math.min(12,zoom*Math.exp(-e.deltaY*.001)));render()},{passive:false});function reset(){ax=-.3;ay=.4;zoom=1;render()}selector.onchange=reset;document.getElementById('reset').onclick=reset;render();</script></html>'''
    (output_dir/'index.html').write_text(page, encoding='utf-8')
    after = [binding(path) for path in inputs+ply_paths]
    unchanged = before == after
    report = {'schema':'learned-geometry-report-v1','status':'NOT_QUALITY_ACCEPTED',
              'scale_status':'UNKNOWN','components_merged':False,'inputs_before':before,
              'inputs_after':after,'inputs_unchanged':unchanged,'renderer':binding(Path(__file__).resolve()),
              'primary_model_id':main['id'],'outputs':['index.html','projections.png']}
    (output_dir/'REPORT_BINDINGS.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    if not unchanged:
        raise RuntimeError('Input drift detected; report is not stable')
    return report
