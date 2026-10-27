"""Create a local, playable report from committed incremental coverage evidence."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from urllib.parse import quote


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def binding(path):
    return dict(path=str(path), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def link(path, root):
    return quote(os.path.relpath(path, root).replace('\\', '/'), safe='/:')


def build(source, output):
    source = source.resolve(strict=True)
    output = output.resolve()
    if output.exists() or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('fresh nonoverlapping report directory required')
    paths = [source / name for name in ('INPUTS.json', 'TRACKS.json', 'RESULT.json')]
    before = [binding(p) for p in paths]
    inputs, tracking, result = map(read, paths)
    if result['status'] != 'PARTIAL_CONDITIONAL_EXPANSION' or not result['source_unchanged']:
        raise ValueError('committed coverage evidence required')
    if tracking['profile'].get('profile_id') != 'whole-video-incremental-v1':
        raise ValueError('incremental tracking evidence required')
    frames = inputs['frames']
    if len(frames) != len(tracking['coverage']):
        raise ValueError('frame statistics differ')
    per_frame = [[] for _ in frames]
    recovered = {(e['track_id'], e['frame_index']) for e in tracking['events']
                 if e['kind'] == 'mask_gap_recovered'}
    for track in tracking['tracks']:
        for obs in track['observations']:
            f = obs['frame_index']
            state = ('unknown' if obs['label'] == 'unknown' else
                     'birth' if f == track['seed_frame'] else
                     'recovered' if (track['track_id'], f) in recovered else 'tracked')
            per_frame[f].append([track['track_id'], obs['identity_segment'],
                                 obs['xy'][0] - .5, obs['xy'][1] - .5, state])
    output.mkdir(parents=True)
    rows = []
    for i, frame in enumerate(frames):
        image = Path(frame['path'])
        if hashlib.sha256(image.read_bytes()).hexdigest() != frame['sha256']:
            raise ValueError('image bytes drifted')
        stats = tracking['coverage'][i]
        if len(per_frame[i]) != stats['active']:
            raise ValueError('active count cannot be reproduced')
        rows.append(dict(index=frame['input_index'], time=frame['time_seconds'],
                         status=frame['original_mask_status'], image=link(image, output),
                         stats=stats, points=per_frame[i]))
    events = Counter(e['kind'] for e in tracking['events'])
    counts = dict(frames=len(rows), candidate_tracks=len(tracking['tracks']),
                  observations=sum(len(p) for p in per_frame),
                  adjacent_flow_edges=sum(max(0, len(t['observations']) - 1) for t in tracking['tracks']),
                  births_after_first_mask=sum(t['seed_frame'] > min(t['seed_frame'] for t in tracking['tracks'])
                                             for t in tracking['tracks']) if tracking['tracks'] else 0,
                  unknown_observations=sum(p[4] == 'unknown' for row in per_frame for p in row),
                  candidate_gap_returns=events['mask_gap_recovered'],
                  verified_cross_gap_reidentifications=0,
                  independently_verified_physical_point_ids=0,
                  supported_points=sum(g['supported_points'] for g in result['groups']),
                  added_supported_points=sum(g['added_strict'] + g['added_relaxed'] for g in result['groups']),
                  provisional_points=sum(g['two_view_provisional'] for g in result['groups']),
                  faces=sum(g['faces'] for g in result['groups']),
                  independent_components=len(result['groups']))
    links = dict(coverage=link(source/'index.html', output),
                 viewer=link(source/'viewer.html', output),
                 blend=link(source.parent/'blender/model.blend', output))
    data = json.dumps(dict(frames=rows, counts=counts, links=links), ensure_ascii=False,
                      allow_nan=False, separators=(',', ':')).replace('<', '\\u003c')
    document = TEMPLATE.replace('__DATA__', data)
    (output/'index.html').write_text(document, encoding='utf-8')
    if before != [binding(p) for p in paths]:
        raise ValueError('source evidence changed')
    summary = dict(status='REAL_TRACK_VISUALIZATION', counts=counts, input_bindings=before,
                   sources_unchanged=True, geometry_accepted=False, quality_status='NOT_EVALUATED',
                   scale_status='unknown', independently_verified_physical_point_ids=0,
                   report=binding(output/'index.html'), implementation=binding(Path(__file__).resolve()))
    (output/'RESULT.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2,
                                                allow_nan=False)+'\n', encoding='utf-8')
    return summary


TEMPLATE = r'''<!doctype html><html lang="zh"><meta charset="utf-8">
<title>逐帧补点与轨迹延续</title><style>
body{margin:24px auto;max-width:1250px;padding:0 20px;background:#10232d;color:#ecf4f7;font:16px/1.65 system-ui}
a{color:#88d3ff}button,input{font:inherit}button{padding:6px 16px;background:#244959;border:1px solid #527381;color:white;border-radius:6px}
.layout{display:grid;grid-template-columns:minmax(310px,576px) 1fr;gap:24px}.panel{padding:16px;background:#18333e;border-radius:10px}
canvas{width:100%;height:auto;background:#07151e;cursor:crosshair}#slider{width:100%}.notice{border-left:4px solid #ffd25e;padding:8px 16px}
.legend span{display:inline-block;margin-right:15px}.stat{font-size:24px;color:#a1ecd3}pre{white-space:pre-wrap;font-size:14px}
@media(max-width:800px){.layout{grid-template-columns:1fr}}</style>
<h1>每帧保留旧点，补充物体内的新点</h1>
<p id="summary"></p><p><a id="coverage">覆盖与几何记录</a> · <a id="viewer">局部三维</a> · <a id="blend">Blender 文件</a></p>
<p class="notice">这里展示实际计算的候选轨迹。编号相同表示跟踪假设，不等于物理身份已经证实。黄色为轮廓缺失时的短期光流候选，不参与三维；紫色返回点另开几何段。超长缺口、已检测的光流失败或已知背景会结束旧轨迹，真实遮挡不保证能被自动识别。</p>
<div class="layout"><div><canvas id="view" width="576" height="1024"></canvas></div><div class="panel">
<div><button id="play">播放</button> <button id="prev">上一帧</button> <button id="next">下一帧</button> <label><input type="checkbox" id="trails" checked>短轨迹</label></div>
<input id="slider" type="range" min="0" step="1"><p id="position"></p>
<div class="legend"><span style="color:#56ddff">● 新补点</span><span style="color:#65ed9b">● 连续点</span><span style="color:#ffcb59">● 未知区间候选</span><span style="color:#ec8eff">● 返回的新几何段</span></div>
<p id="stats" class="stat"></p><p id="status"></p><p>点击画面中的点可查看候选编号与几何段；可拖动时间条检查该点是否继续存在。</p><pre id="selected">尚未选择候选点。</pre>
<p>每一帧都检查旧点，再在已提交的 SAM 轮廓内按 8×12 网格补充角点。新点从首次实际观测开始，未来帧继续核对，不补写过去的坐标。</p>
<p>相机沿用已有局部估计，没有新增相机配准；独立分量没有拼成完整物体。表面含插值，孔洞保留；尺寸、外形与匹配精度尚未验收。</p>
</div></div>
<script>const data=__DATA__;
const frames=data.frames,c=document.getElementById('view'),ctx=c.getContext('2d'),slider=document.getElementById('slider');
const colors={birth:'#56ddff',tracked:'#65ed9b',unknown:'#ffcb59',recovered:'#ec8eff'};
let index=Math.max(0,frames.findIndex(f=>f.points.length)),serial=0,picture=null,chosen=null,playing=false,timer=null;
slider.max=frames.length-1;for(const [k,v]of Object.entries(data.links))document.getElementById(k).href=v;
const n=data.counts;document.getElementById('summary').textContent=`实际处理 ${n.frames} 帧，${n.candidate_tracks} 条候选轨迹；条件支持点 ${n.supported_points}（新增 ${n.added_supported_points}），${n.independent_components} 个独立局部坐标系。`;
const maps=frames.map(f=>new Map(f.points.map(p=>[p[0],p])));
function selectedText(){const p=chosen===null?null:maps[index].get(chosen);document.getElementById('selected').textContent=chosen===null?'尚未选择候选点。':p?`候选轨迹 ${p[0]} / 几何段 ${p[1]}\n像素位置 (${p[2].toFixed(2)}, ${p[3].toFixed(2)})\n状态 ${p[4]}\n跨段物理身份：未确认`:`候选轨迹 ${chosen} 在本帧没有有效观测；不补画预测点。`;}
function draw(){if(!picture)return;ctx.clearRect(0,0,c.width,c.height);ctx.drawImage(picture,0,0);const ps=frames[index].points;
if(document.getElementById('trails').checked)for(const p of ps){ctx.strokeStyle=colors[p[4]];ctx.globalAlpha=.65;ctx.beginPath();ctx.moveTo(p[2],p[3]);for(let j=index-1;j>=Math.max(0,index-6);j--){const q=maps[j].get(p[0]);if(!q||q[1]!==p[1])break;ctx.lineTo(q[2],q[3]);}ctx.stroke();}ctx.globalAlpha=1;
for(const p of ps){ctx.fillStyle=colors[p[4]];ctx.beginPath();ctx.arc(p[2],p[3],p[4]==='birth'?3.1:2.1,0,Math.PI*2);ctx.fill();if(p[0]===chosen){ctx.strokeStyle='white';ctx.lineWidth=2;ctx.beginPath();ctx.arc(p[2],p[3],8,0,Math.PI*2);ctx.stroke();ctx.lineWidth=1;}}selectedText();}
async function show(i){index=Math.max(0,Math.min(frames.length-1,i));slider.value=index;const f=frames[index],s=f.stats,key=++serial;
picture=null;ctx.clearRect(0,0,c.width,c.height);
document.getElementById('position').textContent=`帧 ${f.index} / ${frames.length-1} · ${f.time.toFixed(3)} 秒`;
document.getElementById('stats').textContent=`延续 ${s.tracked} · 新增 ${s.seeded} · 返回 ${s.recovered} · 未知候选 ${s.suspended}`;
document.getElementById('status').textContent=`原分割状态：${f.status}。${f.status==='OK'?'当前物体内点可以进入分段几何核验。':'当前没有已提交的物体轮廓；不播种、不认定物体内点。'}`;
const im=new Image();im.src=f.image;try{await im.decode();if(key===serial){picture=im;c.width=im.naturalWidth;c.height=im.naturalHeight;draw();}}catch(e){if(key===serial){picture=null;ctx.clearRect(0,0,c.width,c.height);document.getElementById('status').textContent='原始图片读取失败：'+String(e);}}}
function stop(){playing=false;clearTimeout(timer);document.getElementById('play').textContent='播放';}
async function tick(){if(!playing)return;await show(index<frames.length-1?index+1:0);if(playing)timer=setTimeout(tick,120);}
document.getElementById('play').onclick=()=>{if(playing)stop();else{playing=true;document.getElementById('play').textContent='暂停';tick();}};
document.getElementById('prev').onclick=()=>{stop();show(index-1);};document.getElementById('next').onclick=()=>{stop();show(index+1);};slider.oninput=()=>{stop();show(Number(slider.value));};document.getElementById('trails').onchange=draw;
c.onclick=e=>{if(!picture)return;const r=c.getBoundingClientRect(),x=(e.clientX-r.left)*c.width/r.width,y=(e.clientY-r.top)*c.height/r.height;let best=null,d=18;for(const p of frames[index].points){const v=Math.hypot(p[2]-x,p[3]-y);if(v<d){d=v;best=p[0];}}chosen=best;draw();};show(index);
</script></html>'''


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--coverage', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.coverage, args.output), ensure_ascii=False))
