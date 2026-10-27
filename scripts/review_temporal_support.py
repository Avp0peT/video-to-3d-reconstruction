"""Bounded CPU A/B of object-scoped candidate tracking; never geometry acceptance."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
import numpy as np
from PIL import Image, ImageDraw
from kozlov_lab.stereo.coverage_tracks import track_coverage, profile_for
from kozlov_lab.stereo.temporal_support import DEFAULT_PROFILE
from integrations.demo_pipeline.temporal_review import load_cv

CLIPS = [(49, 62), (77, 90)]
PROFILES = [('before', 'whole-video-incremental-v1'), ('after', 'whole-video-temporal-support-v2')]
COLORS = {'original_mask': '#65ed9b', 'seed': '#56ddff', 'unknown': '#ffcb59',
          'temporal_supported': '#469cff', 'boundary_supported': '#ed85ee'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + '\n', encoding='utf-8')


def bind(path):
    path = Path(path).resolve(strict=True)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def verify(ref):
    actual = bind(ref['path'])
    if actual['sha256'] != ref['sha256'] or ('bytes' in ref and actual['bytes'] != ref['bytes']):
        raise ValueError('binding drift: ' + ref['path'])
    return actual


def view_rows(tracking, frames):
    observations = [[] for _ in frames]
    for track in tracking['tracks']:
        for obs in track['observations']:
            state = obs.get('mask_status', 'original_mask' if obs['label'] == 'foreground' else 'unknown')
            if obs['frame_index'] == track['seed_frame']:
                state = 'seed'
            observations[obs['frame_index']].append(dict(track_id=track['track_id'],
                segment=obs['identity_segment'], xy=[v - .5 for v in obs['xy']], state=state,
                label=obs['label'], target_verified=obs['target_verified'], proof=obs.get('temporal_support')))
    rows = []
    for i, (frame, points) in enumerate(zip(frames, observations)):
        counts = dict(Counter(p['state'] for p in points))
        if len(points) != tracking['coverage'][i]['active']:
            raise ValueError('observation count mismatch')
        if any(p['target_verified'] for p in points if p['label'] != 'foreground'):
            raise ValueError('unverified point upgraded')
        rows.append(dict(input_index=frame['input_index'], time=frame['time_seconds'],
            source_mask_status=frame['original_mask_status'], counts=counts,
            coverage=tracking['coverage'][i], points=points))
    return rows


def transitions(rows):
    out = []
    for a, b in zip(rows, rows[1:]):
        if b['input_index'] not in (57, 81, 84):
            continue
        previous = {p['track_id'] for p in a['points']}
        surviving = [p for p in b['points'] if p['track_id'] in previous]
        out.append(dict(from_index=a['input_index'], to_index=b['input_index'],
            before_active=len(previous), surviving_old=len(surviving),
            surviving_states=dict(Counter(p['state'] for p in surviving)),
            new_seeds=b['coverage']['seeded'], source_mask_status=b['source_mask_status']))
    return out


def render(frame, rows, path, title):
    with Image.open(frame['path']) as source:
        picture = source.convert('RGB')
    draw = ImageDraw.Draw(picture)
    for p in rows['points']:
        x, y = p['xy']; r = 2.2
        draw.ellipse((x-r, y-r, x+r, y+r), fill=COLORS[p['state']])
    draw.rectangle((0, 0, picture.width, 40), fill='#10232d')
    draw.text((10, 6), title, fill='white')
    draw.text((10, 22), 'SAM: ' + frame['original_mask_status'] + ' | tracked: ' + str(len(rows['points'])), fill='white')
    picture.save(path)


def build(source, output):
    start = time.monotonic()
    source = source.resolve(strict=True); output = output.resolve()
    if output.exists() or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('fresh, nonoverlapping output required')
    inputs = read(source / 'INPUTS.json'); source_result = read(source / 'RESULT.json')
    if source_result['status'] != 'PARTIAL_CONDITIONAL_EXPANSION' or not source_result['source_unchanged']:
        raise ValueError('committed incremental coverage source required')
    if inputs['track_profile']['profile_id'] != PROFILES[0][1]:
        raise ValueError('source profile differs')
    frame_map = {f['input_index']: f for f in inputs['frames']}
    if len(frame_map) != len(inputs['frames']): raise ValueError('duplicate indices')
    source_refs = [bind(source / name) for name in ('INPUTS.json', 'RESULT.json')]
    clips = []
    for lo, hi in CLIPS:
        frames = [frame_map[i] for i in range(lo, hi+1)]
        for a, b in zip(frames, frames[1:]):
            if b['decode_index'] != a['decode_index']+1 or b['pts'] <= a['pts'] or b['time_base'] != a['time_base']:
                raise ValueError('nonadjacent video frames')
        for frame in frames:
            source_refs.append(verify(dict(path=frame['path'], sha256=frame['sha256'])))
            mask = frame.get('mask_binding')
            if (frame['original_mask_status'] == 'OK') != bool(mask):
                raise ValueError('mask provenance mismatch')
            if mask: source_refs.append(verify(mask))
        clips.append(frames)
    implementation = [bind(ROOT / p) for p in (
        'scripts/review_temporal_support.py', 'src/kozlov_lab/stereo/coverage_tracks.py',
        'src/kozlov_lab/stereo/temporal_support.py', 'integrations/demo_pipeline/temporal_review.py')]
    cv_refs = []; cv = load_cv(cv_refs)
    if cv is None: raise ValueError('pinned OpenCV unavailable')
    output.mkdir(parents=True)
    plan = dict(status='FROZEN_BEFORE_TRACKING', clips=CLIPS, profiles={name: profile_for(profile_id=p) for name,p in PROFILES},
        support_profile=DEFAULT_PROFILE, expected_tracker_calls=4, unique_frames=28, variant_frame_evaluations=56,
        geometry_calls=0, model_calls=0, implementation=implementation, source_bindings=source_refs,
        runtime_bindings=cv_refs, argv=sys.argv, cwd=os.getcwd(), input_frames=clips,
        threads=2, geometry_accepted=False, quality_status='NOT_EVALUATED',
        sampling='contiguous slices initialized independently; ids are local to clip and variant')
    save(output/'PLAN.json', plan)
    report_clips = []; ledger = []; all_summaries = []
    for frames in clips:
        name = str(frames[0]['input_index']) + '_' + str(frames[-1]['input_index'])
        clip_dir = output/name; clip_dir.mkdir()
        grays=[]; masks=[]
        for frame in frames:
            with Image.open(frame['path']) as image:
                grays.append(np.array(image.convert('L')))
            mask = frame.get('mask_binding')
            if mask:
                with Image.open(mask['path']) as image:
                    if image.mode != 'L': raise ValueError('mask mode differs')
                    masks.append(np.array(image))
            else: masks.append(None)
        variants = {}
        for variant, profile in PROFILES:
            run = clip_dir/variant; run.mkdir()
            t = time.monotonic()
            entry=dict(clip=name, variant=variant, profile=profile, monotonic_start=t, status='RUNNING')
            ledger.append(entry); save(output/'ATTEMPTS.json', ledger)
            tracking = track_coverage(grays,masks,cv,profile_id=profile)
            entry.update(monotonic_end=time.monotonic(),status='RETURNED')
            entry['wall_seconds']=entry['monotonic_end']-t
            save(output/'ATTEMPTS.json', ledger)
            rows=view_rows(tracking,frames)
            save(run/'TRACKS.json',tracking); save(run/'FRAMES.json',rows)
            reasons=Counter(reason for row in rows for p in row['points'] for reason in (p['proof'] or {}).get('reasons',[]))
            summary=dict(clip=name,variant=variant,tracks=len(tracking['tracks']),observations=sum(len(r['points']) for r in rows),
                states=dict(Counter(p['state'] for r in rows for p in r['points'])),
                support_rejection_reasons=dict(reasons), transitions=transitions(rows), wall_seconds=entry['wall_seconds'])
            save(run/'SUMMARY.json',summary);all_summaries.append(summary)
            compact=[]
            for frame,row in zip(frames,rows):
                png=run/(str(frame['input_index'])+'.png')
                render(frame,row,png,variant.upper()+' | frame '+str(frame['input_index']))
                compact.append(dict(index=frame['input_index'],image=png.relative_to(output).as_posix(),
                    status=row['source_mask_status'],counts=row['counts'],active=len(row['points']),
                    seeded=row['coverage']['seeded'],tracked=row['coverage']['tracked']))
            variants[variant]=compact
        for index in (57,81,84):
            if index not in [f['input_index'] for f in frames]:continue
            panels=[Image.open(clip_dir/v/(str(index)+'.png')).convert('RGB') for v,_ in PROFILES]
            board=Image.new('RGB',(panels[0].width*2,panels[0].height))
            for n,panel in enumerate(panels):board.paste(panel,(n*panel.width,0))
            board.save(output/('comparison_'+str(index)+'.png'))
        report_clips.append(dict(name=name,variants=variants))
    for ref in source_refs+implementation: verify(ref)
    data=json.dumps(dict(clips=report_clips,summaries=all_summaries),ensure_ascii=False,allow_nan=False).replace('<','\\u003c')
    document=TEMPLATE.replace('__DATA__',data)
    (output/'index.html').write_text(document,encoding='utf-8')
    js=document.split('<script>')[1].split('</script>')[0]
    (output/'report.js').write_text(js,encoding='utf-8')
    result=dict(status='REAL_CPU_AB_COMPARISON', tracker_calls=len(ledger), summaries=all_summaries,
        unique_frames=28,variant_frame_evaluations=56,source_bindings_verified=len(source_refs),sources_unchanged=True,
        geometry_calls=0,model_calls=0,geometry_accepted=False,quality_status='NOT_EVALUATED',
        physical_identity_verified=False,wall_seconds=time.monotonic()-start,report=bind(output/'index.html'),
        full_video_run=False,ui_interaction='NOT_VERIFIED',
        limitations=['Slice warmup differs from full-video state','No hand/occluder ground truth',
            'Unknown or boundary-supported points are not new verified foreground or geometry'])
    save(output/'RESULT.json',result)
    return result


TEMPLATE='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>轮廓缺失时的时序支持：真实片段对比</title>
<style>body{background:#10232d;color:#eaf3f7;font:16px/1.7 system-ui;margin:24px auto;max-width:1250px;padding:0 18px}a{color:#8bd8ff}button,select{font:inherit;padding:6px 15px}input{width:100%}.pair{display:grid;grid-template-columns:1fr 1fr;gap:16px}.pair img{width:100%;height:auto}pre{white-space:pre-wrap}.note{border-left:4px solid #ffd25e;padding:10px 18px;background:#18343f}#stats{min-height:90px}</style>
<h1>轮廓未知时，保留经过检查的运动候选</h1>
<p class="note">真实 PPP 视频的两段连续切片：49–62、77–90。左右采用相同图像和原始 SAM 结果，分别重新初始化。蓝点有额外时序证据，但物体身份仍待确认；不会把黄色改成已验证的绿色，也没有生成新三维。</p>
<p><span style="color:#65ed9b">● 原始轮廓内</span>　<span style="color:#56ddff">● 新点</span>　<span style="color:#469cff">● 时序支持候选</span>　<span style="color:#ed85ee">● 边界延续候选</span>　<span style="color:#ffcb59">● 轮廓未知、支持不足</span></p>
<select id="clip"></select> <button id="prev">上一帧</button> <button id="play">播放</button> <button id="next">下一帧</button>
<input type="range" id="slider" min="0" step="1"><p id="position"></p>
<div class="pair"><section><h2>修复前 v1</h2><img id="before" alt="原方案真实轨迹"><pre id="left"></pre></section><section><h2>时序支持 v2</h2><img id="after" alt="新方案真实轨迹"><pre id="right"></pre></section></div>
<p>仅保留实际光流坐标，不画预测点。边界最多延续6像素、3帧；轮廓缺失时支持最多6帧。未知区间不能播种新点，返回轮廓后另开几何段。手指与物体共同运动仍可能通过检查，不能据此宣称没有误匹配。</p>
<p><a href="RESULT.json">结果与限制</a> · <a href="PLAN.json">冻结输入及规则</a> · <a href="ATTEMPTS.json">实际执行记录</a></p>
<script>const data=__DATA__;let ci=0,fi=0,playing=false,timer=null;
const clips=document.getElementById('clip'),slider=document.getElementById('slider');
data.clips.forEach((c,i)=>{const o=document.createElement('option');o.value=i;o.textContent='切片 '+c.name;clips.append(o);});
function show(){const c=data.clips[ci];slider.max=c.variants.before.length-1;slider.value=fi;const b=c.variants.before[fi],a=c.variants.after[fi];document.getElementById('position').textContent=`原始视频帧 ${a.index}；SAM状态 ${a.status}`;for(const [v,id]of [['before','left'],['after','right']]){const row=c.variants[v][fi];document.getElementById(v).src=row.image;document.getElementById(id).textContent=`有效观测 ${row.active}；延续 ${row.tracked}；新增 ${row.seeded}\\n`+JSON.stringify(row.counts,null,2);}}
function stop(){playing=false;clearTimeout(timer);document.getElementById('play').textContent='播放';}
function tick(){if(!playing)return;fi=(fi+1)%data.clips[ci].variants.before.length;show();timer=setTimeout(tick,180);}
clips.onchange=()=>{stop();ci=Number(clips.value);fi=0;show();};slider.oninput=()=>{stop();fi=Number(slider.value);show();};
document.getElementById('prev').onclick=()=>{stop();fi=Math.max(0,fi-1);show();};document.getElementById('next').onclick=()=>{stop();fi=Math.min(data.clips[ci].variants.before.length-1,fi+1);show();};document.getElementById('play').onclick=()=>{if(playing)stop();else{playing=true;document.getElementById('play').textContent='暂停';tick();}};show();</script></html>'''


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    try:
        print(json.dumps(build(args.source,args.output),ensure_ascii=False,allow_nan=False))
    except Exception as exc:
        if args.output.is_dir():save(args.output/'FAILURE.json',dict(error_type=type(exc).__name__,message=str(exc)))
        raise
