"""Read back saved association decisions and render a report without refitting."""
import argparse
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import sys
from urllib.parse import quote
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def verify(ref):
    p=Path(ref['path']).resolve(strict=True)
    if digest(p)!=ref['sha256']:raise ValueError('binding mismatch: '+str(p))
    return p
def link(path,output):return quote(os.path.relpath(path,output).replace('\\','/'),safe='/:')
def dump(path,value):path.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf-8')


def build(source,output):
    source=source.resolve(strict=True);output=output.resolve()
    if output.exists() or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('fresh nonoverlapping output required')
    inputs=[source/'PLAN.json',source/'RESULT.json',source/'DECISIONS.json',source/'SELECTION_LEDGER.json']
    start={str(p):digest(p) for p in inputs}
    plan,result,decisions,ledger=map(read,inputs)
    if result['frames']!=decisions or result['selected_candidates']!=len(ledger):raise ValueError('decision count mismatch')
    if not result['source_unchanged'] or result['publishable_for_geometry']:raise ValueError('unaccepted source state')
    frame_map={f['input_index']:f for f in plan['frames']}
    for ref in plan['source_bindings']+plan['implementation']:verify(ref)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18)
    output.mkdir(parents=True)
    pages=[];pngs=[]
    for row in decisions:
        index=row['input_index'];frame=frame_map[index];evidence=row.get('kernel_evidence') or {}
        reasons=evidence.get('reasons') or row.get('reasons') or ([evidence['reason']] if evidence.get('reason') else [])
        selected=row.get('selection');selected_id=selected['candidate_id'] if selected else None
        candidate_metrics={c['candidate_id']:c for c in evidence.get('candidates',[])}
        if row['original_status']=='NEEDS_SELECTION' or index in (57,82,83):
            tiles=[]
            choices=[]
            for endpoint,label in ((row['before_index'],'前参考'),(index,'当前原图'),(row['after_index'],'后参考')):
                if endpoint is not None:
                    f=frame_map[endpoint];choices.append((f,label+str(endpoint),f.get('mask_binding') if endpoint!=index else None,None))
            for c in row['candidates']:
                title=('关联建议' if c['candidate_id']==selected_id else '原始候选')+' '+str(c['candidate_id'])
                choices.append((frame,title,c['binding'],c['candidate_id']))
            for f,label,mask_ref,cid in choices:
                path=verify(dict(path=f['path'],sha256=f['sha256']))
                with Image.open(path) as im:picture=im.convert('RGB')
                if mask_ref:
                    with Image.open(verify(mask_ref)) as im:mask=np.array(im)
                    color=(60,160,255) if cid==selected_id and selected_id is not None else (35,230,130)
                    picture=Image.composite(Image.new('RGB',picture.size,color),picture,Image.fromarray(((mask>0)*95).astype('uint8')))
                tile=Image.new('RGB',(288,630),'#18313c');picture.thumbnail((288,512));tile.paste(picture,(0,110))
                draw=ImageDraw.Draw(tile);draw.text((5,4),label,font=font,fill='white')
                draw.text((5,30),str(f['original_mask_status']),font=font,fill='#d0e4ec')
                if cid in candidate_metrics:
                    m=candidate_metrics[cid]
                    draw.text((5,56),'IoU %.3f / %.3f'%(m['iou_before'],m['iou_after']),font=font,fill='white')
                    draw.text((5,80),'正点覆盖 %.3f'%m['query_point_coverage'],font=font,fill='white')
                tiles.append(tile)
            columns=min(3,len(tiles));board=Image.new('RGB',(288*columns,630*((len(tiles)+columns-1)//columns)),'#10232d')
            for n,tile in enumerate(tiles):board.paste(tile,((n%columns)*288,(n//columns)*630))
            name='frame_'+str(index)+'.png';board.save(output/name);pngs.append(name)
            details=[]
            for name_ref,ref in evidence.get('references',{}).items():
                details.append(dict(side=name_ref,fit_inlier_fraction=ref.get('fit_inlier_fraction'),
                    holdout_p90_px=ref.get('holdout_p90_px'),holdout_max_px=ref.get('holdout_max_px'),
                    reasons=ref.get('reasons')))
            pages.append('<section id="f'+str(index)+'"><h2>帧 '+str(index)+' · '+html.escape(row['status'])+'</h2>'+
                '<p>原始状态 '+html.escape(row['original_status'])+'；前后参考 '+str(row['before_index'])+' / '+str(row['after_index'])+
                '；可用实际轨迹 '+str(row['eligible_track_count'])+'；建议候选 '+str(selected_id)+'</p>'+
                '<p>判定原因：'+html.escape(', '.join(reasons) or '见下方证据')+'</p>'+f'<img src="{name}" alt="原始图像与候选轮廓">'+
                '<details><summary>数值证据与拒绝理由</summary><pre>'+html.escape(json.dumps(dict(references=details,candidates=evidence.get('candidates'),
                track_exclusions=row.get('track_exclusion_counts'),prediction_agreement_iou=evidence.get('prediction_agreement_iou'),winner_margin=evidence.get('winner_margin')),ensure_ascii=False,indent=2))+'</pre></details></section>')
    counts=dict(Counter(r['status'] for r in decisions))
    body='<h1>前后帧核对 SAM 原始候选</h1><p class="note">这是一份离线二维关联建议。只从已有候选中选择，不重跑SAM、不修改原始NO_MATCH或NEEDS_SELECTION，也不生成补齐轮廓。共同运动的手部仍可能通过，不代表已排除遮挡或得到可靠三维。</p>'
    body+='<p>28帧诊断；派生建议 '+str(len(ledger))+' 项。状态计数：'+html.escape(str(counts))+'</p>'
    body+='<p>绿色：原始轮廓/原始候选；蓝色：获得时序关联建议的原始候选。参考帧始终来自原始OK结果，不使用本轮派生选择递归确认。光学焦距变化没有得到验证。</p>'
    body+='<table><tr><th>帧</th><th>原始SAM</th><th>派生判断</th><th>实际轨迹</th><th>前/后参考</th><th>建议候选</th></tr>'
    for r in decisions:
        ix=r['input_index'];selection=r.get('selection')
        label=f'<a href="#f{ix}">{ix}</a>' if r['original_status']=='NEEDS_SELECTION' or ix in (57,82,83) else str(ix)
        body+='<tr><td>'+label+'</td>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in (r['original_status'],r['status'],r['eligible_track_count'],(r['before_index'],r['after_index']),selection['candidate_id'] if selection else None))+'</tr>'
    body+='</table>'+''.join(pages)
    body+='<p>相似运动拟合使用一组点，另留实际点校验。两侧共享轨迹链，不能称为两个统计独立实验；轮廓外推范围、裁剪、空间覆盖和候选分差均受固定门槛约束。</p>'
    body+='<p><a href="'+link(source/'RESULT.json',output)+'">完整判定证据</a> · <a href="'+link(source/'SELECTION_LEDGER.json',output)+'">派生选择记录</a> · <a href="'+link(source/'PLAN.json',output)+'">冻结输入</a></p>'
    (output/'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>SAM跨帧关联建议</title><style>body{background:#10232d;color:#edf4f6;font:16px/1.8 system-ui;max-width:1200px;margin:24px auto;padding:0 22px}a{color:#88d5ff}img{max-width:100%}table{border-collapse:collapse;width:100%}th,td{border-bottom:1px solid #4a626c;padding:6px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}.note{border-left:4px solid #eeca69;background:#1b3541;padding:12px}section{margin-top:35px}</style>'+body+'</html>',encoding='utf-8')
    assert start=={str(p):digest(p) for p in inputs}
    result=dict(status='READBACK_REPORT_ONLY',source_bindings=start,source_unchanged=True,counts=counts,
        pngs=[dict(path=n,sha256=digest(output/n)) for n in pngs],selected_candidates=len(ledger),
        script_sha256=digest(Path(__file__)),report_sha256=digest(output/'index.html'),
        model_calls=0,refits=0,quality_status='NOT_EVALUATED',browser_interaction='NOT_VERIFIED')
    dump(output/'RESULT.json',result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',required=True,type=Path);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();print(json.dumps(build(args.source,args.output),ensure_ascii=False))
