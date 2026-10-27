"""Read saved boundary/point diagnostics; no model, tracking or selection."""
import argparse
from collections import Counter
import html
import os
from pathlib import Path
import sys
from urllib.parse import quote

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0,str(Path(__file__).resolve().parent))
from review_mask_association import read, save, bind, verify


def link(path, output):
    return quote(os.path.relpath(path,output).replace('\\','/'),safe='/:')


def ratio(x):
    return '未知' if x is None else f'{100*x:.1f}%'


def build(source, output):
    source=source.resolve(strict=True);output=output.resolve()
    if output.exists() or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('fresh nonoverlapping report required')
    inputs=[bind(source/n) for n in ('PLAN.json','RESULT.json','DECISIONS.json','EXTERIOR_PAIRS.json','SELECTION_LEDGER.json')]
    plan,result,decisions,pairs,ledger=[read(r['path']) for r in inputs]
    if result['frames']!=decisions or result['pair_ledgers']!=pairs or ledger or result['selected_candidates']:
        raise ValueError('inconsistent evidence')
    for ref in plan['source_bindings']+plan['implementation']+plan['runtime_bindings']:verify(ref)
    frames={f['input_index']:f['original_frame'] for f in plan['frames']}
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17)
    output.mkdir(parents=True)
    pages=[];table=[];pngs=[];summary=[]
    for row in decisions:
        ix=row['input_index'];assessment=row['assessment']
        if assessment is None:
            table.append(f'<tr><td>{ix}</td><td>{row["original_status"]}</td><td colspan="5">原始无候选，未填补</td></tr>');continue
        candidate_refs={c['candidate_id']:c['binding'] for c in row['candidates']}
        tiles=[]
        for c in assessment['candidates']:
            cid=c['candidate_id'];edge=c['boundary'];pos=c['positive_evidence'];ext=c['exterior_evidence']
            simple=dict(input_index=ix,candidate_id=cid,positive_inside=pos['inside_count'],positive_count=pos['count'],exterior_inside=ext['inside_count'],exterior_count=ext['count'],edge_alignment=edge['image_edge_alignment'],clipped_boundary_px=edge['clipped_px'],reasons=c['reasons'])
            summary.append(simple)
            table.append('<tr>'+''.join(f'<td>{html.escape(str(v))}</td>' for v in (ix,row['original_status'],cid,f'{pos["inside_count"]}/{pos["count"]}',f'{ext["inside_count"]}/{ext["count"]}',ratio(edge['image_edge_alignment']),edge['clipped_px']))+'</tr>')
            if row['original_status']!='NEEDS_SELECTION':continue
            f=frames[ix];verify(dict(path=f['path'],sha256=f['sha256']));verify(candidate_refs[cid])
            with Image.open(f['path']) as im:picture=im.convert('RGB')
            with Image.open(candidate_refs[cid]['path']) as im:mask=np.asarray(im)
            tint=Image.fromarray(((mask>0)*55).astype(np.uint8))
            picture=Image.composite(Image.new('RGB',picture.size,'#2eda83'),picture,tint)
            tile=Image.new('RGB',(picture.width,picture.height+140),'#162c36');tile.paste(picture,(0,140));draw=ImageDraw.Draw(tile)
            labels=[f'帧 {ix} / 原始候选 {cid}',f'正点覆盖 {pos["inside_count"]}/{pos["count"]}；包含参考外点 {ext["inside_count"]}/{ext["count"]}',f'图像边缘邻近 {ratio(edge["image_edge_alignment"])}；裁剪未知边界 {edge["clipped_px"]} px','蓝：正点；紫：被包含外点；灰：未包含外点']
            for j,s in enumerate(labels):draw.text((8,5+j*31),s,font=font,fill='white')
            for p in pos['points']:
                x,y=p['query_point_xy'];draw.ellipse((x-5,y+135,x+5,y+145),outline='#11ccff',width=3)
            for p in ext['points']:
                x,y=p['query_point_xy'];color='#ff44c8' if p['inside'] else '#b7bdc3'
                draw.rectangle((x-3,y+137,x+3,y+143),outline=color,width=2)
            tiles.append(tile)
        if tiles:
            board=Image.new('RGB',(sum(t.width for t in tiles),max(t.height for t in tiles)),'#162c36');x=0
            for tile in tiles:board.paste(tile,(x,0));x+=tile.width
            name=f'frame_{ix}.png';board.save(output/name);pngs.append(bind(output/name))
            pages.append(f'<section><h2>帧 {ix} · 原始歧义保留</h2><p>参考帧 {row["exterior_pair"]}；外点必须经过每个相邻帧的实际测量，终点筛选具有幸存者条件。</p><a href="{name}"><img src="{name}" alt="原始候选与实测正点、参考外点"></a><details><summary>全部测量与原因</summary><pre>'+html.escape(str(assessment))+'</pre></details></section>')
    pair_rows=[]
    for name,pair in pairs.items():
        pair_rows.append('<tr><td>'+name+'</td><td>'+html.escape(str(pair['counts']))+'</td></tr>')
    title='边界与参考外点：哪些候选存在观测冲突'
    body='<h1>'+title+'</h1><p class="note">没有自动提交掩膜。参考掩膜外的纹理点不是已证实的物理背景；强图像边缘也可能来自物体内部纹理、孔洞或手部。无冲突不代表轮廓正确。</p>'
    body+='<p>固定两段参考区间80→82、83→87，每段最多160个外部纹理种子。只跟踪原始参考轮廓外8–48像素带，不按当前候选筛点。记录所有种子、失败链和终点未通过者，查询坐标来自实际光流，不作插值。</p><table><tr><th>参考区间</th><th>完整分母</th></tr>'+''.join(pair_rows)+'</table>'
    body+='<p>边界邻近率：候选内部边界到Canny边缘的距离≤3px。图像最外3行/列记未知。它是图像边缘一致性描述，不是语义准确率或自动选择分数。</p><table><tr><th>帧</th><th>原始SAM</th><th>候选</th><th>正点覆盖</th><th>包含参考外点</th><th>边缘邻近率</th><th>裁剪未知像素</th></tr>'+''.join(table)+'</table>'
    body+=''.join(pages)
    body+='<p class="note">正点来自上一轮已保存的局部仿射双侧留出检验；两类点都不是人工真值。新露出区域、遮挡、手部共同运动、参考SAM自身遗漏仍可能改变判断。原NO_MATCH/NEEDS_SELECTION与旧掩膜未修改。</p>'
    body+='<p><a href="'+link(source/'RESULT.json',output)+'">完整结果</a> · <a href="'+link(source/'EXTERIOR_PAIRS.json',output)+'">全部光流链</a> · <a href="'+link(source/'PLAN.json',output)+'">冻结计划</a> · <a href="../SOURCE_REVIEW.md">书籍与官方资料</a></p>'
    document='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title><style>body{background:#10232c;color:#e5f1f5;font:16px/1.8 system-ui;max-width:1250px;margin:28px auto;padding:0 22px}a{color:#78d2ff}img{width:100%;height:auto}table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #34505d;text-align:left;padding:7px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}.note{background:#26353d;border-left:4px solid #edc66c;padding:15px}section{margin-top:35px}</style>'+body+'</html>'
    (output/'index.html').write_text(document,encoding='utf-8')
    for ref in inputs:verify(ref)
    report=dict(status='READBACK_REPORT_ONLY',source_bindings=inputs,source_unchanged=True,pngs=pngs,candidates=summary,pair_counts={k:v['counts'] for k,v in pairs.items()},selected_candidates=0,tracking_calls=0,model_calls=0,quality_status='NOT_EVALUATED',browser_interaction='NOT_VERIFIED',script=bind(Path(__file__)),report=bind(output/'index.html'))
    save(output/'RESULT.json',report);return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',required=True,type=Path);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();build(args.source,args.output)
