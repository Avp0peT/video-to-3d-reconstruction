"""Read saved evidence only: no model, flow, selection, or geometry execution."""
import argparse
import hashlib
import html
import io
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--coverage', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--indices', default='49..62,77..90')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output must be a fresh directory')
    indices = []
    for part in args.indices.split(','):
        bounds = part.split('..')
        indices.extend(range(int(bounds[0]), int(bounds[1])+1) if len(bounds)==2 else [int(part)])
    indices = sorted(set(indices))
    ledger = {}

    def read(path, expected=None, kind='source'):
        path = Path(path).resolve()
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if expected and digest != expected:
            raise ValueError(f'hash mismatch: {path}')
        key = str(path)
        old = ledger.get(key)
        if old and old['sha256'] != digest:
            raise ValueError(f'input changed: {path}')
        ledger[key] = dict(path=key, sha256=digest, bytes=len(data),
                           expected_sha256=expected or (old or {}).get('expected_sha256'), kind=kind)
        return data

    inputs_path = args.coverage/'INPUTS.json'
    inputs = json.loads(read(inputs_path, kind='coverage_inputs'))
    bindings = [b for b in inputs['source_bindings']
                if Path(b['path']).name=='manifest.json' and 'segmentation' in b['path']]
    if len(bindings)!=1:
        raise ValueError('one bound segmentation manifest required')
    binding = bindings[0]
    manifest_path = Path(binding['path'])
    manifest = json.loads(read(manifest_path, binding['sha256'], 'segmentation_manifest'))
    raw_by_id = {f['frame_id']:f for f in manifest['frames']}
    frames = inputs['frames']
    if not indices or indices[0]<0 or indices[-1]>=len(frames):
        raise ValueError('indices outside coverage')
    rows, masks, pictures, candidates = [], {}, {}, {}

    def metrics(mask):
        ys,xs = np.nonzero(mask)
        return dict(area=int(mask.sum()), bbox_xyxy_exclusive=
                    [int(xs.min()),int(ys.min()),int(xs.max())+1,int(ys.max())+1] if len(xs) else None,
                    centroid_xy=[float(xs.mean()),float(ys.mean())] if len(xs) else None)

    for i in indices:
        f=frames[i]; raw=raw_by_id[f['frame_id']]
        image=Image.open(io.BytesIO(read(f['path'],f['sha256'],'image'))).convert('RGB')
        if image.size!=(f['width'],f['height']): raise ValueError('image dimensions differ')
        pictures[i]=image
        mask=None
        if f.get('mask_binding'):
            b=f['mask_binding']; mask=np.array(Image.open(io.BytesIO(read(b['path'],b['sha256'],'effective_mask'))))>0
            if mask.shape!=(f['height'],f['width']): raise ValueError('mask dimensions differ')
        masks[i]=mask
        candrows=[]; candidates[i]=[]
        for c in raw.get('candidates',[]):
            cm=np.array(Image.open(io.BytesIO(read(manifest_path.parent/c['mask_path'],c['mask_sha256'],'candidate_mask'))))>0
            if cm.shape!=(f['height'],f['width']): raise ValueError('candidate dimensions differ')
            candidates[i].append((c['candidate_id'],cm))
            candrows.append(dict(c,measured=metrics(cm)))
        m=metrics(mask) if mask is not None else dict(area=None,bbox_xyxy_exclusive=None,centroid_xy=None)
        previous=masks.get(i-1); iou=shift=None
        if mask is not None and previous is not None:
            union=int((mask|previous).sum()); iou=float((mask&previous).sum()/union) if union else None
            p=metrics(previous)['centroid_xy']
            if m['centroid_xy'] is not None and p is not None:
                shift=float(np.linalg.norm(np.array(m['centroid_xy'])-p))
        rows.append(dict(frame=i,decode_index=f['decode_index'],frame_id=f['frame_id'],
                         original_status=f['original_mask_status'],raw_status=raw['status'],
                         candidate_count=len(candrows),candidates=candrows,
                         selected_candidate_id=raw.get('selected_candidate_id'),selection=raw.get('selection'),
                         raw_adjacent_iou=iou,centroid_displacement_px=shift,**m))
    args.output.mkdir(parents=True)
    font_path=Path('C:/Windows/Fonts/msyh.ttc')
    font=ImageFont.truetype(str(font_path),18) if font_path.exists() else ImageFont.load_default()
    pngs=[]
    by_index={r['frame']:r for r in rows}
    for group in ((82,83),(80,81,82),(83,84,87)):
        present=[i for i in group if i in pictures]
        if not present: continue
        tiles=[]
        for i in present:
            r=by_index[i]
            choices=[('原图',None),('已提交轮廓',masks[i])]+[(f'候选 {cid}（本核查未选择）',m) for cid,m in candidates[i]]
            for label,mask in choices:
                tile=Image.new('RGB',(288,590),'#17202a'); im=pictures[i].copy()
                if mask is not None:
                    tint=Image.new('RGB',im.size,(35,230,130))
                    alpha=Image.fromarray((mask*95).astype('uint8'))
                    im=Image.composite(tint,im,alpha)
                im.thumbnail((288,512)); tile.paste(im,(0,72))
                d=ImageDraw.Draw(tile); d.text((5,3),f'帧 {i} {r["original_status"]}',font=font,fill='white')
                d.text((5,29),label if mask is not None or label=='原图' else '无已提交轮廓',font=font,fill='white')
                tiles.append(tile)
        columns=4; canvas=Image.new('RGB',(288*columns,590*((len(tiles)+columns-1)//columns)),'#0d131a')
        for n,tile in enumerate(tiles):canvas.paste(tile,((n%columns)*288,(n//columns)*590))
        name='compare_'+'_'.join(map(str,group))+'.png';canvas.save(args.output/name);pngs.append(name)
    interpretation=['二维轮廓突变不能单独证实焦距或镜头变化。','原始IoU没有运动补偿；运动、视角和遮挡变化可导致降低，不能单据此判分割错误。','缺失轮廓保留null；不按面积、IoU或分数自动选择候选。','只读取既有图像和轮廓；未运行SAM、torch、光流、SfM或重建。']
    report=dict(status='SAVED_MASK_CPU_REVIEW_ONLY',argv=[sys.executable,*sys.argv],indices=indices,
                coverage=str(args.coverage.resolve()),frames=rows,interpretation=interpretation,
                hash_counts=dict(unique_inputs=len(ledger),expected_verified=sum(bool(v['expected_sha256']) for v in ledger.values())),
                input_hashes=list(ledger.values()),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),comparison_pngs=pngs)
    (args.output/'MASK_CONTINUITY_REVIEW.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    body='<h1>已有轮廓连续性核查</h1>'+''.join('<p>'+html.escape(x)+'</p>' for x in interpretation)
    body+='<p>完整命令：<code>'+html.escape(repr(report['argv']))+'</code></p><p>核验输入 '+str(report['hash_counts'])+'</p>'
    body+='<table><tr>'+''.join('<th>'+x+'</th>' for x in ['帧','状态','面积','边界框','相邻原始IoU','中心位移px','候选数','选中策略'])+'</tr>'
    for r in rows:
        body+='<tr>'+''.join('<td>'+html.escape(str(r[k]))+'</td>' for k in ['frame','original_status','area','bbox_xyxy_exclusive','raw_adjacent_iou','centroid_displacement_px','candidate_count','selection'])+'</tr>'
    body+='</table>'+''.join(f'<h2>{name}</h2><img src="{name}">' for name in pngs)
    body+='<p><a href="MASK_CONTINUITY_REVIEW.json">完整来源、候选、hash记录</a></p>'
    (args.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>轮廓核查</title><style>body{font:16px sans-serif;background:#101820;color:#eef;margin:24px}td,th{border:1px solid #678;padding:6px}table{border-collapse:collapse}img{max-width:100%}a{color:#8cf}code{overflow-wrap:anywhere}</style>'+body,encoding='utf8')
    print(json.dumps(dict(output=str(args.output.resolve()),hash_counts=report['hash_counts'],frames=len(rows),pngs=pngs),ensure_ascii=False))


if __name__=='__main__':
    main()
