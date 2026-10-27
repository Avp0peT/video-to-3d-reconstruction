"""Read-only audit of region IDs split among native 3D points; no identity verdict."""
from pathlib import Path
from collections import defaultdict
import argparse
import hashlib
import html
import json
import os
import sqlite3
import struct


def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as s:
        for b in iter(lambda:s.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def parse_points(path):
    """Read exported COLMAP text point tracks without loading a solver."""
    points=[]
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if not line.strip() or line.startswith('#'):continue
        p=line.split()
        if len(p)<8 or (len(p)-8)%2:raise ValueError('invalid point track text')
        points.append(dict(point3D_id=int(p[0]),xyz=list(map(float,p[1:4])),error=float(p[7]),observations=[(int(p[i]),int(p[i+1])) for i in range(8,len(p),2)]))
    return points


def group_splits(points, frames):
    """Bind native image/feature rows explicitly; never infer physical identity."""
    lookup={f['image_id']:(f,{int(row):int(t) for t,row in f['row_by_track'].items()}) for f in frames}
    groups=defaultdict(lambda:defaultdict(list))
    point_by_id={p['point3D_id']:p for p in points}
    for p in points:
        for iid,row in p['observations']:
            if iid not in lookup:raise ValueError('native image missing from explicit plan')
            f,mapping=lookup[iid]
            if row not in mapping:continue
            tid=mapping[row]
            groups[tid][p['point3D_id']].append(dict(input_index=f['input_index'],decode_index=f['decode_index'],image_id=iid,feature_row=row,track_id=tid,xy=f['points'][str(tid)]))
    return [dict(region_id=t,identity_status='NOT_VERIFIED',split_is_automatic_mismatch_verdict=False,point_groups=[dict(point3D_id=pid,xyz=point_by_id[pid]['xyz'],native_error=point_by_id[pid]['error'],observations=sorted(obs,key=lambda r:r['input_index'])) for pid,obs in sorted(g.items())]) for t,g in sorted(groups.items()) if len(g)>1]


def render(source_run, output, geometry='geometry02', model_id=0):
    from PIL import Image,ImageDraw,ImageFont
    source=Path(source_run).resolve();out=Path(output).resolve()
    if out.exists():raise FileExistsError('fresh audit output required')
    if out==source or source.is_relative_to(out) or out.is_relative_to(source):raise ValueError('audit must be outside source run')
    native=source/geometry;plan=read(native/'PLAN.json');result=read(native/'RESULT.json');tracking=read(source/'tracking/REGION_TRACKS.json');comparison=read(source/'COMPARISON.json')
    if tracking.get('schema')!='glove-region-tracks-v1':raise ValueError('tracking schema mismatch')
    model=next(m for m in result['models'] if m['model_id']==model_id)
    text=Path(model['text_path'])/'points3D.txt'
    files=[native/'PLAN.json',native/'RESULT.json',native/'database.db',source/'tracking/REGION_TRACKS.json',source/'COMPARISON.json',text]
    bindings={str(p):sha(p) for p in files}
    frames={f['input_index']:f for f in tracking['frames']}
    pf={f['image_id']:f for f in plan['frames']}
    points=parse_points(text);groups=group_splits(points,plan['frames'])
    db=sqlite3.connect((native/'database.db').as_uri()+'?mode=ro',uri=True)
    try:
        db.execute('PRAGMA query_only=ON');images=dict(db.execute('SELECT image_id,name FROM images'))
        for f in plan['frames']:
            if images.get(f['image_id'])!=f['name']:raise ValueError('native image name binding mismatch')
        blobs={iid:(n,c,b) for iid,n,c,b in db.execute('SELECT image_id,rows,cols,data FROM keypoints')}
        for g in groups:
            for pg in g['point_groups']:
                for ob in pg['observations']:
                    n,c,b=blobs[ob['image_id']];r=ob['feature_row']
                    if c not in (2,4,6) or len(b)!=n*c*4 or r>=n:raise ValueError('invalid native keypoint row')
                    native_xy=struct.unpack_from('<ff',b,r*c*4)
                    if max(abs(native_xy[k]-(ob['xy'][k]+.5)) for k in (0,1))>1e-3:raise ValueError('native row/region coordinate mismatch')
                    endpoint=frames[ob['input_index']]
                    tracked={o['track_id']:o['xy'] for o in endpoint['observations']}
                    if tracked.get(g['region_id'])!=ob['xy']:raise ValueError('tracking/plan region binding mismatch')
                    ob['image_path']=endpoint['image_path'];ob['image_sha256']=endpoint['image_sha256']
    finally:db.close()
    comparison_model=next((m for m in comparison.get('models',[]) if m.get('label')=='sift_plus_regions' and m.get('model_id')==model_id),{})
    expected=comparison_model.get('region_ids_split_into_multiple_3d_points')
    if expected is not None and expected!=len(groups):raise ValueError('comparison split count differs')
    fontfile=Path('C:/Windows/Fonts/msyh.ttc');font=ImageFont.truetype(str(fontfile),15) if fontfile.exists() else ImageFont.load_default()
    cache={};out.mkdir(parents=True);bodies=[]
    for g in groups:
        tiles=[]
        for pg in g['point_groups']:
            for ob in pg['observations']:
                p=Path(ob['image_path'])
                if str(p) not in cache:
                    if sha(p)!=ob['image_sha256']:raise ValueError('source image drift')
                    bindings[str(p)]=ob['image_sha256'];cache[str(p)]=Image.open(p).convert('RGB')
                im=cache[str(p)];x,y=ob['xy'];tile=Image.new('RGB',(240,218),(20,25,35));d=ImageDraw.Draw(tile)
                d.text((4,2),f'3D {pg["point3D_id"]} / 端点 {ob["input_index"]}',font=font,fill='white')
                d.text((4,23),f'原帧 {ob["decode_index"]} / row {ob["feature_row"]}',font=font,fill='white')
                crop=im.crop((round(x)-20,round(y)-20,round(x)+21,round(y)+21)).resize((164,164));cd=ImageDraw.Draw(crop);cd.line((74,82,90,82),fill='red');cd.line((82,74,82,90),fill='red');tile.paste(crop,(0,52))
                thumb=im.resize((72,128));td=ImageDraw.Draw(thumb);tx=x*72/im.width;ty=y*128/im.height;td.ellipse((tx-3,ty-3,tx+3,ty+3),outline='red',width=2);tile.paste(thumb,(166,52));tiles.append(tile)
        cols=4;rows=(len(tiles)+cols-1)//cols;sheet=Image.new('RGB',(cols*240,rows*218),(12,16,24))
        for j,tile in enumerate(tiles):sheet.paste(tile,((j%cols)*240,(j//cols)*218))
        name=f'region_{g["region_id"]:06d}.jpg';sheet.save(out/name,quality=91);g['contact_sheet']=name
        summary='；'.join(f'3D {p["point3D_id"]}: {[o["input_index"] for o in p["observations"]]}' for p in g['point_groups'])
        bodies.append(f'<details id="region{g["region_id"]}"><summary>区域ID {g["region_id"]} → {len(g["point_groups"])}个3D点</summary><p>{html.escape(summary)}</p><a href="{name}"><img loading="lazy" src="{name}"></a></details>')
    if any(sha(p)!=h for p,h in bindings.items()):raise ValueError('source changed during audit')
    doc=dict(schema='glove-region-identity-audit-v1',status='SPLIT_GROUPS_VISUALIZED_NOT_IDENTITY_VERIFIED',source_run=str(source),geometry=geometry,model_id=model_id,group_count=len(groups),comparison_expected_groups=expected,groups=groups,identity_verified=False,geometry_accepted=False,automatic_mismatch_labels=0,source_unchanged=True,source_bindings=bindings,crop_policy='41x41 raw RGB pixels enlarged to 164x164; outside-image crop padding is black; all split observations included')
    (out/'IDENTITY_AUDIT.json').write_text(json.dumps(doc,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>手套轨迹分裂身份诊断</title><style>body{background:#101722;color:#eee;font:16px system-ui;max-width:1100px;margin:30px auto}details{border:1px solid #567;padding:10px;margin:12px}img{max-width:100%}a{color:#9cf}</style><h1>轨迹ID与多个3D点：局部纹理核查</h1><p>状态：NOT_IDENTITY_VERIFIED。分裂不自动代表误匹配，也不证明多个点属于同一物理表面。可能涉及不连通观测、遮挡、形变或错误对应，需结合每组原图位置及纹理审核。各3D点观测分组保持独立，未修改轨迹或模型。</p><p>模型 '+str(model_id)+'，共 '+str(len(groups))+' 组。每格左为41×41原始纹理放大，右为原图位置；每个分裂观测均展示。<a href="IDENTITY_AUDIT.json">完整机器账本与来源SHA</a></p>'+''.join(bodies),encoding='utf-8')
    return doc


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--source-run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--geometry',default='geometry02');p.add_argument('--model-id',type=int,default=0);a=p.parse_args()
    r=render(a.source_run,a.output,a.geometry,a.model_id);print(json.dumps(dict(status=r['status'],group_count=r['group_count'],output=str(a.output)),ensure_ascii=False))


if __name__=='__main__':main()
