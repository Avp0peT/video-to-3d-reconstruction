"""Book-inspired radius covers on bound observations; no invented photo points."""
from pathlib import Path
import html
import math
import numpy as np
from PIL import Image, ImageDraw
from kozlov_lab.stereo.point_coverage import greedy_r_cover, raster_radius_coverage
from .parallel_photo_audit import binding, write_json

PROFILE=dict(schema='object-point-radius-cover-v1',radii_at_1024=[16,32],max_centers=4096,
    radius_rule='base_radius * max(image_width,image_height) / 1024',
    point_domain='Unmodified native image observations; mask lookup at floor(x),floor(y)',
    raster_domain='Foreground pixel centres (column+0.5,row+0.5)',
    source_definition='Book print118-120: supplied finite point set A covered by a subset A_r',
    method='deterministic farthest-point greedy; valid radius cover, not minimum ostov',
    quality_threshold=None,minimum_view_counts=[2,3],coordinates_modified=False)
LABELS={'features':'本帧目标内全部特征','registered':'至少2个注册视图轨迹','multiview3':'至少3个注册视图轨迹'}


def populations(image,points,mask,image_id):
    height,width=mask.shape;rows=[];outside=0;ambiguous=0
    for index,(x,y,pid) in enumerate(image['observations']):
        if not (math.isfinite(x) and math.isfinite(y) and 0<=x<width and 0<=y<height):
            raise ValueError('native observation outside canonical image domain')
        if mask[math.floor(y),math.floor(x)]!=255:
            outside+=1;continue
        p=points.get(pid)
        if p is not None and p['track'].get(image_id)!=index:
            raise ValueError('native observation/track identity differs')
        is_ambiguous=bool(p and p.get('ambiguous_observations'))
        ambiguous+=int(is_ambiguous)
        views=sorted(p['track']) if p and not is_ambiguous else []
        rows.append(dict(feature_row=index,xy=[float(x),float(y)],point3D_id=pid if p else None,
            registered_view_ids=views,registered_view_count=len(views),ambiguous=is_ambiguous))
    return {'features':rows,'registered':[r for r in rows if r['registered_view_count']>=2],
            'multiview3':[r for r in rows if r['registered_view_count']>=3]},dict(
        native_observations=len(image['observations']),outside_mask=outside,foreground_observations=len(rows),
        ambiguous_registered_rows=ambiguous,distinct_coordinate_count=len({tuple(r['xy']) for r in rows}))


def write_map(path,raster):
    array=raster.pop('coverage_map')
    Image.fromarray(array).save(path)
    with Image.open(path) as im:
        if im.mode!='L' or not np.array_equal(np.asarray(im),array):
            raise ValueError('coverage map pixel readback differs')
    raster['map']=dict(**binding(path),labels={'0':'background','1':'covered target','2':'uncovered target'},readback=True)
    return array


def overlay(gray,coverage,centres):
    rgb=np.stack([gray]*3,axis=-1).astype(float)
    rgb[coverage==0]*=.45
    for value,colour in [(1,[25,185,125]),(2,[240,75,65])]:
        selected=coverage==value;rgb[selected]=.45*rgb[selected]+.55*np.array(colour)
    result=Image.fromarray(np.rint(rgb).astype(np.uint8));draw=ImageDraw.Draw(result)
    for x,y in centres:draw.ellipse((x-2,y-2,x+2,y+2),fill='#ffe16a')
    return result


def build_point_coverage(source,output):
    output=Path(output);output.mkdir(exist_ok=False)
    result=dict(schema='object-point-coverage-review-v1',status='COMPLETE',profile=PROFILE,
        frames=[],geometry_changed=False,quality_status='NOT_EVALUATED',
        warnings=['Covering existing features does not cover all target pixels or prove correct matches.',
            'View counts use distinct native registered image IDs; they are not independent truth.',
            'Foreground membership is checked in this frame; counts do not certify all other views are in their masks.',
            'These radius values are display/selection profiles, not geometric acceptance thresholds.',
            'Mask errors and invisible surfaces remain outside this guarantee.'])
    write_json(output/'PROFILE.json',PROFILE)
    for iid in sorted(source['images']):
        frame=dict(image_id=iid,input_index=source['frames'][iid]['input_index'],status='NOT_EVALUATED',radii=[])
        result['frames'].append(frame)
        if iid not in source['masks']:
            frame['reason']='No accepted bound mask';continue
        try:
            mask=source['masks'][iid];gray=source['gray'][iid]
            sets,counts=populations(source['images'][iid],source['points'],mask,iid)
            frame.update(counts=counts,source_image=source['frames'][iid]['binding'],source_mask=source['mask_info'][iid]['binding'])
            frame['observation_sets']=sets
            for base in PROFILE['radii_at_1024']:
                radius=base*max(mask.shape)/1024
                scale=dict(base_radius=base,radius=radius,sets={});frame['radii'].append(scale)
                pictures=[]
                for key,rows in sets.items():
                    xy=[r['xy'] for r in rows]
                    cover=greedy_r_cover(xy,radius,PROFILE['max_centers'])
                    selected=[rows[i] for i in cover['selected_indices']]
                    cover['selected_feature_rows']=[r['feature_row'] for r in selected]
                    cover['nearest_selected_feature_rows']=[rows[i]['feature_row'] if i>=0 else None for i in cover['nearest_selected_indices']]
                    full=raster_radius_coverage(mask,xy,radius)
                    sampled=raster_radius_coverage(mask,[r['xy'] for r in selected],radius)
                    stem=f'image_{iid}_r{base}_{key}'
                    fullmap=write_map(output/(stem+'_all.png'),full)
                    write_map(output/(stem+'_representatives.png'),sampled)
                    scale['sets'][key]=dict(point_count=len(rows),cover=cover,
                        all_points_mask_coverage=full,representatives_mask_coverage=sampled)
                    picture=overlay(gray,fullmap,[r['xy'] for r in selected])
                    h=512;w=max(1,round(picture.width*h/picture.height))
                    pictures.append((key,picture.resize((w,h),Image.Resampling.BILINEAR)))
                board=Image.new('RGB',(sum(p.width for _,p in pictures),550),'#eef1f4');draw=ImageDraw.Draw(board);x=0
                for key,picture in pictures:
                    stats=scale['sets'][key]
                    ratio=stats['all_points_mask_coverage']['coverage_ratio']
                    text=f"{key}: {stats['point_count']} pts, cover {ratio:.1%}" if ratio is not None else key+': no target'
                    draw.text((x+5,6),text,fill='#243246');board.paste(picture,(x,35));x+=picture.width
                image_path=output/f'image_{iid}_r{base}_overview.png';board.save(image_path)
                scale['overview']=binding(image_path)
            frame['status']='EVALUATED' if np.any(mask) else 'NO_TARGET'
        except (ValueError,OSError,KeyError,TypeError) as exc:
            frame.update(status='NOT_EVALUATED',reason=f'{type(exc).__name__}: {exc}')
    result['evaluated_frames']=sum(f['status']=='EVALUATED' for f in result['frames'])
    if result['evaluated_frames']!=len(result['frames']):result['status']='PARTIAL'
    write_json(output/'RESULT.json',result)
    page='<!doctype html><meta charset="utf-8"><title>目标区域点覆盖</title><style>body{font:16px/1.6 system-ui;max-width:1100px;margin:30px auto;padding:20px;color:#243246}table{border-collapse:collapse;width:100%}td,th{padding:8px;border-bottom:1px solid #ddd}img{max-width:100%}aside{background:#fff0cb;padding:15px}</style>'
    page+='<h1>目标区域点覆盖与代表点</h1><aside>绿色：在原始观测点的半径范围内；红色：目标mask内缺少附近观测点；黄色：从已有点中选出的代表点。红色不是确认错误或隐藏表面的真值。</aside>'
    page+='<p>书中第118–120页的覆盖条件用于已有特征点集；这里采用有界贪心近似，不声称最小骨架。16/32像素按图像长边归一化，均报告，不挑最好档。比例衡量当前mask的像素覆盖，不是三维准确率。</p>'
    for frame in result['frames']:
        page+=f'<h2>帧 {frame["input_index"]} · {frame["status"]}</h2>'
        if frame['status']!='EVALUATED':page+='<p>'+html.escape(frame.get('reason','No foreground target'))+'</p>'
        page+='<table><tr><th>半径</th><th>观测类型</th><th>点数 → 代表点</th><th>已有点集r覆盖</th><th>mask覆盖：全部点 / 代表点</th></tr>'
        for scale in frame['radii']:
            for key,data in scale['sets'].items():
                pct=lambda v:'未知' if v is None else f'{v:.1%}'
                c=data['cover'];a=data['all_points_mask_coverage'];b=data['representatives_mask_coverage']
                page+=f'<tr><td>{scale["radius"]:g}px</td><td>{LABELS[key]}</td><td>{data["point_count"]} → {len(c["selected_indices"])}</td><td>{c["status"]}</td><td>{pct(a["coverage_ratio"])} / {pct(b["coverage_ratio"])}</td></tr>'
        page+='</table>'
        for scale in frame['radii']:
            if scale.get('overview'):
                page+=f'<h3>半径 {scale["radius"]:g}px</h3><img src="{Path(scale["overview"]["path"]).name}" alt="三组点覆盖对照">'
    page+='<p>三组点均要求本帧位置在目标mask内；视图数按原模型的不同注册视图计数，不保证其他视图的观测均在各自mask内。未添加任何新照片观测或改变已有对应；轨迹身份仍来自同一模型。覆盖缺口提示后续采集或特征工作方向，不能保证补点后重建必然成功。</p><a href="RESULT.json">全部身份、见证与覆盖图</a> · <a href="PROFILE.json">固定参数</a>'
    (output/'index.html').write_text(page,encoding='utf-8')
    return dict(status=result['status'],evaluated_frames=result['evaluated_frames'],frame_count=len(result['frames']),
        record=binding(output/'RESULT.json'),report=binding(output/'index.html'),geometry_changed=False)
