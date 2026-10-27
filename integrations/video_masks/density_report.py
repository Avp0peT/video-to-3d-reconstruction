"""Mask-local sampling and observed sparse support, never dense visibility."""
from __future__ import annotations
from collections import Counter
import html
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from integrations.demo_pipeline.balanced_point_run import binding, check_bindings
from .point_density import DensityPolicy, assess_density
from .density_sampling import default_policy, seed_with_density
from . import region_tracks


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def observed_support(model, image, max_error=2.0):
    """Use actual reciprocal observations, never arbitrary point projections."""
    accepted, rejected = [], Counter()
    pose = image.cam_from_world()
    camera = model.cameras[image.camera_id]
    for row, observation in enumerate(image.points2D):
        if not observation.has_point3D():
            continue
        pid = int(observation.point3D_id)
        point = model.points3D[pid]
        elements = list(point.track.elements)
        if (len({int(e.image_id) for e in elements}) < 2 or
                sum(int(e.image_id) == int(image.image_id) and int(e.point2D_idx) == row for e in elements) != 1):
            rejected['track_support'] += 1; continue
        local = np.asarray(pose * point.xyz)
        if not np.isfinite(local).all() or local[2] <= 0:
            rejected['depth'] += 1; continue
        projection = camera.img_from_cam(local)
        if projection is None:
            rejected['reprojection'] += 1; continue
        projected = np.asarray(projection, dtype=float)
        xy = np.asarray(observation.xy, dtype=float)
        if projected.shape != (2,) or xy.shape != (2,) or not np.isfinite(projected).all() or not np.isfinite(xy).all() or np.linalg.norm(projected-xy) > max_error:
            rejected['reprojection'] += 1; continue
        accepted.append(dict(point3D_id=pid, point2D_idx=row, xy=(xy-.5).tolist()))
    return accepted, dict(rejected)


def render_panel(image, mask, points, density, label):
    canvas = image.convert('RGB')
    tint = np.zeros((*mask.shape, 4), np.uint8)
    for cell in density['cells']:
        x0,y0,x1,y1=cell['bounds']
        color = (232, 69, 56, 68) if cell['deficit'] else (31, 164, 88, 55)
        patch=tint[y0:y1,x0:x1];patch[mask[y0:y1,x0:x1] != 0] = color
    canvas=Image.alpha_composite(canvas.convert('RGBA'),Image.fromarray(tint)).convert('RGB')
    draw=ImageDraw.Draw(canvas)
    for cell in density['cells']:
        x0,y0,x1,y1=cell['bounds']
        draw.rectangle((x0,y0,x1-1,y1-1),outline=(255,205,30),width=1)
        draw.text((x0+1,y0+1),f"{cell['point_count']}/{cell['target_points']}",fill='white',stroke_width=1,stroke_fill='black')
    for x,y in points:
        if 0<=x<canvas.width and 0<=y<canvas.height:
            draw.ellipse((x-2,y-2,x+2,y+2),fill=(0,245,255))
    canvas.thumbnail((400,710))
    out=Image.new('RGB',(400,760),(24,30,40));out.paste(canvas,(0,45))
    draw=ImageDraw.Draw(out);draw.text((10,8),label,fill='white')
    draw.text((10,25),f"area meeting target: {density['met_area_fraction']:.1%}",fill='white')
    return out


def audit_export(export_root, output):
    import pycolmap
    export_root=Path(export_root).resolve(strict=True);out=Path(output).resolve()
    if out.exists() or out.is_relative_to(export_root) or export_root.is_relative_to(out):
        raise ValueError('fresh disjoint density output required')
    refs=[]
    def bind(path, expected=None):
        record=binding(Path(path).resolve(strict=True))
        if expected is not None and record['sha256'] != expected:
            raise ValueError('density input hash mismatch: '+str(path))
        refs.append(record);return Path(record['path'])
    source=bind(export_root/'EXPORT.json')
    doc=json.loads(source.read_text(encoding='utf-8'))
    if doc.get('status')!='EXPORTED_NOT_GEOMETRY_ACCEPTED' or not 1<=len(doc['frames'])<=96:
        raise ValueError('bounded regional export required')
    # Every model/image is checked against the exporter receipt before reading.
    declarations={str(Path(r['path']).resolve()):r for r in doc['outputs']}
    binary=Path(doc['binary_path']).resolve(strict=True)
    if not binary.is_relative_to(export_root):
        raise ValueError('exported model outside root')
    for path in sorted(binary.iterdir()):
        if path.is_file():bind(path,declarations[str(path.resolve())]['sha256'])
    model=pycolmap.Reconstruction(str(binary))
    frames=doc['frames']
    ids=[f['export_image_id'] for f in frames]
    if len(set(ids))!=len(ids) or set(ids)!=set(model.reg_image_ids()):
        raise ValueError('export frame identity set differs')
    for path in (Path(__file__),Path(region_tracks.__file__),Path(__file__).with_name('point_density.py'),
                 Path(__file__).with_name('density_sampling.py')):bind(path)
    cv=region_tracks.load_cv(refs)
    if cv is None:raise RuntimeError('bound CPU OpenCV unavailable')
    cv.setNumThreads(2);cv.ocl.setUseOpenCL(False);cv.setRNGSeed(0)
    out.mkdir(parents=True);(out/'images').mkdir()
    rows=[]
    policy=DensityPolicy()
    for frame in frames:
        iid=frame['export_image_id'];im=model.images[iid]
        if im.name != frame['exported_name']:raise ValueError('image name differs')
        camera=model.cameras[im.camera_id]
        image_path=bind(frame['exported_png_path'],frame['exported_png_sha256'])
        mask_path=bind(frame['source_mask_path'],frame['source_mask_sha256'])
        with Image.open(io.BytesIO(image_path.read_bytes())) as opened:image=opened.convert('RGB')
        with Image.open(io.BytesIO(mask_path.read_bytes())) as opened:
            if opened.mode!='L':raise ValueError('L mask required')
            mask=np.asarray(opened).copy()
        if image.size!=(camera.width,camera.height) or mask.shape!=(camera.height,camera.width) or not np.isin(mask,[0,255]).all():
            raise ValueError('mask/image/camera domain differs')
        gray=np.asarray(image.convert('L')).copy()
        old=region_tracks.seed(gray,mask,[],cv)
        old_density=assess_density(mask,old,policy)
        new,new_density=seed_with_density(gray,mask,[],cv)
        observed,rejected=observed_support(model,im)
        xy=[p['xy'] for p in observed]
        support=assess_density(mask,xy,policy)
        support['evidence_kind']='RECIPROCAL_NATIVE_SPARSE_OBSERVATIONS_POSITIVE_DEPTH_ERROR_LE_2PX'
        name=f'{frame["input_index"]:06d}.png'
        panels=Image.new('RGB',(1200,760))
        for n,(pts,d,label) in enumerate(((old,old_density,'Legacy measured candidates'),
                                         (new,new_density,'Density-first measured candidates'),
                                         (xy,support,'Existing triangulated observations'))):
            panels.paste(render_panel(image,mask,pts,d,label),(400*n,0))
        panels.save(out/'images'/name)
        rows.append(dict(input_index=frame['input_index'],decode_index=frame['decode_index'],image_id=iid,
            old_candidates=old_density,new_candidates=new_density,triangulated_support=support,
            old_xy=np.asarray(old).reshape(-1,2).tolist(),new_xy=np.asarray(new).reshape(-1,2).tolist(),
            observations=observed,rejected_observations=rejected,overlay='images/'+name))
        print(f'DENSITY_FRAME {len(rows)}/{len(frames)}',flush=True)
    drift=check_bindings(refs)
    if drift:raise ValueError('density inputs changed during reading')
    met=all(r['triangulated_support']['status']=='DENSITY_TARGET_MET' for r in rows)
    averages={k:float(np.mean([r[k]['met_area_fraction'] for r in rows])) for k in ('old_candidates','new_candidates','triangulated_support')}
    result=dict(schema='object-point-density-report-v1',status='DENSITY_TARGET_MET' if met else 'BELOW_DENSITY_TARGET',
        density_policy=default_policy(),source_export=str(source),frames=rows,frame_count=len(rows),
        mean_frame_met_area_fraction=averages,source_bindings=refs,source_unchanged=True,
        geometry_accepted=False,dense_visibility_coverage='NOT_EVALUATED',new_matching_executions=0,
        new_triangulation_executions=0,new_mvs_executions=0,quality_status='NOT_EVALUATED',
        limitations=['Candidate comparison is independent per frame; no new identities or geometry certified.',
                     'Sparse mask-relative coverage is not dense surface completeness or unseen surface coverage.',
                     'Default quotas are engineering starting values, not accuracy or physical density standards.'],
        followup_regions=[dict(input_index=r['input_index'],bounds=c['bounds'],missing_points=c['deficit'],
            next_action='MASK_LOCAL_REFILL_THEN_UNCHANGED_MATCHING_AND_GEOMETRY_VALIDATION',status='NOT_EXECUTED')
            for r in rows for c in r['triangulated_support']['cells'] if c['deficit']])
    save(out/'RESULT.json',result)
    blocks=''.join('<article><h2>Frame '+str(r['input_index'])+'</h2><img loading="lazy" src="'+r['overlay']+'"></article>' for r in rows)
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>物体取点密度</title>'
        '<style>body{font:17px system-ui;background:#17202e;color:#eee;max-width:1250px;margin:30px auto}img{width:100%}a{color:#88caff}</style>'
        '<h1>物体局部取点密度</h1><p>'+html.escape(result['status'])+'</p>'
        '<p>左：旧候选；中：密度优先的新候选；右：原模型三角化观测。网格随掩膜包围框定位；红色未达配额，绿色达到配额。'
        '格内标注为点数/目标数。背景不计入；小边界单元仍计入。</p>'
        '<p>新候选尚未匹配或建模。右图仅统计有实际观测、正深度、回投影≤2px的原稀疏点；不是稠密点覆盖或形状验收。</p>'
        '<p>8×8物体网格，完整单元目标4个唯一像素点，边界按前景面积缩放，最少1个。阈值是工程起点。</p>'
        '<p><a href="RESULT.json">逐帧数据、参数与来源</a></p>'+blocks,encoding='utf-8')
    return result
