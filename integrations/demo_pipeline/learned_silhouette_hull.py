"""CPU silhouette diagnostic: no surface, occlusion model, or quality acceptance."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

STATUS = 'OCCLUSION_UNMODELLED'

def record(path):
    p = Path(path).resolve(strict=True)
    return {'path': str(p), 'bytes': p.stat().st_size,
            'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}

def project(xyz, rotation, translation, params, size):
    """COLMAP SIMPLE_RADIAL; floor maps half-pixel COLMAP coordinates to pixels."""
    q = np.asarray(xyz) @ np.asarray(rotation).T + translation
    z = q[:, 2]
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        xy = q[:, :2] / z[:, None]
        r2 = (xy * xy).sum(axis=1)
        scale = 1 + params[3] * r2
        uv = params[0] * scale[:, None] * xy + params[1:3]
    # Restrict the radial map to its orientation-preserving local branch.
    valid = ((z > 0) & np.isfinite(uv).all(axis=1) & (scale > 0)
             & (1 + 3 * params[3] * r2 > 0))
    inside = valid & (uv[:, 0] >= 0) & (uv[:, 0] < size[0]) & (uv[:, 1] >= 0) & (uv[:, 1] < size[1])
    return uv, inside, valid

def support(points, views):
    positive = np.zeros(len(points), dtype=np.uint16)
    observed = np.zeros(len(points), dtype=np.uint16)
    for v in views:
        uv, inside, _ = project(points, v['R'], v['t'], v['params'], v['size'])
        ids = np.flatnonzero(inside)
        ij = np.floor(uv[ids]).astype(np.int64)
        observed[ids] += 1
        positive[ids] += v['mask'][ij[:, 1], ij[:, 0]]
    strict = (observed >= 3) & (positive == observed)
    curves = {str(t): int(((observed >= 3) & (positive >= observed * t)).sum()) for t in (.9, .95, 1.)}
    return strict, positive, observed, curves

def grid_domain(xyz, resolution=64):
    if resolution != 64:
        raise ValueError('frozen diagnostic resolution is 64')
    xyz = np.asarray(xyz, dtype=float)
    lo, hi = xyz.min(axis=0), xyz.max(axis=0)
    extent = (hi - lo) * 1.5
    if not np.isfinite(xyz).all() or (extent <= 0).any():
        raise ValueError('invalid sparse bounds')
    lower = (lo + hi) / 2 - extent / 2
    step = extent / resolution
    axes = [lower[d] + (np.arange(resolution) + .5) * step[d] for d in range(3)]
    points = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 3)
    return points, lower, step

def footprint(points, step, view):
    """Conservative projected corner rectangles, deliberately an upper coverage bound.

    This avoids treating voxel-center sampling gaps as silhouette holes. It is
    not exact silhouette IoU; radial extrema between corners are not certified.
    """
    w, h = view['size']
    out = Image.new('1', (w, h))
    draw = ImageDraw.Draw(out)
    offsets = np.array([[a,b,c] for a in (-.5,.5) for b in (-.5,.5) for c in (-.5,.5)]) * step
    skipped = 0
    for chunk in np.array_split(points, max(1, (len(points) + 2047)//2048)):
        if not len(chunk): continue
        uv, _, valid = project((chunk[:,None,:] + offsets).reshape(-1,3), view['R'], view['t'], view['params'], view['size'])
        uv, valid = uv.reshape(-1,8,2), valid.reshape(-1,8)
        for corners, good in zip(uv, valid):
            if not good.all():
                skipped += 1
                continue
            a = np.floor(corners.min(axis=0)).astype(int)
            b = np.floor(corners.max(axis=0)).astype(int)
            if b[0] < 0 or b[1] < 0 or a[0] >= w or a[1] >= h: continue
            draw.rectangle((max(0,a[0]), max(0,a[1]), min(w-1,b[0]), min(h-1,b[1])), fill=1)
    return np.asarray(out, dtype=bool), skipped

def load(native, masks_path):
    native = Path(native).resolve()
    bindings = []
    def bind(p):
        row = record(p); bindings.append(row); return Path(row['path'])
    def read(p): return json.loads(bind(p).read_text(encoding='utf-8'))
    plan, result = read(native/'PLAN.json'), read(native/'RESULT.json')
    model = next(m for m in result['models'] if m['model_id'] == 3)
    if not model['binary_text_roundtrip_verified']: raise ValueError('unverified text model')
    text = Path(model['text_path'])
    for p in sorted(Path(model['binary_path']).glob('*.bin')): bind(p)
    camera_lines = [s for s in bind(text/'cameras.txt').read_text().splitlines() if s and not s.startswith('#')]
    cameras = {}
    for line in camera_lines:
        a = line.split()
        if a[1] != 'SIMPLE_RADIAL': raise ValueError('SIMPLE_RADIAL required')
        cameras[int(a[0])] = ((int(a[2]),int(a[3])),np.array(a[4:],float))
    lines = [s for s in bind(text/'images.txt').read_text().splitlines() if not s.startswith('#')]
    rows = [s.split() for s in lines[::2]]
    points = np.array([[float(v) for v in s.split()[1:4]] for s in bind(text/'points3D.txt').read_text().splitlines() if s and not s.startswith('#')])
    masks = read(masks_path)['frames']
    declared = [r for r in plan['source_bindings'] if Path(r['path']).resolve() == Path(masks_path).resolve()]
    if len(declared) != 1 or any(bindings[-1][k] != declared[0][k] for k in ('bytes','sha256')):
        raise ValueError('MASKS differs from native PLAN binding')
    views = []
    frame_by_name = {f['image_name']: f for f in plan['frames']}
    for a in rows:
        f = frame_by_name[a[9]]
        matches = [m for m in masks if m['decode_index'] == f['decode_index']]
        if len(matches) != 1 or matches[0]['status'] != 'OK': raise ValueError('mask identity/status')
        m = matches[0]
        mp = bind(m['mask_path'])
        if bindings[-1]['sha256'] != m['mask_sha256']: raise ValueError('mask hash drift')
        ip = bind(m['raw_path'])
        if bindings[-1]['sha256'] != m['raw_sha256']: raise ValueError('raw image hash drift')
        size, params = cameras[int(a[8])]
        with Image.open(mp) as im:
            if im.mode != 'L': raise ValueError('mask must be native L mode')
            raster = np.array(im)
            if not np.isin(raster, [0, 255]).all(): raise ValueError('mask must contain only 0/255')
            if not (raster == 255).any(): raise ValueError('empty foreground mask')
            mask = raster == 255
        with Image.open(ip) as im:
            if im.size != size: raise ValueError('image size')
        if mask.shape != (size[1],size[0]): raise ValueError('mask size')
        w,x,y,z = np.array(a[1:5],float)
        R = np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)], [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)], [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
        views.append({'name':a[9], 'decode_index': f['decode_index'], 'R':R,'t':np.array(a[5:8],float),'params':params,'size':size,'mask':mask,'image':ip})
    if len(views) != model['registered_views']: raise ValueError('view coverage')
    return points, views, bindings

def run(native, masks, output):
    implementation_binding = record(__file__)
    xyz, views, bindings = load(native,masks)
    bindings.insert(0, implementation_binding)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    points, lower, step = grid_domain(xyz)
    strict, positive, observed, curves = support(points,views)
    occupied = points[strict]
    grid = strict.reshape(64,64,64)
    np.savez_compressed(output/'grid.npz', strict=grid, positive=positive.reshape(grid.shape), observed=observed.reshape(grid.shape), lower=lower, step=step)
    with (output/'voxels.ply').open('w',encoding='ascii') as f:
        f.write(f'ply\nformat ascii 1.0\ncomment voxel centers only; OCCLUSION_UNMODELLED; no faces\nelement vertex {len(occupied)}\nproperty float x\nproperty float y\nproperty float z\nend_header\n')
        np.savetxt(f,occupied,fmt='%.9g')
    metrics=[]
    for v in views:
        projected, skipped = footprint(occupied,step,v)
        Image.fromarray(projected.astype('uint8')*255).save(output/(v['name']+'.occupancy.png'))
        with Image.open(v['image']) as im: rgb=np.array(im.convert('RGB'))
        rgb[projected]=((rgb[projected].astype(float)+[0,255,255])/2).astype('uint8')
        Image.fromarray(rgb).save(output/(v['name']+'.overlay.png'))
        metrics.append({'name':v['name'],'decode_index':v['decode_index'],'mask_pixels':int(v['mask'].sum()),'projected_pixels':int(projected.sum()), 'covered_mask_pixels':int((projected & v['mask']).sum()),'mask_coverage':float((projected & v['mask']).sum()/max(1,v['mask'].sum())), 'outside_mask_pixels':int((projected & ~v['mask']).sum()),'skipped_invalid_corner_voxels':skipped})
    drift = [r['path'] for r in bindings if record(r['path']) != r]
    result={'status':STATUS,'acceptance':False,'surface_generated':False,'binary_text_consistency':'Inherited native RESULT binary_text_roundtrip_verified declaration; binary/text geometry equivalence is not rechecked by this diagnostic.','resolution':64,'selected_rule':'all observed views foreground and at least three positive views; outside-image unknown', 'strict_voxels':int(strict.sum()),'support_count_curves':curves,'domain_lower':lower.tolist(),'domain_upper':(lower+64*step).tolist(),'domain_semantics':'sparse min/max centered extent times 1.5; diagnostic domain, not whole-object bounds','boundary_occupancy':{f'{axis}_{side}':int(np.take(grid,index,axis=axis).sum()) for axis in range(3) for side,index in [('low',0),('high',-1)]},'projection_raster':'projected 8-corner axis-aligned rectangles; conservative sampling diagnostic, no certified radial-extrema bound; not exact silhouette IoU','limitations':['Masks assumed amodal without occlusion; assumption unverified and false where hand occludes onion.','Estimated poses/intrinsics; no external calibration or physical scale.','No recovery of silhouette-invisible concavities.','No boundary caps or mesh faces generated.'],'views':metrics,'input_bindings':bindings,'input_drift':drift}
    (output/'RESULT.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    if drift: raise ValueError('input drift')
    return result

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--native',required=True);p.add_argument('--masks',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();r=run(a.native,a.masks,a.output)
    print(json.dumps({'status':r['status'],'strict_voxels':r['strict_voxels'],'surface_generated':False}))
