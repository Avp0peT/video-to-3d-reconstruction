"""Bound fixed-camera global/local raw-surface comparison; never acceptance."""
import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
import numpy as np
from PIL import Image, ImageDraw
from integrations.demo_pipeline.shape_support_audit import context, _geometry
from integrations.demo_pipeline.silhouette_review import project, raster
from integrations.demo_pipeline.region_depth_support import region_raster
from integrations.video_masks.boundary_geometry_binding import _bind, _read, _binary, _need
from integrations.video_masks.face_mask_adapter import verify_depth
from integrations.video_masks.face_gap_gate import bracketed_gap_mask, GapPolicy
from integrations.mvs_surface.scene import read_scene
from integrations.openmvs_depth.dmap import read_dmap


def camera_equal(a, b):
    return (all(a[k] == b[k] for k in ('global_id', 'width', 'height'))
            and all(np.allclose(a[k], b[k], rtol=0, atol=1e-9) for k in ('K', 'R', 'C')))


def counts(pred, fg, bg, gap):
    return dict(foreground_pixels=int(fg.sum()), foreground_covered=int((pred & fg).sum()),
                known_background_pixels=int(bg.sum()), known_background_occupied=int((pred & bg).sum()),
                gap_pixels=int(gap.sum()), gap_occupied=int((pred & gap).sum()))


def find_camera_depth(runpath, camera, refs, check=lambda: None):
    """Native filenames need not be scene ordinals; authenticate image metadata."""
    paths=sorted((Path(runpath)/'work').glob('depth*.dmap'))
    _need(len(paths)<=256,'depth file bound exceeded')
    found=[]
    for path in paths:
        check();d=read_dmap(path)
        _bind(path,refs,dict(bytes=d.file_bytes,sha256=d.sha256))
        if Path(d.image_name_utf8.replace('\\','/')).name==Path(camera['name']).name:
            verify_depth(d,camera);found.append((path,d))
    _need(len(found)<=1,'duplicate depth map for target image')
    return found[0] if found else (None,None)


def panel(rgb, baseline, candidate, fg, bg, path, title):
    h,w=fg.shape
    canvas=Image.new('RGB',(4*w,h+48),(20,28,40)); draw=ImageDraw.Draw(canvas)
    arrays=[rgb.copy() for _ in range(4)]
    for arr,pred in zip(arrays[1:3],(baseline,candidate)):
        arr[pred & fg]=(0.5*arr[pred & fg]+np.array([10,80,120])).astype(np.uint8)
        arr[pred & bg]=[230,60,60]
        arr[~pred & fg]=(0.5*arr[~pred & fg]+np.array([120,65,0])).astype(np.uint8)
    arrays[3][candidate & ~baseline & fg]=[30,220,80]
    arrays[3][baseline & ~candidate & fg]=[255,150,20]
    arrays[3][candidate & ~baseline & bg]=[240,40,50]
    arrays[3][baseline & ~candidate & bg]=[30,180,240]
    for j,(arr,label) in enumerate(zip(arrays,('SOURCE','GLOBAL LOW TEXTURE','LOCAL LOW TEXTURE','CHANGE'))):
        canvas.paste(Image.fromarray(arr),(j*w,48));draw.text((j*w+5,5),label,fill='white')
    draw.text((5,27),title+'; projection coverage only / NOT ACCEPTED',fill='white')
    canvas.save(path)


def run(source, baseline_surface, candidate_surface, selection, region_json, output):
    started=time.monotonic(); refs=[]
    source,baseline_surface,candidate_surface,selection,region_json=map(lambda p:Path(p).resolve(strict=True),
        (source,baseline_surface,candidate_surface,selection,region_json))
    out=Path(output).resolve()
    _need(not out.exists() and not any(out.is_relative_to(p) or p.is_relative_to(out)
        for p in (source,baseline_surface,candidate_surface,selection.parent,region_json.parent)), 'fresh disjoint output required')
    out.mkdir(parents=True)
    def check():
        if time.monotonic()-started>300: raise TimeoutError('300s comparison deadline')
    def save(name,value):
        (out/name).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    _bind(__file__,refs)
    _,original_raw,_,_,views=context(source,None,refs);check()
    oldplan=_read(source/'surface/plan.json',refs)
    oldframes={Path(f['image']['name']).name:f for f in oldplan['frames']}
    by_decode={v['decode_index']:v for v in views}
    selected=_read(selection,refs);spec=_read(region_json,refs)
    rows=selected['rows'];names=[r['exported_name'] for r in rows]
    _need(len(rows)==12 and len(set(names))==12 and selected['target_decode_index']==spec['decode_index']==840,
          'frozen 12-view/840 selection required')
    _need(all(r['decode_index'] in by_decode and Path(by_decode[r['decode_index']]['camera']['name']).name==r['exported_name'] for r in rows), 'selection/source identity mismatch')
    old_by_name={Path(v['camera']['name']).name:v for v in views}
    native={}
    for label,runpath in [('baseline',baseline_surface),('candidate',candidate_surface)]:
        result=_read(runpath/'result.json',refs); plan=_read(runpath/'plan.json',refs)
        _need(result.get('status')=='SURFACE_GENERATED_NOT_ACCEPTED' and result.get('exit_code')==0 and not result.get('input_drift'), 'completed native result required')
        _need(plan['profile']=='openmvs-2.4.0-cpu-low-texture-v1','matched low-texture profiles required')
        scene=read_scene(_bind(plan['scene']['path'],refs,plan['scene']))
        _need(np.array_equal(scene['transform'],np.eye(4)), 'unmodified coordinates required')
        cams={Path(c['name']).name:c for c in scene['images']}
        frames={Path(f['image']['name']).name:f for f in plan['frames']}
        _need(len(cams)==len(scene['images'])==len(frames)==len(plan['frames']) and cams.keys()==frames.keys(), 'plan/scene view sets mismatch')
        _need((set(cams)==set(names)) if label=='candidate' else (set(cams)==set(old_by_name)), 'native camera set differs')
        for name,c in cams.items():
            _need(name in old_by_name and camera_equal(c,old_by_name[name]['camera']) and camera_equal(c,frames[name]['image']), 'calibrated camera mismatch '+name)
            frame=frames[name]
            for role,oldrole in [('image_file','image'),('mask','mask'),('validity','valid')]:
                rec=frame[role];_bind(rec['path'],refs,rec)
                _need(all(rec[k]==oldframes[name][role][k] for k in ('sha256','bytes')),
                      'old/new input byte identity differs '+role+' '+name)
                if role=='image_file':
                    _need(hashlib.sha256(Path(old_by_name[name]['image']).read_bytes()).hexdigest()==rec['sha256'], 'RGB bytes differ')
                else:
                    arr=_binary(rec['path'],old_by_name[name]['mask'].shape)!=0
                    _need(np.array_equal(arr,old_by_name[name][oldrole]),'mask/validity domain differs')
        roles={(s['name'],g['role']):g for s in result['stages'] for g in s.get('outputs',{}).get('geometry',[])}
        data={kind:_geometry(roles[key],refs) for kind,key in [('raw',('mesh','pre_cleanup_surface')),('dense',('dense','dense'))]}
        native[label]=dict(root=runpath,plan=plan,cams=cams,data=data)
        check()
    gate=ROOT/'outputs/delegation/FACE_MASK_GATE/20261024_edge03/real01'
    gate_result=_read(gate/'RESULT.json',refs)
    _need(Path(gate_result['source_run']).resolve()==source and gate_result.get('source_unchanged') is True, 'gap source mismatch')
    measured=[]; pictures=[]
    for chosen in rows:
        check();idx=chosen['decode_index'];name=chosen['exported_name'];v=by_decode[idx];cam=v['camera']
        fg=v['mask']&v['valid'];bg=~v['mask']&v['valid'];h,w=fg.shape
        gp=_bind(gate/'gap_per_view'/f"view_{cam['global_id']}_gap.png",refs)
        gap=(_binary(gp,fg.shape)!=0)&v['valid']
        recomputed,_=bracketed_gap_mask(v['mask'],v['valid'],GapPolicy(**gate_result['gap_policy']),check)
        _need(np.array_equal(gap,recomputed),'saved gate gap differs from bound source-mask policy')
        predictions={};stats={}
        for label,info in native.items():
            mesh=info['data']['raw'];pred=raster(project(mesh['vertices'],cam['R'],cam['C'],cam['K']),np.asarray(mesh['faces']),w,h,check=check)
            predictions[label]=pred;stats[label]=counts(pred,fg,bg,gap)
        b,c=predictions['baseline'],predictions['candidate']
        row=dict(decode_index=idx,global_id=cam['global_id'],baseline=stats['baseline'],candidate=stats['candidate'],
            newly_covered_foreground=int((c&~b&fg).sum()),lost_foreground=int((b&~c&fg).sum()),
            newly_occupied_background=int((c&~b&bg).sum()),freed_background=int((b&~c&bg).sum()),
            newly_occupied_gap=int((c&~b&gap).sum()),freed_gap=int((b&~c&gap).sum()))
        if idx==840:
            roi=region_raster(spec,v)
            oldpred=raster(project(original_raw['vertices'],cam['R'],cam['C'],cam['K']),np.asarray(original_raw['faces']),w,h,check=check)
            missing=roi&~oldpred
            _need(int(missing.sum())==11478,'frozen original raw-missing denominator changed')
            roi_stats={}
            for label,info in native.items():
                # Dmap ordinal belongs to this run's scene, never the old scene.
                frame=next(f for f in info['plan']['frames'] if Path(f['image']['name']).name==name)
                newcam=info['cams'][name]
                path,d=find_camera_depth(info['root'],newcam,refs,check)
                dv=None
                if d is not None:
                    dv=np.isfinite(d.depth)&(d.depth>0)
                pred=predictions[label]
                roi_stats[label]=dict(roi_pixels=int(roi.sum()),roi_covered=int((roi&pred).sum()),
                    original_raw_missing_pixels=int(missing.sum()),original_missing_covered=int((missing&pred).sum()),
                    scene_image_index=frame['image']['index'],depth_path=None if path is None else str(path),
                    depth_identity='authenticated embedded image name and K/R/C; not filename ordinal',depth_available=dv is not None,
                    depth_valid_roi_pixels=None if dv is None else int((dv&roi).sum()),
                    depth_valid_original_missing_pixels=None if dv is None else int((dv&missing).sum()))
            row['fixed_posthoc_roi']=dict(definition='Frozen source side-branch ROI; not anatomical truth',metrics=roi_stats)
        if idx in (840,846):
            with Image.open(v['image']) as image:rgb=np.array(image.convert('RGB'))
            filename=f'frame_{idx:06d}_comparison.png';panel(rgb,b,c,fg,bg,out/filename,str(idx));pictures.append(filename)
        measured.append(row);print('COMPARED',idx,flush=True)
    # Each record was captured when first consumed, including optional depth.
    # Freeze the complete consumed set before the final independent recheck.
    save('BINDINGS_PRE.json',refs)
    post=[]
    for ref in refs:check();_bind(ref['path'],post,ref)
    result=dict(status='TEMPORAL_SURFACE_COMPARISON_NOT_ACCEPTED',geometry_accepted=False,mask_consumption='NOT_VERIFIED',
        new_native_executions=0,selection=str(selection),views=measured,
        geometry_counts={label:{kind:dict(vertices=len(mesh['vertices']),faces=len(mesh['faces'])) for kind,mesh in info['data'].items()} for label,info in native.items()},
        source_unchanged=True,elapsed_seconds=time.monotonic()-started,
        limitations=['Opaque 2x triangle-union projection, not physical accuracy or visibility.',
          'Global and local reconstructions have different observation opportunities and are correlated estimates.',
          'A sparser short-window surface is not automatically better or worse quality.',
          'Coverage gains, gap/background occupation and losses are reported separately; no candidate adoption.',
          'ROI was frozen posthoc from old source; 11478 original missing pixels are the unchanged denominator.'])
    save('RESULT.json',result);save('BINDINGS_POST.json',post)
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>Global / local surface: NOT ACCEPTED</h1>'
        '<p>Green: newly covered foreground; orange: lost foreground; red: newly occupied background; cyan: freed background. Coverage is not accuracy.</p>'
        '<a href="RESULT.json">Bound metrics</a>'+''.join('<img style="max-width:100%" src="'+html.escape(p)+'">' for p in pictures),encoding='utf-8')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','baseline-surface','candidate-surface','selection','region-json','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args();run(**vars(a))


if __name__=='__main__':main()
