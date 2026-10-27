"""Warp real learned-experimental masks; never launch SAM, MVS, or claim acceptance."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from integrations.demo_pipeline.learned_silhouette_hull import record
from integrations.mvs_masks.mask_geometry import warp_binary_mask, ALGORITHM

SCHEMA='learned-experimental-mvs-masks-v1'

def need(ok,message):
    if not ok: raise ValueError(message)

def join_identity(images, native_frames, masks):
    """Join explicit native name/id and decode identity, never list order."""
    need(len(images)==54 and len({i['name'] for i in images})==54,'54 unique registered names required')
    result={}
    for image in images:
        frames=[f for f in native_frames if f['database_image_id']==image['image_id'] and f['image_name']==image['name']]
        need(len(frames)==1,'native image identity ambiguous/missing')
        f=frames[0]
        need(f['decode_index']==image['decode_index'] and f['input_index']==image['input_index'],'export/native frame identity differs')
        selected=[m for m in masks if m['decode_index']==f['decode_index']]
        need(len(selected)==1 and selected[0]['status']=='OK','registered mask missing/ambiguous/not OK')
        need(selected[0].get('current_observed') is True,'mask not currently observed')
        result[image['name']]=selected[0]
    need(len({m['decode_index'] for m in result.values()})==54,'duplicate decoded frame identity')
    return result

def check_pose(source,target):
    a,b=source.cam_from_world(),target.cam_from_world()
    for left,right in [(a.rotation.matrix(),b.rotation.matrix()),(a.translation,b.translation),(source.projection_center(),target.projection_center())]:
        need(np.allclose(left,right,rtol=1e-8,atol=1e-8),'source/undistorted pose differs')

def binary_mask(path,size):
    with Image.open(path) as im:
        need(im.mode=='L' and im.size==size,'mask must be native L at camera size')
        a=np.array(im)
    need(np.isin(a,[0,255]).all() and bool((a==255).any()),'mask must be nonempty binary 0/255')
    return a

def run(bridge,identity_result,native_plan,masks,output):
    import pycolmap as pc
    need(pc.__version__=='4.2.1','pycolmap 4.2.1 required')
    bindings={}
    def bind(path,expected=None):
        r=record(path)
        if expected is not None:
            need(all(r[k]==expected[k] for k in ('bytes','sha256')),'declared byte binding differs: '+str(path))
        bindings[r['path']]=r
        return Path(r['path'])
    def read(path): return json.loads(bind(path).read_text(encoding='utf-8'))
    bind(__file__)
    import integrations.demo_pipeline.learned_silhouette_hull as records_module
    import pycolmap._core as core
    bind(records_module.__file__);bind(core.__file__)
    import integrations.mvs_masks.mask_geometry as geometry
    bind(geometry.__file__)
    bridge=Path(bridge).resolve(strict=True)
    ident=read(identity_result); plan=read(native_plan); manifest=read(masks)
    need(ident.get('schema')=='learned-identity-native-export-v1' and ident.get('status')=='IDENTITY_CONSTRAINED_EXPERIMENTAL','learned experimental identity export required')
    need(ident['source_model_id']==3 and Path(native_plan).resolve()==Path(ident['native_root']).resolve()/'PLAN.json','native producer binding differs')
    for r in ident['source_bindings']+ident['final_bindings']: bind(r['path'],r)
    by_path={str(Path(r['path']).resolve()):r for r in ident['source_bindings']}
    for p in (native_plan,masks):
        need(str(Path(p).resolve()) in by_path,'identity export does not bind native PLAN/MASKS')
        bind(p,by_path[str(Path(p).resolve())])
    selected=join_identity(ident['images'],plan['frames'],manifest['frames'])
    bp=read(bridge/'PLAN.json'); br=read(bridge/'RESULT.json')
    need(br['status']=='PASS' and br['input_bindings_unchanged'] is True,'bridge did not pass')
    for name in ('UNDISTORT_COMPARISON.json','MVS_COMPARISON.json'):
        need(read(bridge/name)['status']=='PASS','bridge numeric comparison did not pass')
    bind(bridge/'scene.mvs')
    need(Path(bp['source_sparse']).resolve()==Path(ident['binary_path']).resolve() and Path(bp['source_images']).resolve()==Path(ident['image_root']).resolve(),'bridge source is not identity export')
    declared=read(bridge/'INPUT_BINDINGS.json')
    for r in declared: bind(r['path'],r)
    exported={str(Path(r['path']).resolve()) for r in ident['final_bindings'] if Path(r['path']).parent.resolve()==Path(ident['binary_path']).resolve()}
    need(exported and exported <= {str(Path(r['path']).resolve()) for r in declared},'bridge missing exported model bindings')
    target_root=bridge/'undistorted'/'sparse'
    for p in target_root.iterdir():
        if p.is_file(): bind(p)
    source=pc.Reconstruction(ident['binary_path']);target=pc.Reconstruction(str(target_root))
    left={source.images[i].name:source.images[i] for i in source.reg_image_ids()};right={target.images[i].name:target.images[i] for i in target.reg_image_ids()}
    need(set(left)==set(right)==set(selected),'all-view camera/mask coverage differs')
    need(len(left)==54,'54 source/target views required')
    need(len({Path(n).stem.casefold() for n in left})==54,'mask basename collision')
    prepared=[]
    for name in sorted(left):
        need(Path(name).name==name and '/' not in name and '\\' not in name,'flat native image basename required')
        si,ti=left[name],right[name];check_pose(si,ti)
        sc,tc=source.cameras[si.camera_id],target.cameras[ti.camera_id]
        need(sc.model_name=='SIMPLE_RADIAL' and (int(sc.width),int(sc.height))==(576,1024),'expected original SIMPLE_RADIAL domain')
        need(tc.model_name in ('PINHOLE','SIMPLE_PINHOLE'),'target must be pinhole')
        m=selected[name]
        mp=bind(m['mask_path']);need(record(mp)['sha256']==m['mask_sha256'],'original mask digest differs')
        raw=bind(m['raw_path']);need(record(raw)['sha256']==m['raw_sha256'],'original RGB digest differs')
        staged=bind(Path(ident['image_root'])/name)
        need(record(staged)['sha256']==record(raw)['sha256'],'native staged RGB does not match mask decoded RGB')
        mask=binary_mask(mp,(int(sc.width),int(sc.height)))
        image=bind(bridge/'undistorted'/'images'/name)
        with Image.open(image) as im: need(im.size==(int(tc.width),int(tc.height)),'target RGB dimensions differ')
        prepared.append((name,m,sc,tc,mask))
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    (output/'masks').mkdir();(output/'validity').mkdir()
    rows=[]
    for name,m,sc,tc,mask in prepared:
        warped=warp_binary_mask(mask,sc,tc)
        need(warped.counts['foreground']>0,'warped foreground empty')
        stem=Path(name).stem
        paths=[output/'masks'/(stem+'.mask.png'),output/'validity'/(stem+'.validity.png')]
        for path,array in zip(paths,[warped.mask,warped.valid]):
            Image.fromarray(array).save(path)
            with Image.open(path) as im:
                need(im.mode=='L' and im.size==(int(tc.width),int(tc.height)) and np.array_equal(np.array(im),array),'warped PNG readback differs')
        need(not ((warped.mask>0)&(warped.valid==0)).any(),'foreground in unknown region')
        rows.append({'native_name':name,'decode_index':m['decode_index'],'mask':record(paths[0]),'validity':record(paths[1]),'counts':warped.counts,'source_camera':{'model':sc.model_name,'params':np.asarray(sc.params).tolist()},'target_camera':{'model':tc.model_name,'width':int(tc.width),'height':int(tc.height),'params':np.asarray(tc.params).tolist()}})
    drift=[r['path'] for r in bindings.values() if record(r['path'])!=r]
    result={'schema':SCHEMA,'status':'AVAILABLE' if len(rows)==54 and not drift else 'UNAVAILABLE','source_kind':'learned-experimental-identity-constrained','registered_views':54,'available_views':len(rows),'mask_consumption':'NOT_VERIFIED','geometry_acceptance':False,'native_launches':0,'algorithm':ALGORITHM,'mask_root':str(output/'masks'),'validity_root':str(output/'validity'),'bridge':str(bridge),'identity_result':str(Path(identity_result).resolve()),'outputs':rows,'source_bindings':list(bindings.values()),'input_drift':drift,'limitations':['SAM predictions are not human ground truth.','No hand occlusion or amodal-mask inference.','Complete warped labels do not prove native MVS consumption or geometric accuracy.']}
    (output/'RESULT.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    need(not drift,'source drift after mask generation')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('bridge','identity-result','native-plan','masks','output'):p.add_argument('--'+key,required=True)
    a=p.parse_args();r=run(a.bridge,a.identity_result,a.native_plan,a.masks,a.output)
    print(json.dumps({'schema':r['schema'],'status':r['status'],'available_views':r['available_views'],'native_launches':0}))
