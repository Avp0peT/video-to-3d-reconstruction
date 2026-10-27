"""Bind every existing calibrated view/depth of a parent MVS reconstruction.

Camera identity and common pixels are checked against the historical local model;
no estimate or registration is performed. Native depth provenance stays historical.
"""
from pathlib import Path
import numpy as np
from PIL import Image
from .global_local_pipeline import binding,load,save,_binary
from .mutual_model_pipeline import load_model_inputs
from .scene import read_scene
from .mask_consumption_probe import authenticate_depth
from integrations.openmvs_depth.dmap import read_dmap


def prepare_full_inputs(parent,baseline_model,output,check=lambda:None):
    parent=Path(parent).resolve(strict=True);base=Path(baseline_model).resolve(strict=True);out=Path(output).resolve()
    if out.exists() or out.is_relative_to(parent) or parent.is_relative_to(out) or out.is_relative_to(base) or base.is_relative_to(out):raise ValueError('fresh disjoint full input spec required')
    paths=[parent/'surface/plan.json',parent/'surface/RESULT.json',parent/'export/FRAME_MAP.json',parent/'masks/RESULT.json']
    plan=load(paths[0]);mask=load(paths[3]);frames=load(paths[2])['frames']
    if mask.get('coverage_status')!='COMPLETE' or len(frames)!=len(plan['frames']):raise ValueError('complete registered mask/frame mapping required')
    if not 1<=len(frames)<=128:raise ValueError('up to128 calibrated views supported')
    refs={}
    def bind(p,expected=None):
        check();item=binding(p)
        if expected is not None and any(item[k]!=expected[k] for k in ('path','bytes','sha256')):raise ValueError('declared source binding differs')
        refs[item['path']]=item;return item
    for path in paths:bind(path)
    bind(plan['scene']['path'],plan['scene'])
    for f in plan['frames']:
        for role in ('image_file','mask','validity'):bind(f[role]['path'],f[role])
    depths=[bind(p) for p in sorted((parent/'surface/work').glob('*.dmap'))]
    if not 1<=len(depths)<=64:raise ValueError('one to64 native depths required; no silent subset')
    spec=dict(schema='full-native-depth-inputs-v1',parent=str(parent),baseline_model=str(base),plan=str(paths[0]),frame_map=str(paths[2]),mask_result=str(paths[3]),depths=depths,input_bindings=list(refs.values()),native_reexecution=False,
        interpretation='Historical full-parent depths, not newly computed depths and not identical to a temporal-subset reconstruction merely because image IDs match.')
    out.parent.mkdir(parents=True,exist_ok=True);save(out,spec)
    # Full load validates the exact serialized spec; no unvalidated prepared claim.
    data=load_full_inputs(out,check)
    return dict(spec=binding(out),view_count=len(data['views']),depth_count=sum(v['depth_available'] for v in data['views']),same_coordinate_frame=data['full_input_provenance'])


def load_full_inputs(spec_path,check=lambda:None):
    spec_path=Path(spec_path).resolve(strict=True);spec_ref=binding(spec_path);spec=load(spec_path)
    if spec.get('schema')!='full-native-depth-inputs-v1':raise ValueError('full input schema mismatch')
    refs={spec_ref['path']:spec_ref}
    for item in spec['input_bindings']:
        check();actual=binding(item['path'])
        if actual!=item or actual['path'] in refs:raise ValueError('changed or duplicate full source binding')
        refs[actual['path']]=actual
    for key in ('plan','frame_map','mask_result'):
        if str(Path(spec[key]).resolve(strict=True)) not in refs:raise ValueError('unbound full source document')
    plan=load(spec['plan']);fmap=load(spec['frame_map']);mask_result=load(spec['mask_result'])
    if mask_result.get('coverage_status')!='COMPLETE':raise ValueError('full masks not complete')
    scene_path=str(Path(plan['scene']['path']).resolve(strict=True))
    if scene_path not in refs or any(refs[scene_path][k]!=plan['scene'][k] for k in ('path','bytes','sha256')):raise ValueError('scene unbound')
    scene=read_scene(scene_path)
    if not np.allclose(np.asarray(scene['transform']),np.eye(4),atol=0,rtol=0):raise ValueError('nonidentity scene transform unsupported')
    cams={x['global_id']:x for x in scene['images']};rows={x['image']['global_id']:x for x in plan['frames']}
    mask_rows={x['export_image_id']:x for x in mask_result['frames']}
    native_names={x['export_image_id']:x for x in fmap['frames']}
    if len(rows)!=len(plan['frames']) or len(native_names)!=len(fmap['frames']) or len(mask_rows)!=len(mask_result['frames']) or cams.keys()!=rows.keys() or rows.keys()!=native_names.keys() or rows.keys()!=mask_rows.keys() or not 1<=len(rows)<=128:raise ValueError('full view identity sets differ')
    depths={}
    for item in spec['depths']:
        check()
        if refs.get(item['path'])!=item:raise ValueError('unbound native depth')
        d=read_dmap(item['path']);iid=d.view_ids[0]
        if iid in depths or iid not in cams:raise ValueError('duplicate or foreign native depth identity')
        authenticate_depth(d,cams[iid]);depths[iid]=d
    if not 1<=len(depths)<=64:raise ValueError('native depth capacity exceeded')
    data=load_model_inputs(spec['baseline_model'],check);old={v['view_id']:v for v in data['views']}
    if not old.keys()<=rows.keys():raise ValueError('baseline is not a restriction of parent views')
    allviews=[];mapping=[]
    for iid in sorted(rows):
        check();f=rows[iid];cam=cams[iid];fcam=f['image'];shape=(cam['height'],cam['width'])
        for key in ('name','width','height'):
            if cam[key]!=fcam[key]:raise ValueError('scene/plan image domain differs')
        for key in ('K','R','C'):
            if not np.allclose(cam[key],fcam[key],atol=1e-9,rtol=0):raise ValueError('scene/plan camera differs')
        if Path(cam['name']).name!=native_names[iid]['exported_name']:raise ValueError('export mapping name differs')
        roots=plan['arguments'];mr=mask_rows[iid]
        if mr['decode_index']!=native_names[iid]['decode_index'] or mr['exported_name']!=native_names[iid]['exported_name']:raise ValueError('mask/frame identity mapping differs')
        roles=(('image_file','image_root',cam['name']),('mask','mask_root',f['mask_name']),('validity','validity_root',f['validity_name']))
        for role,root_key,relative in roles:
            root=Path(roots[root_key]).resolve(strict=True);expected=(root/relative).resolve(strict=True)
            if not expected.is_relative_to(root) or expected!=Path(f[role]['path']).resolve(strict=True):raise ValueError('frame file not at declared scene/role path')
            if role!='image_file' and any(mr[role][k]!=f[role][k] for k in ('path','bytes','sha256')):raise ValueError('mask result/frame binding differs')
        for role in ('image_file','mask','validity'):
            if refs.get(f[role]['path'])!=f[role]:raise ValueError('unbound frame input')
        with Image.open(f['image_file']['path']) as im:rgb=np.array(im.convert('RGB'))
        if rgb.shape!=shape+(3,):raise ValueError('full source image dimensions differ')
        mask=_binary(f['mask'],shape,refs);valid=_binary(f['validity'],shape,refs);d=depths.get(iid)
        if d is not None and d.depth.shape!=shape:raise ValueError('native resized depth not supported by this pipeline')
        v=dict(view_id=iid,image=rgb,mask=mask,validity=valid,depth_available=d is not None,depth=d.depth if d is not None else np.zeros(shape,np.float32),K=np.asarray(cam['K']),R=np.asarray(cam['R']),C=np.asarray(cam['C']))
        if iid in old:
            for key in ('K','R','C'):
                if not np.allclose(v[key],old[iid][key],atol=1e-9,rtol=0):raise ValueError('parent and baseline coordinate frame differ')
            if any(not np.array_equal(v[key],old[iid][key]) for key in ('image','mask','validity')):raise ValueError('common baseline pixels differ')
        allviews.append(v);mapping.append(dict(view_id=iid,decode_index=native_names[iid]['decode_index'],depth_available=d is not None,was_in_local_source=iid in old))
    for item in data['bindings']:refs[item['path']]=item
    if any(binding(item['path'])!=item for item in refs.values()):raise ValueError('full input changed during load')
    data.update(views=allviews,bindings=list(refs.values()),full_input_provenance=dict(schema='full-depth-load-v1',views=len(allviews),native_depths=len(depths),common_views=len(old),additional_views=len(allviews)-len(old),mapping=mapping,coordinate_transform='identity verified through common K/R/C',new_native_execution=False,parent_native_mask_consumption=mask_result.get('mask_consumption'),depth_values_may_differ_in_common_views=True,physical_shape_accepted=False))
    return data
