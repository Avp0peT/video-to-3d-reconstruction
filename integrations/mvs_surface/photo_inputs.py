"""Connect scene view identities to actual T01/dataset files, not geometry truth.

This validates image identities and current bytes. It neither proves that a
scene's geometry was produced by those images nor authenticates an old run.
"""
from __future__ import annotations
import hashlib,json,math,os
from pathlib import Path
from .scene import read_scene

ROOT=Path(__file__).resolve().parents[2]
MODEL_BIN=('cameras.bin','images.bin','points3D.bin','rigs.bin','frames.bin')
MAX_FILE=64*1024**2

def need(ok,message):
    if not ok:raise ValueError('photo input link: '+message)

def _binding(path,limit=MAX_FILE):
    path=Path(path).resolve(strict=True);need(path.is_file(),'expected file')
    with path.open('rb') as f:
        before=os.fstat(f.fileno());need(before.st_size<=limit,'file exceeds byte bound')
        h=hashlib.sha256();size=0
        while block:=f.read(1024**2):
            size+=len(block);need(size<=limit,'file grew beyond bound');h.update(block)
        after=os.fstat(f.fileno())
    fields=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)
    need(fields(before)==fields(after)==fields(path.stat()) and size==before.st_size,'file changed during read')
    return {'path':str(path),'bytes':size,'sha256':h.hexdigest()}

def _json(data):
    def pairs(rows):
        out={}
        for k,v in rows:need(k not in out,'duplicate JSON key');out[k]=v
        return out
    def bad(value):raise ValueError('photo input link: nonfinite JSON')
    def number(value):
        v=float(value);need(math.isfinite(v),'nonfinite JSON');return v
    return json.loads(data,object_pairs_hook=pairs,parse_constant=bad,parse_float=number)

def _under(root,relative):
    need(isinstance(relative,str) and relative and not Path(relative).is_absolute(),'relative path required')
    parts=relative.replace('\\','/').split('/')
    need(all(p not in ('','.','..') and ':' not in p for p in parts),'unsafe relative path')
    root=Path(root).resolve();p=root.joinpath(*parts).resolve(strict=True)
    need(p.is_relative_to(root),'path escapes input root');return p

def _rows_by_index(rows,label):
    need(isinstance(rows,list) and 0<len(rows)<=512,label+' row count invalid')
    need(all(isinstance(r,dict) and type(r.get('input_index')) is int for r in rows),label+' index must be integer')
    need([r['input_index'] for r in rows]==list(range(len(rows))),label+' indices must be unique and ordered')
    return rows

def _native_views(root):
    import pycolmap
    need(pycolmap.__version__=='4.2.1','native reader version mismatch')
    model=pycolmap.Reconstruction(str(root))
    ids=sorted(model.reg_image_ids());need(2<=len(ids)<=256,'native view count outside profile')
    return [{'image_id':int(i),'name':model.images[i].name,'camera_id':int(model.images[i].camera_id),
             'width':int(model.cameras[model.images[i].camera_id].width),
             'height':int(model.cameras[model.images[i].camera_id].height)} for i in ids]

def _pixels(path):
    from PIL import Image
    with Image.open(path) as img:
        need(img.format=='PNG' and img.mode in ('RGB','L') and img.width*img.height<=16_777_216
             and getattr(img,'n_frames',1)==1 and img.getexif().get(274,1)==1 and 'transparency' not in img.info,'invalid canonical PNG')
        return img.mode,img.size,img.tobytes()

def validate(spec_path,scene_path,image_root):
    """Read-only identity check; fixed native model reader, no reconstruction."""
    seen={}
    def bind(path,limit=MAX_FILE):
        b=_binding(path,limit);old=seen.get(b['path']);need(old is None or old==b,'input drift')
        seen[b['path']]=b;return b
    def read(path,lines=False):
        b=bind(path,8*1024**2)
        with Path(b['path']).open('rb') as stream:raw=stream.read(8*1024**2+1)
        need(len(raw)<=8*1024**2 and hashlib.sha256(raw).hexdigest()==b['sha256'],'record drift')
        text=raw.decode('utf-8-sig')
        result=[_json(s) for s in text.splitlines() if s.strip()] if lines else _json(text)
        need(isinstance(result,list if lines else dict),'record root type invalid');return result
    spec=read(spec_path)
    need(spec.get('schema_version')=='surface-photo-input-v1','unknown source specification')
    roots={}
    for name in ('bridge_run','sfm_run','io_run','dataset_manifest'):
        value=spec.get(name);need(isinstance(value,str) and Path(value).is_absolute(),name+' must be absolute')
        roots[name]=Path(value).resolve(strict=True)
    bridge,sfm,io,manifest_path=(roots[n] for n in ('bridge_run','sfm_run','io_run','dataset_manifest'))
    model_id=spec.get('model_id');need(isinstance(model_id,str) and model_id,'explicit model_id required')
    need(Path(scene_path).resolve()==bridge/'scene.mvs' and Path(image_root).resolve()==bridge,'scene/image root does not belong to bridge')
    scene_bound=bind(scene_path);scene=read_scene(scene_path)
    if 'sha256' in scene:need(scene['sha256']==scene_bound['sha256'],'scene parse identity changed')
    views=scene.get('images');need(isinstance(views,list) and 2<=len(views)<=256,'scene view count invalid')
    names=[v.get('name') for v in views];need(all(isinstance(n,str) for n in names) and len(set(names))==len(names),'scene names must be unique')
    plan=read(bridge/'PLAN.json');result=read(bridge/'RESULT.json')
    # INPUT_BINDINGS has an array root, unlike records above.
    bb=bind(bridge/'INPUT_BINDINGS.json',8*1024**2)
    with Path(bb['path']).open('rb') as stream:raw=stream.read(8*1024**2+1)
    need(len(raw)<=8*1024**2 and hashlib.sha256(raw).hexdigest()==bb['sha256'],'bridge bindings drift')
    bridge_inputs=_json(raw.decode('utf-8-sig'))
    need(result.get('status')=='PASS' and result.get('input_bindings_unchanged') is True,'bridge has no accepted input record')
    need(isinstance(plan.get('source_images'),str) and Path(plan['source_images']).is_absolute() and Path(plan['source_images']).resolve()==sfm,'bridge uses another SfM image root')
    value=plan.get('source_sparse');need(isinstance(value,str) and Path(value).is_absolute(),'bridge selected model path missing')
    selected=Path(value).resolve(strict=True)
    sfm_run=read(sfm/'run.json');io_run=read(io/'run.json');manifest=read(manifest_path)
    need(sfm_run.get('contract_version') in ('sfm-native-v1','sfm-native-profile-v1','sfm-native-object-v1','sfm-native-object-video-v1','sfm-native-object-pinhole-v1'),'unsupported SfM record')
    need(io_run.get('contract_version')==manifest.get('contract_version')=='photo-io-v1','unsupported photo IO record')
    # Accepted SfM contracts require a complete SUCCESS T01 input, even when
    # SfM itself is PARTIAL because it produced several components.
    need(sfm_run.get('status') in ('SUCCESS','PARTIAL') and io_run.get('status')=='SUCCESS','upstream run failed or IO incomplete')
    need(sfm_run.get('data_kind')==io_run.get('data_kind')==manifest.get('data_kind')=='real','real input declarations disagree')
    need(isinstance(io_run.get('run_id'),str) and sfm_run.get('upstream_io_run_id')==io_run['run_id'],'upstream IO run mismatch')
    need(io_run.get('manifest_sha256')==seen[str(manifest_path)]['sha256'] and io_run.get('dataset_id')==manifest.get('dataset_id'),'dataset manifest identity mismatch')
    models=sfm_run.get('models');need(isinstance(models,list) and all(isinstance(m,dict) for m in models),'SfM models missing')
    matched=[m for m in models if m.get('model_id')==model_id];need(len(matched)==1 and matched[0].get('usable') is True,'requested SfM model not usable')
    info=matched[0];original=_under(sfm,info.get('binary_path'))
    model_bindings=[bind(directory/f) for directory in (selected,original) for f in MODEL_BIN]
    left=_native_views(selected);right=_native_views(original)
    subset_link=None
    if 'subset_certificate' in spec:
        reference_record=read(spec['subset_certificate']['path'])
        if reference_record.get('schema_version')=='native-general-subset-v1':
            from .general_subset import validate_reference
        else:
            from .subset import validate_reference
        subset_link=validate_reference(spec['subset_certificate'],derived_model=selected,
                                      sfm_run=sfm,io_run=io,original_model=original)
        kept=subset_link.get('retained_native_ids')
        if kept is None:kept=[v['image_id'] for v in right if v['image_id']!=subset_link['excluded']['image_id']]
        need(left==[v for v in right if v['image_id'] in kept], 'selected/source certified subset image identities differ')
        for row in subset_link['bindings']:
            actual=bind(row['path']);need(actual==row,'subset binding drift')
    else:
        need(left==right,'selected/source native image identities differ')
    # InterfaceCOLMAP is given <bridge>/undistorted/images as --image-folder.
    # Its scene stores that folder relative to the bridge plus the full native
    # image name (which may itself contain "images/"). Never join by basename.
    scene_names={v['name']:'undistorted/images/'+v['name'] for v in left}
    need(len(left)==len(names) and len(scene_names)==len(left) and set(scene_names.values())==set(names),'scene and selected model view sets differ')
    scene_by_name={v['name']:v for v in views}
    need(all(type(v.get('index')) is int and v['index']==i for i,v in enumerate(views)),'scene local indices differ')
    for v in left:
        identity=scene_by_name[scene_names[v['name']]].get('global_id')
        need(type(identity) is int and identity==v['image_id'],'scene global/native image identity mismatch')
    declared_views=right if subset_link is not None else left
    need(all(type(i) is int for i in info.get('registered_image_ids',[])) and sorted(info['registered_image_ids'])==sorted(v['image_id'] for v in declared_views),'SfM original registered image identities differ')
    expected={str((selected/f).resolve()) for f in MODEL_BIN}|{str(_under(sfm,n)) for n in scene_names}
    need(isinstance(bridge_inputs,list) and len(bridge_inputs)==len(expected),'bridge input binding count mismatch')
    actual_paths=[]
    for row in bridge_inputs:
        need(isinstance(row,dict) and isinstance(row.get('path'),str) and Path(row['path']).is_absolute(),'bridge binding path invalid')
        actual=bind(row['path']);need(type(row.get('bytes')) is int and row['bytes']==actual['bytes'] and row.get('sha256')==actual['sha256'],'bridge input hash/size mismatch')
        actual_paths.append(actual['path'])
    need(len(set(actual_paths))==len(actual_paths) and set(actual_paths)==expected,'bridge inputs differ from selected model/photos')
    io_frames=_rows_by_index(read(io/'frames.jsonl',True),'IO');maps=_rows_by_index(read(sfm/'frames_map.jsonl',True),'SfM')
    entries=manifest.get('frames');need(isinstance(entries,list) and len(entries)==len(io_frames)==len(maps),'input row count mismatch')
    need(type(io_run.get('input_count')) is int and io_run['input_count']==len(io_frames),'IO input count mismatch')
    ok=sum(f.get('status')=='OK' for f in io_frames)
    need(type(io_run.get('ok_count')) is int and io_run['ok_count']==ok and type(io_run.get('error_count')) is int
         and io_run['error_count']==len(io_frames)-ok==0,'IO status counts mismatch')
    counts=sfm_run.get('counts');need(isinstance(counts,dict) and type(counts.get('input_rows')) is int and counts['input_rows']==len(maps),'SfM input count mismatch')
    root_value=manifest.get('dataset_root');need(isinstance(root_value,str),'dataset root missing')
    dataset=Path(root_value);dataset=(dataset if dataset.is_absolute() else ROOT/dataset).resolve(strict=True)
    frames=[]
    for view in sorted(left,key=lambda v:v['name']):
        name=view['name'];rows=[r for r in maps if r.get('staged_image_name')==name]
        need(rows,'native view has no explicit frame mapping')
        need(all(type(r.get('representative_input_index')) is int for r in rows),'alias representative must be integer')
        representatives={r['representative_input_index'] for r in rows}
        need(len(representatives)==1,'multiple representatives for native view')
        rep=next(iter(representatives));need(type(rep) is int and 0<=rep<len(maps) and maps[rep] in rows
             and maps[rep].get('representative_input_index')==rep,'invalid alias representative')
        need(rep==min(r['input_index'] for r in rows),'representative must be the first byte-identical input')
        staged=bind(_under(sfm,name));image=bind(_under(image_root,scene_names[name]));aliases=[]
        for m in rows:
            idx=m['input_index'];f=io_frames[idx];entry=entries[idx]
            need(isinstance(entry,dict) and f.get('status')=='OK','selected IO frame failed')
            need(isinstance(f.get('frame_id'),str) and m.get('frame_id')==f['frame_id'] and m.get('upstream_io_run_id')==io_run['run_id'],'frame or run identity mismatch')
            need(m.get('upstream_image_output')==f.get('image_output') and f.get('image_path')==entry.get('image'),'frame image path mismatch')
            registrations=m.get('registrations');need(isinstance(registrations,list) and all(isinstance(r,dict) for r in registrations),'missing registrations')
            reg=[r for r in registrations if r.get('model_id')==model_id]
            need(len(reg)==1 and type(reg[0].get('image_id')) is int and type(reg[0].get('camera_id')) is int
                 and reg[0]['image_id']==view['image_id'] and reg[0]['camera_id']==view['camera_id'],'registration does not identify this native view')
            source=bind(_under(dataset,f.get('image_path')));canonical=bind(_under(io,f.get('image_output')))
            need(source['sha256']==f.get('image_sha256') and canonical['sha256']==m.get('canonical_image_sha256'),'source/canonical image hash mismatch')
            pixels=_pixels(canonical['path'])
            need(pixels[1]==(view['width'],view['height']) and type(f.get('canonical_width')) is int and type(f.get('canonical_height')) is int
                 and pixels[1]==(f['canonical_width'],f['canonical_height']),'canonical dimensions mismatch')
            # These SfM profiles group aliases by canonical FILE bytes. Equal
            # decoded pixels with different encodings are separate inputs.
            need(canonical['sha256']==staged['sha256'],'canonical/staged byte identity mismatch')
            aliases.append({'input_index':idx,'frame_id':f['frame_id'],'dataset_image':source,'canonical_image':canonical,
                            'staged_image':staged,'alias_pixel_equivalence':False})
        frames.append({'scene_image':scene_names[name],'native_image':name,'scene_index':scene_by_name[scene_names[name]]['index'],
                       'scene_global_id':scene_by_name[scene_names[name]]['global_id'],'image_id':view['image_id'],'camera_id':view['camera_id'],'representative_input_index':rep,
                       'undistorted_image':image,'aliases':aliases})
    for b in list(seen.values()):need(_binding(b['path'])==b,'input changed during identity validation')
    result={'schema_version':'surface-photo-link-v1','status':'PHOTO_INPUT_IDENTITIES_BOUND','model_id':model_id,
            'scene':scene_bound,'frames':frames,'bindings':sorted(seen.values(),key=lambda b:b['path']),
            'native_reader':'pycolmap 4.2.1 image identities only','geometry_lineage':'NOT_VERIFIED','mask_source_chain':'NOT_VERIFIED',
            'raw_camera_capture':'NOT_VERIFIED','quality_status':'NOT_EVALUATED','photo_reconstruction':'NOT_RUN',
            'limitations':'Current view/file association only. Selected point derivation, undistortion pixels, scene geometry provenance, masks and original capture chain are not certified.'}
    if subset_link is not None:
        result['subset_certificate']={'certificate':subset_link['certificate'],'excluded':subset_link.get('excluded',subset_link.get('excluded_inputs')),
              'relation':subset_link['relation'],'status':'CURRENT_EXPLICIT_SUBSET_BOUND_NOT_PHOTO_TRUTH'}
    return result
