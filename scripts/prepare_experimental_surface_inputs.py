"""Prepare explicit experimental identities or warp an already completed bridge.

Neither mode starts a native process. The bridge must be executed separately
under the caller's supervision and budget; failed preparations are retained.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'integrations'))
from integrations.mvs_surface import experimental_inputs as ex
from integrations.mvs_masks import manifest_io as mio


def save(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')


def identity(args, output):
    producer = Path(args.producer).resolve(strict=True)
    plan = ex.load(producer/'PLAN.json')
    configs = [c for c in plan.get('configs',[]) if c.get('arm') == args.arm]
    ex.need(len(configs) == 1, 'explicit producer arm missing/ambiguous')
    config = configs[0]; arm_root = Path(config['output']).resolve(strict=True)
    model_root = Path(config['models_root']).resolve(strict=True)/str(args.model_id)
    manifests = [mio.parse_object_manifest(p) for p in args.mask_manifest]
    selections = []
    for row in plan.get('inputs',[]):
        if row.get('arm') != args.arm or row.get('kind') != 'mask': continue
        matches = []
        for i,m in enumerate(manifests):
            frames = [f for f in m['frames'] if f['input_index'] == row['input_index'] and f['status'] == 'OK']
            if len(frames) == 1 and Path(frames[0]['chosen']['resolved_mask_path']).resolve() == Path(row['source']['path']).resolve():
                matches.append(i)
        ex.need(len(matches) == 1, 'producer-selected mask must identify exactly one supplied accepted manifest')
        selections.append({'input_index':row['input_index'],'manifest_index':matches[0]})
    spec = {'schema_version':ex.SCHEMA, 'producer_profile':ex.PROFILE,
            'source_kind':'real-experimental-native-reconstruction', 'model_id':args.model_id,
            'producer_plan':ex.record(producer/'PLAN.json'),
            'producer_config':ex.record(arm_root/'CONFIG.json'),
            'producer_entry':ex.record(arm_root/'supervision'/'ENTRY.json'),
            'producer_result':ex.record(arm_root/'RESULT.json'),
            'database':ex.record(config['database_path']),
            'model_files':[ex.record(p) for p in sorted(model_root.iterdir()) if p.is_file()],
            'mask_manifests':[ex.record(p) for p in args.mask_manifest], 'mask_selections':selections,
            'historical_code_copies':[]}
    for value in args.historical_code_copy:
        original,sep,archive=value.partition('=')
        ex.need(bool(sep and original and archive),'historical copy must be ORIGINAL=ARCHIVE')
        spec['historical_code_copies'].append({'original_path':str(Path(original).resolve()),'archive':ex.record(archive)})
    path = output/'IDENTITY_SPEC.json'; save(path,spec)
    result = ex.validate_identity(path, _derivation_parent=getattr(args,'derivation_parent',False))
    save(output/'IDENTITY.json',result)
    return {'status':'DERIVATION_PARENT_AUTHENTICATED_NOT_BRIDGE_ADMITTED' if getattr(args,'derivation_parent',False) else 'IDENTITY_PREPARED',
            'identity_spec':ex.record(path), 'identity':ex.record(output/'IDENTITY.json'),
            'counts':result['counts'],'selected_indices':[f['input_index'] for f in result['frames']],
            'native_processes_started':0, 'not_original_t01_sfm_run':True,
            'bridge_inputs':{'sparse':result['source_sparse'],'image_root':result['source_images']}}


def derive(args, output):
    import pycolmap
    parent_path = Path(args.parent_identity_spec).resolve(strict=True)
    ex.need(ex.load(parent_path).get('schema_version') == ex.SCHEMA, 'original experimental parent required')
    parent = ex.validate_identity(parent_path, _derivation_parent=True)
    source = Path(parent['source_sparse'])
    ex.need(not output.is_relative_to(source), 'derived output must be outside original model')
    child = pycolmap.Reconstruction(str(source))
    for pid, point in list(child.points3D.items()):
        ids = [e.image_id for e in point.track.elements]
        if len(set(ids)) != len(ids) or len(set(ids)) < 2:
            child.delete_point3D(pid)
    sparse = output/'sparse'; sparse.mkdir()
    child.write_binary(str(sparse))
    relation = ex.derived_relation(pycolmap.Reconstruction(str(source)), pycolmap.Reconstruction(str(sparse)))
    codes = ex.derived_code_bindings(); parent_ref = ex.record(parent_path)
    certificate = {'policy':ex.DERIVED_POLICY,'parent_identity_spec':parent_ref,
                   'source_sparse':str(source),'derived_sparse':str(sparse),
                   'relation':relation,'code_bindings':codes}
    save(output/'DERIVATION_CERTIFICATE.json',certificate)
    spec = {'schema_version':ex.DERIVED_SCHEMA,'policy':ex.DERIVED_POLICY,
            'parent_identity_spec':parent_ref,'derived_sparse':str(sparse),
            'model_files':[ex.record(p) for p in sorted(sparse.iterdir()) if p.is_file()],
            'certificate':ex.record(output/'DERIVATION_CERTIFICATE.json'),'code_bindings':codes}
    save(output/'IDENTITY_SPEC.json',spec)
    validated = ex.validate_identity(output/'IDENTITY_SPEC.json')
    save(output/'IDENTITY.json',validated)
    return {'status':'DERIVED_IDENTITY_PREPARED','identity_spec':ex.record(output/'IDENTITY_SPEC.json'),
            'counts':validated['counts'],'derivation':relation,'native_processes_started':0,
            'bridge_inputs':{'sparse':str(sparse),'image_root':validated['source_images']}}


def warp(args, output):
    from scripts.prepare_mvs_masks import bridge_artifacts, model_bindings, scene_binding
    from integrations.mvs_masks.mask_geometry import warp_binary_mask
    identity = ex.validate_identity(args.identity_spec)
    bridge = bridge_artifacts(args.bridge)
    ex.need(str(bridge['source_sparse']) == identity['source_sparse'] and
            str(bridge['source_images']) == str(Path(identity['source_images']).resolve()), 'bridge source conflicts with experimental identity')
    models = model_bindings(bridge); scene_binding(bridge,models['records'])
    source,target = models['source_model'],models['target_model']
    outputs=[]; stems=set()
    for f in identity['frames']:
        name=f['native_name'];stem=Path(name).stem.casefold()
        ex.need(stem not in stems,'mask basename alias');stems.add(stem)
        sc=source.cameras[source.images[f['image_id']].camera_id]
        tc=target.cameras[target.images[f['image_id']].camera_id]
        labels,_=mio.read_label_png(f['mask']['path'],sc.width,sc.height)
        transformed=warp_binary_mask(labels,sc,tc)
        ex.need(transformed.counts['foreground']>0,'empty foreground after warp')
        mask=output/'masks'/(Path(name).stem+'.mask.png')
        valid=output/'validity'/(Path(name).stem+'.validity.png')
        mio.write_label_png(mask,transformed.mask);mio.write_label_png(valid,transformed.valid)
        outputs.append({'native_name':name,'mask':ex.record(mask),'validity':ex.record(valid)})
    spec={'schema_version':ex.SURFACE_SCHEMA,'identity_spec':ex.record(args.identity_spec),
          'bridge_run':str(bridge['bridge']),
          'bridge_bindings':[ex.record(p) for p in sorted(bridge['bridge'].rglob('*')) if p.is_file()],
          'outputs':outputs}
    path=output/'SURFACE_SPEC.json';save(path,spec)
    validated=ex.validate(path,bridge['scene']['path'],bridge['bridge'],output/'masks',output/'validity')
    save(output/'VALIDATION.json',validated)
    return {'status':'EXPERIMENTAL_SURFACE_INPUTS_PREPARED','surface_spec':ex.record(path),
            'native_processes_started':0,'scene':bridge['scene']['path'],'image_root':str(bridge['bridge']),
            'mask_root':str(output/'masks'),'validity_root':str(output/'validity'),
            'input_origin':'experimental-real','object_shape_status':'NOT_EVALUATED','full_object_accepted':False}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='mode',required=True)
    a=sub.add_parser('identity');a.add_argument('--producer',required=True)
    a.add_argument('--arm',choices=['masked28','unmasked28'],required=True)
    a.add_argument('--model-id',type=int,required=True)
    a.add_argument('--historical-code-copy',action='append',default=[],metavar='ORIGINAL=ARCHIVE')
    a.add_argument('--derivation-parent',action='store_true',help='authenticate original geometry only for deterministic whole-point derivation; not bridge admission')
    a.add_argument('--mask-manifest',action='append',required=True);a.add_argument('--output',required=True)
    b=sub.add_parser('warp');b.add_argument('--identity-spec',required=True)
    b.add_argument('--bridge',required=True);b.add_argument('--output',required=True)
    d=sub.add_parser('derive');d.add_argument('--parent-identity-spec',required=True);d.add_argument('--output',required=True)
    args=parser.parse_args(argv);output=Path(args.output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    try:
        result={'identity':identity,'derive':derive,'warp':warp}[args.mode](args,output)
        save(output/'RESULT.json',result);return 0
    except Exception as exc:
        save(output/'RESULT.json',{'status':'INPUT_ERROR','error':f'{type(exc).__name__}: {exc}', 'native_processes_started':0})
        print(f'{type(exc).__name__}: {exc}',file=sys.stderr);return 2


if __name__=='__main__':
    raise SystemExit(main())
