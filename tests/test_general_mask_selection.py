"""Run the existing consumer on a real synthetic native model restriction."""
import json
from test_native_subset import chain43, native, INDICES, save, sha
import test_mvs_mask_bridge as base
from integrations.mvs_surface import general_subset as g


def test_non59_exclusion_and_unregistered_rows_publish_complete(chain43,tmp_path):
    c=chain43; kept=[i for i in INDICES if i not in (59,60)]
    derived=g.derive(c['sfm'],c['t01'],'0',kept,tmp_path/'general',{59:'NO_MATCH',60:'NO_MATCH'})
    sparse=derived['selected_sparse']
    target=native([i+1 for i in kept],count=0,target=True)
    target.write_binary(str(c['bridge']/'undistorted/sparse'))
    base.write_scene(c['bridge']/'scene.mvs',[f'{i:06d}.png' for i in kept],60,44,
                     global_ids=[i+1 for i in kept],centers=[[-float(i+1),0.,0.] for i in kept])
    save(c['bridge']/'PLAN.json',{'source_sparse':sparse,'source_images':str(c['sfm'])})
    from pathlib import Path
    save(c['bridge']/'INPUT_BINDINGS.json',[base.mio.file_record(Path(sparse)/f) for f in g.MODEL_BIN]+
        [base.mio.file_record(c['sfm']/f'images/{i:06d}.png') for i in kept])
    manifest=base.world_manifest(c['root'],n_ok=66,directory=tmp_path/'full_masks')
    doc=json.loads(manifest.read_text())
    # Rebind the synthetic producer to the chain's actual T01 source identities.
    old=json.loads(c['primary'].read_text())
    doc['source_io']=old['source_io']
    frames=[json.loads(x) for x in (c['t01']/'frames.jsonl').read_text().splitlines()]
    for f in doc['frames']:
        i=f['input_index']; canonical=c['t01']/frames[i]['image_output']
        f.update(frame_id=frames[i]['frame_id'],source_image=str(canonical),source_image_sha256=sha(canonical))
        if i in (59,60):f.update(status='NO_MATCH',candidates=[],selected_candidate_id=None,selected_mask=None,selection=None)
    doc['status']='PARTIAL';doc['counts']=dict(requested=66,success=64,no_match=2,needs_selection=0,error=0,not_run=0)
    save(manifest,doc)
    ledger={'schema_version':'object-mask-selection-v1','subset_certificate':derived['subset_certificate'],
            'manifests':[{'path':str(manifest),'sha256':sha(manifest)}],'selections':[], 'excluded_source_rows':[]}
    for f in doc['frames']:
        i=f['input_index']
        if i in kept:ledger['selections'].append(dict(input_index=i,frame_id=f['frame_id'],manifest_path=str(manifest),
            selected_candidate_id=f['selected_candidate_id'],selected_mask_sha256=f['selected_mask']['sha256']))
        else:ledger['excluded_source_rows'].append(dict(manifest_path=str(manifest),manifest_sha256=sha(manifest),
            input_index=i,frame_id=f['frame_id'],reason='CERTIFIED_VIEW_EXCLUDED'))
    path=tmp_path/'ledger.json';save(path,ledger);out=tmp_path/'converted'
    assert base.cli.main(base.cli_args(c,manifest,out,'--selection-ledger',str(path)))==0
    actual=json.loads((out/'manifest.json').read_text())
    assert actual['coverage_status']=='COMPLETE' and actual['publishable_arguments'] is True
    excluded=actual['mask_selection']['unselected_source_rows']
    assert len(excluded)==24 and {59,60,0} <= {v['input_index'] for v in excluded}
    assert next(v for v in excluded if v['input_index']==60)['original_status']=='NO_MATCH'
