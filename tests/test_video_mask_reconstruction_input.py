"""Adoption provenance and consumer gates; no model or native geometry calls."""
import copy
import json
from pathlib import Path

import pytest
from PIL import Image

from integrations.video_masks import reconstruction_input as r


@pytest.fixture
def source(tmp_path, monkeypatch):
    source = tmp_path/'union'; source.mkdir()
    io = tmp_path/'io'; io.mkdir()
    pipeline = tmp_path/'pipeline'; (pipeline/'input').mkdir(parents=True)
    (source/'final_masks').mkdir(); (source/'frames').mkdir()
    old = dict(backend={'name':'sam3'}, object={'object_id':'glove', 'selection_policy':'unique_candidate_or_explicit'},
               counts=dict(requested=2,success=1,no_match=1,needs_selection=0,error=0,not_run=0),
               frames=[dict(input_index=0,status='OK'),dict(input_index=1,status='NO_MATCH')])
    r.save(pipeline/'old.json', old)
    native=[]; ledger=[]; full=[]; fused=[]; decoded=[]
    for n, decode in enumerate((0, 7)):
        image=io/f'{n}.png'; Image.new('RGB',(4,5),(n*100,30,40)).save(image)
        mask=source/'final_masks'/f'{decode:06d}.png'; Image.new('L',(4,5),255).save(mask)
        jpeg=source/'frames'/f'{decode:06d}.jpg'; Image.new('RGB',(4,5),(n*100,30,40)).save(jpeg)
        decoded.append(dict(decode_index=decode,path=str(jpeg),sha256=r.sha(jpeg)))
        native.append(dict(input_index=n,frame_id=f'frame{n}',image_output=f'{n}.png',image_sha256=r.sha(image),
            status='OK',canonical_width=4,canonical_height=5,raw_to_canonical=[[1,0,0],[0,1,0],[0,0,1]],orientation_applied=1))
        ledger.append(dict(input_index=n,frame_id=f'frame{n}',decode_index=decode,pts=decode*3000,time_base=[1,90000],
            time_seconds=decode/30,original_status=old['frames'][n]['status'],source_image=str(image),
            source_image_sha256=r.sha(image),width=4,height=5))
        fusion=dict(decode_index=decode,mode='union',removed_pixels=0)
        fused.append(fusion)
        full.append(dict(decode_index=decode,input_index=n,pts=decode*3000,time_base=[1,90000],
            original_status=ledger[-1]['original_status'],placeholder=False,fusion=fusion,path=str(mask),
            sha256=r.sha(mask),mask_path=str(mask),mask_sha256=r.sha(mask),foreground_pixels=20,
            pixel_sha256=r.mask_pixels(mask,(4,5))[1]))
    r.save(io/'run.json',dict(contract_version='photo-io-v1',status='SUCCESS',run_id='io',input_count=2,ok_count=2,error_count=0))
    (io/'frames.jsonl').write_text('\n'.join(json.dumps(v) for v in native),encoding='utf-8')
    r.save(pipeline/'PLAN.json',{})
    r.save(pipeline/'dataset.json',{})
    r.save(pipeline/'input/RESULT.json',dict(photo_io=dict(manifest=dict(path=str(pipeline/'dataset.json'),sha256=r.sha(pipeline/'dataset.json')))))
    bindings={}
    for key,path in [('pipeline_plan',pipeline/'PLAN.json'),('source_manifest',pipeline/'old.json'),('io_run',io/'run.json'),('io_frames',io/'frames.jsonl')]:
        bindings[key+'_path']=str(path); bindings[key+'_sha256']=r.sha(path)
    plan=dict(bindings,schema='video-mask-full-plan-v1',coverage_ledger=ledger,width=4,height=5,
              expected_decode_indexes=[0,7],frame_count=2)
    r.save(source/'PLAN.json',plan)
    r.save(source/'RESULT.json',dict(status='FULL_TIMELINE_FUSED_EXPORTED',mode='union',model_entries=0,input_drift=[],code_drift=[]))
    r.save(source/'MANIFEST.json',dict(schema='full-video-mask-manifest-v1',rows=full))
    r.save(source/'FUSION.json',dict(summary={'mode':'union'},rows=fused))
    r.save(source/'DECODE.json',dict(frames=decoded))
    r.save(source/'START_BINDINGS.json',dict(source_run=str(source),input_files={str(source/'DECODE.json'):r.sha(source/'DECODE.json')}))
    from integrations.video_masks import planning
    monkeypatch.setattr(planning,'_validated_evidence',lambda _: (copy.deepcopy(ledger),copy.deepcopy(bindings)))
    return source


def test_prepare_preserves_history_and_consumer_readback(source,tmp_path):
    result=r.prepare(source,tmp_path/'adopted')
    assert result['frames']==2 and result['model_entries']==0
    assert result['historical_counts']=={'OK':1,'NO_MATCH':1}
    assert result['dataset_manifest'].endswith('dataset.json')
    doc=r.read(result['manifest_path'])
    assert doc['backend']['name']=='video_temporal_union'
    assert doc['backend']['model_executed'] is False
    assert doc['frames'][1]['historical_status']=='NO_MATCH'
    from integrations.mvs_masks.manifest_io import parse_object_manifest, InputError
    assert parse_object_manifest(result['manifest_path'])['counts']['success']==2
    assert r.read(source/'PLAN.json')['coverage_ledger'][1]['original_status']=='NO_MATCH'


@pytest.mark.parametrize('fault',['pts','index','placeholder','binary','jpeg','transform','ledger'])
def test_rejects_wrong_coordinate_or_raster_evidence(source,tmp_path,fault):
    if fault in ('pts','index','placeholder'):
        doc=r.read(source/'MANIFEST.json')
        key={'pts':'pts','index':'input_index','placeholder':'placeholder'}[fault]
        doc['rows'][0][key]=True if fault=='placeholder' else 999
        r.save(source/'MANIFEST.json',doc)
    elif fault=='binary':
        Image.new('L',(4,5),127).save(source/'final_masks/000000.png')
    elif fault=='jpeg':
        Image.new('RGB',(4,5),'red').save(source/'frames/000000.jpg')
    elif fault=='transform':
        plan=r.read(source/'PLAN.json'); io=Path(plan['io_frames_path'])
        rows=[json.loads(v) for v in io.read_text().splitlines()]
        rows[0]['raw_to_canonical']=[[0,1,0],[1,0,0],[0,0,1]]
        io.write_text('\n'.join(json.dumps(v) for v in rows))
    else:
        doc=r.read(source/'PLAN.json'); doc['coverage_ledger'][0]['decode_index']=99;r.save(source/'PLAN.json',doc)
    with pytest.raises(ValueError):r.prepare(source,tmp_path/'adopted')
    assert not (tmp_path/'adopted').exists()


@pytest.mark.parametrize('fault',['mask','history','backend','lineage'])
def test_consumers_reject_spoofed_derived_source(source,tmp_path,fault):
    result=r.prepare(source,tmp_path/'adopted'); path=Path(result['manifest_path']);doc=r.read(path)
    if fault=='mask':Image.new('L',(4,5),0).save(path.parent/'masks/000000.png')
    if fault=='history':doc['frames'][1]['historical_status']='OK'
    if fault=='backend':doc['backend']['name']='sam3'
    if fault=='lineage':doc.pop('derivation')
    r.save(path,doc)
    from integrations.mvs_masks.manifest_io import parse_object_manifest, InputError
    from integrations.mask_review.review import load_inputs,Bindings
    with pytest.raises((InputError,ValueError,KeyError)):parse_object_manifest(path)
    with pytest.raises((ValueError,KeyError)):load_inputs([path],Bindings())


def test_output_never_overwrites(source,tmp_path):
    output=tmp_path/'adopted';r.prepare(source,output)
    digest=r.sha(output/'manifest.json')
    with pytest.raises(ValueError,match='new'):r.prepare(source,output)
    assert r.sha(output/'manifest.json')==digest
