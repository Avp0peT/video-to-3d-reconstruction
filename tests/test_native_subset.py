"""TEST_DATA native models; no solving/reconstruction or real source derivation."""
import copy,json,hashlib,importlib.util
from pathlib import Path
import numpy as np
import pycolmap
import pytest
import test_mvs_mask_bridge as base
from integrations.mvs_surface import subset,photo_inputs

INDICES=list(range(3,9))+list(range(17,24))+list(range(34,50))+list(range(51,66))
ROOT=Path(__file__).resolve().parents[1]

def save(p,value):p.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def native(ids, *, duplicates=0, count=3, observed=1, target=False):
    rec=pycolmap.Reconstruction()
    width,height=(60,44) if target else (64,48)
    rec.add_camera_with_trivial_rig(pycolmap.Camera(model='PINHOLE' if target else 'SIMPLE_RADIAL',width=width,height=height,params=np.array([50.,50.,30.,22.] if target else [53.34,32.,24.,0.]),camera_id=1))
    trackrows=[];pair=ids[:2]
    for n in range(count):trackrows.append([(pair[0],n),(pair[1],n)]+([(60,n)] if 60 in ids and n<observed else []))
    for n in range(duplicates):trackrows.append([(pair[0],count+2*n),(pair[0],count+2*n+1),(pair[1],count+n)])
    lengths={iid:0 for iid in ids}
    for track in trackrows:
        for iid,idx in track:lengths[iid]=max(lengths[iid],idx+1)
    for iid in ids:
        pts=pycolmap.Point2DList()
        for i in range(lengths[iid]):pts.append(pycolmap.Point2D(xy=np.array([float(i%20+1),float(i%15+1)])))
        image=pycolmap.Image(name=f'images/{iid-1:06d}.png',camera_id=1,image_id=iid,points2D=pts)
        rec.add_image_with_trivial_frame(image,pycolmap.Rigid3d(pycolmap.Rotation3d(),np.array([float(iid),0.,0.])))
    for n,rows in enumerate(trackrows):
        track=pycolmap.Track()
        for iid,idx in rows:track.add_element(iid,idx)
        rec.add_point3D(np.array([float(n+1),1.,5.]),track,np.array([10,20,30],dtype=np.uint8))
    return rec


def test_small_native_relation_and_roundtrip(tmp_path):
    origin=native([4,5,6,60],count=3,observed=1)
    a=tmp_path/'original';a.mkdir();origin.write_binary(str(a))
    clean=pycolmap.Reconstruction(str(a));derived=pycolmap.Reconstruction(str(a));derived.deregister_frame(60)
    b=tmp_path/'derived';b.mkdir();derived.write_binary(str(b));back=pycolmap.Reconstruction(str(b))
    result=subset.compare_models(origin,clean,back,60)
    assert result['derived_registered_ids']==[4,5,6] and len(result['removed_observations'])==1
    assert result['derived_points']==3


@pytest.mark.parametrize('fault',['extra_view','pose','xyz','rgb','track','points2D','parent_pose','multi_frame'])
def test_small_native_relation_refuses_unapproved_changes(tmp_path,fault):
    original=native([4,5,6,60],count=3,observed=1);a=tmp_path/'source';a.mkdir();original.write_binary(str(a))
    clean=pycolmap.Reconstruction(str(a));child=pycolmap.Reconstruction(str(a));child.deregister_frame(60)
    if fault=='extra_view':child.deregister_frame(6)
    elif fault=='pose':child.frame(4).rig_from_world=pycolmap.Rigid3d(pycolmap.Rotation3d(),np.array([99.,0.,0.]))
    elif fault=='parent_pose':clean.frame(4).rig_from_world=pycolmap.Rigid3d(pycolmap.Rotation3d(),np.array([99.,0.,0.]))
    elif fault=='xyz':child.point3D(1).xyz=np.array([99.,1.,5.])
    elif fault=='rgb':child.point3D(1).color=np.array([1,2,3],dtype=np.uint8)
    elif fault=='track':child.delete_observation(4,0)
    elif fault=='points2D':child.image(4).points2D[0].xy=np.array([99.,0.])
    elif fault=='multi_frame':
        # Independent native data-ID frame validation rejects multi-image frames.
        expected=subset.views(clean);expected[60]['frame_data_ids'].append(('SensorType.CAMERA',1,6))
        from unittest.mock import patch
        real=subset.views
        with patch.object(subset,'views',side_effect=lambda rec:expected if rec is clean or rec is original else real(rec)):
            with pytest.raises(ValueError,match='single-image'):subset.compare_models(original,clean,child,60)
        return
    with pytest.raises(ValueError):subset.compare_models(original,clean,child,60)


@pytest.fixture
def chain43(tmp_path):
    root=tmp_path/'TEST_DATA_43';world=base.build_world(root,model_images=66,extra_frames=0,n_ok=66)
    sfm,io=world['sfm'],world['t01']
    original_path=sfm/'models/0/binary';original_path.mkdir(parents=True)
    ids=[i+1 for i in INDICES]
    original=native(ids,count=2393,observed=185,duplicates=20);original.write_binary(str(original_path))
    clean=pycolmap.Reconstruction(str(original_path))
    for pid,p in list(clean.points3D.items()):
        track=list(p.track.elements)
        if len({e.image_id for e in track})!=len(track):clean.delete_point3D(pid)
    clean.write_binary(str(root/'source_sparse'))
    derived=pycolmap.Reconstruction(str(root/'source_sparse'));derived.deregister_frame(60)
    derived_path=root/'subset_sparse';derived_path.mkdir();derived.write_binary(str(derived_path))
    t01=[json.loads(x) for x in (io/'frames.jsonl').read_text().splitlines()]
    maps=[json.loads(x) for x in (sfm/'frames_map.jsonl').read_text().splitlines()]
    for f,m in zip(t01,maps):
        f['image_sha256']=sha(io/f['image_output'])
        m['registrations']=[{'model_id':'0','image_id':f['input_index']+1,'camera_id':1}] if f['input_index'] in INDICES else []
        m['representative_input_index']=f['input_index']
    (io/'frames.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in t01),encoding='utf-8')
    (sfm/'frames_map.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in maps),encoding='utf-8')
    dataset=io/'dataset.json';save(dataset,{'contract_version':'photo-io-v1','data_kind':'real','dataset_root':str(io),'dataset_id':'TEST_DATA','frames':[{'image':f['image_path']} for f in t01]})
    meta=json.loads((io/'run.json').read_text());meta.update(data_kind='real',error_count=0,dataset_id='TEST_DATA',manifest_sha256=sha(dataset));save(io/'run.json',meta)
    sfm_meta=json.loads((sfm/'run.json').read_text());sfm_meta.update(data_kind='real',models=[{'model_id':'0','usable':True,'binary_path':'models/0/binary','registered_image_ids':ids}],counts={'input_rows':66});save(sfm/'run.json',sfm_meta)
    kept=[i for i in INDICES if i!=59];target=native([i+1 for i in kept],count=0,target=True);target.write_binary(str(world['bridge']/'undistorted/sparse'))
    base.write_scene(world['bridge']/'scene.mvs',[f'{i:06d}.png' for i in kept],60,44,global_ids=[i+1 for i in kept],centers=[[-float(i+1),0.,0.] for i in kept])
    save(world['bridge']/'PLAN.json',{'source_sparse':str(derived_path),'source_images':str(sfm)})
    save(world['bridge']/'RESULT.json',{'status':'PASS','input_bindings_unchanged':True})
    inputs=[base.mio.file_record(derived_path/f) for f in subset.MODEL_BIN]+[base.mio.file_record(sfm/f'images/{i:06d}.png') for i in kept]
    save(world['bridge']/'INPUT_BINDINGS.json',inputs)
    certificate=root/'subset_certificate.json';save(certificate,subset.evaluate(root/'source_sparse',derived_path,sfm,io))
    ref={'path':str(certificate),'sha256':sha(certificate)}
    primary=base.world_manifest(root,n_ok=66)
    doc=json.loads(primary.read_text());doc['frames']=[f for f in doc['frames'] if f['input_index'] in INDICES]
    for f in doc['frames']:
        if f['input_index'] in (59,60):f.update(status='NO_MATCH',candidates=[],selected_candidate_id=None,selected_mask=None,selection=None)
    doc['status']='PARTIAL';doc['counts']=dict(requested=44,success=42,no_match=2,needs_selection=0,error=0,not_run=0);save(primary,doc)
    supplementary=base.world_manifest(root,n_ok=66,directory=root/'supplement')
    sup=json.loads(supplementary.read_text());sup['frames']=[f for f in sup['frames'] if f['input_index'] in (59,60)]
    sup['frames'][0].update(status='NO_MATCH',candidates=[],selected_candidate_id=None,selected_mask=None,selection=None)
    sup['status']='PARTIAL';sup['counts']=dict(requested=2,success=1,no_match=1,needs_selection=0,error=0,not_run=0);save(supplementary,sup)
    selections=[];exclusions=[]
    for path,d in [(primary,doc),(supplementary,sup)]:
        for f in d['frames']:
            if f['status']=='OK':selections.append({'input_index':f['input_index'],'frame_id':f['frame_id'],'manifest_path':str(path),'selected_candidate_id':f['selected_candidate_id'],'selected_mask_sha256':f['selected_mask']['sha256']})
            elif f['input_index']==59:exclusions.append({'manifest_path':str(path),'manifest_sha256':sha(path),'input_index':59,'frame_id':f['frame_id'],'reason':'NO_MATCH_VIEW_EXCLUDED'})
    ledger=root/'selection.json';save(ledger,{'schema_version':'object-mask-selection-v1','manifests':[{'path':str(x),'sha256':sha(x)} for x in (primary,supplementary)],'selections':selections,'subset_certificate':ref,'excluded_source_rows':exclusions})
    photo=root/'PHOTO_SOURCE.json';save(photo,{'schema_version':'surface-photo-input-v1','bridge_run':str(world['bridge']),'sfm_run':str(sfm),'io_run':str(io),'dataset_manifest':str(dataset),'model_id':'0','subset_certificate':ref})
    return dict(world,primary=primary,supplement=supplementary,certificate=certificate,reference=ref,derived=derived_path,ledger=ledger,photo=photo)


def test_production_certificate_recomputes_native43_and_photo_link(chain43):
    c=chain43;result=subset.validate_reference(c['reference'],derived_model=c['derived'],sfm_run=c['sfm'],io_run=c['t01'])
    assert result['relation']['derived_points']==2393 and len(result['relation']['removed_observations'])==185
    link=photo_inputs.validate(c['photo'],c['bridge']/'scene.mvs',c['bridge'])
    assert len(link['frames'])==43 and link['subset_certificate']['excluded']['input_index']==59
    assert link['geometry_lineage']=='NOT_VERIFIED'


def test_real_consumer_cli43_preserves_both_original_no_match_rows(chain43,tmp_path):
    c=chain43;out=tmp_path/'converted'
    args=base.cli_args(c,c['primary'],out,'--mask-manifest',str(c['supplement']),'--selection-ledger',str(c['ledger']))
    assert base.cli.main(args)==0
    result=json.loads((out/'manifest.json').read_text())
    assert result['counts']==dict(expected=43,ok=43,missing=0,error=0,not_run=0)
    assert 59 not in {f['input_index'] for f in result['frames']}
    excluded=[f for f in result['mask_selection']['unselected_source_rows'] if f['input_index']==59]
    assert len(excluded)==2 and all(f['original_status']=='NO_MATCH' and f['replacement_manifest'] is None and f['subset_certificate']['sha256']==c['reference']['sha256'] for f in excluded)
    assert result['unbound_object_mask_frames']==[] and all(m['status']=='PARTIAL' for m in result['object_mask_manifests'])


@pytest.mark.parametrize('fault',['hash','wrong_parent','extra_exclusion','missing_exclusion','wrong_frame'])
def test_certificate_or_exclusion_corruption_refused(chain43,tmp_path,fault):
    c=chain43
    if fault=='hash':
        ref=dict(c['reference'],sha256='0'*64)
        with pytest.raises(ValueError):subset.validate_reference(ref)
    elif fault=='wrong_parent':
        with pytest.raises(ValueError):subset.validate_reference(c['reference'],clean_parent=c['derived'])
    else:
        ledger=json.loads(c['ledger'].read_text())
        if fault=='extra_exclusion':ledger['excluded_source_rows'].append(copy.deepcopy(ledger['excluded_source_rows'][0]))
        elif fault=='missing_exclusion':ledger['excluded_source_rows'].pop()
        else:ledger['excluded_source_rows'][0]['frame_id']='WRONG'
        save(c['ledger'],ledger)
        assert base.cli.main(base.cli_args(c,c['primary'],tmp_path/'reject','--mask-manifest',str(c['supplement']),'--selection-ledger',str(c['ledger'])))==2


def test_default_photo_and_default_selection_do_not_allow_subset(chain43,tmp_path):
    c=chain43;spec=json.loads(c['photo'].read_text());spec.pop('subset_certificate');save(c['photo'],spec)
    with pytest.raises(ValueError,match='identities differ'):photo_inputs.validate(c['photo'],c['bridge']/'scene.mvs',c['bridge'])
    ledger=json.loads(c['ledger'].read_text());ledger.pop('subset_certificate');ledger.pop('excluded_source_rows');save(c['ledger'],ledger)
    assert base.cli.main(base.cli_args(c,c['primary'],tmp_path/'reject','--mask-manifest',str(c['supplement']),'--selection-ledger',str(c['ledger'])))==2


def test_forged_self_declared_relation_not_authoritative(chain43):
    c=chain43;cert=json.loads(c['certificate'].read_text());cert['relation']['derived_points']=123;save(c['certificate'],cert)
    with pytest.raises(ValueError,match='independently recomputed'):subset.validate_reference({'path':str(c['certificate']),'sha256':sha(c['certificate'])})


def test_derive_cli_only_on_synthetic_native_fixture(chain43,tmp_path,monkeypatch):
    c=chain43
    spec=importlib.util.spec_from_file_location('derive_under_test',ROOT/'scripts/derive_native_subset.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    # Keep the CLI's real output-root rule active within this isolated fixture.
    monkeypatch.setattr(module,'ROOT',tmp_path)
    out=tmp_path/'outputs/TEST_SYNTHETIC_DERIVE'
    assert module.main(['--clean-parent',str(c['root']/'source_sparse'),'--sfm-run',str(c['sfm']),'--io-run',str(c['t01']),'--output',str(out)])==0
    cert=json.loads((out/'subset_certificate.json').read_text())
    assert cert['relation']['derived_registered_ids']==[i+1 for i in INDICES if i!=59]
    assert len(cert['relation']['removed_observations'])==185
    assert module.main(['--clean-parent',str(c['root']/'source_sparse'),'--sfm-run',str(c['sfm']),'--io-run',str(c['t01']),'--output',str(out)])==2


@pytest.mark.parametrize('fault',['exclude_ok','wrong_canonical','no_source_image','wrong59_frame'])
def test_cli_excluded_source_row_never_bypasses_normal_identity(chain43,tmp_path,fault):
    c=chain43;doc=json.loads(c['primary'].read_text());frame=next(v for v in doc['frames'] if v['input_index']==59)
    if fault=='exclude_ok':
        frame.update(status='OK',selected_candidate_id='c00',selected_mask={'path':'masks/000059_c00.png','sha256':sha(c['primary'].parent/'masks/000059_c00.png')},selection={'method':'unique_candidate','source':'automatic','mask_source':'automatic_sam3'},candidates=[{'candidate_id':'c00','mask_path':'masks/000059_c00.png','mask_sha256':sha(c['primary'].parent/'masks/000059_c00.png'),'foreground_pixels':int(np.count_nonzero(base._mask(59)))}])
        doc['counts']['success']=43;doc['counts']['no_match']=1
    elif fault=='wrong_canonical':frame['source_image_sha256']='0'*64
    elif fault=='no_source_image':frame['source_image']=None
    else:frame['frame_id']='WRONG_NATIVE59'
    save(c['primary'],doc)
    ledger=json.loads(c['ledger'].read_text());ledger['manifests'][0]['sha256']=sha(c['primary']);ledger['excluded_source_rows'][0]['manifest_sha256']=sha(c['primary']);save(c['ledger'],ledger)
    assert base.cli.main(base.cli_args(c,c['primary'],tmp_path/'reject','--mask-manifest',str(c['supplement']),'--selection-ledger',str(c['ledger'])))==2
