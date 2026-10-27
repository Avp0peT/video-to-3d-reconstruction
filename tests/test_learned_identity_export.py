import copy
import json
import numpy as np
import pytest
from PIL import Image
from integrations.demo_pipeline import learned_identity_export as export

pc=pytest.importorskip('pycolmap')


def fixture(tmp_path):
    native=tmp_path/'native';native.mkdir();images=native/'images';images.mkdir()
    rec=pc.Reconstruction()
    camera=pc.Camera(model='PINHOLE',width=64,height=48,params=np.array([50.,50.,32.,24.]),camera_id=1)
    rec.add_camera_with_trivial_rig(camera)
    xyzs=[np.array([0.,0.,5.]),np.array([.2,.1,5.])]
    frames=[]; records=[dict(query_id=i,xyz=x.tolist(),status='FITTED_NOT_IDENTITY_VERIFIED',observations=[]) for i,x in enumerate(xyzs)]
    for iid in range(1,6):
        pose=pc.Rigid3d(pc.Rotation3d(),np.array([-iid*.1,0.,0.]))
        points=pc.Point2DList()
        for row,xyz in enumerate(xyzs):
            xy=camera.img_from_cam(pose*xyz)
            if row==1 and iid==5: xy=xy+[10.,0.]
            points.append(pc.Point2D(xy=xy))
            records[row]['observations'].append(dict(image_id=iid,feature_row=row,xy=xy.tolist(),
                split='heldout' if iid==5 else 'train',positive_depth=True,
                within_threshold=not (row==1 and iid==5)))
        name=f'frame_{iid:06d}.png'
        rec.add_image_with_trivial_frame(pc.Image(name=name,camera_id=1,image_id=iid,points2D=points),pose)
        Image.new('RGB',(64,48),(iid*10,20,30)).save(images/name)
        frames.append(dict(database_image_id=iid,image_name=name,track_ids=[0,1],
                           input_index=iid-1,decode_index=100+iid,frame_id=f'raw_decode_{100+iid}'))
    old=pc.Track()
    old.add_element(1,0);old.add_element(2,0)
    rec.add_point3D(np.array([99.,88.,77.]),old,np.array([1,2,3],dtype=np.uint8))
    binary=native/'models'/'3'/'binary';binary.mkdir(parents=True);rec.write_binary(str(binary))
    masks=tmp_path/'MASKS.json';masks.write_text('{"test_data":true}')
    (native/'database.db').write_bytes(b'TEST_DATA synthetic binding only')
    plan=dict(frames=frames,source_bindings=[export._hash(masks)])
    (native/'PLAN.json').write_text(json.dumps(plan))
    (native/'RESULT.json').write_text(json.dumps(dict(models=[dict(model_id=3,binary_path=str(binary),registered_views=5)])))
    identity=dict(schema='learned-identity-geometry-v2',model_id=3,registered_image_ids=list(range(1,6)),
        heldout_image_ids=[5],records=records,physical_identity_verified=False,geometry_accepted=False,
        conditional_holdout='Synthetic conditional camera holdout')
    identity['source_bindings']=[export._hash(p) for p in [native/'PLAN.json',native/'RESULT.json',native/'database.db',*sorted(binary.iterdir())]]
    identity_path=tmp_path/'identity.json';identity_path.write_text(json.dumps(identity))
    return rec,identity,plan,images,identity_path,native


def test_native_export_roundtrip_and_real_color(tmp_path):
    original,identity,plan,images,identity_path,native=fixture(tmp_path)
    before=export._hash(native/'PLAN.json')
    result=export.worker(identity_path,native,tmp_path/'export')
    assert export.infer_native_root(identity_path)==native
    assert result['status']=='IDENTITY_CONSTRAINED_EXPERIMENTAL'
    assert result['registered_views']==5 and result['points']==2
    assert result['binary_readback']['exact_xyz_rgb_tracks_and_reciprocity']
    assert result['text_readback']['exact_xyz_rgb_tracks_and_reciprocity']
    assert result['source_unchanged'] and export._hash(native/'PLAN.json')==before
    records=result['query_native_point_map']
    assert records[0]['color_rgb']==[30,20,30]
    assert records[1]['color_rgb']==[25,20,30]
    assert len(records[0]['observations'])==5 and len(records[1]['observations'])==4
    assert records[0]['observations'][-1]['split']=='heldout'
    assert all(export._hash(b['path'])==b for b in result['final_bindings'])
    assert result['images'][0]['decode_index']==101
    assert result['mask_manifest_paths']==[str(tmp_path/'MASKS.json')]
    assert any(b['path']==str(tmp_path/'MASKS.json') for b in result['source_bindings'])
    assert len(original.points3D)==1 and np.array_equal(next(iter(original.points3D.values())).xyz,[99.,88.,77.])


@pytest.mark.parametrize('fault',['duplicate_query','duplicate_view','xy','query_row','train_support','source_drift'])
def test_rejects_identity_contract_breaks(tmp_path,fault):
    rec,identity,plan,images,identity_path,native=fixture(tmp_path)
    if fault=='duplicate_query': identity['records'].append(copy.deepcopy(identity['records'][0]))
    elif fault=='duplicate_view': identity['records'][0]['observations'].append(copy.deepcopy(identity['records'][0]['observations'][0]))
    elif fault=='xy': identity['records'][0]['observations'][0]['xy'][0]+=.5
    elif fault=='query_row': plan['frames'][0]['track_ids']=[1,0]
    elif fault=='train_support':
        identity['records'][0]['observations']=identity['records'][0]['observations'][:2]
    elif fault=='source_drift':
        (native/'PLAN.json').write_text('changed')
        with pytest.raises(ValueError,match='binding changed'):
            export.worker(identity_path,native,tmp_path/'export')
        return
    with pytest.raises(ValueError): export.derive(rec,identity,plan,images,pc)
