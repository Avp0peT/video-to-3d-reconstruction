"""Bounded synthetic known-projection integration plus pure input counterexamples."""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
from integrations.demo_pipeline import balanced_native as b


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def fixture(tmp_path,n=10):
    rng=np.random.default_rng(11);world=np.column_stack((rng.uniform(-.8,.8,90),rng.uniform(-.6,.6,90),rng.uniform(4.5,7.,90)))
    frames=[];w,h=640,480;f=1.2*w
    for i in range(n):
        image=tmp_path/f'image{i}.png';mask=tmp_path/f'mask{i}.png'
        Image.new('RGB',(w,h),(70,80,90)).save(image);Image.new('L',(w,h),255).save(mask)
        center=np.array([(i-(n-1)/2)*.20,.08*np.sin(i*.6),0.])
        camera=world-center;xy=f*camera[:,:2]/camera[:,2,None]+[w/2,h/2]-.5
        frames.append(dict(input_index=i*2,decode_index=i*7,frame_id=f'TEST_DATA_{i}',image_path=str(image),image_sha256=digest(image),
            mask_path=str(mask),mask_sha256=digest(mask),width=w,height=h,original_mask_status='OK',
            observations=[dict(track_id=int(t),xy=p.tolist()) for t,p in enumerate(xy)]))
    path=tmp_path/'TRACKS.json';path.write_text(json.dumps(dict(schema=b.SCHEMA,arm='balanced',frames=frames)),encoding='utf-8')
    return path


@pytest.mark.parametrize('fault',['bool_id','nan_xy','duplicate_track','source_drift','duplicate_frame','mask_status'])
def test_input_counterexamples(tmp_path,fault):
    path=fixture(tmp_path,3);doc=json.loads(path.read_text());f=doc['frames'][0]
    if fault=='bool_id':f['observations'][0]['track_id']=True
    elif fault=='nan_xy':f['observations'][0]['xy'][0]=float('nan')
    elif fault=='duplicate_track':f['observations'].append(f['observations'][0])
    elif fault=='source_drift':Path(f['image_path']).write_bytes(b'changed')
    elif fault=='duplicate_frame':doc['frames'][1]['frame_id']=f['frame_id']
    else:f['original_mask_status']='NO_MATCH'
    path.write_text(json.dumps(doc),encoding='utf-8')
    with pytest.raises(ValueError):b.prepare(path)


@pytest.mark.parametrize('arm',['balanced','sift_control'])
def test_track_gate_zero_rows_and_sequence_distance(tmp_path,arm):
    path=fixture(tmp_path,5);doc=json.loads(path.read_text())
    doc['arm']=arm;doc['frames'][-1]['observations']=[dict(track_id=900,xy=[100.,100.])]
    path.write_text(json.dumps(doc),encoding='utf-8');plan=b.prepare(path)
    assert len(plan['frames'])==5 and plan['frames'][-1]['keypoints'].shape==(0,2)
    assert all(p['sequence_b']-p['sequence_a']<=3 for p in plan['pairs'])
    assert np.array_equal(plan['frames'][0]['keypoints'],np.array([o['xy'] for o in doc['frames'][0]['observations']],np.float32)+.5)


def test_known_projection_no_descriptors_native_roundtrip(tmp_path,monkeypatch):
    for key in b.THREAD_ENV:monkeypatch.setenv(key,'2')
    path=fixture(tmp_path);sources={str(p):digest(p) for p in tmp_path.glob('*.png')}
    result=b.run(path,tmp_path/'native')
    assert result['status']=='MODELS_GENERATED_NOT_ACCEPTED'
    assert result['verification_calls']==result['mapping_calls']==1
    assert result['unique_registered_count']==10
    assert len(result['models'])==1 and result['models'][0]['points']==90
    assert result['models'][0]['reprojection_error_px']['max']<.001
    assert result['descriptors_written'] is False and result['source_unchanged'] is True
    assert all(m['binary_text_roundtrip_verified'] for m in result['models'])
    assert all(digest(p)==h for p,h in sources.items())
