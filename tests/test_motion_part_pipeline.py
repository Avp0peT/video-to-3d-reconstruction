import json
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
from integrations.video_masks import motion_part_pipeline as p


def fixture(root, n=4):
    root.mkdir()
    rows=[]
    for i in range(n):
        im=root/f'{i}.jpg'; mask=root/f'{i}.png'
        Image.new('RGB',(40,32),(130,130,130)).save(im)
        m=np.zeros((32,40),np.uint8); m[5:28,4:36]=255; m[12:17,12:17]=0
        Image.fromarray(m).save(mask)
        rows.append(dict(frame_id=f'original-{i*3}',time_seconds=i*.2,
            image=dict(path=str(im),sha256=p.sha(im)),
            mask=dict(path=str(mask),sha256=p.sha(mask))))
    manifest=root/'input.json'
    p.save(manifest,dict(schema='motion-part-input-v1',object_id='test-object',frames=rows))
    return manifest


def test_generic_windows_cover_all_without_fixed_frame_ids():
    for n in (1,2,3,6,7,12,31):
        w=list(p.windows(n,6))
        assert sorted(set(sum(w,[])))==list(range(n))
        for a,b in zip(w,w[1:]): assert a[-2:]==b[:2]


def test_real_cpu_pipeline_retains_low_texture_object_and_hole(tmp_path):
    source=fixture(tmp_path/'source',7)
    out=tmp_path/'run'
    result=p.run(source,out)
    assert result['native_executions']==result['neural_inferences']==0
    assert result['window_count']==2 and not result['geometry_accepted']
    assert result['cross_window_identity']=='NOT_ESTABLISHED'
    rows=json.loads(source.read_text())['frames']
    for window in result['windows']:
        for r in window['frames']:
            original=rows[r['input_position']]['mask']
            assert Path(r['whole']['path']).read_bytes()==Path(original['path']).read_bytes()
            assert r['coverage_loss_pixels']==r['added_background_pixels']==0
            assert not r['parts']
            with Image.open(r['fallback']['path']) as im: m=np.asarray(im)
            assert m[13,13]==0
    assert (out/'index.html').exists()


@pytest.mark.parametrize('bad',['hash','duplicate','time','shape','labels'])
def test_bad_inputs_refused_before_output(tmp_path,bad):
    source=fixture(tmp_path/'source')
    doc=json.loads(source.read_text()); rows=doc['frames']
    if bad=='hash': rows[0]['mask']['sha256']='0'*64
    if bad=='duplicate': rows[1]['frame_id']=rows[0]['frame_id']
    if bad=='time': rows[1]['time_seconds']=True
    if bad in ('shape','labels'):
        mask=Path(rows[1]['mask']['path'])
        Image.new('L',(30,30) if bad=='shape' else (40,32),255 if bad=='shape' else 1).save(mask)
        rows[1]['mask']['sha256']=p.sha(mask)
    p.save(source,doc); out=tmp_path/'bad'
    with pytest.raises(ValueError): p.run(source,out)
    assert not out.exists()


def test_missing_mask_is_not_silently_skipped(tmp_path):
    source=fixture(tmp_path/'source')
    doc=json.loads(source.read_text()); doc['frames'][0].pop('mask'); p.save(source,doc)
    with pytest.raises(ValueError): p.run(source,tmp_path/'bad')


def test_strict_duplicate_key_and_bool_window(tmp_path):
    f=tmp_path/'bad.json'; f.write_text('{"schema":1,"schema":2}')
    with pytest.raises(ValueError,match='duplicate'): p.strict_json(f)
    with pytest.raises(ValueError): list(p.windows(5,True))


@pytest.mark.parametrize('shape',['round','ring','branched','two_components'])
def test_category_free_shapes_retain_entire_domain_and_holes(shape):
    yy,xx=np.mgrid[:96,:120]
    if shape=='round': mask=(xx-60)**2+(yy-48)**2<30**2
    elif shape=='ring':
        d=(xx-60)**2+(yy-48)**2
        mask=(d<35**2)&(d>17**2)
    elif shape=='branched':
        mask=((abs(xx-60)<7)&(yy>20)&(yy<85))|((abs(yy-30)<6)&(xx>15)&(xx<105))
    else: mask=((xx-25)**2+(yy-48)**2<14**2)|((xx-90)**2+(yy-48)**2<18**2)
    rgb=np.zeros((96,120,3),np.uint8)
    r=p.propose_motion_parts([rgb]*3,[mask]*3)
    assert not r['part_masks']
    for whole,fallback,unknown in zip(r['global_masks'],r['fallback_masks'],r['unknown_masks']):
        assert np.array_equal(whole,mask)
        assert np.array_equal(fallback,mask)
        assert np.array_equal(unknown,mask)
