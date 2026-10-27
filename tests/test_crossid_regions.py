"""CPU region appearance tests, never native geometry or physical acceptance."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image
from integrations.demo_pipeline.temporal_review import load_cv
from integrations.demo_pipeline import crossid_regions as c
cv=load_cv([])
if cv is None:raise RuntimeError('bound CPU OpenCV required')
cv.setNumThreads(2);cv.ocl.setUseOpenCL(False)


def fixture(root,kind='translation',shared=False):
    rng=np.random.default_rng(14020)
    raw=cv.GaussianBlur(rng.integers(30,190,(256,256),dtype=np.uint8),(3,3),.65)
    positions=np.float32([[x,y] for y in range(48,210,24) for x in range(48,210,24)])
    if kind=='rotation':
        matrix=cv.getRotationMatrix2D((128,128),90,1)
        target=cv.warpAffine(raw,matrix,(256,256));mapped=positions@matrix[:,:2].T+matrix[:,2]
    elif kind=='unrelated':
        target=cv.GaussianBlur(rng.integers(30,190,(256,256),dtype=np.uint8),(3,3),.65);mapped=positions.copy()
    elif kind=='blank':target=raw=np.full((256,256),128,np.uint8);mapped=positions.copy()
    elif kind=='repeat':
        tile=rng.integers(30,190,(24,24),dtype=np.uint8);raw=np.tile(tile,(11,11))[:256,:256].copy();target=raw.copy();mapped=positions.copy()
    else:
        target=cv.warpAffine(raw,np.float32([[1,0,3],[0,1,2]]),(256,256))
        target=np.clip(target.astype(np.int16)+20,0,255).astype(np.uint8);mapped=positions+[3,2]
    frames=[]
    for i in range(4):
        image=root/f'im{i}.png';mask=root/f'mask{i}.png';gray=raw if i<2 else target
        Image.fromarray(np.repeat(gray[:,:,None],3,axis=2)).save(image);Image.new('L',(256,256),255).save(mask)
        points=positions if i<2 else mapped
        observations=[dict(track_id=(j if i<2 or (shared and j<2) else j+1000),xy=list(map(float,p)),measured=True) for j,p in enumerate(points)]
        frames.append(dict(input_index=i,decode_index=i*7,frame_id=f'IMAGE_{i}',width=256,height=256,
            original_mask_status='OK',image_path=str(image),image_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
            mask_path=str(mask),mask_sha256=hashlib.sha256(mask.read_bytes()).hexdigest(),observations=observations))
    tracks=root/'TRACKS.json';tracks.write_text(json.dumps(dict(schema=c.native.SCHEMA,arm='balanced',frames=frames,predicted_observations=0)),encoding='utf8')
    return tracks,frames


class CrossIdTests(unittest.TestCase):
    def measure(self,kind,shared=False):
        folder=tempfile.TemporaryDirectory();self.addCleanup(folder.cleanup);root=Path(folder.name)
        tracks,frames=fixture(root,kind,shared);before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir()}
        result=c.run(tracks,root/'result')
        self.assertTrue(result['source_unchanged']);self.assertFalse(result['geometry_verified'])
        self.assertEqual(before,{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir() if p.is_file()})
        self.assertEqual(len(result['pairs']),4)
        self.assertTrue(all(p['component_a']!=p['component_b'] for p in result['pairs']))
        self.assertTrue(all((root/'result'/p['visualization']).is_file() for p in result['pairs']))
        return root,result

    def test_translation_brightness_real_correspondence_and_no_id_fusion(self):
        _,result=self.measure('translation')
        pair=result['pairs'][0];self.assertGreaterEqual(len(pair['matches']),10)
        for m in pair['matches']:
            self.assertEqual(m['track_id_b'],m['track_id_a']+1000)
            np.testing.assert_allclose(np.subtract(m['evidence']['xy_b'],m['evidence']['xy_a']),[3,2])
            self.assertLess(m['evidence']['ratio_a'],.8);self.assertLess(m['evidence']['ratio_b'],.8)
        self.assertEqual(result['id_merges'],0)

    def test_rotation_with_measured_orientation(self):
        _,result=self.measure('rotation')
        matches=result['pairs'][0]['matches'];self.assertGreaterEqual(len(matches),8)
        self.assertTrue(all(m['track_id_b']==m['track_id_a']+1000 for m in matches))

    def test_unrelated_repeated_and_blank_are_not_invented_matches(self):
        for kind in ('unrelated','repeat','blank'):
            with self.subTest(kind=kind):
                root,result=self.measure(kind)
                self.assertLessEqual(sum(len(p['matches']) for p in result['pairs']),2 if kind=='unrelated' else 0)
                diag=json.loads((root/'result/FRAME_DIAGNOSTICS.json').read_text('utf8'))
                if kind=='blank':self.assertTrue(all(f['descriptor_valid']==0 for f in diag['frames']))

    def test_same_id_endpoints_reserved_even_when_cross_id_descriptor_agrees(self):
        _,result=self.measure('translation',True)
        self.assertTrue(all(p['reserved_same_id_count']==2 for p in result['pairs']))
        self.assertTrue(all(m['track_id_a'] not in (0,1) and m['track_id_b'] not in (0,1) for p in result['pairs'] for m in p['matches']))

    def test_mask_support_omissions_are_not_labelled_unmatched(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);path,frames=fixture(root)
            doc=json.loads(path.read_text('utf8'))
            for f in doc['frames']:
                mask=np.zeros((256,256),np.uint8);mask[40:220,40:220]=255
                Image.fromarray(mask).save(f['mask_path']);f['mask_sha256']=hashlib.sha256(Path(f['mask_path']).read_bytes()).hexdigest()
            path.write_text(json.dumps(doc));c.run(path,root/'result')
            diag=json.loads((root/'result/FRAME_DIAGNOSTICS.json').read_text('utf8'))
            self.assertTrue(any(f['support_valid']<f['eligible'] for f in diag['frames']))
            self.assertTrue(any('SUPPORT_NOT_FULLY_IN_ORIGINAL_MASK' in f['reasons'] for f in diag['frames']))

    def test_source_binding_fault_and_fresh_output_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);path,frames=fixture(root);Path(frames[0]['image_path']).write_bytes(b'drift')
            with self.assertRaises(ValueError):c.run(path,root/'result')
            existing=root/'existing';existing.mkdir()
            with self.assertRaises(ValueError):c.run(path,existing)

    def test_one_real_sift_compute_per_multikey_frame(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);path,_=fixture(root)
            frame=c.native.prepare(path,'all_shared_tracks')['frames'][0]
            calls=[]
            class SiftSpy:
                def __init__(self):self.inner=cv.SIFT_create()
                def compute(self,image,keys):
                    calls.append(len(keys));return self.inner.compute(image,keys)
            class CvSpy:
                def __getattr__(self,key):return getattr(cv,key)
                def SIFT_create(self,**kwargs):return SiftSpy()
            diag,rows,desc=c.describe(frame,CvSpy())
            self.assertEqual(len(calls),1);self.assertGreater(calls[0],10)
            self.assertEqual(diag['descriptor_valid'],len(rows));self.assertEqual(desc.shape,(len(rows),128))


if __name__=='__main__':unittest.main()
