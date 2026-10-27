"""Coverage candidates tested with real CPU OpenCV images."""
import json
from pathlib import Path
import sys
from collections import Counter
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'outputs/toolchains/opencv-4.13.0.92'))
import cv2 as cv
from kozlov_lab.stereo.coverage_tracks import track_coverage, PROFILE


def texture():
    return cv.GaussianBlur(np.random.default_rng(12).integers(0,256,(240,320),dtype=np.uint8),(3,3),.6)


def test_real_translation_labels_counts_and_immutable():
    base=texture()
    moved=cv.warpAffine(base,np.float32([[1,0,2],[0,1,1]]),(320,240))
    mask=np.zeros_like(base);mask[:,:160]=255
    originals=[base.copy(),moved.copy(),mask.copy()]
    result=track_coverage([base,moved],[mask,None],cv,selection_scope='whole-frame')
    long=[t for t in result['tracks'] if len(t['observations'])==2]
    assert len(long)>100
    delta=[np.array(t['observations'][1]['xy'])-t['observations'][0]['xy'] for t in long]
    assert np.linalg.norm(np.median(delta,axis=0)-[2,1])<.1
    assert all(t['observations'][1]['label']=='unknown' for t in long)
    assert all(t['observations'][1]['fb_error_px'] is not None for t in long)
    assert {t['observations'][0]['label'] for t in result['tracks'] if t['seed_frame']==0}=={'foreground','background'}
    for frame,stats in enumerate(result['coverage']):
        obs=[o for t in result['tracks'] for o in t['observations'] if o['frame_index']==frame]
        assert stats['active']==len(obs)==sum(map(sum,stats['grid_counts']))
        assert Counter(o['label'] for o in obs)==Counter({k:stats[k] for k in ('foreground','background','unknown') if stats[k]})
    assert all(np.array_equal(a,b) for a,b in zip([base,moved,mask],originals))
    assert result['status']=='CANDIDATE_TRACKS_ONLY'
    json.dumps(result,allow_nan=False)


def test_grid_seeds_cover_all_cells_and_coordinate_convention():
    result=track_coverage([texture()],[None],cv,selection_scope='whole-frame')
    grid=result['coverage'][0]['grid_counts']
    assert all(0<n<=6 for row in grid for n in row)
    assert len(result['tracks'])<=576
    for t in result['tracks']:
        o=t['observations'][0]
        assert np.array_equal(np.asarray(o['xy'])-.5,np.rint(np.asarray(o['xy'])-.5))
        assert o['label']=='unknown' and o['fb_error_px'] is None and not o['strong']


def test_occlusion_failure_new_ids():
    base=texture();result=track_coverage([base,np.zeros_like(base),base],[None]*3,cv,selection_scope='whole-frame')
    first=[t for t in result['tracks'] if t['seed_frame']==0]
    restored=[t for t in result['tracks'] if t['seed_frame']==2]
    assert first and restored
    assert all(len(t['observations'])==1 for t in first)
    assert all(t['termination']['reason']!='SEQUENCE_END' for t in first)
    assert not {t['track_id'] for t in first}&{t['track_id'] for t in restored}


def test_permissive_candidates_are_distinct_from_strong():
    class MeasuredFlow:
        ocl=cv.ocl
        setNumThreads=staticmethod(cv.setNumThreads)
        setRNGSeed=staticmethod(cv.setRNGSeed)
        circle=staticmethod(cv.circle)
        goodFeaturesToTrack=staticmethod(cv.goodFeaturesToTrack)
        TERM_CRITERIA_EPS=cv.TERM_CRITERIA_EPS
        TERM_CRITERIA_COUNT=cv.TERM_CRITERIA_COUNT
        calls=0
        @classmethod
        def calcOpticalFlowPyrLK(cls,a,b,p,d,**kw):
            cls.calls+=1
            q,status,error=cv.calcOpticalFlowPyrLK(a,b,p,d,**kw)
            if cls.calls==1:
                error[:]=30 # Controlled measured LK output; still below loose gate.
            else:
                q[:,:,0]+=1.5 # Controlled real FB displacement, below loose gate.
            return q,status,error
    base=texture();result=track_coverage([base,base],[None,None],MeasuredFlow,selection_scope='whole-frame')
    obs=[t['observations'][1] for t in result['tracks'] if len(t['observations'])==2]
    assert obs and all(1<o['fb_error_px']<3 and o['lk_error']==30 and not o['strong'] for o in obs)


def test_validation_and_budgets(monkeypatch):
    base=texture()
    for grays,masks in [([],[]),([base],[]),([base.astype(float)],[None]),([base],[np.ones_like(base)]),([base,base[:10]],[None,None]),([np.zeros((1,1025),np.uint8)],[None])]:
        with pytest.raises(ValueError):track_coverage(grays,masks,cv,selection_scope='whole-frame')
    monkeypatch.setitem(PROFILE,'max_total_tracks',1)
    with pytest.raises(ValueError,match='budget'):track_coverage([base],[None],cv,selection_scope='whole-frame')
    monkeypatch.setitem(PROFILE,'max_active',769)
    with pytest.raises(ValueError,match='budget'):track_coverage([base],[None],cv,selection_scope='whole-frame')


def test_round_robin_small_active_budget(monkeypatch):
    monkeypatch.setitem(PROFILE,'max_active',96)
    result=track_coverage([texture()],[None],cv,selection_scope='whole-frame')
    grid=result['coverage'][0]['grid_counts']
    assert result['coverage'][0]['active']==96
    assert sum(n>0 for row in grid for n in row)>=80
    assert sum(map(sum,grid[6:]))>=35


def test_endpoint_conflict_keeps_deterministic_best_track():
    class ConflictFlow:
        ocl=cv.ocl
        setNumThreads=staticmethod(cv.setNumThreads)
        setRNGSeed=staticmethod(cv.setRNGSeed)
        circle=staticmethod(cv.circle)
        goodFeaturesToTrack=staticmethod(cv.goodFeaturesToTrack)
        TERM_CRITERIA_EPS=cv.TERM_CRITERIA_EPS
        TERM_CRITERIA_COUNT=cv.TERM_CRITERIA_COUNT
        calls=0
        before=None
        @classmethod
        def calcOpticalFlowPyrLK(cls,a,b,p,d,**kw):
            cls.calls+=1
            if cls.calls==1:
                cls.before=p.copy()
                q=p.copy();q[1]=q[0]
                return q,np.ones((len(p),1),np.uint8),np.full((len(p),1),10,np.float32)
            return cls.before.copy(),np.ones((len(p),1),np.uint8),np.zeros((len(p),1),np.float32)
    base=texture();result=track_coverage([base,base],[None,None],ConflictFlow,selection_scope='whole-frame')
    assert len(result['tracks'][0]['observations'])==2
    assert result['tracks'][1]['termination']['reason']=='ENDPOINT_CONFLICT'
    assert len(result['tracks'][1]['observations'])==1



@pytest.mark.parametrize('shape',[(97,131),(240,320),(513,769),(12,14)])
def test_roi_offsets_sizes_and_bounded_corner_work(shape):
    image=np.random.default_rng(3).integers(0,256,shape,dtype=np.uint8)
    class RoiCV:
        ocl=cv.ocl
        setNumThreads=staticmethod(cv.setNumThreads)
        setRNGSeed=staticmethod(cv.setRNGSeed)
        circle=staticmethod(cv.circle)
        TERM_CRITERIA_EPS=cv.TERM_CRITERIA_EPS
        TERM_CRITERIA_COUNT=cv.TERM_CRITERIA_COUNT
        pixels=0
        expected=set()
        @classmethod
        def goodFeaturesToTrack(cls,roi,**kwargs):
            cls.pixels+=roi.size
            assert roi.shape==kwargs['mask'].shape
            assert np.shares_memory(roi,image)
            offset=roi.__array_interface__['data'][0]-image.__array_interface__['data'][0]
            y0,x0=divmod(offset,image.strides[0])
            points=cv.goodFeaturesToTrack(roi,**kwargs)
            if points is not None:
                cls.expected.update((float(p[0])+x0,float(p[1])+y0) for p in points[:,0,:])
            return points
    result=track_coverage([image],[None],RoiCV,selection_scope='whole-frame')
    assert RoiCV.pixels < image.size*3
    if min(shape)<=16:
        assert not result['tracks']
        return
    assert sum(n>0 for row in result['coverage'][0]['grid_counts'] for n in row)>=60
    for t in result['tracks']:
        x,y=np.asarray(t['observations'][0]['xy'])-.5
        assert 8<=x<shape[1]-8 and 8<=y<shape[0]-8
        assert (x,y) in RoiCV.expected


def test_v2_migration_retires_budget_and_reseeds_vacated_cells():
    class MigrationCV:
        ocl=cv.ocl
        setNumThreads=staticmethod(cv.setNumThreads)
        setRNGSeed=staticmethod(cv.setRNGSeed)
        circle=staticmethod(cv.circle)
        goodFeaturesToTrack=staticmethod(cv.goodFeaturesToTrack)
        TERM_CRITERIA_EPS=cv.TERM_CRITERIA_EPS
        TERM_CRITERIA_COUNT=cv.TERM_CRITERIA_COUNT
        calls=0
        before=None
        @classmethod
        def calcOpticalFlowPyrLK(cls,a,b,p,d,**kw):
            cls.calls+=1
            if cls.calls%2:
                cls.before=p.copy();q=p.copy()
                if cls.calls==1:
                    for n in range(min(24,len(q))):
                        q[n,0]=[245+(n%8)*4,205+(n//8)*4]
                else:
                    selected=(q[:,0,0]>=240)&(q[:,0,0]<320)&(q[:,0,1]>=200)&(q[:,0,1]<240)
                    q[selected,0,0]+=80
                return q,np.ones((len(p),1),np.uint8),np.full((len(p),1),10,np.float32)
            return cls.before.copy(),np.ones((len(p),1),np.uint8),np.zeros((len(p),1),np.float32)
    image=cv.GaussianBlur(np.random.default_rng(2).integers(0,256,(480,640),dtype=np.uint8),(3,3),.6)
    result=track_coverage([image]*3,[None]*3,MigrationCV,selection_scope='whole-frame')
    assert result['profile']['version']=='coverage-grid-lk-v3'
    retired=[t for t in result['tracks'] if t['termination']['reason']=='GRID_BUDGET_RETIRED']
    assert retired and result['coverage'][1]['grid_budget_retired']>0
    assert len(retired)==sum(c['grid_budget_retired'] for c in result['coverage'])
    assert all(t['termination']['fb_error_px']==0 for t in retired)
    assert all(t['termination']['lk_error']==10 for t in retired)
    # Frame 2 moves the crowded cell away, so its corner-rich vacancy is replanted.
    restored=[t for t in result['tracks'] if t['seed_frame']==2 and
        240<=t['observations'][0]['xy'][0]-.5<320 and 200<=t['observations'][0]['xy'][1]-.5<240]
    assert restored
    for c in result['coverage']:
        assert max(n for row in c['grid_counts'] for n in row)<=8
        assert c['active']<=768
    for frame,c in enumerate(result['coverage']):
        assert c['active']==sum(o['frame_index']==frame for t in result['tracks'] for o in t['observations'])


def test_v2_stationary_survivors_cannot_consume_other_cell_budgets():
    image=cv.GaussianBlur(np.random.default_rng(17).integers(0,256,(480,640),dtype=np.uint8),(3,3),.6)
    result=track_coverage([image]*5,[None]*5,cv,selection_scope='whole-frame')
    assert result['coverage'][1]['seeded']>0
    assert result['coverage'][-1]['active']>result['coverage'][0]['active']
    for c in result['coverage']:
        assert max(n for row in c['grid_counts'] for n in row)<=8
        assert sum(n>0 for row in c['grid_counts'] for n in row)==96


def assert_object_observations(result, masks):
    for frame, stats in enumerate(result['coverage']):
        observations=[o for t in result['tracks'] for o in t['observations'] if o['frame_index']==frame]
        assert stats['active']==len(observations)==sum(map(sum,stats['grid_counts']))
        assert max(n for row in stats['grid_counts'] for n in row)<=8
        assert stats['background']==stats['unknown']==0
        for o in observations:
            x,y=np.floor(o['xy']).astype(int)
            assert masks[frame] is not None and masks[frame][y,x]==255
            assert o['label']=='foreground'
        mask=masks[frame]
        if mask is not None and mask.any():
            ys,xs=np.nonzero(mask)
            assert stats['grid_bounds']==[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)]


def test_object_default_tracks_moving_mask_and_holes():
    base=texture();mask=np.zeros_like(base);mask[30:210,60:240]=255;mask[80:150,115:180]=0
    moved=cv.warpAffine(base,np.float32([[1,0,2],[0,1,1]]),(320,240))
    moved_mask=cv.warpAffine(mask,np.float32([[1,0,2],[0,1,1]]),(320,240))
    before=[base.copy(),moved.copy(),mask.copy(),moved_mask.copy()]
    result=track_coverage([base,moved],[mask,moved_mask],cv)
    assert result['profile']['selection_scope']=='object'
    assert result['profile']['version']=='coverage-grid-lk-v3'
    assert_object_observations(result,[mask,moved_mask])
    long=[t for t in result['tracks'] if len(t['observations'])==2]
    assert len(long)>50
    delta=[np.asarray(t['observations'][1]['xy'])-t['observations'][0]['xy'] for t in long]
    assert np.linalg.norm(np.median(delta,axis=0)-[2,1])<.1
    assert all(np.array_equal(a,b) for a,b in zip(before,[base,moved,mask,moved_mask]))
    json.dumps(result,allow_nan=False)


@pytest.mark.parametrize('missing,reason',[(None,'MASK_UNAVAILABLE'),('empty','EMPTY_FOREGROUND')])
def test_object_mask_gap_ends_tracks_without_flow_or_bridge(missing,reason):
    base=texture();mask=np.zeros_like(base);mask[20:220,30:290]=255
    gap=np.zeros_like(base) if missing=='empty' else None
    class NoFlow:
        def __getattr__(self,name):return getattr(cv,name)
        def calcOpticalFlowPyrLK(self,*a,**k):raise AssertionError('unknown or empty frame must end tracks before LK')
    result=track_coverage([base]*3,[mask,gap,mask],NoFlow())
    first=[t for t in result['tracks'] if t['seed_frame']==0]
    restored=[t for t in result['tracks'] if t['seed_frame']==2]
    assert first and restored
    assert all(len(t['observations'])==1 and t['termination']['reason']==reason for t in first)
    assert result['coverage'][1]['active']==result['coverage'][1]['seeded']==0
    assert result['coverage'][1]['selection_status']==reason
    assert all(o['frame_index']!=1 for t in result['tracks'] for o in t['observations'])
    assert not {t['track_id'] for t in first}&{t['track_id'] for t in restored}
    assert_object_observations(result,[mask,gap,mask])


@pytest.mark.parametrize('height,width',[(1,1),(5,3),(13,9)])
def test_object_small_bbox_uses_nonempty_integer_cells(height,width):
    base=texture();mask=np.zeros_like(base);mask[60:60+height,80:80+width]=255
    result=track_coverage([base],[mask],cv)
    assert_object_observations(result,[mask])
    stats=result['coverage'][0]
    assert stats['grid_bounds']==[80,60,80+width,60+height]
    assert len(stats['grid_counts'])==12 and all(len(row)==8 for row in stats['grid_counts'])
    for row,cells in enumerate(stats['grid_counts']):
        for col,count in enumerate(cells):
            if (col*width//8==(col+1)*width//8 or row*height//12==(row+1)*height//12):
                assert count==0


def test_object_rejects_left_foreground_before_conflict():
    base=texture();mask=np.zeros_like(base);mask[30:210,40:280]=255
    class OutsideFlow:
        calls=0;before=None
        def __getattr__(self,name):return getattr(cv,name)
        def calcOpticalFlowPyrLK(self,a,b,p,d,**kwargs):
            self.calls+=1
            if self.calls%2:
                self.before=p.copy();q=p.copy();q[0,0]=[10,10]
                return q,np.ones((len(p),1),np.uint8),np.full((len(p),1),30,np.float32)
            return p.copy(),np.ones((len(p),1),np.uint8),np.zeros((len(p),1),np.float32)
    result=track_coverage([base,base],[mask,mask],OutsideFlow())
    assert result['tracks'][0]['termination']['reason']=='LEFT_FOREGROUND'
    assert len(result['tracks'][0]['observations'])==1
    assert_object_observations(result,[mask,mask])


def test_object_keeps_permissive_flow_candidates():
    base=texture();mask=np.zeros_like(base);mask[30:210,40:280]=255
    class WeakFlow:
        calls=0;before=None
        def __getattr__(self,name):return getattr(cv,name)
        def calcOpticalFlowPyrLK(self,a,b,p,d,**kwargs):
            self.calls+=1
            if self.calls%2:
                self.before=p.copy();q=p.copy();error=30
            else:
                q=self.before.copy();q[:,:,0]+=1.5;error=0
            return q,np.ones((len(p),1),np.uint8),np.full((len(p),1),error,np.float32)
    result=track_coverage([base,base],[mask,mask],WeakFlow())
    measured=[o for t in result['tracks'] for o in t['observations'] if o['fb_error_px'] is not None]
    assert measured and all(o['fb_error_px']==1.5 and o['lk_error']==30 and not o['strong'] for o in measured)
    assert_object_observations(result,[mask,mask])


def test_invalid_selection_scope_is_rejected():
    with pytest.raises(ValueError,match='scope'):
        track_coverage([texture()],[None],cv,selection_scope='invalid')
