"""Measured-image region proposals between existing immutable feature rows.

No new coordinates, geometry, propagated masks or track-ID unions are created.
SIFT descriptors are computed on raw grayscale, never on masked-out images.
"""
from dataclasses import asdict, dataclass
from collections import Counter
from pathlib import Path
import math
import time
import numpy as np
from PIL import Image, ImageDraw
from . import connectivity_native as native
from . import connectivity_graph, temporal_review
from .connectivity_graph import components
from .temporal_review import load_cv


@dataclass(frozen=True)
class Profile:
    size: float = 4.0
    orientation_radius: int = 9
    support_radius: int = 31
    ratio: float = .8
    max_sequence_distance: int = 3
    minimum_graph_shared_ids: int = 15
    max_rows_per_frame: int = 20000
    seconds: float = 120.

    def __post_init__(self):
        # One frozen experiment, not an unrecorded parameter sweep.
        if (self.size,self.orientation_radius,self.support_radius,self.ratio,
            self.max_sequence_distance,self.minimum_graph_shared_ids,self.max_rows_per_frame)!=(4.,9,31,.8,3,15,20000):
            raise ValueError('fixed region profile required')
        if not math.isfinite(self.seconds) or not 0<self.seconds<=180: raise ValueError('seconds')


def _direction(gray, point, profile):
    """Measured 36-bin gradient mode; CV clockwise degrees, never angle=-1."""
    x,y=point; radius=profile.orientation_radius
    # Orientation is measured at a rounded centre only; immutable keypoint
    # coordinates remain unrounded for descriptor compute and export.
    ix,iy=np.floor(np.asarray(point)+.5).astype(int)
    patch=gray[iy-radius-1:iy+radius+2,ix-radius-1:ix+radius+2].astype(float)
    dx=(patch[1:-1,2:]-patch[1:-1,:-2])*.5
    dy=(patch[2:,1:-1]-patch[:-2,1:-1])*.5
    yy,xx=np.mgrid[-radius:radius+1,-radius:radius+1]
    sigma=1.5*profile.size/2
    weights=np.exp(-(xx*xx+yy*yy)/(2*sigma*sigma))*np.hypot(dx,dy)
    bins=((np.degrees(np.arctan2(dy,dx))%360)/10).astype(int)%36
    hist=np.bincount(bins.ravel(),weights=weights.ravel(),minlength=36)
    for _ in range(2):hist=(np.roll(hist,1)+2*hist+np.roll(hist,-1))/4
    peak=int(np.argmax(hist))
    if float(hist[peak])<=1e-6:return None,'orientation_zero_texture'
    alternative=hist.copy()
    for offset in range(-2,3):alternative[(peak+offset)%36]=0
    if float(alternative.max())>=.9*float(hist[peak]):return None,'orientation_ambiguous'
    left,mid,right=hist[(peak-1)%36],hist[peak],hist[(peak+1)%36]
    denom=left-2*mid+right
    shift=.5*(left-right)/denom if abs(denom)>1e-12 else 0.
    return float(((peak+.5+np.clip(shift,-.5,.5))*10)%360),'measured'


def describe(frame, cv, profile=Profile()):
    """Return descriptors plus explicit support/texture omissions for every row."""
    with Image.open(frame['image_path']) as image:rgb=np.asarray(image.convert('RGB'))
    gray=cv.cvtColor(rgb,cv.COLOR_RGB2GRAY)
    mask=frame['mask'];h,w=gray.shape;radius=profile.support_radius
    if mask.shape!=gray.shape:raise ValueError('mask/image shape differs')
    sift=cv.SIFT_create(nOctaveLayers=3,sigma=1.6); accepted=[]; descriptors=[]; rows=[];support_valid=0;keys_to_compute=[]
    for row,tid in enumerate(frame['track_ids']):
        # Source prepare stores float32 COLMAP rows. Read original float64 CV
        # coordinates from validated observations to avoid a needless roundtrip.
        xy=np.asarray(frame['observations'][tid],float)
        ix,iy=np.floor(xy+.5).astype(int)
        detail=dict(feature_row=row,track_id=int(tid),xy=xy.tolist(),status='UNEXAMINED')
        if ix-radius<0 or iy-radius<0 or ix+radius>=w or iy+radius>=h:
            detail['status']='SUPPORT_OUTSIDE_IMAGE';rows.append(detail);continue
        if not bool(np.all(mask[iy-radius:iy+radius+1,ix-radius:ix+radius+1])):
            detail['status']='SUPPORT_NOT_FULLY_IN_ORIGINAL_MASK';rows.append(detail);continue
        support_valid+=1
        angle,reason=_direction(gray,xy,profile)
        if angle is None:detail['status']=reason.upper();rows.append(detail);continue
        key=cv.KeyPoint(float(xy[0]),float(xy[1]),profile.size,angle,0.,0,row)
        detail.update(status='DESCRIPTOR_MISSING',orientation_degrees=angle,sift_size=profile.size)
        rows.append(detail);keys_to_compute.append(key)
    if keys_to_compute:
        keys,computed=sift.compute(gray,keys_to_compute)
        if computed is None:
            if len(keys):raise ValueError('SIFT key/descriptor output mismatch')
        else:
            if computed.shape!=(len(keys),128):raise ValueError('SIFT descriptor output shape')
            seen=set();requested={k.class_id for k in keys_to_compute}
            for key,vec in zip(keys,computed):
                row=key.class_id
                if row not in requested or row in seen:raise ValueError('SIFT row identity changed/duplicated')
                seen.add(row);detail=rows[row]
                if not np.allclose(key.pt,detail['xy'],rtol=0,atol=1e-4):raise ValueError('SIFT modified immutable coordinate')
                norm=float(np.linalg.norm(vec))
                if not np.isfinite(vec).all():detail['status']='DESCRIPTOR_INVALID';continue
                if norm<=1e-12:detail['status']='DESCRIPTOR_ZERO_TEXTURE';continue
                detail['status']='DESCRIPTOR_VALID';accepted.append(detail);descriptors.append(vec.astype(np.float32)/norm)
    return dict(sequence_index=frame['sequence_index'],input_index=frame['input_index'],
        frame_id=frame['frame_id'],eligible=len(rows),support_valid=support_valid,
        descriptor_valid=len(accepted),reasons=dict(Counter(r['status'] for r in rows)),rows=rows),accepted,np.asarray(descriptors,np.float32).reshape(-1,128)


def match_descriptors(rows_a,desc_a,rows_b,desc_b,reserved,ratio=.8):
    """Bidirectional two-neighbour ratio and mutual unique row assignment."""
    ia=[i for i,r in enumerate(rows_a) if r['track_id'] not in reserved]
    ib=[i for i,r in enumerate(rows_b) if r['track_id'] not in reserved]
    if len(ia)<2 or len(ib)<2:return [],dict(status='INSUFFICIENT_UNRESERVED_DESCRIPTORS',a=len(ia),b=len(ib))
    a=desc_a[ia];b=desc_b[ib]
    # Bound memory independently of the source schema cap.
    if len(a)>20000 or len(b)>20000:raise ValueError('descriptor row bound')
    # Process rows without an N x M x 128 distance tensor. The selected video
    # has <=768 active rows; a 20k x 20k matrix is explicitly refused.
    if len(a)*len(b)>4_000_000:raise ValueError('descriptor pair work bound')
    distances=np.sqrt(np.maximum(np.sum(a*a,axis=1)[:,None]+np.sum(b*b,axis=1)[None,:]-2*a@b.T,0))
    forward=np.argsort(distances,axis=1,kind='stable')[:,:2]
    reverse=np.argsort(distances,axis=0,kind='stable')[:2,:]
    matches=[]
    for x,(y,second) in enumerate(forward):
        if reverse[0,y]!=x:continue
        d1,d2=float(distances[x,y]),float(distances[x,second])
        r1,r2=float(distances[x,y]),float(distances[reverse[1,y],y])
        if d2<=1e-8 or r2<=1e-8 or not (d1<ratio*d2 and r1<ratio*r2):continue
        ra,rb=rows_a[ia[x]],rows_b[ib[y]]
        if ra['track_id']==rb['track_id']:raise ValueError('reserved same ID unexpectedly reached matcher')
        matches.append(dict(track_id_a=ra['track_id'],track_id_b=rb['track_id'],
            evidence=dict(feature_row_a=ra['feature_row'],feature_row_b=rb['feature_row'],
                xy_a=ra['xy'],xy_b=rb['xy'],orientation_a=ra['orientation_degrees'],orientation_b=rb['orientation_degrees'],
                sift_size=4.,distance=d1,second_distance_a=d2,second_distance_b=r2,
                ratio_a=d1/d2,ratio_b=r1/r2,mutual=True,geometry_verified=False)))
    if len({m['track_id_a'] for m in matches})!=len(matches) or len({m['track_id_b'] for m in matches})!=len(matches):raise ValueError('matching not one-to-one')
    return matches,dict(status='REGION_CANDIDATES_MEASURED',a=len(ia),b=len(ib))


def _visualize(a,b,matches,path):
    with Image.open(a['image_path']) as image:im_a=image.convert('RGB')
    with Image.open(b['image_path']) as image:im_b=image.convert('RGB')
    board=Image.new('RGB',(im_a.width+im_b.width,max(im_a.height,im_b.height)+35),'#171717')
    board.paste(im_a,(0,35));board.paste(im_b,(im_a.width,35));draw=ImageDraw.Draw(board)
    draw.text((5,8),f"{a['input_index']} -> {b['input_index']}: {len(matches)} cross-ID proposals. NOT GEOMETRY / ID TRUTH",fill='white')
    for n,m in enumerate(matches):
        x,y=m['evidence']['xy_a'];u,v=m['evidence']['xy_b'];color=(255,190,20) if n%2 else (0,255,220)
        draw.line((x,y+35,u+im_a.width,v+35),fill=color,width=1)
        draw.ellipse((x-2,y+33,x+2,y+37),fill=color);draw.ellipse((u+im_a.width-2,v+33,u+im_a.width+2,v+37),fill=color)
    board.save(path,quality=88)


def run(tracks_path,output,*,cv=None,profile=Profile()):
    """Freeze all scheduled cross-component pairs, measure and save proposals."""
    start=time.monotonic();output=Path(output).resolve()
    if output.exists():raise ValueError('fresh output required')
    plan=native.prepare(tracks_path,'all_shared_tracks')
    refs=list(plan['source_bindings']);refs.append(native._hash(Path(__file__).resolve()))
    cvrefs=[];verified_cv=load_cv(cvrefs)
    if verified_cv is None:raise RuntimeError('bound CPU OpenCV unavailable')
    if cv is not None and cv is not verified_cv:raise ValueError('caller OpenCV differs from bound module')
    cv=verified_cv;cv.setNumThreads(2);cv.ocl.setUseOpenCL(False);refs.extend(cvrefs)
    refs.extend(native._hash(Path(module.__file__).resolve()) for module in (native,connectivity_graph,temporal_review))
    for ref in refs:
        if Path(ref['path']).resolve().is_relative_to(output):raise ValueError('output overlaps source')
    frames=plan['frames'];nodes=list(range(len(frames)))
    comps=components(nodes,[(p['sequence_a'],p['sequence_b']) for p in plan['pairs']])
    component={n:i for i,c in enumerate(comps) for n in c}
    scheduled=[dict(sequence_a=a,sequence_b=b,component_a=component[a],component_b=component[b],
        input_index_a=frames[a]['input_index'],input_index_b=frames[b]['input_index'])
        for a in nodes for b in range(a+1,min(a+profile.max_sequence_distance+1,len(frames))) if component[a]!=component[b]]
    output.mkdir(parents=True);(output/'pair_visualizations').mkdir()
    frozen=dict(schema='crossid-region-plan-v1',profile=asdict(profile),original_components=comps,pairs=scheduled,
        tracks_binding=refs[0],source_bindings=refs,selection='All sequence-distance 1..3 pairs across >=15 actual shared-ID graph components; frozen before descriptors.')
    native._save(output/'PAIR_PLAN.json',frozen)
    needed=sorted({p[k] for p in scheduled for k in ('sequence_a','sequence_b')});cache={};diagnostics=[]
    for i in needed:
        if time.monotonic()-start>profile.seconds:raise TimeoutError('region descriptor budget exhausted')
        diag,rows,desc=describe(frames[i],cv,profile);cache[i]=(rows,desc);diagnostics.append(diag)
    result_pairs=[]
    for p in scheduled:
        if time.monotonic()-start>profile.seconds:raise TimeoutError('region matching budget exhausted')
        a,b=p['sequence_a'],p['sequence_b'];reserved=set(frames[a]['track_ids'])&set(frames[b]['track_ids'])
        matches,stats=match_descriptors(*cache[a],*cache[b],reserved,profile.ratio)
        visual=f'pair_visualizations/pair_{a:03d}_{b:03d}.jpg';_visualize(frames[a],frames[b],matches,output/visual)
        result_pairs.append(dict(**p,matches=matches,matching=stats,reserved_same_id_count=len(reserved),visualization=visual))
    for ref in refs:
        if native._hash(ref['path'])!=ref:raise ValueError('source changed during region matching: '+ref['path'])
    diagnostics.extend(dict(sequence_index=i,input_index=frames[i]['input_index'],frame_id=frames[i]['frame_id'],
        eligible=len(frames[i]['track_ids']),support_valid=None,descriptor_valid=None,
        reasons={'NOT_SCHEDULED_NOT_A_DESCRIPTOR_FAILURE':len(frames[i]['track_ids'])},rows=[])
        for i in sorted(set(nodes)-set(needed)))
    diagnostics.sort(key=lambda d:d['sequence_index'])
    native._save(output/'FRAME_DIAGNOSTICS.json',dict(frames=diagnostics,unneeded_sequences=sorted(set(nodes)-set(needed)),
        unneeded_reason='NO_FROZEN_CROSS_COMPONENT_PAIR_NOT_DESCRIPTOR_FAILURE'))
    result=dict(schema='crossid-region-matches-v1',tracks_binding=refs[0],source_bindings=refs,pairs=result_pairs,
        profile=asdict(profile),source_unchanged=True,coordinate_changes=0,id_merges=0,geometry_verified=False,
        elapsed_seconds=time.monotonic()-start,support_policy='Original grayscale; full 63x63 original-mask square. SIFT size4, octave0/layer0, sigma1.6; descriptor sample radius rounded21 + central-gradient1 + conservative initial Gaussian halo8 + rounding1 =31; orientation radius9. Conservative engineering support, not physical truth.',
        support_source='https://github.com/opencv/opencv/blob/4.13.0/modules/features2d/src/sift.simd.hpp#L646-L701',
        orientation_policy='Gaussian-weighted smoothed36-bin local gradient mode, ambiguity rejection; explicitly supplied angle.',
        limitations=['Descriptor support rejects near-boundary rows; undescribed is not unmatched.','The 8px prefilter halo is an engineering bound for pinned OpenCV4.13.0 octave0/layer0 sigma1.6, not a mathematical footprint proof for arbitrary library implementations.','The mask itself is not semantic truth; full support membership cannot exclude mislabeled hand/background.','Only frozen nearby pairs across old ID graph components are searched.','Appearance ambiguity, perspective and reflection can still produce wrong candidates; native validation required.'],
        supplement_path=str(output/'SUPPLEMENT.json'))
    native._save(output/'SUPPLEMENT.json',result)
    return result
