"""CPU LK video candidate tracks, never target identity or geometric truth."""
from collections import Counter
from copy import deepcopy
import math
import numpy as np
from .point_coverage import greedy_r_cover

PROFILE=dict(max_frames=96,max_long_side=1024,max_active=256,max_seed_candidates=1024,
    cover_radius_at_1024=16.,erosion_pixels=10,quality_level=.01,min_distance=5,
    fb_max_pixels=1.,win_size=21,max_level=3,max_iterations=30,epsilon=.01,
    min_eigenvalue=1e-4,unknown_mask_max_steps=12,directions=['forward','backward'])


def _profile(profile):
    config=deepcopy(PROFILE)
    if profile is not None:
        if type(profile) is not dict or not set(profile).issubset(config):raise ValueError('known profile fields required')
        config.update(deepcopy(profile))
    for key in ('max_frames','max_long_side','max_active','max_seed_candidates','unknown_mask_max_steps'):
        if type(config[key]) is not int or not 1<=config[key]<=PROFILE[key]:raise ValueError('bounded integer profile field: '+key)
    for key in set(PROFILE)-{'max_frames','max_long_side','max_active','max_seed_candidates','unknown_mask_max_steps','directions'}:
        if isinstance(config[key],bool) or config[key]!=PROFILE[key]:raise ValueError('fixed LK/seeding profile field: '+key)
    if type(config['directions']) is not list or not config['directions'] or len(set(config['directions']))!=len(config['directions']) or any(v not in ('forward','backward') for v in config['directions']):
        raise ValueError('unique forward/backward direction list required')
    return config


def _validate(grays,masks,config):
    if type(grays) is not list or not 1<=len(grays)<=config['max_frames'] or type(masks) is not list or len(masks)!=len(grays):
        raise ValueError('bounded equal-length frame and mask lists required')
    shape=None
    for gray,mask in zip(grays,masks):
        if not isinstance(gray,np.ndarray) or gray.dtype!=np.uint8 or gray.ndim!=2 or not gray.size:
            raise ValueError('nonempty uint8 HxW grayscale required')
        if max(gray.shape)>config['max_long_side'] or (shape is not None and gray.shape!=shape):
            raise ValueError('same-size frames within frozen long-side bound required')
        shape=gray.shape
        if mask is not None and (not isinstance(mask,np.ndarray) or mask.dtype!=np.uint8 or mask.shape!=shape or not np.isin(mask,[0,255]).all()):
            raise ValueError('known masks require same-size uint8 0/255 arrays')


def _in_bounds(xy,shape):
    return bool(np.isfinite(xy).all() and 0<=xy[0]<=shape[1]-1 and 0<=xy[1]<=shape[0]-1)


def _inside(mask,xy):
    # CV image pixels have integer centres. Require nearest source mask pixel.
    x,y=np.floor(np.asarray(xy)+.5).astype(int)
    return bool(mask[y,x]==255)


def track_sequence(grays,masks,cv,profile=None):
    """Run independent bounded direction passes with disjoint string track IDs.

    Internal coordinates use OpenCV integer pixel centres. Output observations
    add +0.5 to both coordinates for the canonical COLMAP centre convention.
    Seeds are actual goodFeaturesToTrack corners inside a 10px-eroded known mask;
    no seeds are invented in unknown masks. The 16px radius is scaled by original
    image long side /1024. No resizing, RANSAC, model load or triangulation occurs.
    """
    config=_profile(profile);_validate(grays,masks,config);shape=grays[0].shape
    cv.ocl.setUseOpenCL(False);cv.setNumThreads(2);cv.setRNGSeed(0)
    radius=config['cover_radius_at_1024']*max(shape)/1024.;passes=[]
    lk=dict(winSize=(config['win_size'],config['win_size']),maxLevel=config['max_level'],
        criteria=(cv.TERM_CRITERIA_EPS|cv.TERM_CRITERIA_COUNT,config['max_iterations'],config['epsilon']),
        minEigThreshold=config['min_eigenvalue'])
    for direction in config['directions']:
        order=list(range(len(grays)))
        if direction=='backward':order.reverse()
        active={};tracks=[];steps=[];counter=0;previous=None
        def terminate(track,frame,reason,fb=None):
            track.update(status='TERMINATED',termination=dict(frame_index=frame,reason=reason,fb_error_pixels=fb))
        for frame in order:
            mask=masks[frame];counts=Counter();attempted=len(active);failures=Counter()
            if previous is not None and active:
                ids=list(active);before=np.asarray([active[i]['_xy'] for i in ids],dtype=np.float32).reshape(-1,1,2)
                flow,status,_=cv.calcOpticalFlowPyrLK(grays[previous],grays[frame],before,None,**lk)
                eligible=[]
                for n,ident in enumerate(ids):
                    track=active[ident];reason=None
                    if flow is None or status is None or not bool(status[n,0]):reason='FORWARD_LK_FAILED'
                    elif not _in_bounds(flow[n,0],shape):reason='FORWARD_NONFINITE_OR_OUT_OF_BOUNDS'
                    if reason:
                        terminate(track,frame,reason);failures[reason]+=1;del active[ident]
                    else:eligible.append((n,ident))
                if eligible:
                    forward=np.asarray([flow[n,0] for n,_ in eligible],dtype=np.float32).reshape(-1,1,2)
                    back,bstatus,_=cv.calcOpticalFlowPyrLK(grays[frame],grays[previous],forward,None,**lk)
                    for position,(n,ident) in enumerate(eligible):
                        track=active[ident];reason=None;error=None;xy=forward[position,0]
                        if back is None or bstatus is None or not bool(bstatus[position,0]):reason='BACKWARD_LK_FAILED'
                        elif not _in_bounds(back[position,0],shape):reason='BACKWARD_NONFINITE_OR_OUT_OF_BOUNDS'
                        else:
                            error=float(np.hypot(*(back[position,0]-before[n,0])))
                            if error>config['fb_max_pixels']:reason='FORWARD_BACKWARD_ERROR'
                        unknown=track['_unknown']+1 if mask is None else 0
                        if reason is None and mask is not None and not _inside(mask,xy):reason='TARGET_MASK_REJECTED'
                        if reason is None and unknown>config['unknown_mask_max_steps']:reason='UNKNOWN_MASK_LIMIT'
                        if reason:
                            terminate(track,frame,reason,error);failures[reason]+=1;del active[ident];continue
                        track['_xy']=xy.copy();track['_unknown']=unknown
                        track['observations'].append(dict(frame_index=frame,xy=(xy.astype(float)+.5).tolist(),
                            fb_error_pixels=error,target_verified=mask is not None,
                            target_status='TARGET_MASK_SUPPORTED' if mask is not None else 'TARGET_UNVERIFIED',
                            consecutive_unknown_steps=unknown))
                        counts['accepted_lk']+=1
            seed_count=0;available_count=0;cover=None;candidate_count=0;candidate_xy=[]
            if mask is not None and len(active)<config['max_active']:
                width=2*config['erosion_pixels']+1
                eroded=cv.erode(mask,np.ones((width,width),dtype=np.uint8),borderType=cv.BORDER_CONSTANT,borderValue=0)
                candidates=cv.goodFeaturesToTrack(grays[frame],maxCorners=config['max_seed_candidates'],
                    qualityLevel=config['quality_level'],minDistance=config['min_distance'],mask=eroded)
                if candidates is not None:
                    candidates=np.asarray(candidates,dtype=np.float32).reshape(-1,2);candidate_count=len(candidates)
                    kept=[]
                    for xy in candidates:
                        if not _in_bounds(xy,shape) or not _inside(eroded,xy):continue
                        if active and min(float(np.hypot(*(xy-t['_xy']))) for t in active.values())<radius:continue
                        kept.append(xy)
                    # CV response ties/order do not become arbitrary source labels.
                    kept=sorted(kept,key=lambda xy:(float(xy[1]),float(xy[0])))
                    candidate_xy=[(xy.astype(float)+.5).tolist() for xy in kept]
                    available_count=len(kept)
                    cover=greedy_r_cover(np.asarray(kept,dtype=float).reshape(-1,2),radius,
                        max_centers=config['max_active']-len(active))
                    for selected in cover['selected_indices']:
                        xy=np.array(kept[selected],dtype=np.float32);ident=f'{direction}:{counter:06d}';counter+=1
                        track=dict(track_id=ident,direction=direction,seed_frame_index=frame,seed_candidate_index=selected,
                            seed_xy=(xy.astype(float)+.5).tolist(),status='ACTIVE',termination=None,
                            observations=[dict(frame_index=frame,xy=(xy.astype(float)+.5).tolist(),fb_error_pixels=None,
                                target_verified=True,target_status='TARGET_MASK_SUPPORTED',consecutive_unknown_steps=0)],
                            _xy=xy,_unknown=0)
                        tracks.append(track);active[ident]=track;seed_count+=1
            steps.append(dict(frame_index=frame,previous_frame_index=previous,known_target_mask=mask is not None,
                attempted_lk=attempted if previous is not None else 0,accepted_lk=counts['accepted_lk'],
                terminated_lk=sum(failures.values()),termination_reasons=dict(failures),seed_candidates=candidate_count,
                candidates_after_active_exclusion=available_count,seeded=seed_count,active_count=len(active),seed_cover=cover))
            steps[-1]['cover_candidate_xy']=candidate_xy
            previous=frame
        for track in active.values():terminate(track,order[-1],'DIRECTION_BOUNDARY')
        for track in tracks:
            del track['_xy'];del track['_unknown']
            track['seed_frame']=track['seed_frame_index']
            track['termination']['frame']=track['termination']['frame_index']
            for observation in track['observations']:
                observation['frame']=observation['frame_index']
                observation['fb_error']=observation['fb_error_pixels']
        for step in steps:
            step['direction']=direction;step['frame']=step['frame_index']
        passes.append(dict(direction=direction,frame_order=order,tracks=tracks,steps=steps,
            candidate_track_count=len(tracks),max_active_observed=max(s['active_count'] for s in steps)))
    return dict(schema='bounded-temporal-lk-candidates-v1',status='CANDIDATE_TRACKING_COMPLETE_NOT_VERIFIED',
        profile=config,frame_count=len(grays),image_shape=list(shape),cover_radius_pixels=radius,passes=passes,
        tracks=[track for item in passes for track in item['tracks']],steps=[step for item in passes for step in item['steps']],
        backend=dict(name='OpenCV CPU',version=str(cv.__version__),opencl_enabled=bool(cv.ocl.useOpenCL()),threads=int(cv.getNumThreads())),
        counts=dict(seed_tracks=sum(p['candidate_track_count'] for p in passes),
            accepted_lk_observations=sum(s['accepted_lk'] for p in passes for s in p['steps']),
            failed_tracks=sum(s['terminated_lk'] for p in passes for s in p['steps']),
            direction_boundary_tracks=sum(t['termination']['reason']=='DIRECTION_BOUNDARY' for p in passes for t in p['tracks'])),
        coordinate_convention='canonical pixel centre = OpenCV xy + 0.5',source_modified=False,
        target_identity_verified=False,correspondence_verified=False,geometry_acceptance=False,
        limitations=['Known-mask support is predicted target-region evidence, not semantic or correspondence truth.',
            'Unknown-mask observations remain TARGET_UNVERIFIED and cannot certify target identity.',
            'Forward/backward passes are separate candidate tracks, not independent truth; no deduplication.',
            'Every failed ID terminates permanently; new corner seeds always receive new IDs.'])
