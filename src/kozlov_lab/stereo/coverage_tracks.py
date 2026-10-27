"""Mask-scoped optical-flow candidates; no geometric acceptance is implied."""
from copy import deepcopy
import numpy as np

PROFILE = dict(version='coverage-grid-lk-v3', max_frames=96, max_long_side=1024,
    grid_columns=8, grid_rows=12, corners_per_cell=6, max_active_per_cell=8, margin=8,
    quality_level=.005, min_distance=6, exclusion_radius=8., max_active=768,
    max_total_tracks=40000, block_size=7, win_size=21, max_level=3,
    max_iterations=30, epsilon=.01, conflict_radius=2., fb_max=3., error_max=40., strong_fb_max=1., strong_error_max=25.)


INCREMENTAL_PROFILE_ID = 'whole-video-incremental-v1'
TEMPORAL_SUPPORT_PROFILE_ID = 'whole-video-temporal-support-v2'


def profile_for(selection_scope='object', profile_id='default'):
    if selection_scope not in ('object', 'whole-frame'):
        raise ValueError('selection_scope must be object or whole-frame')
    if profile_id not in ('default', INCREMENTAL_PROFILE_ID, TEMPORAL_SUPPORT_PROFILE_ID):
        raise ValueError('unknown coverage tracking profile')
    if profile_id == TEMPORAL_SUPPORT_PROFILE_ID:
        return dict(profile_for(selection_scope, INCREMENTAL_PROFILE_ID),
            profile_id=profile_id, boundary_max_distance_px=6., boundary_max_age=3,
            temporal_max_age=6, temporal_min_interior_distance_px=3.,
            temporal_min_measured_foreground=2)
    if profile_id == INCREMENTAL_PROFILE_ID:
        if selection_scope != 'object': raise ValueError('incremental profile requires object selection')
        return dict(profile_for(selection_scope), profile_id=profile_id, max_frames=192,
            missing_mask_policy='bounded_unknown_lk_no_seeds', unknown_mask_max_steps=12,
            max_unknown_active=768, birth_reason_scope='observed-new-mask-area',
            unknown_observations_eligible_for_3d=False,
            cross_gap_identity='UNVERIFIED', geometry_grouping='track_id_and_identity_segment')
    return dict(deepcopy(PROFILE), selection_scope=selection_scope,
        grid_domain='object_bbox' if selection_scope=='object' else 'whole_image',
        missing_mask_policy='end_tracks_no_seeds' if selection_scope=='object' else 'label_unknown',
        mask_membership='candidate_center_pixel', coordinate_convention='COLMAP_pixel_center')


def grid_bounds(mask, shape, selection_scope='object'):
    """Half-open integer bounds. A missing/empty object has no seed grid."""
    profile_for(selection_scope)
    if selection_scope=='whole-frame':
        return [0, 0, shape[1], shape[0]]
    if mask is None:
        return None
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max())+1, int(ys.max())+1]


def grid_edges(bounds, columns=8, rows=12):
    x0,y0,x1,y1 = bounds
    return (x0+np.arange(columns+1)*(x1-x0)//columns,
            y0+np.arange(rows+1)*(y1-y0)//rows)


def grid_cell(p, bounds, columns=8, rows=12):
    """Assign the measured centre's pixel to the exact ROI partition."""
    x,y = np.floor(np.asarray(p)+.5).astype(int)
    x0,y0,x1,y1 = bounds
    if not (x0 <= x < x1 and y0 <= y < y1):
        raise ValueError('point outside seed-grid bounds')
    xe, ye = grid_edges(bounds,columns,rows)
    return int(np.searchsorted(ye,y,side='right')-1), int(np.searchsorted(xe,x,side='right')-1)


def _validate(grays, masks, config=None):
    config = PROFILE if config is None else config
    if type(grays) is not list or not 1 <= len(grays) <= config['max_frames']:
        raise ValueError('nonempty list of at most '+str(config['max_frames'])+' frames required')
    if type(masks) is not list or len(masks) != len(grays):
        raise ValueError('equal-length mask list required')
    shape = None
    for gray, mask in zip(grays, masks):
        if not isinstance(gray, np.ndarray) or gray.dtype != np.uint8 or gray.ndim != 2 or not gray.size:
            raise ValueError('nonempty uint8 grayscale required')
        if max(gray.shape) > config['max_long_side'] or (shape is not None and shape != gray.shape):
            raise ValueError('same shape with long side at most 1024 required')
        shape = gray.shape
        if mask is not None and (not isinstance(mask, np.ndarray) or mask.dtype != np.uint8 or mask.shape != shape or not np.isin(mask, [0,255]).all()):
            raise ValueError('mask must be None or same-size uint8 0/255')
    for name, limit in [('max_active',768), ('max_total_tracks',40000), ('max_active_per_cell',8)]:
        if type(config[name]) is not int or not 1 <= config[name] <= limit:
            raise ValueError('invalid track budget: '+name)


def _inside(p, shape):
    return bool(np.isfinite(p).all() and 0 <= p[0] <= shape[1]-1 and 0 <= p[1] <= shape[0]-1)


def _label(mask, p):
    if mask is None:
        return 'unknown'
    x,y = np.floor(np.asarray(p)+.5).astype(int)
    return 'foreground' if mask[y,x] else 'background'


def track_coverage(grays, masks, cv, selection_scope='object', profile_id='default'):
    """Track adjacent frames, replenish grid corners, and emit COLMAP centres.

    Seed observations have null flow errors and strong=False because no flow
    measurement exists. All subsequent measurements retain actual LK/FB values.
    """
    if profile_id != 'default':
        config = profile_for(selection_scope, profile_id)
        return _track_coverage_incremental(grays, masks, cv, config)
    _validate(grays, masks)
    config = profile_for(selection_scope)
    cv.ocl.setUseOpenCL(False)
    cv.setNumThreads(2)
    cv.setRNGSeed(0)
    shape = grays[0].shape
    tracks, active, coverage = [], {}, []
    columns, rows = config['grid_columns'], config['grid_rows']

    def cell_of(p):
        return grid_cell(p,bounds,columns,rows)
    lk = dict(winSize=(config['win_size'],config['win_size']), maxLevel=config['max_level'],
        criteria=(cv.TERM_CRITERIA_EPS|cv.TERM_CRITERIA_COUNT,config['max_iterations'],config['epsilon']))

    def observation(frame, p, fb=None, error=None):
        return dict(frame_index=frame, xy=[float(p[0])+.5,float(p[1])+.5],
            label=_label(masks[frame],p), fb_error_px=fb, lk_error=error,
            strong=bool(fb is not None and fb <= config['strong_fb_max'] and error <= config['strong_error_max']))

    def end(ident, frame, reason, fb=None, error=None):
        active.pop(ident)
        tracks[ident]['termination'] = dict(frame_index=frame,reason=reason,
            fb_error_px=fb,lk_error=error)

    for frame, gray in enumerate(grays):
        grid_budget_retired = 0
        bounds = grid_bounds(masks[frame],shape,selection_scope)
        selection_status = 'TRACKING_OBJECT' if selection_scope=='object' else 'TRACKING_WHOLE_FRAME'
        if bounds is None:
            selection_status = 'MASK_UNAVAILABLE' if masks[frame] is None else 'EMPTY_FOREGROUND'
            for ident in list(active):
                end(ident,frame,selection_status)
            coverage.append(dict(frame_index=frame,active=0,seeded=0,grid_budget_retired=0,
                foreground=0,background=0,unknown=0,strong=0,grid_counts=[[0]*columns for _ in range(rows)],
                grid_bounds=None,grid_domain=config['grid_domain'],selection_status=selection_status))
            continue
        if frame and active:
            ids = sorted(active)
            before = np.asarray([active[i] for i in ids],np.float32).reshape(-1,1,2)
            flow, status, errors = cv.calcOpticalFlowPyrLK(grays[frame-1],gray,before,None,**lk)
            eligible = []
            for n, ident in enumerate(ids):
                error = None if errors is None or not np.isfinite(errors[n]).all() else float(errors[n,0])
                if flow is None or status is None or not status[n,0] or not _inside(flow[n,0],shape) or error is None:
                    end(ident,frame,'FORWARD_LK_FAILED',error=error)
                elif selection_scope=='object' and _label(masks[frame],flow[n,0])!='foreground':
                    end(ident,frame,'LEFT_FOREGROUND',error=error)
                else:
                    eligible.append((n,ident,error))
            candidates = []
            if eligible:
                forward = np.asarray([flow[n,0] for n,_,_ in eligible],np.float32).reshape(-1,1,2)
                back, status_back, _ = cv.calcOpticalFlowPyrLK(gray,grays[frame-1],forward,None,**lk)
                for j,(n,ident,error) in enumerate(eligible):
                    fb = None if back is None or not np.isfinite(back[j,0]).all() else float(np.linalg.norm(back[j,0]-before[n,0]))
                    if back is None or status_back is None or not status_back[j,0] or not _inside(back[j,0],shape):
                        end(ident,frame,'BACKWARD_LK_FAILED',fb,error)
                    elif fb > config['fb_max'] or error > config['error_max']:
                        end(ident,frame,'FLOW_THRESHOLD',fb,error)
                    else:
                        candidates.append((fb,error,ident,flow[n,0].copy()))
            accepted = []
            by_cell = {}
            for fb,error,ident,p in sorted(candidates,key=lambda item:item[:3]):
                if any(np.linalg.norm(p-q) <= config['conflict_radius'] for q in accepted):
                    end(ident,frame,'ENDPOINT_CONFLICT',fb,error)
                else:
                    accepted.append(p)
                    by_cell.setdefault(cell_of(p),[]).append((fb,error,ident,p))
            for group in by_cell.values():
                group.sort(key=lambda item:(-len(tracks[item[2]]['observations']),item[0],item[1],item[2]))
                for rank,(fb,error,ident,p) in enumerate(group):
                    if rank >= config['max_active_per_cell']:
                        end(ident,frame,'GRID_BUDGET_RETIRED',fb,error)
                        grid_budget_retired += 1
                    else:
                        active[ident] = p
                        tracks[ident]['observations'].append(observation(frame,p,fb,error))

        # Build exclusion once; ROI corner responses include a block-radius halo.
        cell_counts = [[0]*columns for _ in range(rows)]
        for p in active.values():
            row,col = cell_of(p)
            cell_counts[row][col] += 1
        margin, halo = config['margin'], config['block_size']//2
        allowed = np.zeros(shape,np.uint8)
        if shape[0] > 2*margin and shape[1] > 2*margin:
            allowed[margin:shape[0]-margin,margin:shape[1]-margin] = 255
        if selection_scope=='object':
            allowed[masks[frame]==0] = 0
        for p in active.values():
            cv.circle(allowed,tuple(np.floor(p+.5).astype(int)),
                int(np.ceil(config['exclusion_radius'])),0,-1)
        cells = []
        xedges,yedges = grid_edges(bounds,columns,rows)
        for row in range(rows):
            for col in range(columns):
                x0,x1 = int(xedges[col]),int(xedges[col+1])
                y0,y1 = int(yedges[row]),int(yedges[row+1])
                if cell_counts[row][col] >= config['max_active_per_cell'] or x1 <= x0 or y1 <= y0 or not allowed[y0:y1,x0:x1].any():
                    cells.append([])
                    continue
                rx0,rx1 = max(0,x0-halo),min(shape[1],x1+halo)
                ry0,ry1 = max(0,y0-halo),min(shape[0],y1+halo)
                region = np.zeros((ry1-ry0,rx1-rx0),np.uint8)
                region[y0-ry0:y1-ry0,x0-rx0:x1-rx0] = allowed[y0:y1,x0:x1]
                points = cv.goodFeaturesToTrack(gray[ry0:ry1,rx0:rx1],
                    maxCorners=config['corners_per_cell'],qualityLevel=config['quality_level'],
                    minDistance=config['min_distance'],mask=region,blockSize=config['block_size'])
                cells.append([] if points is None else list(points[:,0,:]+np.float32([rx0,ry0])))
        seeded = 0
        for slot in range(config['corners_per_cell']):
            for points in cells:
                if slot >= len(points) or len(active) >= config['max_active']:
                    continue
                p = points[slot]
                if not _inside(p,shape) or (selection_scope=='object' and _label(masks[frame],p)!='foreground'):
                    continue
                row,col = cell_of(p)
                if cell_counts[row][col] >= config['max_active_per_cell']:
                    continue
                if any(np.linalg.norm(p-q) <= config['exclusion_radius'] for q in active.values()):
                    continue
                if len(tracks) >= config['max_total_tracks']:
                    raise ValueError('total track budget exhausted')
                ident = len(tracks)
                tracks.append(dict(track_id=ident,seed_frame=frame,
                    observations=[observation(frame,p)],termination=None))
                active[ident] = p.copy()
                seeded += 1
                cell_counts[row][col] += 1
        observations = [tracks[i]['observations'][-1] for i in active]
        grid_counts = [[0]*columns for _ in range(rows)]
        for o in observations:
            row,col = cell_of(np.asarray(o['xy'])-.5)
            grid_counts[row][col] += 1
        coverage.append(dict(frame_index=frame,active=len(active),seeded=seeded,
            grid_budget_retired=grid_budget_retired,
            grid_bounds=bounds,grid_domain=config['grid_domain'],selection_status=selection_status,
            foreground=sum(o['label']=='foreground' for o in observations),
            background=sum(o['label']=='background' for o in observations),
            unknown=sum(o['label']=='unknown' for o in observations),
            strong=sum(o['strong'] for o in observations),grid_counts=grid_counts))
    for ident in active:
        tracks[ident]['termination'] = dict(frame_index=len(grays)-1,reason='SEQUENCE_END',fb_error_px=None,lk_error=None)
    return dict(status='CANDIDATE_TRACKS_ONLY',profile=config,tracks=tracks,coverage=coverage)


def _original_foreground_anchor(history, frame, ident, age_limit, config):
    """Bound an appearance chain to its last independently observed foreground.

    Support or neighbor conclusions never confer anchor eligibility. Background
    breaks the chain, and unknown frames never reset the original source age.
    """
    indices = [i for i,o in enumerate(history) if o['label']=='foreground']
    if not indices: return None
    last_index = indices[-1]
    last = history[last_index]
    age = frame-last['frame_index']
    if (age>age_limit or not last['strong'] or
        (last.get('interior_distance_px') or 0)<config['temporal_min_interior_distance_px']):
        return None
    measured = 0
    for o in reversed(history[:last_index+1]):
        if o['label']!='foreground' or o['fb_error_px'] is None: break
        measured += 1
    if measured<config['temporal_min_measured_foreground']: return None
    independent_failures = {'flow_failed','nonfinite_flow','fb_error','lk_error',
        'outside_patch','patch_low_texture','patch_appearance'}
    for o in history[last_index+1:]:
        proof = o.get('temporal_support') or {}
        ncc,std = proof.get('patch_ncc'),proof.get('patch_std')
        if (o['label']!='unknown' or not o['strong'] or ncc is None or std is None or
            not np.isfinite(ncc) or not np.isfinite(std) or ncc<.75 or std<5 or
            independent_failures.intersection(proof.get('reasons',[]))):
            return None
    return dict(track_id=ident,last_foreground_frame=last['frame_index'],age=age,
        source='original_foreground_with_bounded_patch_chain')


def _track_coverage_incremental(grays, masks, cv, config):
    """Track adjacent frames, replenish grid corners, and emit COLMAP centres.

    Seed observations have null flow errors and strong=False because no flow
    measurement exists. All subsequent measurements retain actual LK/FB values.
    """
    temporal = config.get('profile_id') == TEMPORAL_SUPPORT_PROFILE_ID
    if temporal:
        from .temporal_support import assess_temporal_support, boundary_distance
    selection_scope = 'object'
    _validate(grays, masks, config)
    cv.ocl.setUseOpenCL(False)
    cv.setNumThreads(2)
    cv.setRNGSeed(0)
    shape = grays[0].shape
    tracks, active, coverage = [], {}, []
    events, gaps, segments = [], {}, {}
    support_by_id = {}
    interior = None
    columns, rows = config['grid_columns'], config['grid_rows']

    def cell_of(p):
        return grid_cell(p,bounds,columns,rows)
    lk = dict(winSize=(config['win_size'],config['win_size']), maxLevel=config['max_level'],
        criteria=(cv.TERM_CRITERIA_EPS|cv.TERM_CRITERIA_COUNT,config['max_iterations'],config['epsilon']))

    def observation(frame, p, fb=None, error=None, identity_segment=0):
        result = dict(frame_index=frame, identity_segment=identity_segment, xy=[float(p[0])+.5,float(p[1])+.5],
            label=_label(masks[frame],p), target_verified=_label(masks[frame],p)=='foreground', fb_error_px=fb, lk_error=error,
            strong=bool(fb is not None and fb <= config['strong_fb_max'] and error <= config['strong_error_max']))
        if temporal:
            x,y = np.floor(np.asarray(p)+.5).astype(int)
            result.update(tracking_status='seed' if fb is None else 'flow_tracked',
                mask_status='original_mask' if result['label']=='foreground' else 'unknown',
                temporal_support=None, interior_distance_px=float(interior[y,x]) if interior is not None and result['label']=='foreground' else None)
        return result

    def end(ident, frame, reason, fb=None, error=None):
        active.pop(ident)
        gaps.pop(ident,None)
        segments.pop(ident,None)
        events.append(dict(kind='track_ended',track_id=ident,frame_index=frame,reason=reason))
        tracks[ident]['termination'] = dict(frame_index=frame,reason=reason,
            fb_error_px=fb,lk_error=error)

    for frame, gray in enumerate(grays):
        grid_budget_retired = 0
        recovered = 0
        unknown_frame = masks[frame] is None
        bounds = grid_bounds(masks[frame],shape,selection_scope)
        selection_status = 'TRACKING_OBJECT' if selection_scope=='object' else 'TRACKING_WHOLE_FRAME'
        visible_bounds = bounds
        if temporal:
            interior = None if unknown_frame else cv.distanceTransform(masks[frame],cv.DIST_L2,cv.DIST_MASK_PRECISE)
            exterior = None if unknown_frame else boundary_distance(masks[frame],cv)
            support_by_id = {}
            boundary_bounds = None if bounds is None else [max(0,bounds[0]-6),max(0,bounds[1]-6),min(shape[1],bounds[2]+6),min(shape[0],bounds[3]+6)]
        if bounds is None and not unknown_frame:
            selection_status = 'MASK_UNAVAILABLE' if masks[frame] is None else 'EMPTY_FOREGROUND'
            for ident in list(active):
                end(ident,frame,selection_status)
            coverage.append(dict(frame_index=frame,active=0,seeded=0,grid_budget_retired=0,
                tracked=0,recovered=0,suspended=0,foreground=0,background=0,unknown=0,strong=0,grid_counts=[[0]*columns for _ in range(rows)],
                grid_bounds=None,grid_domain=config['grid_domain'],selection_status=selection_status))
            continue
        if unknown_frame:
            selection_status = 'MASK_UNKNOWN_SUSPENDED_LK'
            bounds = [0,0,shape[1],shape[0]]  # scheduling only; never a target/seed mask
            for ident in list(active):
                if gaps.get(ident,0) >= config['unknown_mask_max_steps']:
                    end(ident,frame,'MASK_GAP_EXCEEDED')
        if frame and active:
            ids = sorted(active)
            before = np.asarray([active[i] for i in ids],np.float32).reshape(-1,1,2)
            flow, status, errors = cv.calcOpticalFlowPyrLK(grays[frame-1],gray,before,None,**lk)
            eligible = []
            for n, ident in enumerate(ids):
                error = None if errors is None or not np.isfinite(errors[n]).all() else float(errors[n,0])
                if flow is None or status is None or not status[n,0] or not _inside(flow[n,0],shape) or error is None:
                    end(ident,frame,'FORWARD_LK_FAILED',error=error)
                elif not temporal and not unknown_frame and _label(masks[frame],flow[n,0])!='foreground':
                    end(ident,frame,'LEFT_FOREGROUND',error=error)
                else:
                    eligible.append((n,ident,error))
            candidates = []
            if eligible:
                forward = np.asarray([flow[n,0] for n,_,_ in eligible],np.float32).reshape(-1,1,2)
                back, status_back, _ = cv.calcOpticalFlowPyrLK(gray,grays[frame-1],forward,None,**lk)
                for j,(n,ident,error) in enumerate(eligible):
                    fb = None if back is None or not np.isfinite(back[j,0]).all() else float(np.linalg.norm(back[j,0]-before[n,0]))
                    if back is None or status_back is None or not status_back[j,0] or not _inside(back[j,0],shape):
                        end(ident,frame,'BACKWARD_LK_FAILED',fb,error)
                    elif fb > config['fb_max'] or error > config['error_max']:
                        end(ident,frame,'FLOW_THRESHOLD',fb,error)
                    else:
                        candidates.append((fb,error,ident,flow[n,0].copy()))
            if temporal and candidates:
                # Support sees all ordinary-valid endpoints before mask rejection.
                anchors = []
                anchor_sources = []
                histories = {}
                for fb,error,ident,p in candidates:
                    history = tracks[ident]['observations']
                    known = [o for o in history if o['label']=='foreground']
                    last = known[-1]
                    age = frame-last['frame_index']
                    measured = 0
                    for o in reversed(history[:history.index(last)+1]):
                        if o['label']!='foreground' or o['fb_error_px'] is None: break
                        measured += 1
                    limit = config['temporal_max_age'] if unknown_frame else config['boundary_max_age']
                    source = _original_foreground_anchor(history,frame,ident,limit,config)
                    anchors.append(source is not None)
                    anchor_sources.append(source)
                    histories[ident] = (last,age,measured)
                evidence = assess_temporal_support(grays[frame-1],gray,
                    np.asarray([active[i] for _,_,i,_ in candidates],np.float32),
                    np.asarray([p for _,_,_,p in candidates],np.float32),
                    np.asarray([fb for fb,_,_,_ in candidates]),
                    np.asarray([e for _,e,_,_ in candidates]),np.asarray(anchors,bool),cv,
                    flow_valid=np.ones(len(candidates),bool))
                retained = []
                for item,proof in zip(candidates,evidence):
                    fb,error,ident,p = item
                    proof = dict(proof)
                    proof['neighbor_anchor_track_ids'] = [int(candidates[j][2]) for j in proof.get('neighbor_indices',[]) if candidates[j][2]!=ident]
                    proof['agreeing_anchor_track_ids'] = [int(candidates[j][2]) for j in proof.get('agreeing_neighbor_indices',[]) if candidates[j][2]!=ident]
                    proof['neighbor_anchors'] = [anchor_sources[j] for j in proof.get('neighbor_indices',[]) if candidates[j][2]!=ident]
                    proof['eligible_anchors'] = proof['neighbor_anchors']
                    proof['agreeing_anchors'] = [anchor_sources[j] for j in proof.get('agreeing_neighbor_indices',[]) if candidates[j][2]!=ident]
                    last,age,measured = histories[ident]
                    supported = bool(proof.get('supported',False))
                    label = _label(masks[frame],p)
                    status_name = 'original_mask' if label=='foreground' else 'unknown'
                    if unknown_frame:
                        if (supported and age<=config['temporal_max_age'] and
                            (last.get('interior_distance_px') or 0)>=config['temporal_min_interior_distance_px'] and
                            measured>=config['temporal_min_measured_foreground']):
                            status_name = 'temporal_supported'
                    elif label!='foreground':
                        x,y = np.floor(p+.5).astype(int)
                        if not (supported and age<=config['boundary_max_age'] and exterior[y,x]<=config['boundary_max_distance_px']):
                            end(ident,frame,'LEFT_FOREGROUND',fb,error)
                            continue
                        status_name = 'boundary_supported'
                    support_by_id[ident] = (status_name,proof)
                    retained.append(item)
                candidates = retained
            accepted = []
            by_cell = {}
            for fb,error,ident,p in sorted(candidates,key=lambda item:item[:3]):
                if any(np.linalg.norm(p-q) <= config['conflict_radius'] for q in accepted):
                    end(ident,frame,'ENDPOINT_CONFLICT',fb,error)
                else:
                    accepted.append(p)
                    key = ('unknown',ident) if unknown_frame else (('boundary',)+grid_cell(p,boundary_bounds,columns,rows) if temporal and _label(masks[frame],p)!='foreground' else cell_of(p))
                    by_cell.setdefault(key,[]).append((fb,error,ident,p))
            for group in by_cell.values():
                group.sort(key=lambda item:(-len(tracks[item[2]]['observations']),item[0],item[1],item[2]))
                for rank,(fb,error,ident,p) in enumerate(group):
                    if rank >= config['max_active_per_cell']:
                        end(ident,frame,'GRID_BUDGET_RETIRED',fb,error)
                        grid_budget_retired += 1
                    else:
                        active[ident] = p
                        if unknown_frame or (temporal and _label(masks[frame],p)!='foreground'):
                            if not gaps.get(ident,0):
                                events.append(dict(kind='mask_gap_started',track_id=ident,frame_index=frame))
                            gaps[ident] = gaps.get(ident,0)+1
                        else:
                            if gaps.get(ident,0):
                                recovered += 1
                                previous_segment = segments[ident]
                                segments[ident] += 1
                                events.append(dict(kind='mask_gap_recovered',track_id=ident,frame_index=frame,
                                    unknown_steps=gaps[ident],cross_gap_identity='UNVERIFIED',
                                    previous_identity_segment=previous_segment,identity_segment=segments[ident]))
                            gaps[ident] = 0
                        obs = observation(frame,p,fb,error,segments[ident])
                        if temporal:
                            obs['mask_status'],obs['temporal_support'] = support_by_id[ident]
                        tracks[ident]['observations'].append(obs)

        tracked = len(active)
        if unknown_frame:
            if len(active) > config['max_unknown_active']: raise ValueError('unknown track budget exceeded')
            observations = [tracks[i]['observations'][-1] for i in active]
            coverage.append(dict(frame_index=frame,active=len(active),tracked=tracked,seeded=0,recovered=0,
                suspended=len(active),grid_budget_retired=grid_budget_retired,foreground=0,background=0,
                unknown=len(active),strong=sum(o['strong'] for o in observations),
                grid_counts=[[0]*columns for _ in range(rows)],grid_bounds=None,
                grid_domain=config['grid_domain'],selection_status=selection_status))
            continue
        # Build exclusion once; ROI corner responses include a block-radius halo.
        cell_counts = [[0]*columns for _ in range(rows)]
        for p in active.values():
            if temporal and _label(masks[frame],p)!='foreground': continue
            row,col = cell_of(p)
            cell_counts[row][col] += 1
        margin, halo = config['margin'], config['block_size']//2
        allowed = np.zeros(shape,np.uint8)
        if shape[0] > 2*margin and shape[1] > 2*margin:
            allowed[margin:shape[0]-margin,margin:shape[1]-margin] = 255
        if selection_scope=='object':
            allowed[masks[frame]==0] = 0
        for p in active.values():
            cv.circle(allowed,tuple(np.floor(p+.5).astype(int)),
                int(np.ceil(config['exclusion_radius'])),0,-1)
        cells = []
        xedges,yedges = grid_edges(bounds,columns,rows)
        for row in range(rows):
            for col in range(columns):
                x0,x1 = int(xedges[col]),int(xedges[col+1])
                y0,y1 = int(yedges[row]),int(yedges[row+1])
                if cell_counts[row][col] >= config['max_active_per_cell'] or x1 <= x0 or y1 <= y0 or not allowed[y0:y1,x0:x1].any():
                    cells.append([])
                    continue
                rx0,rx1 = max(0,x0-halo),min(shape[1],x1+halo)
                ry0,ry1 = max(0,y0-halo),min(shape[0],y1+halo)
                region = np.zeros((ry1-ry0,rx1-rx0),np.uint8)
                region[y0-ry0:y1-ry0,x0-rx0:x1-rx0] = allowed[y0:y1,x0:x1]
                points = cv.goodFeaturesToTrack(gray[ry0:ry1,rx0:rx1],
                    maxCorners=config['corners_per_cell'],qualityLevel=config['quality_level'],
                    minDistance=config['min_distance'],mask=region,blockSize=config['block_size'])
                cells.append([] if points is None else list(points[:,0,:]+np.float32([rx0,ry0])))
        seeded = 0
        for slot in range(config['corners_per_cell']):
            for points in cells:
                if slot >= len(points) or len(active) >= config['max_active']:
                    continue
                p = points[slot]
                if not _inside(p,shape) or (selection_scope=='object' and _label(masks[frame],p)!='foreground'):
                    continue
                row,col = cell_of(p)
                if cell_counts[row][col] >= config['max_active_per_cell']:
                    continue
                if any(np.linalg.norm(p-q) <= config['exclusion_radius'] for q in active.values()):
                    continue
                if len(tracks) >= config['max_total_tracks']:
                    raise ValueError('total track budget exhausted')
                ident = len(tracks)
                previous_mask = masks[frame-1] if frame else None
                newly_observed = previous_mask is not None and _label(previous_mask,p)=='background'
                reason = 'NEWLY_VISIBLE_REGION' if newly_observed else 'COVERAGE_REPLENISHMENT'
                birth = dict(reason=reason,reason_scope='observed-new-mask-area',physical_surface_visibility_inferred=False)
                tracks.append(dict(track_id=ident,seed_frame=frame,birth=birth,
                    observations=[observation(frame,p)],termination=None))
                gaps[ident] = 0
                segments[ident] = 0
                events.append(dict(kind='birth',track_id=ident,frame_index=frame,**birth))
                active[ident] = p.copy()
                seeded += 1
                cell_counts[row][col] += 1
        observations = [tracks[i]['observations'][-1] for i in active]
        grid_counts = [[0]*columns for _ in range(rows)]
        for o in observations:
            if temporal and o['label']!='foreground': continue
            row,col = cell_of(np.asarray(o['xy'])-.5)
            grid_counts[row][col] += 1
        coverage.append(dict(frame_index=frame,active=len(active),tracked=tracked,seeded=seeded,recovered=recovered,suspended=0,
            grid_budget_retired=grid_budget_retired,
            grid_bounds=visible_bounds,grid_domain=config['grid_domain'],selection_status=selection_status,
            foreground=sum(o['label']=='foreground' for o in observations),
            background=sum(o['label']=='background' for o in observations),
            unknown=sum(o['label']=='unknown' for o in observations),
            strong=sum(o['strong'] for o in observations),grid_counts=grid_counts))
    for ident in active:
        tracks[ident]['termination'] = dict(frame_index=len(grays)-1,reason='SEQUENCE_END',fb_error_px=None,lk_error=None)
    return dict(status='CANDIDATE_TRACKS_ONLY',profile=config,tracks=tracks,coverage=coverage,events=events)
