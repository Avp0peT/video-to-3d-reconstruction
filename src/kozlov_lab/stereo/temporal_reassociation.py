"""Bounded gap-link hypotheses on measured tracklets; never identity truth."""
from collections import Counter, defaultdict
from copy import deepcopy
import math
import numpy as np

PROFILE = dict(max_frames=96, max_long_side=1024, max_tracks=50000,
    min_frame_delta=2, max_frame_delta=6, max_endpoints_per_pair=256,
    max_comparisons=2000000, max_candidate_edges=100000,
    fb_max_pixels=1., seed_max_pixels=2., runner_up_separation_pixels=4.,
    patch_size=15, patch_min_std=5., patch_min_zncc=.85)
PROFILE.update(patch_ambiguity_radius_pixels=16., patch_ambiguity_score_margin=.05,
               max_patch_comparisons=100000)


class _DistanceBudget:
    def __init__(self, limit):
        self.limit=limit; self.count=0

    def add(self, count):
        if self.count+count>self.limit:raise ValueError('candidate comparison budget exceeded')
        self.count+=count


class _SpatialIndex:
    """Full universe, original-index stable exact radius queries; no truncation."""
    def __init__(self, points, budget, cell_size=16.):
        self.points=np.asarray(points,dtype=float).reshape(-1,2)
        self.budget=budget;self.cell_size=cell_size;self.bins=defaultdict(list)
        for i,xy in enumerate(self.points):self.bins[tuple(np.floor(xy/cell_size).astype(int))].append(i)

    def query(self, xy, radius):
        xy=np.asarray(xy,dtype=float);lo=np.floor((xy-radius)/self.cell_size).astype(int);hi=np.floor((xy+radius)/self.cell_size).astype(int)
        indices=[]
        for x in range(int(lo[0]),int(hi[0])+1):
            for y in range(int(lo[1]),int(hi[1])+1):indices.extend(self.bins.get((x,y),()))
        self.budget.add(len(indices))
        if not indices:return []
        values=np.linalg.norm(self.points[indices]-xy,axis=1)
        return sorted(((float(d),i) for d,i in zip(values,indices) if d<=radius),key=lambda item:(item[0],item[1]))


def _frame(observation):
    value = observation.get('frame_index', observation.get('frame'))
    if type(value) is not int or ('frame' in observation and observation['frame'] != value):
        raise ValueError('consistent integer frame identity required')
    return value


def _xy(observation):
    value = observation.get('xy')
    if not isinstance(value, (list, tuple)) or len(value) != 2 or any(
        isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.integer, np.floating))
        or not math.isfinite(float(v)) for v in value):
        raise ValueError('finite real measured xy required')
    return np.asarray(value, dtype=float) - .5


def _bounds(xy, shape, margin=0):
    return bool(np.isfinite(xy).all() and margin <= xy[0] <= shape[1]-1-margin
                and margin <= xy[1] <= shape[0]-1-margin)


def _support(mask, xy):
    if mask is None or not _bounds(xy, mask.shape, 7): return False
    # Include all bilinear-support pixels of subpixel getRectSubPix samples.
    left,top=np.floor(xy-7).astype(int);right,bottom=np.ceil(xy+7).astype(int)
    return bool(np.all(mask[top:bottom+1, left:right+1] == 255))


def _patch_metrics(first, second, a, b, cv):
    p = cv.getRectSubPix(first, (15, 15), tuple(map(float, a))).astype(float)
    q = cv.getRectSubPix(second, (15, 15), tuple(map(float, b))).astype(float)
    p -= p.mean(); q -= q.mean()
    s, t = float(p.std()), float(q.std())
    score = float(np.sum(p*q) / (225*s*t)) if s and t else None
    return s, t, score


def reassociate_tracklets(tracks, grays, masks, cv):
    """Return conservative links and unmodified-observation joined hypotheses.

    All comparisons use original first/last observations in each direction.
    Missing intervals remain UNKNOWN, with no generated observations. Independent
    forward/backward passes never share IDs or edges. Resource excess fails closed.
    """
    if type(grays) is not list or not 1 <= len(grays) <= 96 or type(masks) is not list or len(masks) != len(grays):
        raise ValueError('one to 96 equal-length grayscale/mask lists required')
    shape = None
    for image, mask in zip(grays, masks):
        if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 2 or not image.size or max(image.shape) > 1024:
            raise ValueError('bounded uint8 grayscale frames required')
        if shape is not None and image.shape != shape: raise ValueError('same-size frames required')
        shape = image.shape
        if mask is not None and (not isinstance(mask, np.ndarray) or mask.dtype != np.uint8
            or mask.shape != shape or not np.isin(mask, [0,255]).all()):
            raise ValueError('same-size uint8 binary mask required')
    if type(tracks) is not list or len(tracks) > 50000: raise ValueError('at most 50000 tracks required')
    starts = defaultdict(list); ends = defaultdict(list); by_id = {}; rejects = Counter()
    for track in tracks:
        ident = track.get('track_id'); direction = track.get('direction'); observations = track.get('observations')
        if not isinstance(ident, str) or not ident or ident in by_id or direction not in ('forward', 'backward'):
            raise ValueError('unique nonempty track IDs and known directions required')
        if type(observations) is not list or not 1 <= len(observations) <= 96: raise ValueError('bounded original observations required')
        frames = [_frame(o) for o in observations]; sign = 1 if direction == 'forward' else -1
        if any(not 0 <= f < len(grays) for f in frames) or any(sign*(b-a) <= 0 for a,b in zip(frames, frames[1:])):
            raise ValueError('ordered unique original observation frames required')
        for observation in observations:
            if not _bounds(_xy(observation), shape): raise ValueError('original observation outside canonical frame')
        if 'seed_xy' in track and list(track['seed_xy']) != list(observations[0]['xy']):
            raise ValueError('first observation differs from measured seed')
        by_id[ident] = track
        for bucket, observation, frame in ((starts, observations[0], frames[0]), (ends, observations[-1], frames[-1])):
            if observation.get('target_verified') is not True or not _support(masks[frame], _xy(observation)):
                rejects['ENDPOINT_MASK_UNAVAILABLE_OR_UNSUPPORTED'] += 1
            else: bucket[(direction, frame)].append((ident, _xy(observation)))
    cv.ocl.setUseOpenCL(False); cv.setNumThreads(2); cv.setRNGSeed(0)
    proposals = []; budget=_DistanceBudget(PROFILE['max_comparisons']); pair_count = 0; patch_comparisons=0
    lk = dict(winSize=(21,21), maxLevel=3, criteria=(cv.TERM_CRITERIA_EPS|cv.TERM_CRITERIA_COUNT,30,.01), minEigThreshold=1e-4)
    for (direction, end_frame), old in sorted(ends.items()):
        sign = 1 if direction == 'forward' else -1
        for delta in range(2,7):
            start_frame = end_frame + sign*delta
            new = starts.get((direction, start_frame), [])
            if not new: continue
            if len(old)>256 or len(new)>256: raise ValueError('frame-pair endpoint budget exceeded')
            pair_count += 1
            source = np.asarray([xy for _,xy in old],dtype=np.float32).reshape(-1,1,2)
            forward, status, _ = cv.calcOpticalFlowPyrLK(grays[end_frame],grays[start_frame],source,None,**lk)
            if forward is None or status is None:
                rejects['FORWARD_LK_FAILED'] += len(old); continue
            good=[]
            for i in range(len(old)):
                if not bool(status[i,0]) or not _bounds(forward[i,0],shape): rejects['FORWARD_LK_FAILED_OR_OUT_OF_BOUNDS']+=1
                else:good.append(i)
            if not good:continue
            back, bstatus, _ = cv.calcOpticalFlowPyrLK(grays[start_frame],grays[end_frame],forward[good],None,**lk)
            if back is None or bstatus is None:
                rejects['BACKWARD_LK_FAILED']+=len(good);continue
            valid=[]
            for position,i in enumerate(good):
                fb=float(np.linalg.norm(back[position,0]-source[i,0]))
                if not bool(bstatus[position,0]) or not _bounds(back[position,0],shape) or not math.isfinite(fb) or fb>1:
                    rejects['BACKWARD_LK_OR_FB_REJECTED']+=1;continue
                if not _support(masks[start_frame],forward[i,0]): rejects['PREDICTED_MASK_UNSUPPORTED']+=1;continue
                valid.append((i,fb))
            if not valid:continue
            prediction=np.asarray([forward[i,0] for i,_ in valid],dtype=float)
            measured=np.asarray([xy for _,xy in new],dtype=float)
            seed_back, seed_status, _ = cv.calcOpticalFlowPyrLK(grays[start_frame],grays[end_frame],
                measured.astype(np.float32).reshape(-1,1,2),None,**lk)
            if seed_back is None or seed_status is None:
                rejects['MEASURED_SEED_BACKWARD_FAILED']+=len(valid);continue
            new_index=_SpatialIndex(measured,budget)
            prediction_index=_SpatialIndex(prediction,budget)
            old_index=_SpatialIndex([xy for _,xy in old],budget)
            for row,(i,fb) in enumerate(valid):
                neighbors=new_index.query(prediction[row],6.)
                if not neighbors or neighbors[0][0]>2: rejects['NO_MEASURED_SEED_WITHIN_2PX']+=1;continue
                distance,j=neighbors[0]
                runner=neighbors[1][0] if len(neighbors)>1 else None
                incoming=prediction_index.query(measured[j],6.)
                incoming_runner=incoming[1][0] if len(incoming)>1 else None
                if (runner is not None and runner-distance<4) or not incoming or incoming[0][1]!=row or (incoming_runner is not None and incoming_runner-distance<4):
                    rejects['AMBIGUOUS_ENDPOINT']+=1;continue
                if not bool(seed_status[j,0]) or not _bounds(seed_back[j,0],shape):
                    rejects['MEASURED_SEED_BACKWARD_FAILED']+=1;continue
                reverse=old_index.query(seed_back[j,0],5.)
                seed_error=next((d for d,index in reverse if index==i),None)
                if seed_error is None or seed_error>1:
                    rejects['MEASURED_SEED_BACKWARD_DISAGREEMENT']+=1;continue
                if reverse[0][1]!=i or (len(reverse)>1 and reverse[1][0]-seed_error<4):
                    rejects['AMBIGUOUS_REVERSE_ENDPOINT']+=1;continue
                s,t,score=_patch_metrics(grays[end_frame],grays[start_frame],old[i][1],new[j][1],cv)
                patch_comparisons+=1
                if patch_comparisons>PROFILE['max_patch_comparisons']:raise ValueError('patch comparison budget exceeded')
                if min(s,t)<5:rejects['LOW_PATCH_TEXTURE']+=1;continue
                if score is None or score<.85:rejects['PATCH_ZNCC_REJECTED']+=1;continue
                ambiguous_texture=False
                for _,alternative in new_index.query(prediction[row],16.):
                    if alternative==j:continue
                    patch_comparisons+=1
                    if patch_comparisons>PROFILE['max_patch_comparisons']:raise ValueError('patch comparison budget exceeded')
                    astd,bstd,other_score=_patch_metrics(grays[end_frame],grays[start_frame],old[i][1],new[alternative][1],cv)
                    if min(astd,bstd)>=5 and other_score is not None and other_score>=.85 and other_score>=score-.05:
                        ambiguous_texture=True;break
                if not ambiguous_texture:
                    for _,alternative in old_index.query(seed_back[j,0],16.):
                        if alternative==i:continue
                        patch_comparisons+=1
                        if patch_comparisons>PROFILE['max_patch_comparisons']:raise ValueError('patch comparison budget exceeded')
                        astd,bstd,other_score=_patch_metrics(grays[end_frame],grays[start_frame],old[alternative][1],new[j][1],cv)
                        if min(astd,bstd)>=5 and other_score is not None and other_score>=.85 and other_score>=score-.05:
                            ambiguous_texture=True;break
                if ambiguous_texture:rejects['AMBIGUOUS_LOCAL_PATCH_TEXTURE']+=1;continue
                if old[i][0]==new[j][0]:raise ValueError('self edge violates track chronology')
                metrics=dict(fb_error_pixels=fb,predicted_to_measured_seed_pixels=distance,
                    seed_back_to_source_pixels=seed_error,outgoing_runner_distance_pixels=runner,
                    incoming_runner_distance_pixels=incoming_runner,source_patch_std=s,target_patch_std=t,patch_zncc=score)
                metrics.update(outgoing_runner_distance_lower_bound_pixels=6. if runner is None else None,
                    incoming_runner_distance_lower_bound_pixels=6. if incoming_runner is None else None,
                    reverse_runner_distance_pixels=reverse[1][0] if len(reverse)>1 else None,
                    reverse_runner_distance_lower_bound_pixels=5. if len(reverse)<2 else None)
                proposals.append(dict(source_track_id=old[i][0],target_track_id=new[j][0],direction=direction,
                    from_track_id=old[i][0],to_track_id=new[j][0],from_frame=end_frame,to_frame=start_frame,
                    xy_from=(old[i][1]+.5).tolist(),xy_to=(new[j][1]+.5).tolist(),
                    missing_frame_indices=list(range(end_frame+sign,start_frame,sign)),missing_count=delta-1,
                    metrics=metrics,status='CANDIDATE_NOT_IDENTITY_VERIFIED',
                    end_frame=end_frame,start_frame=start_frame,frame_delta=delta,
                    missing_frames=list(range(end_frame+sign,start_frame,sign)),gap_status='UNKNOWN',
                    fb_error_pixels=fb,predicted_to_measured_seed_pixels=distance,
                    seed_back_to_source_pixels=seed_error,
                    outgoing_runner_distance_pixels=runner,incoming_runner_distance_pixels=incoming_runner,
                    source_patch_std=s,target_patch_std=t,patch_zncc=score,
                    measured_source_xy=(old[i][1]+.5).tolist(),measured_target_xy=(new[j][1]+.5).tolist(),
                    prediction_is_observation=False,correspondence_verified=False,geometry_acceptance=False))
                if len(proposals)>PROFILE['max_candidate_edges']: raise ValueError('candidate edge budget exceeded')
    outgoing=Counter(p['source_track_id'] for p in proposals); incoming=Counter(p['target_track_id'] for p in proposals)
    links=[]
    for p in proposals:
        if outgoing[p['source_track_id']]!=1 or incoming[p['target_track_id']]!=1:
            rejects['GLOBAL_COMPETING_ENDPOINT']+=1
        else: links.append(p)
    next_link={p['source_track_id']:p for p in links}; previous={p['target_track_id'] for p in links}
    joined=[]; visited=set()
    for ident in by_id:
        if ident in previous:continue
        current=ident; ids=[]; observations=[]; gaps=[]
        while True:
            if current in visited:raise ValueError('cycle or merge in hypotheses')
            visited.add(current);ids.append(current);observations.extend(deepcopy(by_id[current]['observations']))
            if current not in next_link:break
            edge=next_link[current];gaps.append(deepcopy(edge));current=edge['target_track_id']
        if len({_frame(o) for o in observations})!=len(observations):raise ValueError('duplicate frame in hypothesis')
        joined.append(dict(source_track_ids=ids,direction=by_id[ident]['direction'],observations=observations,
            gaps=gaps,correspondence_verified=False,geometry_acceptance=False))
    if len(visited)!=len(tracks):raise ValueError('unvisited track cycle')
    return dict(schema='bounded-temporal-reassociation-v1',status='CANDIDATE_REASSOCIATION_COMPLETE_NOT_VERIFIED',
        profile=deepcopy(PROFILE),links=links,joined_tracks=joined,rejection_counts=dict(rejects),
        counts=dict(original_tracks=len(tracks),joined_tracks=len(joined),proposed_links=len(proposals),accepted_links=len(links),
            rejected_links=len(proposals)-len(links),frame_pair_batches=pair_count,endpoint_comparisons=budget.count),
        proposal_summary=dict(proposed_links=len(proposals),globally_rejected_links=len(proposals)-len(links)),
        original_track_count=len(tracks),joined_track_count=len(joined),candidate_edge_count=len(proposals),
        accepted_link_count=len(links),frame_pair_batches=pair_count,endpoint_comparisons=budget.count,
        distance_count_rule='Actual exact norm evaluations in queried bins, including radius-filtered-out candidates and repeated queries; no virtual NxM charge.',
        patch_comparisons=patch_comparisons,
        original_tracks_modified=False,missing_observations_synthesized=False,
        correspondence_verified=False,geometry_acceptance=False,
        limitations=['Gap cause is UNKNOWN; no occlusion classification.',
            'Local LK and patches cannot establish physical identity or reject every repeated texture.',
            'Forward/backward passes remain independent; no reconstructed geometry is changed.'])
