"""Native foreground depth availability and sampled correlated cross-view evidence."""
import numpy as np
from .local_depth_search import _view, _rays
from .point_depth_evidence import assess_points

COVERAGE={0:'BACKGROUND_VALID',1:'INVALID_PIXEL',2:'FOREGROUND_WITHOUT_NATIVE_DEPTH',
          3:'NATIVE_DEPTH_UNSAMPLED',4:'NATIVE_NO_SUFFICIENT_OTHER_SUPPORT',
          5:'NATIVE_SUPPORTED',6:'NATIVE_CONFLICT',7:'NATIVE_OCCLUDED',8:'NATIVE_UNKNOWN'}


def classify_depth_coverage(views,*,stride=3,max_points=200000,check=lambda:None):
    """Classify full native domains, then audit a fixed native pixel sampling grid.

    CAPACITY_LIMIT returns availability maps without selecting a truncated subset.
    Background and invalid pixels are never counted as object depth gaps.
    """
    check()
    if type(stride) is not int or stride<1 or type(max_points) is not int or not 1<=max_points<=200000:
        raise ValueError('positive integer stride and max_points<=200000 required')
    if not isinstance(views,list) or not 1<=len(views)<=128:raise ValueError('one to 128 views required')
    selected=[];ids=[];maps={};samples=[];summaries=[];requested=0
    for raw in views:
        check();v=_view(raw)
        valid=v.get('validity')
        if not isinstance(valid,np.ndarray) or valid.dtype!=bool or valid.shape!=v['mask'].shape:raise ValueError('explicit native validity required')
        available=v.get('depth_available')
        if type(available) is not bool:raise ValueError('explicit depth_available boolean required')
        iid=v['view_id']
        if iid in ids:raise ValueError('duplicate view ID')
        selected.append(v);ids.append(iid)
        fg=v['mask']&valid;native=fg&(v['depth']>0)&available
        image_map=np.zeros(valid.shape,np.uint8);image_map[~valid]=1
        image_map[fg]=2;image_map[native]=3
        maps['coverage_view_'+str(iid)]=image_map
        sample=np.zeros(valid.shape,bool);sample[::stride,::stride]=native[::stride,::stride]
        y,x=np.nonzero(sample);samples.append((x,y));requested+=len(x)
        summaries.append(dict(view_id=iid,total_pixels=int(valid.size),mask_foreground_pixels=int(v['mask'].sum()),valid_mask_foreground_pixels=int(fg.sum()),foreground_without_native_depth=int((fg&~native).sum()),native_depth_foreground_pixels=int(native.sum()),requested_sampled_native_pixels=len(x),depth_available=available,processed_sampled_native_pixels=0,coverage_map_counts={name:int(np.count_nonzero(image_map==code)) for code,name in COVERAGE.items()},
            background_pixels=int((~v['mask']).sum()),invalid_pixels=int((~valid).sum()),background_valid_pixels=int((valid&~v['mask']).sum()),
            denominator_semantics='Depth-availability denominator is valid mask foreground; sampling denominator is sampled native foreground. Background/invalid counts are not object holes. Background and invalid totals can overlap.'))
    summary=dict(total_valid_mask_foreground_pixels=sum(s['valid_mask_foreground_pixels'] for s in summaries),total_foreground_without_native_depth=sum(s['foreground_without_native_depth'] for s in summaries),total_native_depth_foreground_pixels=sum(s['native_depth_foreground_pixels'] for s in summaries),requested_sampled_native_pixels=requested,processed_sampled_native_pixels=0,
        sampling_denominator='Native depth foreground pixels on the fixed stride grid, not all pixels and not the entire object',per_view=summaries)
    common=dict(**maps,view_ids=ids,coverage_legend=dict(COVERAGE),stride=stride,max_points=max_points,summary=summary,geometry_accepted=False,complete=False,
        semantics='Availability plus sampled native-depth consistency only. Shared depths/cameras are correlated; supported does not establish independent physical truth. No new points or depths are inferred.',
        classification_precedence='supported; conflict; some but insufficient support; occluded; unknown; no other evidence. Native unsampled pixels have no cross-view label.')
    if requested>max_points:
        return dict(**common,status='CAPACITY_LIMIT',capacity_required=requested,capacity_available=max_points,points=np.empty((0,3),np.float64),source_view_ids=np.empty(0,np.int64),source_uv=np.empty((0,2),np.int32),cross_view_evaluated=False)
    pointblocks=[];origins=[];uvblocks=[]
    for v,(x,y) in zip(selected,samples):
        for start in range(0,len(x),8192):
            check();bx,by=x[start:start+8192],y[start:start+8192]
            rays=_rays(v,bx,by);z=v['depth'][by,bx]
            pointblocks.append((rays*z[:,None])@v['R']+v['C'])
            uvblocks.append(np.column_stack((bx,by)).astype(np.int32))
            origins.append(np.full(len(bx),v['view_id'],np.int64))
    points=np.concatenate(pointblocks) if pointblocks else np.empty((0,3),np.float64)
    sourceids=np.concatenate(origins) if origins else np.empty(0,np.int64)
    uvs=np.concatenate(uvblocks) if uvblocks else np.empty((0,2),np.int32)
    evidence=assess_points(points,selected,sourceids,check=check)
    classification=np.full(len(points),4,np.uint8)
    support=evidence['depth_support_count'];occlusion=evidence['occluded_count']
    unknown=evidence['out_of_view_count']+evidence['no_native_depth_count']+evidence['unknown_discontinuity_count']
    classification[(support==0)&(unknown>0)]=8
    classification[(support==0)&(occlusion>0)]=7
    classification[evidence['conflict']]=6;classification[evidence['supported']]=5
    for s in summaries:
        iid=s['view_id'];own=sourceids==iid;uv=uvs[own]
        maps['coverage_view_'+str(iid)][uv[:,1],uv[:,0]]=classification[own]
        s['processed_sampled_native_pixels']=int(own.sum())
        s['coverage_map_counts']={name:int(np.count_nonzero(maps['coverage_view_'+str(iid)]==code)) for code,name in COVERAGE.items()}
        s['sampled_classification_counts']={name:int(np.count_nonzero(classification[own]==code)) for code,name in COVERAGE.items() if code>=4}
    summary['processed_sampled_native_pixels']=len(points)
    summary['sampled_classification_counts']={name:int(np.count_nonzero(classification==code)) for code,name in COVERAGE.items() if code>=4}
    arrays={'audit_'+key:value for key,value in evidence.items() if isinstance(value,np.ndarray)}
    metadata={key:value for key,value in evidence.items() if not isinstance(value,np.ndarray)}
    return dict(**common,**arrays,points=points,source_view_ids=sourceids,source_uv=uvs,sampled_coverage_classification=classification,
        audit_metadata=metadata,status='NATIVE_DEPTH_COVERAGE_CLASSIFIED_NOT_GEOMETRY_ACCEPTED',cross_view_evaluated=True)
