"""Radius covers of supplied points/pixels; never a minimum skeleton or truth."""
import math
import numpy as np

MAX_POINTS = 20000
MAX_PIXELS = 4000000
MAX_RASTER_WORK = 100000000


def _points(points):
    try:
        for row in points:
            if len(row)!=2 or any(isinstance(v,(bool,np.bool_)) or
                not isinstance(v,(int,float,np.integer,np.floating)) for v in row):
                raise ValueError('finite real Nx2 points required; bool rejected')
        array=np.array(points,dtype=float,copy=True)
    except (TypeError,OverflowError) as exc:
        raise ValueError('finite real Nx2 points required') from exc
    if array.size==0 and array.shape==(0,):array=array.reshape(0,2)
    if array.ndim!=2 or array.shape[1]!=2 or len(array)>MAX_POINTS or not np.isfinite(array).all():
        raise ValueError('at most 20000 finite real Nx2 points required')
    return array


def _radius(radius,maximum=None):
    if isinstance(radius,(bool,np.bool_)) or not isinstance(radius,(int,float,np.integer,np.floating)):
        raise ValueError('finite nonnegative radius required; bool rejected')
    try:value=float(radius)
    except OverflowError as exc:raise ValueError('finite nonnegative radius required') from exc
    if not math.isfinite(value) or value<0 or (maximum is not None and value>maximum):
        raise ValueError('finite nonnegative radius within declared bound required')
    return value


def greedy_r_cover(points_xy,radius,max_centers=4096):
    """Bounded implementation of the source's r-cover condition using existing rows.

    Uses O(N) distance/witness storage and deterministic farthest-point selection;
    this is a different centre-selection strategy from official.greedy_sketch,
    not reuse of its selection algorithm. A COMPLETE result satisfies the same
    supplied-point r-cover condition, not a minimum sketch/ostov claim.

    Start at row 0; farthest-distance ties choose smallest original row index.
    Nearest witnesses are original selected row IDs, with ties choosing minimum
    original row index. Duplicate source rows are retained as separate witnesses.
    max_centers is the explicit budget; zero is allowed and returns PARTIAL for
    nonempty input. No-point distances are None rather than nonfinite numbers.
    """
    points=_points(points_xy);r=_radius(radius)
    if type(max_centers) is not int or not 0<=max_centers<=4096:
        raise ValueError('integer max_centers in 0..4096 required; bool rejected')
    n=len(points);chosen=[];nearest=np.full(n,-1,dtype=np.int64);distance=None
    while n and len(chosen)<max_centers:
        index=0 if not chosen else int(np.argmax(distance))
        chosen.append(index)
        # hypot avoids square overflow for large finite source coordinates.
        with np.errstate(over='ignore',invalid='ignore'):
            new=np.hypot(points[:,0]-points[index,0],points[:,1]-points[index,1])
        if not np.isfinite(new).all():raise ValueError('coordinate separation exceeds finite numerical range')
        if distance is None:
            distance=new;nearest.fill(index)
        else:
            update=(new<distance)|((new==distance)&(index<nearest))
            distance=np.minimum(distance,new);nearest[update]=index
        if float(distance.max())<=r:break
    maximum=0. if not n else (float(distance.max()) if distance is not None else None)
    complete=not n or (maximum is not None and maximum<=r)
    return dict(status='COMPLETE' if complete else 'PARTIAL',complete=bool(complete),
        stop_reason='ALL_POINTS_COVERED' if complete else 'MAX_CENTERS_EXHAUSTED',
        radius=r,input_count=n,max_centers=max_centers,selected_indices=chosen,
        nearest_selected_indices=nearest.tolist(),nearest_distances=[] if not n else
            (distance.tolist() if distance is not None else [None]*n),max_distance=maximum,
        minimum_cover_claim=False,coordinates_modified=False)


def raster_radius_coverage(mask,points_xy,radius):
    """Intersect radius disks with a target mask at pixel centres (x+.5,y+.5).

    Input points use the same continuous image domain [0,width)x[0,height).
    Every disk is evaluated in its clipped pixel bounding box; summed box areas
    are the work bound, including repeated/overlapping disks. No HW x N matrix.
    coverage_map: background=0, covered target=1, uncovered target=2.
    No target yields NO_TARGET and ratio None. Empty points are valid.
    """
    if not isinstance(mask,np.ndarray) or mask.dtype!=np.uint8 or mask.ndim!=2 or not mask.size or mask.size>MAX_PIXELS:
        raise ValueError('nonempty uint8 2D mask with at most four million pixels required')
    if not np.isin(mask,[0,255]).all():raise ValueError('binary mask labels must be 0/255')
    points=_points(points_xy);r=_radius(radius,128);height,width=mask.shape
    if len(points) and (np.any(points[:,0]<0) or np.any(points[:,0]>=width) or np.any(points[:,1]<0) or np.any(points[:,1]>=height)):
        raise ValueError('point outside continuous image bounds')
    boxes=[];work=0
    for x,y in points:
        left=max(0,math.ceil(x-r-.5));right=min(width-1,math.floor(x+r-.5))
        top=max(0,math.ceil(y-r-.5));bottom=min(height-1,math.floor(y+r-.5))
        area=max(0,right-left+1)*max(0,bottom-top+1);work+=area
        boxes.append((left,right,top,bottom))
    if work>MAX_RASTER_WORK:raise ValueError('summed disk bounding-box work exceeds 100 million pixels')
    covered=np.zeros(mask.shape,dtype=bool)
    for (x,y),(left,right,top,bottom) in zip(points,boxes):
        if right<left or bottom<top:continue
        dx=np.arange(left,right+1,dtype=float)+.5-x
        dy=np.arange(top,bottom+1,dtype=float)+.5-y
        disk=dy[:,None]**2+dx[None,:]**2<=r*r
        covered[top:bottom+1,left:right+1]|=disk
    target=mask==255;total=int(target.sum());count=int(np.count_nonzero(target&covered))
    output=np.zeros(mask.shape,dtype=np.uint8);output[target]=2;output[target&covered]=1
    return dict(status='NO_TARGET' if not total else 'EVALUATED',radius=r,point_count=len(points),
        target_pixels=total,covered_pixels=count,uncovered_pixels=total-count,
        coverage_ratio=float(count/total) if total else None,coverage_map=output,
        bounding_box_work=work,pixel_center_convention='(column+0.5,row+0.5)',
        coordinates_modified=False,physical_accuracy_claim=False)
