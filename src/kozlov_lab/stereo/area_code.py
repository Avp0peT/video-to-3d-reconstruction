"""Ordinary unsigned triangle-area ratios, with explicit engineering gates.

Full code compares every ordered pair of distinct labelled triangles. Four-point
scope retains only pairs whose union has four vertices; it is not the full code
and has no same-epsilon numerical equivalence guarantee.
"""
from itertools import combinations
import math
import numpy as np


def _points(value):
    objects=np.asarray(value,dtype=object)
    if any(isinstance(v,(bool,np.bool_)) for v in objects.flat):
        raise ValueError('bool coordinates rejected')
    raw=np.asarray(value)
    if raw.ndim!=2 or raw.shape[1]!=2 or not 4<=len(raw)<=12 or raw.dtype.kind not in 'iuf' or raw.dtype.kind=='b':
        raise ValueError('4..12 finite real non-bool Nx2 points required')
    points=np.array(raw,dtype=float,copy=True)
    if not np.isfinite(points).all() or np.any(np.abs(points)>1000000):
        raise ValueError('finite coordinates bounded by 1000000 required')
    return points


def _area(points,triangle):
    a,b,c=points[list(triangle)]
    return abs(float((b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])))/2


def compare_area_codes(source,target,*,scope='full',epsilon=.08,min_denominator_area=32.0):
    """Compare every declared slot; thresholds are engineering acceptance gates.

    A zero denominator in both views is undefined, not fabricated as a ratio.
    Any unsupported denominator or nonfinite ratio prevents PASSED. No slot is
    skipped on a previous failure. Physical correspondence remains unverified.
    """
    a=_points(source);b=_points(target)
    if a.shape!=b.shape:raise ValueError('equal labelled point counts required')
    if scope not in ('full','four_point'):raise ValueError('scope must be full or four_point')
    for name,value in (('epsilon',epsilon),('min_denominator_area',min_denominator_area)):
        if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,float,np.integer,np.floating)) or not math.isfinite(float(value)) or value<0:
            raise ValueError('finite nonnegative '+name+' required')
    triangles=list(combinations(range(len(a)),3));left=[_area(a,t) for t in triangles];right=[_area(b,t) for t in triangles]
    total=checked=undefined=zero_mismatch=unstable=nonfinite=0;maximum=None;worst=None
    for denominator,den_ids in enumerate(triangles):
        for numerator,num_ids in enumerate(triangles):
            if numerator==denominator or (scope=='four_point' and len(set(den_ids)|set(num_ids))!=4):continue
            total+=1;d1=left[denominator];d2=right[denominator]
            if d1==0 and d2==0:undefined+=1;continue
            if d1==0 or d2==0:zero_mismatch+=1;continue
            if min(d1,d2)<min_denominator_area:unstable+=1;continue
            q1=left[numerator]/d1;q2=right[numerator]/d2;difference=abs(q1-q2)
            if not all(math.isfinite(v) for v in (q1,q2,difference)):nonfinite+=1;continue
            checked+=1
            if maximum is None or difference>maximum:
                maximum=difference;worst=dict(numerator_triangle=list(num_ids),denominator_triangle=list(den_ids),
                    source_numerator_area=left[numerator],target_numerator_area=right[numerator],
                    source_denominator_area=d1,target_denominator_area=d2,source_ratio=q1,target_ratio=q2,difference=difference)
    status=('NONFINITE_RATIO' if nonfinite else 'ZERO_DENOMINATOR_MISMATCH' if zero_mismatch else
            'UNSTABLE_DENOMINATOR' if unstable else 'NO_DEFINED_STABLE_SLOTS' if not checked else
            'CODE_MISMATCH' if maximum>epsilon else 'PASSED')
    return dict(status=status,scope=scope,code_scope='ordinary_unsigned',point_count=len(a),triangle_count=len(triangles),
        epsilon=float(epsilon),min_denominator_area=float(min_denominator_area),
        denominator_gate='Engineering stability gate; not part of the exact book code definition.',
        total_slots=total,checked=checked,checked_slots=checked,comparisons=checked,
        undefined=undefined,undefined_slots=undefined,zero_denominator_mismatch_slots=zero_mismatch,
        unstable_denominator_slots=unstable,nonfinite_slots=nonfinite,
        unsupported_slots=zero_mismatch+unstable+nonfinite,max_difference=maximum,worst_slot=worst,
        identity_verified=False,geometry_acceptance=False,native_injected=False)
