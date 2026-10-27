"""Same-frame background topology and narrow opposed-bank diagnostics.

Nothing here edits segmentation or proves 3D free space. Bank-centre distance
is a directional upper bound, not a medial-axis or physical gap width. Even
subresolution source background is protected against a later foreground union.
"""
from dataclasses import asdict, dataclass
import hashlib
import math
import numpy as np


@dataclass(frozen=True)
class GapPolicyNew:
    max_span_fraction: float = .04
    min_span_px: float = 6.
    max_span_px: float = 48.
    uncertainty_px: float = 3.
    uncertainty_width_fraction: float = .2
    subresolution_px: float = 3.
    max_pixels: int = 4_194_304
    max_skew_pixels: int = 8_388_608

    def __post_init__(self):
        for key in ('max_span_fraction','min_span_px','max_span_px','uncertainty_px',
                    'uncertainty_width_fraction','subresolution_px'):
            value=getattr(self,key)
            if type(value) not in (int,float) or not math.isfinite(value):
                raise ValueError('finite narrow-gap policy required: '+key)
        if (not 0 < self.max_span_fraction <= .5 or not 0 < self.min_span_px <= self.max_span_px <= 128
                or not 0 <= self.uncertainty_px <= 100 or not 0 < self.uncertainty_width_fraction <= .5
                or not 0 < self.subresolution_px <= 20):
            raise ValueError('invalid narrow-gap policy range')
        for key,ceiling in (('max_pixels',16_777_216),('max_skew_pixels',33_554_432)):
            value=getattr(self,key)
            if type(value) is not int or not 1 <= value <= ceiling:
                raise ValueError('bounded positive pixel budget required: '+key)


KIND_NAMES={0:'NOT_VALID_BACKGROUND',1:'EXTERIOR',2:'ENCLOSED',3:'UNRESOLVED'}
DIRECTION_NAMES={-1:'NONE',0:'HORIZONTAL',1:'VERTICAL',2:'DIAGONAL_DOWN',3:'DIAGONAL_UP'}
PRESERVED='GAP_PRESERVATION_PASSED_NOT_GEOMETRY_ACCEPTED'
REJECTED='GAP_PRESERVATION_REJECTED'


def _bool_mask(value,shape=None):
    array=np.asarray(value)
    if (array.dtype!=np.bool_ or array.ndim!=2 or not array.size
            or (shape is not None and array.shape!=shape)):
        raise ValueError('nonempty same-grid bool masks required')
    return array


def _cv_module(cv):
    if cv is None:
        from integrations.demo_pipeline.temporal_review import load_cv
        cv=load_cv([])
    if cv is None:raise RuntimeError('existing CPU OpenCV unavailable')
    return cv


def _digest(array):
    return hashlib.sha256(array.tobytes(order='C')).hexdigest()


def _components(mask,valid,cv):
    background=(~mask)&valid
    count,labels,stats,_=cv.connectedComponentsWithStats(background.astype(np.uint8),connectivity=4)
    border=np.zeros(count,bool)
    border[np.unique(np.r_[labels[0],labels[-1],labels[:,0],labels[:,-1]])]=True
    border[0]=False
    unknown=np.pad(~valid,1,constant_values=False)
    adjacent=unknown[:-2,1:-1]|unknown[2:,1:-1]|unknown[1:-1,:-2]|unknown[1:-1,2:]
    touches_unknown=np.zeros(count,bool)
    touches_unknown[np.unique(labels[background&adjacent])]=True;touches_unknown[0]=False
    kinds=np.zeros(count,np.uint8)
    kinds[1:]=2
    kinds[touches_unknown]=3
    # A known valid path already proves exterior membership. Unknown adjacency
    # elsewhere cannot invalidate that path; record both facts independently.
    kinds[border]=1
    return dict(labels=labels,component_classes=kinds,component_kind=kinds[labels],
                component_touches_border=border,component_touches_unknown=touches_unknown,
                component_areas=stats[:,cv.CC_STAT_AREA].copy())


def _scan_last(foreground,valid):
    """Nearest opposed foreground sites with no invalid pixel between the banks."""
    length=foreground.shape[-1];x=np.arange(length,dtype=np.int32)
    left=np.maximum.accumulate(np.where(foreground,x,-1),axis=-1)
    right=np.minimum.accumulate(np.where(foreground,x,length)[...,::-1],axis=-1)[...,::-1]
    bad_left=np.maximum.accumulate(np.where(~valid,x,-1),axis=-1)
    bad_right=np.minimum.accumulate(np.where(~valid,x,length)[...,::-1],axis=-1)[...,::-1]
    bracket=((left>=0)&(right<length)&(left<x)&(x<right)&(bad_left<left)&(bad_right>right)
             &valid&~foreground)
    return left,right,bracket


def detect_narrow_gaps(mask,validity,policy=GapPolicyNew(),cv=None,check=lambda:None):
    if not isinstance(policy,GapPolicyNew):raise ValueError('GapPolicyNew required')
    mask=_bool_mask(mask);valid=_bool_mask(validity,mask.shape)
    if mask.size>policy.max_pixels:raise ValueError('narrow-gap pixel budget exceeded')
    if np.any(mask&~valid):raise ValueError('foreground cannot occupy unknown domain')
    h,w=mask.shape
    if h*(w+h-1)>policy.max_skew_pixels:raise ValueError('four-direction diagonal budget exceeded')
    check();cv=_cv_module(cv);topology=_components(mask,valid,cv)
    kind=topology['component_kind'];exterior=kind==1;enclosed=kind==2
    unresolved_background=kind==3;unresolved=unresolved_background|~valid
    yy,xx=np.indices(mask.shape,dtype=np.int32)
    width=np.full(mask.shape,np.inf,np.float64);open_span=np.full(mask.shape,np.nan,np.float64)
    direction=np.full(mask.shape,-1,np.int8)
    bank_a=np.full((*mask.shape,2),-1,np.int32);bank_b=np.full((*mask.shape,2),-1,np.int32)
    def accept(left,right,bracket,code,step,ax,ay,bx,by):
        span=(right-left)*step
        take=bracket&(span<width)
        width[take]=span[take];open_span[take]=(span-step)[take];direction[take]=code
        bank_a[:,:,0][take]=ax[take];bank_a[:,:,1][take]=ay[take]
        bank_b[:,:,0][take]=bx[take];bank_b[:,:,1][take]=by[take]
    left,right,bracket=_scan_last(mask,valid)
    accept(left,right,bracket,0,1.,left,yy,right,yy);check()
    left,right,bracket=(x.T for x in _scan_last(mask.T,valid.T))
    accept(left,right,bracket,1,1.,xx,left,xx,right);check()
    for code,dx in ((2,1),(3,-1)):
        column=xx+(h-1-yy) if dx==1 else xx+yy
        fg_skew=np.zeros((h,w+h-1),bool);valid_skew=np.zeros_like(fg_skew)
        fg_skew[yy,column]=mask;valid_skew[yy,column]=valid
        l,r,b=_scan_last(fg_skew.T,valid_skew.T)
        left,right,bracket=l.T[yy,column],r.T[yy,column],b.T[yy,column]
        accept(left,right,bracket,code,math.sqrt(2),xx+dx*(left-yy),left,xx+dx*(right-yy),right)
        del fg_skew,valid_skew,l,r,b
        check()
    has_span=np.isfinite(width)
    y,x=np.nonzero(mask)
    extent=int(max(x.max()-x.min()+1,y.max()-y.min()+1)) if len(x) else 0
    maximum=float(np.clip(extent*policy.max_span_fraction,policy.min_span_px,policy.max_span_px))
    narrow_external=exterior&has_span&(width<=maximum)
    protected=narrow_external|enclosed
    subresolution=has_span&(open_span<policy.subresolution_px)
    margin=np.full(mask.shape,np.nan,np.float64)
    margin[has_span]=np.minimum(policy.uncertainty_px,width[has_span]*policy.uncertainty_width_fraction)
    width[~has_span]=np.nan
    distance=cv.distanceTransform(np.pad((~mask&valid).astype(np.uint8),1),cv.DIST_L2,cv.DIST_MASK_PRECISE)[1:-1,1:-1]
    local_max=cv.dilate(distance,np.ones((3,3),np.uint8))
    ridge=narrow_external&(distance>=local_max)&(distance>0)
    classes=topology['component_classes'];known_width=width[narrow_external]
    stats=dict(pixels=mask.size,foreground_pixels=int(mask.sum()),valid_background_pixels=int((~mask&valid).sum()),
        invalid_pixels=int((~valid).sum()),background_components=len(classes)-1,
        exterior_components=int((classes==1).sum()),enclosed_components=int((classes==2).sum()),
        unresolved_components=int((classes==3).sum()),exterior_pixels=int(exterior.sum()),enclosed_pixels=int(enclosed.sum()),
        unresolved_background_pixels=int(unresolved_background.sum()),narrow_external_pixels=int(narrow_external.sum()),
        protected_background_pixels=int(protected.sum()),subresolution_narrow_pixels=int((narrow_external&subresolution).sum()),
        centerline_candidate_pixels=int(ridge.sum()),object_extent_pixels=extent,maximum_bank_span_pixels=maximum,
        narrow_bank_span_min=float(known_width.min()) if len(known_width) else None,
        narrow_bank_span_median=float(np.median(known_width)) if len(known_width) else None,
        narrow_bank_span_max=float(known_width.max()) if len(known_width) else None,
        directions_tested=4,connectivity=4,
        width_semantics='Minimum observed foreground pixel-centre bank distance over four directions; directional upper bound, not true gap width.',
        open_span_semantics='Bank span minus one direction step; sampling convention for subresolution flag, not a measured physical width.',
        uncertainty_semantics='Display/operation limit only; reduced margin does not certify background or 3D free space.',
        centerline_semantics='3x3 distance-ridge candidates with plateaus, not a strict medial axis.',
        component_semantics='Known valid path to image boundary proves EXTERIOR; otherwise unknown-adjacent is UNRESOLVED, remainder ENCLOSED.')
    check()
    return dict(status='NARROW_GAP_DIAGNOSTIC_NOT_GEOMETRY_ACCEPTED',geometry_accepted=False,
        source_mask_unchanged=True,source_mask_sha256=_digest(mask),validity_sha256=_digest(valid),policy=asdict(policy),
        narrow_external=narrow_external,protected_background=protected,exterior=exterior,enclosed=enclosed,
        unresolved=unresolved,unresolved_background=unresolved_background,width_upper_bound=width,open_span=open_span,
        uncertainty_margin=margin,local_uncertainty_margin=margin,subresolution=subresolution,
        centerline_candidate=ridge,bank_a_xy=bank_a,bank_b_xy=bank_b,direction=direction,
        kind_names=KIND_NAMES.copy(),direction_names=DIRECTION_NAMES.copy(),stats=stats,**topology)


def evaluate_gap_preservation(source,candidate,valid,diagnostics,cv=None):
    """Refuse added foreground that erases source gaps or cuts known exterior paths.

    Compare only this frame. No cross-frame hole-count constraint is imposed.
    Source foreground removal is reported but is the caller's separate policy.
    """
    source=_bool_mask(source);candidate=_bool_mask(candidate,source.shape);valid=_bool_mask(valid,source.shape)
    if source.size>16_777_216:raise ValueError('preservation pixel budget exceeded')
    if np.any((source|candidate)&~valid):raise ValueError('foreground cannot occupy unknown domain')
    if (diagnostics.get('source_mask_sha256')!=_digest(source)
            or diagnostics.get('validity_sha256')!=_digest(valid)):
        raise ValueError('gap diagnostics do not bind these source/validity pixels')
    cv=_cv_module(cv);original=_components(source,valid,cv);after=_components(candidate,valid,cv)
    narrow=_bool_mask(diagnostics['narrow_external'],source.shape)
    enclosed=_bool_mask(diagnostics['enclosed'],source.shape)
    protected=_bool_mask(diagnostics['protected_background'],source.shape)
    exterior=original['component_kind']==1
    if (not np.array_equal(enclosed,original['component_kind']==2)
            or not np.array_equal(protected,narrow|enclosed)
            or np.any(narrow&~exterior) or np.any(protected&(source|~valid))
            or not np.array_equal(diagnostics['labels'],original['labels'])):
        raise ValueError('inconsistent source background diagnostic arrays')
    gained=candidate&~source&valid;removed=source&~candidate&valid
    filled=protected&gained
    surviving_bg=~candidate&valid
    disconnected=exterior&surviving_bg&(after['component_kind']!=1)
    labels=original['labels'];component_count=len(original['component_classes'])
    survivors=np.bincount(labels[surviving_bg],minlength=component_count)
    erased=np.flatnonzero((np.arange(component_count)>0)&(survivors==0))
    violation=[]
    if filled.any():violation.append('PROTECTED_BACKGROUND_FILLED')
    if disconnected.any():violation.append('EXTERIOR_BACKGROUND_PATH_DISCONNECTED')
    if len(erased):violation.append('SOURCE_BACKGROUND_COMPONENT_ERASED')
    stats=dict(foreground_added_pixels=int(gained.sum()),foreground_removed_pixels=int(removed.sum()),
        protected_background_pixels=int(protected.sum()),protected_filled_pixels=int(filled.sum()),
        narrow_external_filled_pixels=int((narrow&gained).sum()),enclosed_filled_pixels=int((enclosed&gained).sum()),
        exterior_path_disconnected_pixels=int(disconnected.sum()),
        disconnected_source_component_ids=np.unique(labels[disconnected]).astype(int).tolist(),
        disconnected_candidate_component_ids=np.unique(after['labels'][disconnected]).astype(int).tolist(),
        erased_source_component_ids=erased.astype(int).tolist(),erased_source_components=len(erased),
        source_background_components=component_count-1,candidate_background_components=len(after['component_classes'])-1,
        source_enclosed_components=int((original['component_classes']==2).sum()),
        candidate_enclosed_components=int((after['component_classes']==2).sum()),
        source_narrow_subresolution_pixels=int((narrow&np.asarray(diagnostics['subresolution'],bool)).sum()),
        comparison_scope='SAME_FRAME_SOURCE_BACKGROUND_PRESERVATION_ONLY')
    ok=not violation
    return dict(status=PRESERVED if ok else REJECTED,preserved=ok,ok=ok,geometry_accepted=False,
        source_unchanged=True,violations=violation,stats=stats,protected_filled=filled,
        path_disconnected=disconnected,source_mask_sha256=_digest(source),candidate_mask_sha256=_digest(candidate),
        validity_sha256=_digest(valid),foreground_removal_policy='REPORTED_SEPARATELY_NOT_ENFORCED_HERE')
