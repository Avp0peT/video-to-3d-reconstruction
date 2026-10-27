"""Bounded diagnostic hypotheses, never a calibration or BA authorization.

Normalized eight-point source: Hartley, ICCV 1995 fundamental.pdf.
Distortion convention: COLMAP SIMPLE_RADIAL, xd = u * (1 + k*r*r).
"""
import numbers
import numpy as np
from .focal_profile import profile_fundamental


def _numeric(value, shape=None):
    if any(isinstance(x, (bool, np.bool_)) for x in np.asarray(value, dtype=object).flat):
        raise ValueError('Boolean geometry input')
    a = np.asarray(value)
    if a.dtype.kind not in 'iuf' or (shape is not None and a.shape != shape):
        raise ValueError('Invalid numeric shape or type')
    a = a.astype(float, copy=True)
    if not np.isfinite(a).all():
        raise ValueError('Nonfinite geometry input')
    return a


def invert_radial(xy, principal, focal: float, k: float) -> np.ndarray:
    """Invert only the monotonic branch using 80 bounded bisections."""
    xy = _numeric(xy)
    p = _numeric(principal, (2,))
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError('Expected Nx2 pixels')
    if (isinstance(focal, (bool, np.bool_)) or isinstance(k, (bool, np.bool_))
            or not isinstance(focal, numbers.Real) or not isinstance(k, numbers.Real)
            or not np.isfinite(focal) or focal <= 0 or focal > 1e8 or not np.isfinite(k)):
        raise ValueError('Invalid radial parameters')
    if k == 0:
        return xy.copy()
    d = (xy-p)/focal
    rd = np.linalg.norm(d, axis=1)
    hi = np.full(len(rd), np.sqrt(-1/(3*k))) if k < 0 else rd.copy()
    if k < 0 and np.any(rd >= hi*(1+k*hi*hi)):
        raise ValueError('Outside negative-k monotonic radial domain')
    lo = np.zeros(len(rd))
    for _ in range(80):
        mid = (lo+hi)/2
        less = mid*(1+k*mid*mid) < rd
        lo = np.where(less, mid, lo)
        hi = np.where(less, hi, mid)
    r = (lo+hi)/2
    reconstructed = r*(1+k*r*r)
    if not np.isfinite(reconstructed).all() or not np.allclose(reconstructed, rd, rtol=1e-12, atol=1e-12):
        raise ValueError('Radial inverse forward substitution failed')
    scale = np.divide(r, rd, out=np.ones_like(r), where=rd > 0)
    answer = p + focal*d*scale[:, None]
    if not np.isfinite(answer).all():
        raise ValueError('Nonfinite radial inverse')
    return answer


def _hom(x):
    return np.column_stack((x, np.ones(len(x))))


def _normalize(x):
    c = x.mean(axis=0)
    radius = np.linalg.norm(x-c, axis=1).mean()
    if radius <= 1e-12:
        raise ValueError('Collapsed coordinates')
    s = np.sqrt(2)/radius
    T = np.array([[s, 0, -s*c[0]], [0, s, -s*c[1]], [0, 0, 1.]])
    return (_hom(x)@T.T)[:, :2], T


def _fit(a, b, model):
    x, T = _normalize(a); y, U = _normalize(b)
    if len(x) < 8:
        raise ValueError('At least eight training rows required')
    if model == 'F':
        A = np.einsum('ni,nj->nij', _hom(y), _hom(x)).reshape(-1, 9)
    else:
        X = _hom(x); z = np.zeros_like(X)
        A = np.stack((np.column_stack((-X,z,y[:,0,None]*X)),
                      np.column_stack((z,-X,y[:,1,None]*X))), axis=1).reshape(-1,9)
    rank = int(np.linalg.matrix_rank(A))
    if rank < 8:
        raise ValueError('Design rank below eight')
    _, _, V = np.linalg.svd(A, full_matrices=False)
    M = V[-1].reshape(3,3)
    if model == 'F':
        u,s,v = np.linalg.svd(M); s[-1] = 0
        M = U.T @ (u@np.diag(s)@v) @ T
    else:
        M = np.linalg.solve(U, M@T)
    norm = np.linalg.norm(M)
    if not np.isfinite(M).all() or not np.isfinite(norm) or norm <= 0:
        raise ValueError('Invalid fitted matrix')
    M /= norm
    return M, rank


def _stats(values, total):
    v = np.asarray(values)
    v = v[np.isfinite(v)]
    scale = float(np.max(np.abs(v))) if len(v) else 0.
    normalized = v/scale if scale else v
    return dict(valid_count=int(len(v)), failed_or_undefined_count=int(total-len(v)),
                mean=float(scale*normalized.mean()) if len(v) else None,
                median=float(np.median(v)) if len(v) else None,
                rms=float(scale*np.sqrt(np.mean(normalized*normalized))) if len(v) else None,
                maximum=float(v.max()) if len(v) else None)


def _profile(F,p,edge,config=3):
    return profile_fundamental(F,p,p,edge,config=config)


def _summary(profile):
    result = {k:profile[k] for k in ('status','reason_code','minimum_grid_index',
              'minimum_qess','focal_observability_status') if k in profile}
    if profile['curve']:
        result['minimum'] = profile['curve'][profile['minimum_grid_index']]
    return result


def _fold(a,b,ids,train,test,p,f,edge):
    out = dict(status='EVALUATED', training_feature_pairs=ids[train].tolist(),
               heldout_feature_pairs=ids[test].tolist(), models={})
    if len(train)<12 or len(test)<12:
        out['status']='INSUFFICIENT'; return out
    x,y = _hom(a[test]),_hom(b[test])
    for kind in ('F','H','R'):
        m = dict(status='FAILED', failed_or_undefined_count=int(len(test)))
        out['models'][kind]=m
        try:
            if kind in ('F','H'):
                M,rank = _fit(a[train],b[train],kind)
                m.update(matrix=M.tolist(), design_rank=rank, rank2_enforced_at_fit=kind=='F')
                if kind=='F':
                    fx=x@M.T; fy=y@M
                    den=(fx[:,:2]**2).sum(1)+(fy[:,:2]**2).sum(1)
                    values=np.divide(np.abs((y*fx).sum(1)),np.sqrt(den),
                                     out=np.full(len(test),np.nan),where=den>1e-24)
                    m.update(unit='corrected_pixels',metric='sqrt_sampson',
                             focal_profile_summary=_summary(_profile(M,p,edge)))
                else:
                    q=x@M.T; r=y@np.linalg.inv(M).T
                    valid=(np.abs(q[:,2])>1e-12)&(np.abs(r[:,2])>1e-12)
                    values=np.full(len(test),np.nan)
                    values[valid]=np.sqrt((((q[valid,:2]/q[valid,2,None]-b[test][valid])**2).sum(1)+
                        ((r[valid,:2]/r[valid,2,None]-a[test][valid])**2).sum(1))/2)
                    m.update(unit='corrected_pixels',metric='symmetric_transfer_rms',
                             fraction_le_2_pixels=float(np.sum(values<=2)/len(test)))
            else:
                def bearing(z):
                    v=_hom((z-p)/f); return v/np.linalg.norm(v,axis=1)[:,None]
                u= bearing(a[train]); v=bearing(b[train])
                U,_,V=np.linalg.svd(v.T@u)
                M=U@np.diag([1,1,np.linalg.det(U@V)])@V
                predicted=bearing(a[test])@M.T; target=bearing(b[test])
                valid=predicted[:,2]>0
                values=np.full(len(test),np.nan)
                values[valid]=np.degrees(np.arccos(np.clip((predicted[valid]*target[valid]).sum(1),-1,1)))
                m.update(matrix=M.tolist(),metric='bearing_angular_residual',unit='degrees',classification=None)
            m.update(status='EVALUATED', **_stats(values,len(test)))
        except (ValueError,np.linalg.LinAlgError,FloatingPointError) as exc:
            m['reason']=str(exc)
    return out


def analyse_pair(xy1,xy2,feature_pairs,principal,image_size,reference_focal,stored_f=None,config=3) -> dict:
    """Return finite JSON data. principal is (cx,cy), image_size (width,height)."""
    out=dict(status='SENSITIVITY_REFUSED',calibration_accepted=False,new_ba_authorized=False,
             independent_truth=False,degeneracy_unresolved=True,principal_profiles=[],distortion_scenarios=[],
             interpretation='Fixed diagnostic hypotheses; no best k, model selection, parallax or calibration acceptance',
             homography_interpretation='Both folds >=0.8 within 2 corrected pixels with zero invalids is compatibility only; not proof of a plane or pure rotation',
             config=int(config) if isinstance(config,numbers.Integral) and not isinstance(config,(bool,np.bool_)) else None)
    try:
        raw_ids = np.asarray(feature_pairs)
        if raw_ids.dtype.kind not in 'iu':
            raise ValueError('Feature identities require original integer dtype')
        a=_numeric(xy1); b=_numeric(xy2); ids=_numeric(feature_pairs)
        p=_numeric(principal,(2,)); size=_numeric(image_size,(2,))
        if a.ndim!=2 or a.shape[1]!=2 or b.shape!=a.shape or ids.shape!=a.shape or len(a)>4096:
            raise ValueError('Coordinates and feature identities must be Nx2')
        if np.any(ids > 2**53-1) or np.any(ids<0) or np.any(ids!=np.floor(ids)) or any(len(np.unique(ids[:,i]))!=len(ids) for i in (0,1)):
            raise ValueError('Feature identities must be nonnegative unique integers in each image')
        if isinstance(reference_focal,(bool,np.bool_)) or not isinstance(reference_focal,numbers.Real):
            raise ValueError('Invalid focal')
        f=float(reference_focal)
        if not np.isfinite(f) or f<=0 or f>1e8:
            raise ValueError('Invalid focal')
        if np.any(size<=0) or np.any(size>16384) or np.any(size != np.floor(size)):
            raise ValueError('Image dimensions must be positive integers <=16384')
        if np.any(p<0) or np.any(p>=size):
            raise ValueError('Principal point outside image bounds')
        if any(np.any(x<0) or np.any(x>=size) for x in (a,b)):
            raise ValueError('Raw pixels outside image bounds')
        if isinstance(config,(bool,np.bool_)) or not isinstance(config,numbers.Integral) or config not in (2,3,6):
            raise ValueError('Unsupported configuration')
        order=np.lexsort((ids[:,1],ids[:,0])); a=a[order]; b=b[order]; ids=ids[order].astype(np.int64)
        edge=float(size.max())
        if stored_f is not None and config in (2,3):
            for name,delta in [('center',(0,0)),('cx_minus',(-.02*size[0],0)),('cx_plus',(.02*size[0],0)),
                               ('cy_minus',(0,-.02*size[1])),('cy_plus',(0,.02*size[1]))]:
                pp=p+delta; prof=_profile(stored_f,pp,edge,config)
                out['principal_profiles'].append(dict(setting=name,principal=pp.tolist(),summary=_summary(prof),profile=prof,
                                                      source='stored_f_only'))
        for k in (-.1,0.,.1):
            scenario=dict(k=k,status='REFUSED',reference_focal=f,principal=p.tolist(),hypothesis_only=True,folds=[],
                          homography_compatible=None,homography_compatibility_reason='SCENARIO_REFUSED')
            out['distortion_scenarios'].append(scenario)
            try:
                aa=invert_radial(a,p,f,k); bb=invert_radial(b,p,f,k)
                rows=np.arange(len(a)); even=rows[::2]; odd=rows[1::2]
                scenario.update(status='EVALUATED',undistorted_pixels1=aa.tolist(),undistorted_pixels2=bb.tolist(),
                                sorted_feature_pairs=ids.tolist())
                scenario['folds']=[_fold(aa,bb,ids,t,h,p,f,edge) for t,h in ((even,odd),(odd,even))]
                hs = [fold['models'].get('H') for fold in scenario['folds']]
                if any(fold['status']!='EVALUATED' for fold in scenario['folds']):
                    scenario['homography_compatibility_reason']='INSUFFICIENT_FOLD'
                elif any(h is None or h['status']!='EVALUATED' for h in hs):
                    scenario['homography_compatibility_reason']='H_MODEL_FAILED'
                elif any(h['failed_or_undefined_count']>0 for h in hs):
                    scenario['homography_compatibility_reason']='H_INVALID_HOLDOUT'
                else:
                    scenario['homography_compatible']=all(h['fraction_le_2_pixels']>=.8 for h in hs)
                    scenario['homography_compatibility_reason']='BOTH_FOLDS_VALID'
            except (ValueError,np.linalg.LinAlgError) as exc:
                scenario['reason']=str(exc)
        out['status']='SENSITIVITY_EVALUATED'
    except (ValueError,TypeError,OverflowError,np.linalg.LinAlgError) as exc:
        out['reason']=str(exc)
    def finite_tree(value):
        if isinstance(value, dict):
            return all(finite_tree(v) for v in value.values())
        if isinstance(value, (list, tuple)):
            return all(finite_tree(v) for v in value)
        return not isinstance(value, (float, np.floating)) or bool(np.isfinite(value))
    if not finite_tree(out):
        out.update(status='SENSITIVITY_REFUSED', reason='NONFINITE_DERIVED_OUTPUT',
                   principal_profiles=[], distortion_scenarios=[])
    return out
