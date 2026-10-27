"""Project engineering affine surrogate, not an author exact/noisy algorithm.

Training SVD and frozen validation are separate. Exact rank of the rationalized
surrogate is imposed by construction; it does not certify original observations,
held-out generalization, a physical camera model, or reconstruction accuracy.
"""
from fractions import Fraction as F
import math
import time
import numpy as np

GRID_BITS=18
GRID=1 << GRID_BITS
SCHEMA='engineering-affine-flat-v1'
SCOPE='ENGINEERING_APPROXIMATION'


class _Exhausted(Exception):pass


class _Budget:
    def __init__(self,work,seconds):
        if type(work) is not int or not 0<=work<=2000000 or type(seconds) not in (int,float) or not math.isfinite(seconds) or not 0<=seconds<=30:
            raise ValueError('bounded work 0..2M and time 0..30 required')
        self.maximum,self.used,self.deadline=work,0,time.monotonic()+seconds
    def spend(self,n=1):
        self.used+=n
        if self.used>self.maximum or time.monotonic()>=self.deadline:raise _Exhausted('work/time budget exhausted')


def _array(raw,minimum=1,maximum=4096,magnitude_limit=1e9):
    if type(raw) not in (tuple,list,np.ndarray):raise ValueError('complete numeric Nx4 rows required')
    # Inspect scalars before dtype coercion: mixed lists must not hide bools.
    for row in raw:
        if type(row) not in (tuple,list,np.ndarray) or len(row)!=4:raise ValueError('Nx4 rows required')
        for v in row:
            if isinstance(v,(bool,np.bool_)) or not isinstance(v,(int,float,np.integer,np.floating)):
                raise ValueError('finite real numeric coordinates; bool/string rejected')
    a=np.asarray(raw,dtype=float)
    if a.ndim!=2 or a.shape[1]!=4 or not minimum<=len(a)<=maximum or not np.isfinite(a).all():
        raise ValueError('bounded complete finite Nx4 input required')
    if magnitude_limit is not None and np.max(np.abs(a))>magnitude_limit:raise ValueError('observation magnitude exceeds1e9')
    return a


def _scalar(v,low,high,name):
    if type(v) not in (int,float) or not math.isfinite(v) or not low<=v<=high:
        raise ValueError('invalid '+name)
    return float(v)


def _model(model):
    keys={'schema','scope','scale','mean','normal','singular_values','dependent_component',
          'free_components','min_support_ratio','grid_bits','training_count'}
    if type(model) is not dict or set(model)!=keys or model['schema']!=SCHEMA or model['scope']!=SCOPE:
        raise ValueError('complete frozen engineering model required')
    scale=_scalar(model['scale'],1e-9,1e9,'scale')
    mean=_array((model['mean'],),magnitude_limit=None)[0];normal=_array((model['normal'],))[0]
    singular=_array((model['singular_values'],),magnitude_limit=None)[0]
    threshold=_scalar(model['min_support_ratio'],1e-12,0.1,'support ratio')
    dependent=model['dependent_component']
    if type(dependent) is not int or dependent!=int(np.argmax(np.abs(normal))):raise ValueError('frozen maximum normal component required')
    if type(model['grid_bits']) is not int or model['grid_bits']!=GRID_BITS:raise ValueError('fixed dyadic grid18 required; no retuning')
    if type(model['free_components']) is not list or any(type(i) is not int for i in model['free_components']) or model['free_components']!=[i for i in range(4) if i!=dependent]:raise ValueError('complete free components')
    if not np.isclose(np.linalg.norm(normal),1,rtol=0,atol=1e-10) or normal[dependent]<=0:raise ValueError('canonical unit normal required')
    if type(model['training_count']) is not int or not 5<=model['training_count']<=4096:raise ValueError('training support count')
    if any(singular[i]<singular[i+1] for i in range(3)) or min(singular)<0 or singular[2]<=threshold*singular[0]:raise ValueError('weak/invalid frozen rank support')
    return scale,mean,normal


def _finish(status,budget,**data):
    return dict(status=status,scope=SCOPE,work_used=budget.used if budget else 0,
                original_exact_model_decided=False,physical_model_accepted=False,**data)


def fit_affine_flat(train,*,scale=1024.0,min_support_ratio=1e-6,max_work=2000000,seconds=30):
    """Fit one 3D affine flat in normalized4D; never evaluate held-out inputs."""
    budget=None
    try:
        budget=_Budget(max_work,seconds);budget.spend(0)
        scale=_scalar(scale,1e-9,1e9,'scale');threshold=_scalar(min_support_ratio,1e-12,0.1,'support ratio')
        raw=_array(train,5);budget.spend(len(raw)*64)
        normalized=raw/scale;mean=normalized.mean(axis=0)
        _,singular,right=np.linalg.svd(normalized-mean,full_matrices=False)
        budget.spend(0)
        singular=tuple(float(v) for v in singular)
        ratios=dict(rank_support_ratio=singular[2]/singular[0] if singular[0] else None,
            flatness_ratio_smallest_over_third=singular[3]/singular[2] if singular[2] else None,
            flatness_ratio_smallest_over_first=singular[3]/singular[0] if singular[0] else None)
        if singular[2]<=threshold*singular[0]:
            return _finish('WEAK_GEOMETRY',budget,model=None,singular_values=singular,**ratios,reason='third singular value lacks frozen rank3 support')
        normal=right[-1];dependent=int(np.argmax(np.abs(normal)))
        if normal[dependent]<0:normal=-normal
        model=dict(schema=SCHEMA,scope=SCOPE,scale=scale,mean=[float(v) for v in mean],
            normal=[float(v) for v in normal],singular_values=list(singular),
            dependent_component=dependent,free_components=[i for i in range(4) if i!=dependent],
            min_support_ratio=threshold,grid_bits=GRID_BITS,training_count=len(raw))
        residual=(normalized-mean)@normal*scale
        return _finish('ENGINEERING_APPROXIMATION_FITTED',budget,model=model,**ratios,
            training_observations=tuple(tuple(float(v) for v in row) for row in raw),
            training_residuals_pixels=tuple(float(v) for v in residual),
            training_rms_pixels=float(np.sqrt(np.mean(residual**2))),
            note='Training residuals are fit diagnostics, not held-out generalization.')
    except _Exhausted as e:return _finish('UNKNOWN',budget,model=None,reason=str(e))
    except (ValueError,TypeError,OverflowError,np.linalg.LinAlgError) as e:return _finish('INVALID_INPUT',budget,model=None,reason=str(e))


def evaluate_affine_flat(model,validation,*,max_work=2000000,seconds=30):
    """Residuals against frozen training normal/mean only; never refit or decide quality."""
    budget=None
    try:
        budget=_Budget(max_work,seconds);budget.spend(0)
        scale,mean,normal=_model(model);raw=_array(validation);budget.spend(len(raw)*16)
        residual=(raw/scale-mean)@normal*scale
        return _finish('ENGINEERING_APPROXIMATION_VALIDATED',budget,
            observations=tuple(tuple(float(v) for v in row) for row in raw),
            signed_residuals_pixels=tuple(float(v) for v in residual),
            rms_pixels=float(np.sqrt(np.mean(residual**2))),max_abs_pixels=float(np.max(np.abs(residual))),
            model=model,refitted=False,quality_threshold_decided=False)
    except _Exhausted as e:return _finish('UNKNOWN',budget,reason=str(e))
    except (ValueError,TypeError,OverflowError) as e:return _finish('INVALID_INPUT',budget,reason=str(e))


def _grid(v):
    # Exact rounding of the stored float to nearest grid point, ties to even.
    return F(round(F.from_float(float(v))*GRID),GRID)


def build_exact_surrogate(model,observations,labels,*,max_work=2000000,seconds=30):
    """Impose one quantized affine relation on all frozen supplied4..7 rows.

    Coordinates remain normalized; denominator/numerator caps are checked before
    publishing AlgT inputs. No point reselection, scale/grid retuning or pixels
    multiplication into AlgT occurs. All modifications are reported in pixels.
    """
    budget=None
    try:
        budget=_Budget(max_work,seconds);budget.spend(0)
        scale,mean,normal=_model(model);raw=_array(observations,4,7)
        if type(labels) not in (tuple,list) or len(labels)!=len(raw) or any(type(i) is not int or not -(2**63)<=i<2**63 for i in labels) or len(set(labels))!=len(labels):raise ValueError('complete unique int64 labels required')
        dep=model['dependent_component'];free=tuple(model['free_components'])
        offset=_grid(np.dot(mean,normal)/normal[dep]);slopes=tuple(_grid(-normal[i]/normal[dep]) for i in free)
        plane=dict(dependent_component=dep,free_components=free,offset=offset,slopes=slopes,grid_bits=GRID_BITS)
        source={};target={};ledger=[];bound_ok=True
        for label,row in zip(labels,raw):
            budget.spend(200)
            original=row/scale;residual=float(np.dot(original-mean,normal))
            projected=original-residual*normal
            rational=[None]*4
            for i in free:rational[i]=_grid(projected[i])
            rational[dep]=offset+sum(s*rational[i] for s,i in zip(slopes,free))
            rational=tuple(rational)
            change=tuple((float(rational[i])-original[i])*scale for i in range(4))
            ledger.append(dict(label=label,original_pixels=tuple(float(v) for v in row),
                original_normalized=tuple(float(v) for v in original),projected_normalized=tuple(float(v) for v in projected),
                rational_normalized=rational,training_flat_residual_pixels=residual*scale,
                projection_change_pixels=tuple(float((projected[i]-original[i])*scale) for i in range(4)),
                quantization_change_pixels=tuple(float((float(rational[i])-projected[i])*scale) for i in range(4)),
                total_change_pixels=change))
            if any(abs(v.numerator)>10**12 or v.denominator>10**12 for v in rational):bound_ok=False
            source[label]=rational[:2];target[label]=rational[2:]
        budget.spend(0)
        if not bound_ok:return _finish('UNSUPPORTED_RATIONAL_BOUND',budget,plane=plane,ledger=tuple(ledger),source_xy=None,target_xy=None,mapping=None,
            reason='fixed grid/scale violates existing AlgT1e12 bounds; no retuning or reselection')
        return _finish('ENGINEERING_APPROXIMATION_SURROGATE',budget,plane=plane,ledger=tuple(ledger),
            source_xy=source,target_xy=target,mapping={i:i for i in labels},model=model,
            affine_rank_upper_bound=3,exact_rank_three_proved=False,
            max_change_pixels=max(math.sqrt(sum(v*v for v in item['total_change_pixels'])) for item in ledger),
            rms_change_pixels=math.sqrt(sum(sum(v*v for v in item['total_change_pixels']) for item in ledger)/len(ledger)),
            note='Rank<=3 imposed by surrogate plane; call independent exact given-mapping gate. Original observations not certified.')
    except _Exhausted as e:return _finish('UNKNOWN',budget,reason=str(e),source_xy=None,target_xy=None,mapping=None)
    except (ValueError,TypeError,OverflowError) as e:return _finish('INVALID_INPUT',budget,reason=str(e),source_xy=None,target_xy=None,mapping=None)
