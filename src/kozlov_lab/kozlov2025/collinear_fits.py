"""Complete block containment with certified nonzero stable collinear centres.

Fixed transverse affine extension: sufficient witness, never general not-fits.
"""
from .affine_subimage import _rows
from .zero_error_fits import _parts, _budget, _call, _center, NonRational
from .exact_center import solve_exact_center, assess_exact_center_result
from .primary_futlyar import assess_primary_futlyar, assess_primary_futlyar_certificate
from .collinear_center_alignment import align_collinear_centers, assess_collinear_alignment_certificate
from .independent_cover import BudgetExceeded
from fractions import Fraction as F

SCHEMA='collinear-fits-v1'


def _whole(points,parts):
    raw=dict(_rows(points,7))
    return raw,_parts(raw,parts)


def _containment(source,parts,centers,radius2,alignment,budget):
    m,t=alignment['matrix'],alignment['translation']
    mapping=dict(alignment['mapping']); rows=[]
    for owner,part in enumerate(parts):
        target=mapping[owner]
        for label in part:
            budget.spend(10+8*len(centers))
            p=source[label]
            q=tuple(m[k][0]*p[0]+m[k][1]*p[1]+t[k] for k in (0,1))
            distances=tuple(sum((q[k]-c[k])**2 for k in (0,1)) for c in centers)
            if distances[target]>=radius2 or any(distances[target]>=v for k,v in enumerate(distances) if k!=target):
                return False,()
            rows.append((label,owner,target,q,distances))
    return True,tuple(rows)


def collinear_fits(reference,reference_parts,source,source_parts,*,max_work=2000000,seconds=30):
    budget=None
    def finish(status,reason,**kw):
        return dict(schema=SCHEMA,status=status,reason=reason,work_used=budget.used if budget else 0,
                    general_fits_decided=False,**kw)
    try:
        budget=_budget(max_work,seconds)
        a,ap=_whole(reference,reference_parts); b,bp=_whole(source,source_parts)
        if len(ap)!=len(bp): raise ValueError('same m required')
        primary=_call(budget,assess_primary_futlyar,a,ap)
        if primary['status']!='TRUE':
            return finish('UNKNOWN' if primary['status']=='UNKNOWN' else 'UNSUPPORTED','given reference primary not certified')
        answers=[]; centers=[]
        for part in bp:
            answer=_call(budget,solve_exact_center,{i:b[i] for i in part})
            if answer['complete'] is not True or answer['exact_optimum'] is not True:
                raise NonRational('full source block center unresolved')
            answers.append(answer); centers.append(_center(answer,budget))
        ca={i:tuple(c) for i,c in enumerate(primary['certificate']['centers'])}
        cb=dict(enumerate(centers))
        alignment=_call(budget,align_collinear_centers,ca,cb)
        if alignment['status']!='GLOBAL_OPTIMAL_STABLE':
            return finish('UNKNOWN' if alignment['status']=='UNKNOWN' else 'UNSUPPORTED','collinear stable alignment unavailable')
        c=dict(schema=SCHEMA,reference_points=tuple(sorted(a.items())),source_points=tuple(sorted(b.items())),
               reference_parts=ap,source_parts=bp,primary_certificate=primary['certificate'],
               source_blocks=tuple(answers),alignment_certificate=alignment['certificate'])
        check=_call(budget,assess_collinear_fits_certificate,a,b,c)
        if check['status']!='VERIFIED':
            return finish('UNKNOWN' if check['status']=='UNKNOWN' else 'NO_SUPPORTED_WITNESS',check['reason'])
        c['point_checks']=check['point_checks']
        return finish('COLLINEAR_FITS_SUFFICIENT_WITNESS','stable global center alignment and all complete blocks strictly contained',certificate=c)
    except (BudgetExceeded,NonRational) as exc: return finish('UNKNOWN',str(exc))
    except (ValueError,TypeError,KeyError,IndexError,AttributeError,ArithmeticError) as exc: return finish('UNSUPPORTED',str(exc))


def assess_collinear_fits_certificate(reference,source,certificate,*,max_work=2000000,seconds=30):
    """Independent primary/centres/two LP bounds/full point checks; no producer."""
    budget=None
    def finish(status,reason,**kw): return dict(status=status,reason=reason,work_used=budget.used if budget else 0,**kw)
    try:
        budget=_budget(max_work,seconds); c=certificate
        from .finite_environment import _canonical_rows
        a,ap=_whole(reference,c['reference_parts']); b,bp=_whole(source,c['source_parts'])
        if c['schema']!=SCHEMA or len(ap)!=len(bp) or not _canonical_rows(c['reference_points']) or not _canonical_rows(c['source_points']) or c['reference_points']!=tuple(sorted(a.items())) or c['source_points']!=tuple(sorted(b.items())):
            return finish('INVALID','full external input binding mismatch')
        pc=c['primary_certificate']
        primary=_call(budget,assess_primary_futlyar_certificate,a,pc)
        if primary['status']!='VERIFIED':
            return finish('UNKNOWN' if primary['status']=='UNKNOWN' else 'INVALID','reference primary certificate invalid')
        if tuple(tuple(p) for p in pc['parts'])!=ap or type(c['source_blocks']) is not tuple or len(c['source_blocks'])!=len(bp):
            return finish('INVALID','complete block certificate identities mismatch')
        centers=[]
        for part,answer in zip(bp,c['source_blocks']):
            check=_call(budget,assess_exact_center_result,{i:b[i] for i in part},answer)
            if check['status']!='VERIFIED': return finish('UNKNOWN' if check['status']=='UNKNOWN' else 'INVALID','full source center certificate invalid')
            centers.append(_center(answer,budget))
        ca={i:tuple(p) for i,p in enumerate(pc['centers'])}; cb=dict(enumerate(centers))
        ac=c['alignment_certificate']
        check=_call(budget,assess_collinear_alignment_certificate,ca,cb,ac)
        if check['status']!='VERIFIED': return finish('UNKNOWN' if check['status']=='UNKNOWN' else 'INVALID','stable global center alignment certificate invalid')
        valid,rows=_containment(b,bp,tuple(ca.values()),pc['radius_squared'],ac,budget)
        if not valid: return finish('INVALID','fixed transverse extension puts a whole-block point outside strict corresponding circle')
        if 'point_checks' in c:
            cached=c['point_checks']
            if type(cached) is not tuple or any(type(row) is not tuple or len(row)!=5 or any(type(x) is not int for x in row[:3]) or any(type(v) is not tuple or any(type(x) is not F for x in v) for v in row[3:]) for row in cached) or cached!=rows:
                return finish('INVALID','complete point checks malformed or tampered')
        budget.spend(0)
        return finish('VERIFIED','independent stable centers and whole-block containment verified',point_checks=rows)
    except (BudgetExceeded,NonRational) as exc: return finish('UNKNOWN',str(exc))
    except (ValueError,TypeError,KeyError,IndexError,AttributeError,ArithmeticError) as exc: return finish('INVALID',str(exc))
