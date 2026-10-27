"""Complete-block fits using sufficient multi-centre circuit certificates."""
from fractions import Fraction as F
from .zero_error_fits import _budget,_call,_center,NonRational
from .collinear_fits import _whole,_containment
from .primary_futlyar import assess_primary_futlyar,assess_primary_futlyar_certificate
from .exact_center import solve_exact_center,assess_exact_center_result,_tree
from .finite_environment import _canonical_rows
from .independent_cover import BudgetExceeded
from .circuit_center_alignment import align_circuit_centers,assess_circuit_alignment_certificate

SCHEMA='circuit-fits-v1'
SUCCESS='CIRCUIT_FITS_SUFFICIENT_WITNESS'

def _transform(ac):
    permutation=ac['ledger'][ac['chosen']]['permutation']
    return dict(matrix=ac['matrix'],translation=ac['translation'],
                mapping=tuple((ac['source'][i][0],ac['target'][j][0]) for i,j in enumerate(permutation)))

def circuit_fits(reference,reference_parts,source,source_parts,*,max_work=2000000,seconds=30):
    budget=None
    def finish(status,reason,**kw):
        return dict(schema=SCHEMA,status=status,reason=reason,work_used=budget.used if budget else 0,
                    general_fits_decided=False,**kw)
    try:
        budget=_budget(max_work,seconds);budget.spend()
        a,ap=_whole(reference,reference_parts);b,bp=_whole(source,source_parts)
        if len(ap)!=len(bp) or not 4<=len(ap)<=7:raise ValueError('equal 4..7 full parts required')
        primary=_call(budget,assess_primary_futlyar,a,ap)
        if primary['status']!='TRUE':
            return finish('UNKNOWN' if primary['status']=='UNKNOWN' else 'UNSUPPORTED','reference primary not certified')
        pc=primary['certificate'];blocks=[];cb={}
        for i,part in enumerate(bp):
            answer=_call(budget,solve_exact_center,{j:b[j] for j in part})
            if answer['complete'] is not True or answer['exact_optimum'] is not True:
                raise NonRational('complete source-block center unresolved')
            blocks.append(answer);cb[i]=_center(answer,budget)
        ca={i:tuple(p) for i,p in enumerate(pc['centers'])}
        alignment=_call(budget,align_circuit_centers,ca,cb)
        if alignment['status']!='GLOBAL_OPTIMAL_STABLE_SUFFICIENT':
            return finish('UNKNOWN','multi-centre sufficient stable witness not established',
                          alignment_status=alignment['status'],alignment_progress={k:alignment.get(k) for k in
                              ('permutations_completed','circuits_evaluated','candidates_checked')})
        ac=alignment['certificate']
        valid,checks=_containment(b,bp,tuple(ca.values()),pc['radius_squared'],_transform(ac),budget)
        if not valid:
            return finish('NO_SUPPORTED_WITNESS','selected certified center transform fails full-block strict containment; other transforms not excluded')
        c=dict(schema=SCHEMA,reference_points=tuple(sorted(a.items())),source_points=tuple(sorted(b.items())),
               reference_parts=ap,source_parts=bp,primary_certificate=pc,source_blocks=tuple(blocks),
               alignment_certificate=ac,point_checks=checks)
        check=_call(budget,assess_circuit_fits_certificate,a,b,c)
        if check['status']=='UNKNOWN':raise BudgetExceeded('independent full fits verification incomplete')
        if check['status']!='VERIFIED':return finish('INVALID_INTERNAL_CERTIFICATE',check['reason'])
        return finish(SUCCESS,'all correspondences bounded, selected stable optimum and complete blocks strictly contained',certificate=c)
    except (BudgetExceeded,NonRational) as exc:return finish('UNKNOWN',str(exc))
    except (ValueError,TypeError,KeyError,IndexError,ArithmeticError,AttributeError) as exc:return finish('UNSUPPORTED',str(exc))

def assess_circuit_fits_certificate(reference,source,certificate,*,max_work=2000000,seconds=30):
    """No producer/search; full child certificates and all point substitutions."""
    budget=None
    def finish(status,reason):return dict(status=status,reason=reason,work_used=budget.used if budget else 0)
    try:
        budget=_budget(max_work,seconds);budget.spend();c=certificate
        expected={'schema','reference_points','source_points','reference_parts','source_parts',
                  'primary_certificate','source_blocks','alignment_certificate','point_checks'}
        if type(c) is not dict or set(c)!=expected or c['schema']!=SCHEMA:raise ValueError('strict complete fits fields required')
        a,ap=_whole(reference,c['reference_parts']);b,bp=_whole(source,c['source_parts'])
        if (not 4<=len(ap)<=7 or len(ap)!=len(bp) or c['reference_parts']!=ap or c['source_parts']!=bp):
            raise ValueError('equal canonical complete 4..7 partitions required')
        if (not _canonical_rows(c['reference_points']) or not _canonical_rows(c['source_points'])
            or c['reference_points']!=tuple(sorted(a.items())) or c['source_points']!=tuple(sorted(b.items()))):
            raise ValueError('complete input identity mismatch')
        pc=c['primary_certificate'];check=_call(budget,assess_primary_futlyar_certificate,a,pc)
        if check['status']!='VERIFIED':return finish('UNKNOWN' if check['status']=='UNKNOWN' else 'INVALID','reference primary invalid')
        if tuple(tuple(p) for p in pc['parts'])!=ap:raise ValueError('reference partition changed')
        if type(c['source_blocks']) is not tuple or len(c['source_blocks'])!=len(bp):raise ValueError('complete source block proofs required')
        cb={}
        for i,(part,answer) in enumerate(zip(bp,c['source_blocks'])):
            check=_call(budget,assess_exact_center_result,{j:b[j] for j in part},answer)
            if check['status']!='VERIFIED':return finish('UNKNOWN' if check['status']=='UNKNOWN' else 'INVALID','source block center invalid')
            cb[i]=_center(answer,budget)
        ca={i:tuple(p) for i,p in enumerate(pc['centers'])};ac=c['alignment_certificate']
        check=_call(budget,assess_circuit_alignment_certificate,ca,cb,ac)
        if check['status']!='VERIFIED':return finish('UNKNOWN' if check['status']=='UNKNOWN' else 'INVALID','multi-centre certificate invalid')
        valid,checks=_containment(b,bp,tuple(ca.values()),pc['radius_squared'],_transform(ac),budget)
        if not valid:raise ValueError('complete transformed blocks not strictly contained')
        cached=c['point_checks'];_tree(cached,budget)
        if (type(cached) is not tuple or len(cached)!=len(b) or any(type(row) is not tuple or len(row)!=5
            or any(type(i) is not int for i in row[:3])
            or any(type(vec) is not tuple or any(type(q) is not F for q in vec) for vec in row[3:]) for row in cached)
            or cached!=checks):raise ValueError('complete exact point checks differ')
        budget.spend(0)
        return finish('VERIFIED','stable circuit optimum and all complete blocks independently verified')
    except (BudgetExceeded,NonRational) as exc:return finish('UNKNOWN',str(exc))
    except (ValueError,TypeError,KeyError,IndexError,ArithmeticError,AttributeError) as exc:return finish('INVALID',str(exc))
