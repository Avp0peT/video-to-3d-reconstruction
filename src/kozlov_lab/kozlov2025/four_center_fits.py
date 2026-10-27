"""Whole-block S-2025 fits with an exact noncollinear four-centre proof.

Every globally minimal invertible certified correspondence is tried. Failure
of this sufficient global-stability profile never disproves local fits.
"""
from .zero_error_fits import _budget, _call, _center, NonRational
from .collinear_fits import _whole, _containment
from .primary_futlyar import assess_primary_futlyar, assess_primary_futlyar_certificate
from .exact_center import solve_exact_center, assess_exact_center_result, _tree
from .exact_four_center_alignment import align_four_centers, assess_four_center_alignment_certificate, zdet
from .finite_environment import _canonical_rows
from .independent_cover import BudgetExceeded
from fractions import Fraction as F

SCHEMA='four-center-fits-v1'


def _map_for(ac,index):
    if type(index) is not int or not 0<=index<24:
        raise ValueError('explicit four-centre ledger index required')
    row=ac['ledger'][index]
    if row['error_squared']!=ac['error_squared'] or not zdet(row['matrix']):
        raise ValueError('selected mapping must attain the global minimum invertibly')
    # Centre labels are canonical block positions, not original point labels.
    mapping=tuple((ac['source'][i][0],ac['target'][j][0]) for i,j in enumerate(row['permutation']))
    return dict(matrix=row['matrix'],translation=row['translation'],mapping=mapping)


def four_center_fits(reference,reference_parts,source,source_parts,*,max_work=2000000,seconds=30):
    budget=None;tried=[]
    def finish(status,reason,**fields):
        return dict(schema=SCHEMA,status=status,reason=reason,work_used=budget.used if budget else 0,
                    general_fits_decided=False,tried_ledger_indices=tuple(tried),**fields)
    try:
        budget=_budget(max_work,seconds);budget.spend()
        a,ap=_whole(reference,reference_parts);b,bp=_whole(source,source_parts)
        if len(ap)!=4 or len(bp)!=4:raise ValueError('exactly four complete parts on each side required')
        primary=_call(budget,assess_primary_futlyar,a,ap)
        if primary['status']!='TRUE':
            return finish('UNKNOWN' if primary['status']=='UNKNOWN' else 'UNSUPPORTED','reference primary not certified')
        blocks=[];cb={}
        for i,part in enumerate(bp):
            answer=_call(budget,solve_exact_center,{j:b[j] for j in part})
            if answer['complete'] is not True or answer['exact_optimum'] is not True:
                raise NonRational('full source block exact centre unresolved')
            blocks.append(answer);cb[i]=_center(answer,budget)
        pc=primary['certificate'];ca={i:tuple(p) for i,p in enumerate(pc['centers'])}
        alignment=_call(budget,align_four_centers,ca,cb)
        if alignment['status']!='GLOBAL_OPTIMAL_STABLE':
            return finish('UNKNOWN' if alignment['status']=='UNKNOWN' else 'NO_SUPPORTED_WITNESS',
                          'strong global centre stability unavailable; original local fits not decided',
                          alignment_status=alignment['status'])
        ac=alignment['certificate']
        for index,row in enumerate(ac['ledger']):
            budget.spend()
            if row['error_squared']!=ac['error_squared'] or not zdet(row['matrix']):continue
            tried.append(index)
            transform=_map_for(ac,index)
            valid,checks=_containment(b,bp,tuple(ca.values()),pc['radius_squared'],transform,budget)
            if not valid:continue
            c=dict(schema=SCHEMA,reference_points=tuple(sorted(a.items())),source_points=tuple(sorted(b.items())),
                   reference_parts=ap,source_parts=bp,primary_certificate=pc,source_blocks=tuple(blocks),
                   alignment_certificate=ac,selected_index=index,point_checks=checks)
            check=_call(budget,assess_four_center_fits_certificate,a,b,c)
            if check['status']=='VERIFIED':
                return finish('FOUR_CENTER_FITS_SUFFICIENT_WITNESS',
                              'noncollinear stable global centres and all complete blocks strictly contained',certificate=c)
            if check['status']=='UNKNOWN':raise BudgetExceeded('independent complete fits verification incomplete')
            return finish('INVALID_INTERNAL_CERTIFICATE',check.get('reason','constructed witness rejected'))
        return finish('NO_SUPPORTED_WITNESS','all supported global-minimum mappings failed complete-block containment')
    except (BudgetExceeded,NonRational) as exc:return finish('UNKNOWN',str(exc))
    except (ValueError,TypeError,KeyError,IndexError,ArithmeticError,AttributeError) as exc:return finish('UNSUPPORTED',str(exc))


def assess_four_center_fits_certificate(reference,source,certificate,*,max_work=2000000,seconds=30):
    """Only independently certified centres, all-24 bounds and point substitution."""
    budget=None
    def finish(status,reason):return dict(status=status,reason=reason,work_used=budget.used if budget else 0)
    try:
        budget=_budget(max_work,seconds);budget.spend();c=certificate;_tree(c,budget)
        fields={'schema','reference_points','source_points','reference_parts','source_parts','primary_certificate',
                'source_blocks','alignment_certificate','selected_index','point_checks'}
        if type(c) is not dict or set(c)!=fields or c['schema']!=SCHEMA:raise ValueError('strict complete fits certificate required')
        a,ap=_whole(reference,c['reference_parts']);b,bp=_whole(source,c['source_parts'])
        if len(ap)!=4 or len(bp)!=4 or c['reference_parts']!=ap or c['source_parts']!=bp:
            raise ValueError('four canonical complete partitions required')
        if (not _canonical_rows(c['reference_points']) or not _canonical_rows(c['source_points'])
            or c['reference_points']!=tuple(sorted(a.items())) or c['source_points']!=tuple(sorted(b.items()))):
            raise ValueError('complete input identity mismatch')
        pc=c['primary_certificate'];check=_call(budget,assess_primary_futlyar_certificate,a,pc)
        if check['status']!='VERIFIED':return finish('UNKNOWN' if check['status']=='UNKNOWN' else 'INVALID','reference primary certificate invalid')
        if tuple(tuple(p) for p in pc['parts'])!=ap:raise ValueError('reference partition changed')
        if type(c['source_blocks']) is not tuple or len(c['source_blocks'])!=4:raise ValueError('four source block certificates required')
        cb={}
        for i,(part,answer) in enumerate(zip(bp,c['source_blocks'])):
            check=_call(budget,assess_exact_center_result,{j:b[j] for j in part},answer)
            if check['status']!='VERIFIED':return finish('UNKNOWN' if check['status']=='UNKNOWN' else 'INVALID','source full-block centre invalid')
            cb[i]=_center(answer,budget)
        ca={i:tuple(p) for i,p in enumerate(pc['centers'])};ac=c['alignment_certificate']
        check=_call(budget,assess_four_center_alignment_certificate,ca,cb,ac)
        if check['status']!='VERIFIED':return finish('UNKNOWN' if check['status']=='UNKNOWN' else 'INVALID','four-centre global certificate invalid')
        if check['classification']!='GLOBAL_OPTIMAL_STABLE':raise ValueError('global value alone is not a stable fits witness')
        transform=_map_for(ac,c['selected_index'])
        valid,rows=_containment(b,bp,tuple(ca.values()),pc['radius_squared'],transform,budget)
        if not valid:raise ValueError('complete transformed block leaves corresponding strict truncated circle')
        cached=c['point_checks']
        if (type(cached) is not tuple or any(type(row) is not tuple or len(row)!=5
            or any(type(v) is not int for v in row[:3])
            or any(type(vec) is not tuple or any(type(v) is not F for v in vec) for vec in row[3:]) for row in cached)
            or cached!=rows):raise ValueError('full point check cache changed or has wrong types')
        budget.spend(0)
        return finish('VERIFIED','strong stable four-centre minimax plus full-block strict containment')
    except (BudgetExceeded,NonRational) as exc:return finish('UNKNOWN',str(exc))
    except (ValueError,TypeError,KeyError,IndexError,ArithmeticError,AttributeError) as exc:return finish('INVALID',str(exc))
