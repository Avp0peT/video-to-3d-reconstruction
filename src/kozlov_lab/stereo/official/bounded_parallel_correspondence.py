"""S-1996 print338 (local PDF19): bounded unknown marking, exact parallel domain.

Joint affine rank is SOURCE_DERIVED. Actual AlgT' runs on both views.
"""
from fractions import Fraction as F
from itertools import combinations, permutations
import math
import time
from . import parallel_construction as pc
from ...kozlov2025.independent_cover import Budget, BudgetExceeded

SCHEMA='bounded-parallel-correspondence-v1'
SCOPE='SOURCE_DERIVED_nonplanar_full2D_injective_parallel_views_distinct_kernel_directions'


class UnreturnedPreparationBudget(ValueError):
    def __init__(self,allocated):
        self.allocated=allocated
        super().__init__('prepare budget exhausted before returning counter; full allocation conservatively charged, not measured work')


def _input(points):
    if type(points) is not dict or not 4<=len(points)<=7: raise ValueError('complete 4..7 labelled points required')
    out={}
    for label,p in points.items():
        if type(label) is not int or label.bit_length()>64 or type(p) not in (tuple,list) or len(p)!=2:
            raise ValueError('64-bit integer identities and two coordinates required')
        if any(type(x) not in (int,F) for x in p): raise ValueError('int/Fraction only; no bool/float/string identities')
        q=tuple(F(x) for x in p)
        if any(abs(x.numerator)>10**12 or x.denominator>10**12 for x in q): raise ValueError('existing AlgT input rational bounds exceeded')
        out[label]=q
    if len(set(out.values()))!=len(out): raise ValueError('duplicate coordinates rejected')
    if not any(_orientation(*(out[i] for i in triple)) for triple in combinations(out,3)):
        raise ValueError('supported view must have full affine dimension two')
    return dict(sorted(out.items()))


def _budget(work,seconds):
    if type(work) is not int or not 0<=work<=20000000 or type(seconds) not in (int,float) or not math.isfinite(seconds) or not 0<=seconds<=30:
        raise ValueError('bounded shared work/time required')
    return Budget(work,seconds)


def _det(matrix):
    n=len(matrix)
    return sum(((-1)**sum(p[i]>p[j] for i in range(n) for j in range(i+1,n)))
               *math.prod(matrix[i][p[i]] for i in range(n)) for p in permutations(range(n)))


def _solve(matrix,rhs):
    n=len(matrix); rows=[list(row)+[v] for row,v in zip(matrix,rhs)]
    for col in range(n):
        pivot=next((i for i in range(col,n) if rows[i][col]),None)
        if pivot is None: return None
        rows[col],rows[pivot]=rows[pivot],rows[col]
        v=rows[col][col]; rows[col]=[x/v for x in rows[col]]
        for i in range(n):
            if i!=col:
                v=rows[i][col]; rows[i]=[x-v*y for x,y in zip(rows[i],rows[col])]
    return tuple(row[-1] for row in rows)


def _rank_witness(points,budget):
    ids=tuple(points); origin=ids[0]; differences={i:tuple(x-y for x,y in zip(points[i],points[origin])) for i in ids[1:]}
    dimension=len(points[origin])
    for rank in range(min(dimension,len(ids)-1),0,-1):
        for basis in combinations(ids[1:],rank):
            for cols in combinations(range(dimension),rank):
                budget.spend(30)
                matrix=tuple(tuple(differences[i][k] for i in basis) for k in cols)
                if not _det(matrix): continue
                coords=[]
                for i in ids:
                    budget.spend(30)
                    rhs=tuple(points[i][k]-points[origin][k] for k in cols)
                    coeff=_solve(matrix,rhs)
                    if any(sum(coeff[j]*differences[basis[j]][k] for j in range(rank))!=points[i][k]-points[origin][k] for k in range(dimension)): break
                    coords.append((i,coeff))
                if len(coords)==len(ids): return dict(rank=rank,origin=origin,basis=basis,columns=cols,coordinates=tuple(coords))
    raise ValueError('zero affine rank impossible for distinct supplied points')


def _verify_rank(points,witness,budget):
    r=witness['rank']; origin=witness['origin']; basis=witness['basis']; cols=witness['columns']
    if type(r) is not int or not 1<=r<=len(next(iter(points.values()))) or type(origin) is not int or origin not in points:
        return False
    if type(basis) is not tuple or type(cols) is not tuple or len(basis)!=r or len(cols)!=r or len(set(basis))!=r or origin in basis or len(set(cols))!=r or any(type(i) is not int or i not in points for i in basis) or any(type(k) is not int or not 0<=k<len(points[origin]) for k in cols): return False
    vectors=tuple(tuple(x-y for x,y in zip(points[i],points[origin])) for i in basis)
    budget.spend(30)
    if not _det(tuple(tuple(v[k] for v in vectors) for k in cols)): return False
    coordinates=witness['coordinates']
    if type(coordinates) is not tuple or tuple(row[0] for row in coordinates)!=tuple(points): return False
    for label,c in coordinates:
        budget.spend(30)
        if type(label) is not int or type(c) is not tuple or len(c)!=r or any(type(x) is not F for x in c): return False
        if any(sum(c[j]*vectors[j][k] for j in range(r))!=points[label][k]-points[origin][k] for k in range(len(points[origin]))): return False
    return True


def _orientation(a,b,c): return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])


def _anchors(first,second,joint,budget):
    for triple in combinations(first,3):
        budget.spend(10)
        if not _orientation(*(first[i] for i in triple)) or not _orientation(*(second[i] for i in triple)): continue
        for e in first:
            if e in triple: continue
            ids=triple+(e,)
            if _rank_witness({i:joint[i] for i in ids},budget)['rank']==3: return ids
    raise ValueError('no common triangle/off-plane anchor in supported nonplane domain')


def _affine_rows(body,target,anchors):
    matrix=tuple(body[i]+(F(1),) for i in anchors)
    return tuple(_solve(matrix,tuple(target[i][k] for i in anchors)) for k in range(len(next(iter(target.values())))))


def _apply(rows,p): return tuple(sum(row[k]*p[k] for k in range(len(p)))+row[-1] for row in rows)


def _construct(first,second,joint,budget):
    anchors=_anchors(first,second,joint,budget)
    tokens={i:'L:'+str(i) for i in first}
    bodies=[]; traces=[]
    for left,right in ((first,second),(second,first)):
        budget.spend(0)
        prep=None
        allocated=max(0,budget.maximum-budget.used)
        try:
            prep=pc.prepare_views({tokens[i]:p for i,p in left.items()},{tokens[i]:p for i,p in right.items()},
                *(tokens[i] for i in anchors),op_budget=allocated)
            result=pc.construct_representative(prep)
        except pc.BudgetExceeded:
            if prep is None:
                # No returned budget counter exists. Reserve its full remaining
                # allocation as an explicit upper-bound account, never actual CPU work.
                budget.used+=allocated
                raise UnreturnedPreparationBudget(allocated)
            raise
        finally:
            if prep is not None: budget.spend(prep.budget.spent)
        if result.status!='OK' or result.unsupported or set(result.points)!=set(tokens.values()): raise ValueError('actual AlgT branch construction not complete')
        bodies.append({i:result.points[tokens[i]] for i in first})
        traces.append(tuple((i,result.witnesses[tokens[i]].branch,result.witnesses[tokens[i]].route,result.witnesses[tokens[i]].completeness) for i in first))
    body=bodies[0]
    budget.spend(1200)
    projections=(_affine_rows(body,first,anchors),_affine_rows(body,second,anchors))
    second_to_first=_affine_rows(bodies[1],body,anchors)
    return dict(body=tuple(body.items()),second_body=tuple(bodies[1].items()),anchors=anchors,
        projections=projections,second_to_first=second_to_first,tokens=tuple(tokens.items()),
        branch_traces=tuple(traces),algorithm='AlgT_prime_both_views',scale_status='unknown')


def search_bounded_parallel_correspondence(source_xy,target_xy,*,max_candidates=5040,max_work=20000000,seconds=30):
    budget=None; ledger=[]; accepted=[]; complete=False
    def finish(status,reason):
        return dict(schema=SCHEMA,status=status,reason=reason,complete=complete,ledger=tuple(ledger),accepted=tuple(accepted),
            solutions_count=len(accepted),unsupported_count=sum(row['status']=='UNSUPPORTED_PLANAR' for row in ledger),
            work_used=budget.used if budget else 0,source=tuple(first.items()) if 'first' in locals_outer else (),
            target=tuple(second.items()) if 'second' in locals_outer else (),true_unique_correspondence=False,
            scope=SCOPE)
    locals_outer={}
    try:
        budget=_budget(max_work,seconds)
        first=_input(source_xy); second=_input(target_xy); locals_outer.update(first=first,second=second)
        if len(first)!=len(second): raise ValueError('equal cardinality required')
        if type(max_candidates) is not int or not 0<=max_candidates<=5040: raise ValueError('candidate cap 0..5040 required')
        for index,targets in enumerate(permutations(second)):
            budget.spend()
            if index>=max_candidates: return finish('PARTIAL_UNKNOWN','candidate cap reached without dropping remaining mappings')
            mapping=tuple(zip(first,targets)); paired={i:second[j] for i,j in mapping}
            joint={i:first[i]+paired[i] for i in first}
            rank=_rank_witness(joint,budget)
            row=dict(mapping=mapping,rank_certificate=rank,status='REJECTED_RANK4' if rank['rank']==4 else 'UNSUPPORTED_PLANAR' if rank['rank']<3 else 'UNKNOWN_BRANCH')
            if rank['rank']==3:
                try:
                    row['certificate']=_construct(first,paired,joint,budget)
                    if not _verify_candidate(first,second,row,budget): raise ValueError('independent candidate geometry verification failed')
                    row['status']='ACCEPTED'; accepted.append(row)
                except (ValueError,pc.ValidationError,pc.BudgetExceeded) as exc:
                    row['reason']=str(exc)
                    if isinstance(exc,UnreturnedPreparationBudget):
                        row['work_accounting']=dict(kind='conservative_unreturned_prepare_allocation',charged_upper_bound=exc.allocated,actual_counter_available=False)
                    ledger.append(row)
                    return finish('PARTIAL_UNKNOWN','rank3 branch unresolved')
            ledger.append(row)
        complete=True
        return finish('SEARCH_COMPLETE','all n! mappings classified; planar classifications outside nonplane theorem')
    except BudgetExceeded as exc: return finish('PARTIAL_UNKNOWN',str(exc))
    except (ValueError,TypeError,KeyError,IndexError,ArithmeticError) as exc: return finish('UNSUPPORTED_INPUT',str(exc))


def _verify_candidate(first,second,row,budget):
    mapping=row['mapping']
    if type(mapping) is not tuple or tuple(i for i,j in mapping)!=tuple(first) or any(type(i) is not int or type(j) is not int for i,j in mapping) or set(j for i,j in mapping)!=set(second): return False
    paired={i:second[j] for i,j in mapping}; joint={i:first[i]+paired[i] for i in first}
    if not _verify_rank(joint,row['rank_certificate'],budget): return False
    r=row['rank_certificate']['rank']
    if row['status']=='REJECTED_RANK4': return r==4
    if row['status']=='UNSUPPORTED_PLANAR': return r<3
    if row['status']=='UNKNOWN_BRANCH' and 'certificate' not in row: return r==3
    if row['status']!='ACCEPTED' and not (row['status']=='UNKNOWN_BRANCH' and 'certificate' in row): return False
    if r!=3: return False
    c=row['certificate']; body=dict(c['body']); second_body=dict(c['second_body']); anchors=c['anchors']
    if any(type(c[field]) is not tuple or len(c[field])!=len(first) or any(type(row) is not tuple or len(row)!=2 for row in c[field]) or tuple(row[0] for row in c[field])!=tuple(first) for field in ('body','second_body')): return False
    if tuple(body)!=tuple(first) or tuple(second_body)!=tuple(first) or type(anchors) is not tuple or len(anchors)!=4 or len(set(anchors))!=4 or any(type(i) is not int or i not in first for i in anchors): return False
    if c['algorithm']!='AlgT_prime_both_views' or c['scale_status']!='unknown' or c['tokens']!=tuple((i,'L:'+str(i)) for i in first): return False
    if type(c['tokens']) is not tuple or any(type(row) is not tuple or len(row)!=2 or type(row[0]) is not int or type(row[1]) is not str for row in c['tokens']): return False
    if any(type(i) is not int or type(p) is not tuple or len(p)!=3 or any(type(x) is not F for x in p) for i,p in (*c['body'],*c['second_body'])): return False
    if not _det(tuple(body[i]+(F(1),) for i in anchors)) or not _det(tuple(second_body[i]+(F(1),) for i in anchors)): return False
    if type(c['branch_traces']) is not tuple or len(c['branch_traces'])!=2: return False
    known_branches={getattr(pc,name) for name in pc.__all__ if name.startswith('BRANCH_')}
    for trace in c['branch_traces']:
        if type(trace) is not tuple or len(trace)!=len(first) or any(type(row) is not tuple or len(row)!=4 for row in trace): return False
        if tuple(row[0] for row in trace)!=tuple(first) or any(type(row[0]) is not int or type(row[1]) is not str or row[1] not in known_branches or type(row[2]) is not tuple or any(type(x) is not str for x in row[2]) or row[3]!='CONSTRUCTED' for row in trace): return False
    if type(c['projections']) is not tuple or len(c['projections'])!=2: return False
    for rows,view in zip(c['projections'],(first,paired)):
        if type(rows) is not tuple or len(rows)!=2 or any(type(row) is not tuple or len(row)!=4 or any(type(x) is not F for x in row) for row in rows): return False
        if not any(rows[0][i]*rows[1][j]-rows[0][j]*rows[1][i] for i,j in combinations(range(3),2)): return False
        for i in first:
            budget.spend(20)
            if _apply(rows,body[i])!=view[i]: return False
    align=c['second_to_first']
    if type(align) is not tuple or len(align)!=3 or any(type(row) is not tuple or len(row)!=4 or any(type(x) is not F for x in row) for row in align): return False
    if not _det(tuple(row[:3] for row in align)): return False
    for i in first:
        budget.spend(20)
        if _apply(align,second_body[i])!=body[i]: return False
    return True


def assess_bounded_parallel_correspondence_result(source_xy,target_xy,result,*,max_work=20000000,seconds=30):
    """Complete identities/rank determinants/projection equations; no constructors."""
    budget=None
    def finish(status,reason): return dict(status=status,reason=reason,work_used=budget.used if budget else 0)
    try:
        budget=_budget(max_work,seconds); budget.spend()
        # Bound all stored numeric operands before arithmetic.
        from ...kozlov2025.exact_center import _tree
        # Ledger can exceed legacy tree array bound; gate each bounded candidate.
        first=_input(source_xy); second=_input(target_xy); r=result
        from ...kozlov2025.finite_environment import _canonical_rows
        if r['schema']!=SCHEMA or r['scope']!=SCOPE or len(first)!=len(second) or not _canonical_rows(r['source']) or not _canonical_rows(r['target']) or r['source']!=tuple(first.items()) or r['target']!=tuple(second.items()) or r['true_unique_correspondence'] is not False: return finish('INVALID','external complete input binding changed')
        if type(r['complete']) is not bool or type(r['ledger']) is not tuple or len(r['ledger'])>5040 or type(r['accepted']) is not tuple: return finish('INVALID','bounded ledger format invalid')
        if type(r['work_used']) is not int or r['work_used']<0:
            return finish('INVALID','work accounting must be a nonnegative integer')
        seen=set(); solutions=[]
        for row in r['ledger']:
            _tree(row,budget)
            if 'work_accounting' in row:
                account=row['work_accounting']
                if (row['status']!='UNKNOWN_BRANCH' or type(account) is not dict
                    or set(account)!={'kind','charged_upper_bound','actual_counter_available'}
                    or account['kind']!='conservative_unreturned_prepare_allocation'
                    or type(account['charged_upper_bound']) is not int
                    or not 0<=account['charged_upper_bound']<=r['work_used']
                    or account['actual_counter_available'] is not False):
                    return finish('INVALID','unreturned preparation is a conservative allocation, not measured work')
            if row['mapping'] in seen or not _verify_candidate(first,second,row,budget): return finish('INVALID','candidate rank/projection/identity certificate invalid')
            seen.add(row['mapping'])
            if row['status']=='ACCEPTED': solutions.append(row)
        if type(r['solutions_count']) is not int or r['solutions_count']!=len(solutions) or tuple(solutions)!=r['accepted']: return finish('INVALID','accepted candidate set mismatch')
        for row in r['accepted']:
            _tree(row,budget)
            if row['status']!='ACCEPTED' or not _verify_candidate(first,second,row,budget): return finish('INVALID','accepted cache type/geometry mismatch')
        if type(r['unsupported_count']) is not int or r['unsupported_count']!=sum(row['status']=='UNSUPPORTED_PLANAR' for row in r['ledger']): return finish('INVALID','planar count mismatch')
        if r['complete']:
            if r['status']!='SEARCH_COMPLETE' or len(seen)!=math.factorial(len(first)) or any(row['status']=='UNKNOWN_BRANCH' for row in r['ledger']): return finish('INVALID','complete n! bijection count not proved')
        elif r['status']!='PARTIAL_UNKNOWN': return finish('INVALID','partial status mismatch')
        budget.spend(0)
        return finish('VERIFIED','all recorded candidates independently certified; completeness exactly matches flag')
    except BudgetExceeded as exc: return finish('UNKNOWN',str(exc))
    except (ValueError,TypeError,KeyError,IndexError,ArithmeticError,AttributeError) as exc: return finish('INVALID',str(exc))
