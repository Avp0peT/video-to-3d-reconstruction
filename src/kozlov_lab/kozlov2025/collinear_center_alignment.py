"""S-2025 PDF9--10: globally optimal stable collinear centre alignment.

Rational LP primal/dual witnesses, no whole-block fits claim.
"""
from fractions import Fraction as F
from itertools import combinations
from .affine_subimage import _rows, _sub, _det, _solve, _linear
from .zero_error_fits import _budget
from .independent_cover import BudgetExceeded
from .exact_center import _tree

SCHEMA='collinear-center-alignment-v1'


def _coordinates(points):
    raw=dict(_rows(points,7))
    if len(raw)<2: raise ValueError('2..7 distinct complete centres required')
    order=sorted(raw,key=lambda i:raw[i]); origin=raw[order[0]]
    axis=_sub(raw[order[-1]],origin)
    k=0 if axis[0] else 1
    if any(_det(axis,_sub(p,origin)) for p in raw.values()):
        raise ValueError('exact collinear profile required')
    rows=tuple((i,(raw[i][k]-origin[k])/axis[k]) for i in order)
    return raw,origin,axis,rows


def _problem(xrows,yrows,sign):
    ys=tuple(sorted(((i,sign*y) for i,y in yrows),key=lambda row:row[1]))
    rows=[]
    for (_,x),(_,y) in zip(xrows,ys):
        rows.extend([((y,F(1),F(-1)),x),((-y,F(-1),F(-1)),-x)])
    rows.extend([((F(-1),F(0),F(0)),F(0)),((F(0),F(0),F(-1)),F(0))])
    return ys,tuple(rows)


def _dot(a,b): return sum(x*y for x,y in zip(a,b))


def _linear3(matrix,rhs):
    a=[list(row)+[v] for row,v in zip(matrix,rhs)]
    for col in range(3):
        pivot=next((i for i in range(col,3) if a[i][col]),None)
        if pivot is None: return None
        a[col],a[pivot]=a[pivot],a[col]
        v=a[col][col]; a[col]=[x/v for x in a[col]]
        for i in range(3):
            if i!=col:
                v=a[i][col]; a[i]=[x-v*y for x,y in zip(a[i],a[col])]
    return tuple(row[3] for row in a)


def _branch(xrows,yrows,sign,budget):
    ys,rows=_problem(xrows,yrows,sign)
    for active in combinations(range(len(rows)),3):
        budget.spend(180+10*len(rows))
        matrix=tuple(rows[i][0] for i in active)
        primal=_linear3(matrix,tuple(rows[i][1] for i in active))
        if primal is None or any(_dot(row,primal)>rhs for row,rhs in rows): continue
        dual=_linear3(tuple(tuple(matrix[j][i] for j in range(3)) for i in range(3)),(F(0),F(0),F(-1)))
        if dual is None or any(v<0 for v in dual): continue
        if -sum(v*rows[i][1] for i,v in zip(active,dual))!=primal[2]: continue
        if primal[0]<=0: raise BudgetExceeded('collapsed slope contradicts strict finite-image improvement proof')
        return dict(sign=sign,primal=primal,dual=tuple(zip(active,dual)))
    raise BudgetExceeded('no bounded primal-dual vertex certified')


def align_collinear_centers(target,source,*,max_work=2000000,seconds=30):
    budget=None
    try:
        budget=_budget(max_work,seconds); budget.spend()
        xr,x0,ux,xrows=_coordinates(target); yr,y0,uy,yrows=_coordinates(source)
        if len(xr)!=len(yr): raise ValueError('bijection requires equal cardinality')
        branches=tuple(_branch(xrows,yrows,s,budget) for s in (1,-1))
        chosen=min(range(2),key=lambda k:branches[k]['primal'][2])
        z=branches[chosen]; a,b,e=z['primal']; sign=z['sign']
        matrix=_solve(uy,(-uy[1],uy[0]),tuple(a*sign*v for v in ux),(-ux[1],ux[0]))
        translation=tuple(x0[k]+b*ux[k]-_linear(matrix,y0)[k] for k in (0,1))
        ys,_=_problem(xrows,yrows,sign)
        mapping=tuple((j,i) for (i,_),(j,_) in zip(xrows,ys))
        certificate=dict(schema=SCHEMA,target=tuple(sorted(xr.items())),source=tuple(sorted(yr.items())),
            branches=branches,chosen=chosen,matrix=matrix,translation=translation,mapping=mapping,
            error_squared=e*e*_dot(ux,ux),epsilon1=F(1),epsilon2=F(1))
        check=assess_collinear_alignment_certificate(target,source,certificate,
            max_work=max(0,budget.maximum-budget.used),seconds=max(0,budget.deadline-__import__('time').monotonic()))
        budget.spend(check['work_used'])
        if check['status']!='VERIFIED': raise BudgetExceeded('independent certificate verification incomplete')
        return dict(status='GLOBAL_OPTIMAL_STABLE',certificate=certificate,work_used=budget.used)
    except BudgetExceeded as exc:
        return dict(status='UNKNOWN',reason=str(exc),work_used=budget.used if budget else 0)
    except (ValueError,TypeError,KeyError,ArithmeticError,IndexError) as exc:
        return dict(status='UNSUPPORTED',reason=str(exc),work_used=budget.used if budget else 0)


def assess_collinear_alignment_certificate(target,source,certificate,*,max_work=2000000,seconds=30):
    """Only inequalities, dual identities and complete affine images; no LP solve."""
    budget=None
    def finish(status,reason): return dict(status=status,reason=reason,work_used=budget.used if budget else 0)
    try:
        budget=_budget(max_work,seconds); budget.spend()
        # Refuse oversized stored rationals before any certificate arithmetic.
        _tree(certificate,budget)
        xr,x0,ux,xrows=_coordinates(target); yr,y0,uy,yrows=_coordinates(source)
        c=certificate
        from .finite_environment import _canonical_rows
        if (c['schema']!=SCHEMA or len(xr)!=len(yr) or not _canonical_rows(c['target']) or not _canonical_rows(c['source'])
            or c['target']!=tuple(sorted(xr.items())) or c['source']!=tuple(sorted(yr.items()))):
            return finish('INVALID','complete input binding mismatch')
        if type(c['branches']) is not tuple or len(c['branches'])!=2: return finish('INVALID','both direction lower bounds required')
        errors=[]
        for sign,z in zip((1,-1),c['branches']):
            budget.spend(200)
            if type(z['sign']) is not int or z['sign']!=sign or type(z['primal']) is not tuple or len(z['primal'])!=3 or any(type(x) is not F for x in z['primal']):
                return finish('INVALID','typed primal/sign required')
            p=z['primal']; _,rows=_problem(xrows,yrows,sign)
            if p[0]<=0 or p[2]<0 or any(_dot(row,p)>rhs for row,rhs in rows): return finish('INVALID','primal infeasible/collapsed')
            if type(z['dual']) is not tuple or len(z['dual'])!=3: return finish('INVALID','three dual entries required')
            ids=[]; total=[F(0)]*3; bound=F(0)
            for entry in z['dual']:
                if type(entry) is not tuple or len(entry)!=2: return finish('INVALID','dual shape')
                i,v=entry
                if type(i) is not int or not 0<=i<len(rows) or type(v) is not F or v<0: return finish('INVALID','dual type/sign')
                ids.append(i)
                for k in range(3): total[k]+=v*rows[i][0][k]
                bound-=v*rows[i][1]
            if len(set(ids))!=3 or total!=[0,0,-1] or bound!=p[2]: return finish('INVALID','dual lower bound identity failed')
            errors.append(p[2])
        chosen=c['chosen']
        if type(chosen) is not int or chosen not in (0,1) or errors[chosen]!=min(errors): return finish('INVALID','chosen branch not globally minimal')
        m,t=c['matrix'],c['translation']
        if type(m) is not tuple or len(m)!=2 or any(type(r) is not tuple or len(r)!=2 or any(type(x) is not F for x in r) for r in m) or type(t) is not tuple or len(t)!=2 or any(type(x) is not F for x in t): return finish('INVALID','typed affine map required')
        if m[0][0]*m[1][1]-m[0][1]*m[1][0]==0: return finish('INVALID','map singular')
        a,b,e=c['branches'][chosen]['primal']; sign=c['branches'][chosen]['sign']
        ys,_=_problem(xrows,yrows,sign)
        expected=tuple((j,i) for (i,_),(j,_) in zip(xrows,ys))
        if type(c['mapping']) is not tuple or any(type(row) is not tuple or len(row)!=2 or any(type(i) is not int for i in row) for row in c['mapping']) or c['mapping']!=expected: return finish('INVALID','complete ordered bijection mismatch')
        for j,y in yrows:
            budget.spend(20)
            actual=tuple(_linear(m,yr[j])[k]+t[k] for k in (0,1))
            wanted=tuple(x0[k]+(a*sign*y+b)*ux[k] for k in (0,1))
            if actual!=wanted: return finish('INVALID','full affine center image mismatch')
        for field in ('epsilon1','epsilon2','error_squared'):
            if type(c[field]) is not F: return finish('INVALID','typed proof values required')
        if c['epsilon1']!=1 or c['epsilon2']!=1 or c['error_squared']!=e*e*_dot(ux,ux): return finish('INVALID','error/stability proof fields mismatch')
        budget.spend(0)
        return finish('VERIFIED','both LP primal-dual bounds and full affine centers verified')
    except BudgetExceeded as exc: return finish('UNKNOWN',str(exc))
    except (ValueError,TypeError,KeyError,ArithmeticError,IndexError,AttributeError) as exc: return finish('INVALID',str(exc))
