"""Exact complete four-centre minimax alignment; no whole-block fits claim."""
from fractions import Fraction as F
from itertools import combinations, permutations
from .affine_subimage import _rows, _sub, _det, _solve, _linear
from .zero_error_fits import _budget
from .independent_cover import BudgetExceeded
from .exact_center import _tree

SCHEMA = 'exact-four-center-alignment-v1'


def _inputs(target, source):
    x, y = _rows(target, 4), _rows(source, 4)
    if len(x) != 4 or len(y) != 4:
        raise ValueError('exactly four distinct centres required')
    if any(not _det(_sub(b[1], a[1]), _sub(c[1], a[1])) for a,b,c in combinations(y,3)):
        raise ValueError('every source triple must be noncollinear')
    if not any(_det(_sub(b[1], a[1]), _sub(c[1], a[1])) for a,b,c in combinations(x,3)):
        raise ValueError('target affine rank two required')
    return x,y


def _relation(y):
    # Signed 3x3 minors of columns (1,y_x,y_y); fixed canonical scale.
    c=[]
    for i in range(4):
        a,b,d=[p for j,(_,p) in enumerate(y) if j != i]
        c.append(F((-1)**i)*_det(_sub(b,a),_sub(d,a)))
    return tuple(q/c[0] for q in c)


def _energy(v): return sum(q*q for q in v)


def _summary(ledger):
    e=min(z['error_squared'] for z in ledger)
    minima=[i for i,z in enumerate(ledger) if z['error_squared']==e]
    attained=[i for i in minima if zdet(ledger[i]['matrix'])]
    chosen=attained[0] if attained else minima[0]
    v=ledger[chosen]['vector']
    strong=bool(attained) and all(ledger[i]['vector'] in (v,tuple(-q for q in v)) for i in minima)
    eps=F(0)
    if strong:
        eps=min([F(1,4)]+[(z['error_squared']-e)/(8*(z['error_squared']+e)) for z in ledger if z['error_squared']>e])
    status=('GLOBAL_OPTIMAL_STABLE' if strong else 'GLOBAL_OPTIMAL_STABILITY_UNRESOLVED') if attained else 'GLOBAL_INFIMUM_UNATTAINED'
    return e,chosen,eps,status


def zdet(m): return m[0][0]*m[1][1]-m[0][1]*m[1][0]


def _point_radius(x, epsilon):
    """Exact matrix-entry radius bridge from the canonical target affine basis."""
    a,b,c=next((a,b,c) for a,b,c in combinations(x,3)
               if _det(_sub(b[1],a[1]),_sub(c[1],a[1])))
    u,v=_sub(b[1],a[1]),_sub(c[1],a[1]); d=_det(u,v)
    inverse=((v[1]/d,-v[0]/d),(-u[1]/d,u[0]/d))
    k=max(sum(abs(inverse[i][j]) for i in range(2)) for j in range(2))
    return epsilon/(4*max(F(1),k))


def _typed(value):
    if type(value) in (int,F,str): return True
    if type(value) is tuple: return all(_typed(q) for q in value)
    if type(value) is dict: return all(type(k) is str and _typed(q) for k,q in value.items())
    return False


def _verify(x,y,c,budget):
    _tree(c,budget)
    if not _typed(c): raise ValueError('strict int/Fraction/tuple certificate required; bool rejected')
    if c['schema']!=SCHEMA or c['target']!=x or c['source']!=y:
        raise ValueError('complete input binding mismatch')
    relation=c['relation']
    if type(relation) is not tuple or len(relation)!=4 or any(type(q) is not F for q in relation) or relation!=_relation(y):
        raise ValueError('canonical affine relation mismatch')
    s=sum(abs(q) for q in relation)
    if c['weight_sum']!=s or type(c['weight_sum']) is not F: raise ValueError('weight sum mismatch')
    ledger=c['ledger']; perms=tuple(permutations(range(4)))
    if type(ledger) is not tuple or len(ledger)!=24: raise ValueError('all 24 permutations required')
    for p,z in zip(perms,ledger):
        budget.spend(300)
        if z['permutation']!=p or any(type(i) is not int for i in z['permutation']): raise ValueError('permutation ledger mismatch')
        v=z['vector']; m=z['matrix']; t=z['translation']
        if type(v) is not tuple or len(v)!=2 or any(type(q) is not F for q in v): raise ValueError('typed vector required')
        if type(m) is not tuple or len(m)!=2 or any(type(r) is not tuple or len(r)!=2 or any(type(q) is not F for q in r) for r in m): raise ValueError('typed matrix required')
        if type(t) is not tuple or len(t)!=2 or any(type(q) is not F for q in t): raise ValueError('typed translation required')
        if v!=tuple(-sum(relation[i]*x[p[i]][1][k] for i in range(4))/s for k in range(2)): raise ValueError('lower bound vector mismatch')
        if type(z['error_squared']) is not F or z['error_squared']!=_energy(v): raise ValueError('squared lower bound mismatch')
        for i,(_,point) in enumerate(y):
            actual=tuple(_linear(m,point)[k]+t[k]-x[p[i]][1][k] for k in range(2))
            if actual!=tuple((1 if relation[i]>0 else -1)*q for q in v): raise ValueError('complete affine residual substitution failed')
    e,chosen,eps,status=_summary(ledger)
    if type(c['chosen']) is not int or c['chosen']!=chosen or type(c['error_squared']) is not F or c['error_squared']!=e or type(c['epsilon']) is not F or c['epsilon']!=eps or c['classification']!=status:
        raise ValueError('global minimum or stability fields mismatch')
    if (type(c['epsilon1']) is not F or c['epsilon1']!=1
        or type(c['epsilon_point_displacement']) is not F
        or c['epsilon_point_displacement']!=_point_radius(x,eps)):
        raise ValueError('point-displacement stability bridge mismatch')
    budget.spend(0)
    return status


def align_four_centers(target, source, *, max_work=2000000, seconds=30):
    budget=None
    try:
        budget=_budget(max_work,seconds); budget.spend()
        x,y=_inputs(target,source); relation=_relation(y); s=sum(abs(q) for q in relation)
        ledger=[]
        for p in permutations(range(4)):
            budget.spend(300)
            v=tuple(-sum(relation[i]*x[p[i]][1][k] for i in range(4))/s for k in range(2))
            images=tuple(tuple(x[p[i]][1][k]+(1 if relation[i]>0 else -1)*v[k] for k in range(2)) for i in range(4))
            m=_solve(_sub(y[1][1],y[0][1]),_sub(y[2][1],y[0][1]),_sub(images[1],images[0]),_sub(images[2],images[0]))
            t=tuple(images[0][k]-_linear(m,y[0][1])[k] for k in range(2))
            ledger.append(dict(permutation=p,vector=v,matrix=m,translation=t,error_squared=_energy(v)))
        ledger=tuple(ledger); e,chosen,eps,status=_summary(ledger)
        c=dict(schema=SCHEMA,target=x,source=y,relation=relation,weight_sum=s,ledger=ledger,chosen=chosen,error_squared=e,epsilon=eps,epsilon1=F(1),epsilon_point_displacement=_point_radius(x,eps),classification=status)
        _verify(x,y,c,budget)
        return dict(status=status,certificate=c,work_used=budget.used)
    except BudgetExceeded as exc: return dict(status='UNKNOWN',reason=str(exc),work_used=budget.used if budget else 0)
    except (ValueError,TypeError,KeyError,ArithmeticError,IndexError) as exc: return dict(status='UNSUPPORTED',reason=str(exc),work_used=budget.used if budget else 0)


def assess_four_center_alignment_certificate(target, source, certificate, *, max_work=2000000, seconds=30):
    """Independent complete matrix substitution/minimum verification; no producer or solve."""
    budget=None
    try:
        budget=_budget(max_work,seconds); budget.spend()
        x,y=_inputs(target,source)
        status=_verify(x,y,certificate,budget)
        return dict(status='VERIFIED',classification=status,work_used=budget.used)
    except BudgetExceeded as exc: return dict(status='UNKNOWN',reason=str(exc),work_used=budget.used if budget else 0)
    except (ValueError,TypeError,KeyError,ArithmeticError,IndexError,AttributeError) as exc: return dict(status='INVALID',reason=str(exc),work_used=budget.used if budget else 0)
