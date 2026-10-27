"""Primary futlyar: one supplied independent cover, rational strict interior.

S-2025 PDF10 fixes R to minimum center separation, not half separation.
This is neither cover search nor a generalized-radius or fits procedure.
"""
from fractions import Fraction as F
import math
import time
from .independent_cover import (Budget, BudgetExceeded, normalize,
    verify_independent_cover_2d, verify_cover_certificate)
from .exact_triangle_ellipse import _digest
from .exact_center import solve_exact_center, assess_exact_center_result, _tree
from .real_algebraic import Root, AN

MAX_BITS=16384


class UnsupportedDomain(Exception):
    pass


def _budget(max_work,seconds):
    if (type(max_work) is not int or not 0<=max_work<=2000000 or
        type(seconds) not in (int,float) or not math.isfinite(seconds) or not 0<=seconds<=30):
        raise ValueError('invalid bounded shared work/time budget')
    return Budget(max_work,seconds)


def _remaining(b):
    b.spend(0)
    return dict(max_work=max(0,b.maximum-b.used),seconds=max(0,b.deadline-time.monotonic()))


def _input(points,parts):
    if type(points) is not dict or not 1<=len(points)<=7:
        raise ValueError('complete 1..7 rational labelled points required')
    for p in points.values():
        if type(p) not in (tuple,list) or len(p)!=2 or any(type(v) not in (int,F) for v in p):
            raise ValueError('int/Fraction coordinates required; bool/float rejected')
    raw,groups,missing=normalize(points,parts)
    if missing: raise ValueError('cover omits supplied labels')
    return raw,groups


def _q(x,b):
    b.spend()
    if type(x) not in (int,F): raise ValueError('strict rational required')
    x=F(x)
    if max(x.numerator.bit_length(),x.denominator.bit_length())>MAX_BITS:
        raise BudgetExceeded('primary rational intermediate bit bound')
    return x


def _distance(a,c,b):
    return _q(sum(_q(_q(a[k]-c[k],b)**2,b) for k in (0,1)),b)


def _rational_an(data,root,b):
    value=AN.from_data(data,root)
    if len(value.numerator)==1 and value.denominator==(1,):
        return _q(value.numerator[0],b)
    raise UnsupportedDomain('algebraic expression not proven rational in supported normalization')


def _piece_data(answer,b):
    context=answer['number_context']
    if context==dict(domain='rational'):
        center=answer['center']
        if type(center) not in (tuple,list) or len(center)!=2:
            raise ValueError('two-dimensional center required')
        center=tuple(_q(x,b) for x in center)
        size=answer['major_axis_length']
        if answer['profile'] in ('point','segment'):
            if set(size)!={'expression','length_squared'} or size['expression']!='sqrt(length_squared)':
                raise ValueError('degenerate length formula required')
            return center,dict(kind='squared',squared=_q(size['length_squared'],b))
        if set(size)!={'expression','trace','discriminant'} or size['expression']!='sqrt(2*(trace+sqrt(discriminant)))':
            raise ValueError('ellipse length formula required')
        trace,disc=_q(size['trace'],b),_q(size['discriminant'],b)
    elif context['domain']=='real-algebraic':
        root=Root.from_data(context['algebraic_root'],b)
        center=tuple(_rational_an(x,root,b) for x in answer['center'])
        if len(center)!=2: raise ValueError('two-dimensional center required')
        size=answer['major_axis_length']
        if set(size)!={'expression','trace','discriminant'} or size['expression']!='sqrt(2*(trace+sqrt(discriminant)))':
            raise ValueError('ellipse length formula required')
        trace,disc=(_rational_an(size[key],root,b) for key in ('trace','discriminant'))
    else:
        raise UnsupportedDomain('unsupported exact-center number context')
    if trace<=0 or disc<0: raise ValueError('invalid ellipse shape length')
    return center,dict(kind='ellipse',trace=trace,discriminant=disc)


def _radius_screen(radius_squared,size,b):
    """R >= length/2; compare sqrt(discriminant) without approximation."""
    if size['kind']=='squared':
        return _q(4*radius_squared,b)>=size['squared']
    # R² >= (trace+sqrt(disc))/2 iff u=2R²-trace >=0 and u²>=disc.
    u=_q(2*radius_squared-size['trace'],b)
    return u>=0 and _q(u*u,b)>=size['discriminant']


def _geometry(raw,groups,centers,sizes,b):
    if len(centers)<2: raise UnsupportedDomain('m=1 has undefined minimum center separation')
    radius=min(_distance(a,c,b) for i,a in enumerate(centers) for c in centers[i+1:])
    if radius==0: raise UnsupportedDomain('coincident centers have unresolved primary radius')
    screens=[_radius_screen(radius,size,b) for size in sizes]
    checks=[]
    failure=None
    boundary=False
    for owner,group in enumerate(groups):
        for label in group:
            ds=[_distance(raw[label],center,b) for center in centers]
            own=ds[owner]
            unique=all(own<d for j,d in enumerate(ds) if j!=owner)
            inside=own<radius
            checks.append(dict(label=label,owner=owner,distances_squared=ds,
                               uniquely_nearest=unique,strictly_inside=inside))
            if not unique: failure='point is not strictly nearest to its assigned center'
            if own>radius: failure='point is outside assigned primary circle'
            if own==radius: boundary=True
    if not all(screens): failure='R is less than half the supplied cover size'
    if failure: return 'FALSE',failure,radius,screens,checks
    if boundary: return 'UNSUPPORTED','primary circle equality boundary unresolved by source',radius,screens,checks
    return 'TRUE','given cover satisfies supported primary strict interior conditions',radius,screens,checks


def assess_primary_futlyar(points,parts,*,max_work=2000000,seconds=30):
    result=dict(schema='primary-futlyar-v1',status='INVALID_INPUT',complete=False,work_used=0,
                scope='one complete supplied cover; rational-center strict-interior primary domain')
    b=None
    try:
        b=_budget(max_work,seconds); b.spend()
        raw,groups=_input(points,parts)
        result.update(parts=[list(g) for g in groups],declared_labels=sorted(raw),source_coordinate_sha256=_digest(raw))
        cover=verify_independent_cover_2d(raw,groups,**_remaining(b)); b.spend(cover['work_used'])
        if cover['status']=='NOT_INDEPENDENT':
            return dict(result,status='FALSE',complete=True,reason='given cover is not independent',cover_certificate=cover,work_used=b.used)
        if cover['status']!='VALID_INDEPENDENT_COVER':
            return dict(result,status='UNKNOWN',reason='independence not certified within budget',work_used=b.used)
        b.spend(500+len(raw)*len(groups)**2)
        if not verify_cover_certificate(raw,groups,cover): raise ValueError('cover witness invalid')
        if len(groups)<2: raise UnsupportedDomain('m=1 has undefined minimum center separation')
        blocks=[]; centers=[]; sizes=[]
        for group in groups:
            block={i:raw[i] for i in group}
            answer=solve_exact_center(block,**_remaining(b)); b.spend(answer['work_used'])
            if not answer['complete']:
                return dict(result,status='UNKNOWN',reason='block exact center unavailable: '+answer['status'],work_used=b.used)
            check=assess_exact_center_result(block,answer,**_remaining(b)); b.spend(check['work_used'])
            if check['status']=='UNKNOWN': raise BudgetExceeded('independent center verification unfinished')
            if check['status']!='VERIFIED': raise ValueError('block center certificate invalid')
            center,size=_piece_data(answer,b)
            blocks.append(dict(labels=list(group),result=answer)); centers.append(center); sizes.append(size)
        status,reason,radius,screens,checks=_geometry(raw,groups,centers,sizes,b)
        cert=dict(schema='primary-futlyar-certificate-v1',status=status,parts=[list(g) for g in groups],
                  declared_labels=sorted(raw),source_coordinate_sha256=_digest(raw),
                  cover_certificate=cover,blocks=blocks,centers=centers,radius_squared=radius,
                  size_descriptors=sizes,radius_screens=screens,point_checks=checks)
        if status=='TRUE':
            check=assess_primary_futlyar_certificate(raw,cert,**_remaining(b)); b.spend(check['work_used'])
            if check['status']=='UNKNOWN': raise BudgetExceeded('primary independent verification unfinished')
            if check['status']!='VERIFIED': raise ValueError('primary certificate failed independent verification')
        result.update(status=status,reason=reason,complete=status in ('TRUE','FALSE'),certificate=cert,
                      centers=centers,radius_squared=radius,work_used=b.used)
        b.spend(0)
        return result
    except UnsupportedDomain as exc:
        return dict(result,status='UNSUPPORTED',reason=str(exc),work_used=b.used if b else 0)
    except BudgetExceeded as exc:
        return dict(result,status='UNKNOWN',complete=False,reason=str(exc),work_used=b.used if b else 0)
    except (ValueError,TypeError,KeyError,IndexError,AttributeError,ArithmeticError) as exc:
        return dict(result,status='INVALID_INPUT',reason=str(exc),work_used=b.used if b else 0)


def assess_primary_futlyar_certificate(points,certificate,*,max_work=2000000,seconds=30):
    """Recompute identities and distances; no solver, hull or constructor."""
    b=None
    try:
        b=_budget(max_work,seconds); b.spend()
        z=certificate; _tree(z,b)
        fields={'schema','status','parts','declared_labels','source_coordinate_sha256','cover_certificate',
                'blocks','centers','radius_squared','size_descriptors','radius_screens','point_checks'}
        if type(z) is not dict or set(z)!=fields or z['schema']!='primary-futlyar-certificate-v1' or z['status']!='TRUE':
            raise ValueError('strict successful primary certificate fields required')
        raw,groups=_input(points,z['parts'])
        if z['declared_labels']!=sorted(raw) or any(type(i) is not int for i in z['declared_labels']) or z['source_coordinate_sha256']!=_digest(raw):
            raise ValueError('complete source binding mismatch')
        b.spend(500+len(raw)*len(groups)**2)
        if z['cover_certificate']['status']!='VALID_INDEPENDENT_COVER' or not verify_cover_certificate(raw,groups,z['cover_certificate']):
            raise ValueError('independent cover witness invalid')
        if type(z['blocks']) is not list or len(z['blocks'])!=len(groups): raise ValueError('complete block witnesses required')
        centers=[]; sizes=[]
        for group,row in zip(groups,z['blocks']):
            if type(row) is not dict or set(row)!={'labels','result'} or row['labels']!=list(group) or any(type(i) is not int for i in row['labels']):
                raise ValueError('block identity mismatch')
            check=assess_exact_center_result({i:raw[i] for i in group},row['result'],**_remaining(b)); b.spend(check['work_used'])
            if check['status']=='UNKNOWN': raise BudgetExceeded('center verifier unfinished')
            if check['status']!='VERIFIED': raise ValueError('center witness invalid')
            center,size=_piece_data(row['result'],b); centers.append(center); sizes.append(size)
        if type(z['centers']) is not list or any(type(c) not in (list,tuple) or len(c)!=2 or any(type(x) not in (int,F) for x in c) for c in z['centers']):
            raise ValueError('strict rational centers required')
        if centers!=[tuple(c) for c in z['centers']] or sizes!=z['size_descriptors']:
            raise ValueError('center/size fields mismatch')
        if type(z['size_descriptors']) is not list:
            raise ValueError('size descriptor list required')
        for size in z['size_descriptors']:
            if type(size) is not dict:
                raise ValueError('size descriptor mapping required')
            if size.get('kind')=='squared' and set(size)=={'kind','squared'}:
                _q(size['squared'],b)
            elif size.get('kind')=='ellipse' and set(size)=={'kind','trace','discriminant'}:
                _q(size['trace'],b); _q(size['discriminant'],b)
            else:
                raise ValueError('size descriptor fields invalid')
        radius_field=_q(z['radius_squared'],b)
        status,_,radius,screens,checks=_geometry(raw,groups,centers,sizes,b)
        if type(z['radius_screens']) is not list or any(type(v) is not bool for v in z['radius_screens']):
            raise ValueError('screen boolean list required')
        # Explicit check types, since bool compares equal to rational 0/1 in Python.
        if type(z['point_checks']) is not list: raise ValueError('point check list required')
        for row in z['point_checks']:
            if type(row) is not dict or set(row)!={'label','owner','distances_squared','uniquely_nearest','strictly_inside'}:
                raise ValueError('point check fields invalid')
            if type(row['label']) is not int or type(row['owner']) is not int or type(row['uniquely_nearest']) is not bool or type(row['strictly_inside']) is not bool:
                raise ValueError('point check types invalid')
            if type(row['distances_squared']) is not list: raise ValueError('distance list required')
            for d in row['distances_squared']: _q(d,b)
        if status!='TRUE' or radius_field!=radius or z['radius_screens']!=screens or z['point_checks']!=checks:
            raise ValueError('radius or full point checks mismatch')
        b.spend(0)
        return dict(status='VERIFIED',work_used=b.used)
    except BudgetExceeded as exc:
        return dict(status='UNKNOWN',reason=str(exc),work_used=b.used if b else 0)
    except (UnsupportedDomain,ValueError,TypeError,KeyError,IndexError,AttributeError,ArithmeticError) as exc:
        return dict(status='INVALID',reason=str(exc),work_used=b.used if b else 0)


def verify_primary_futlyar_certificate(points,certificate,*,max_work=2000000,seconds=30):
    return assess_primary_futlyar_certificate(points,certificate,max_work=max_work,seconds=seconds)['status']=='VERIFIED'
