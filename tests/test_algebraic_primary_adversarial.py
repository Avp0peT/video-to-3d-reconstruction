"""Independent single-domain primary witnesses from the source cubic formulas."""
from fractions import Fraction as F
from copy import deepcopy
import pytest
from kozlov_lab.kozlov2025 import algebraic_primary_futlyar as primary
from kozlov_lab.kozlov2025.real_algebraic import Root,AN
from kozlov_lab.kozlov2025.independent_cover import Budget
from kozlov_lab.kozlov2025.exact_triangle_ellipse import _digest
from kozlov_lab.kozlov2025.exact_center import assess_exact_center_result
from kozlov_lab.kozlov2025.exact_contact4_verify import PROOF


def point_witness(label,point):
    raw={label:tuple(F(x) for x in point)}
    size=dict(expression='sqrt(length_squared)',length_squared=F(0))
    cert=dict(schema='exact-center-degenerate-certificate-v1',kind='point',declared_labels=[label],
              source_coordinate_sha256=_digest(raw),endpoint_labels=[label],center=raw[label],major_axis_length=deepcopy(size))
    return dict(schema='exact-center-v1',status='EXACT_DEGENERATE_COMPLETE_INPUT',complete=True,
                exact_optimum=True,declared_labels=[label],source_coordinate_sha256=_digest(raw),
                profile='point',number_context={'domain':'rational'},center=raw[label],major_axis_length=size,certificate=cert)


def cubic_witness():
    raw={0:(F(0),F(0)),1:(F(1),F(0)),2:(F(0),F(1)),3:(F(2),F(3,2))}
    root=Root((-256,-1440,864,27),F(87,50),F(7,4),Budget())
    a=root.num((0,1)); cross=-F(1,3)-a/8
    k=a-cross**2; j=5*a*(4+3*a)/48
    c=(a*(1-cross)/(2*k),(a-cross)/(2*k))
    s=((j*a/k**2,-j*cross/k**2),(-j*cross/k**2,j/k**2))
    h=((k/j,k*cross/j),(k*cross/j,k*a/j))
    w3=(s[0][1]/2+c[0]*c[1])/3
    weights=(1-c[0]-c[1]+F(5,2)*w3,c[0]-2*w3,c[1]-F(3,2)*w3,w3)
    det=s[0][0]*s[1][1]-s[0][1]**2; trace=s[0][0]+s[1][1]
    disc=(s[0][0]-s[1][1])**2+4*s[0][1]**2
    def q(point):
        x,y=point[0]-c[0],point[1]-c[1]
        return h[0][0]*x*x+2*h[0][1]*x*y+h[1][1]*y*y
    cert=dict(schema='complete-input-contact4-certificate-v1',declared_labels=sorted(raw),
              source_coordinate_sha256=_digest(raw),support_labels=sorted(raw),algebraic_root=root.to_data(),
              center=[x.to_data() for x in c],shape_matrix=[[x.to_data() for x in row] for row in s],
              inverse_shape_matrix=[[x.to_data() for x in row] for row in h],contact_weights=[x.to_data() for x in weights],
              all_input_quadratic_values={i:q(v).to_data() for i,v in raw.items()},determinant_shape=det.to_data(),
              covariance_determinant=(det/4).to_data(),area_squared_over_pi_squared=det.to_data(),
              semiaxis_squared=dict(expression='(trace +/- sqrt(discriminant))/2',trace=trace.to_data(),discriminant=disc.to_data()),
              major_axis_length=dict(expression='sqrt(2*(trace+sqrt(discriminant)))',trace=trace.to_data(),discriminant=disc.to_data()),
              proof=PROOF,exact_optimum=True)
    result=dict(schema='exact-center-v1',status='EXACT_OPTIMUM_COMPLETE_INPUT',complete=True,exact_optimum=True,
                declared_labels=sorted(raw),source_coordinate_sha256=_digest(raw),profile='contact4',
                number_context=dict(domain='real-algebraic',algebraic_root=deepcopy(cert['algebraic_root'])),
                center=deepcopy(cert['center']),major_axis_length=deepcopy(cert['major_axis_length']),certificate=cert)
    return raw,result,root,c


def independent_primary():
    raw,ellipse,root,c=cubic_witness()
    raw[4]=(F(100),F(100)); centers=[c,tuple(root.num((100,)) for _ in (0,1))]
    radius=sum((v-100)**2 for v in c)
    checks=[]
    for label,point in raw.items():
        distances=[sum((point[k]-center[k])**2 for k in (0,1)) for center in centers]
        checks.append(dict(label=label,owner=0 if label<4 else 1,
                           distances_squared=[x.to_data() for x in distances],uniquely_nearest=True,strictly_inside=True))
    cover=dict(schema='independent-cover-certificate-v1',contract='independent-cover-v1',
               status='VALID_INDEPENDENT_COVER',coverage_complete=True,decision_complete=True,
               pairwise_independence=True,missing_labels=[],pair_certificates=[dict(left_part=0,right_part=1,
                   kind='strict_separation',axis=(F(1),F(0)))])
    cert=dict(schema='algebraic-primary-certificate-v1',status='TRUE',declared_labels=sorted(raw),
              source_coordinate_sha256=_digest(raw),parts=[[0,1,2,3],[4]],cover_certificate=cover,
              blocks=[dict(labels=[0,1,2,3],result=ellipse),dict(labels=[4],result=point_witness(4,raw[4]))],
              number_context=dict(domain='real-algebraic',algebraic_root=deepcopy(ellipse['number_context']['algebraic_root'])),
              centers=[[x.to_data() for x in v] for v in centers],radius_squared=radius.to_data(),
              radius_screens=[True,True],point_checks=checks)
    return raw,cert


def test_exact_center_algebraic_context_bool_is_rejected():
    raw,result,_,_=cubic_witness()
    assert assess_exact_center_result(raw,result)['status']=='VERIFIED'
    result['number_context']['algebraic_root']['poly'][0][1]=True
    assert assess_exact_center_result(raw,result)['status']=='INVALID'


def test_hand_derived_genuine_cubic_primary_and_verifier_independence(monkeypatch):
    raw,cert=independent_primary()
    # Mod 13 this cubic has no root: a degree-three factor is irreducible.
    assert all((27*x**3+864*x**2-1440*x-256)%13 for x in range(13))
    def forbidden(*a,**k): raise AssertionError('producer called by verifier')
    monkeypatch.setattr(primary,'solve_exact_center',forbidden)
    monkeypatch.setattr(primary,'verify_independent_cover_2d',forbidden)
    assert primary.assess_algebraic_primary_certificate(raw,cert)['status']=='VERIFIED'
    before=deepcopy(cert['number_context']['algebraic_root'])
    assert primary.assess_algebraic_primary_certificate(raw,cert)['status']=='VERIFIED'
    assert cert['number_context']['algebraic_root']==before


def test_genuine_cubic_geometry_true_false_against_independent_bounds():
    raw,_,root,c=cubic_witness()
    # These bounds from the source derivation independently justify nearest
    # rejection at p=(1,0) for a singleton (3/2,0): own > .78² > 1/4.
    assert F(93,100)<c[0]<F(95,100)
    assert F(78,100)<c[1]<F(82,100)
    far={**raw,4:(F(100),F(100))}
    assert primary.assess_algebraic_primary_futlyar(far,[[0,1,2,3],[4]])['status']=='TRUE'
    near={**raw,4:(F(3,2),F(0))}
    answer=primary.assess_algebraic_primary_futlyar(near,[[0,1,2,3],[4]])
    assert answer['status']=='FALSE'
    # The singleton is strictly separated by n=(3/2,-1).
    assert max(F(3,2)*p[0]-p[1] for p in raw.values())<F(9,4)
    nearest_check=next(row for row in answer['certificate']['point_checks'] if row['label']==1)
    assert nearest_check['uniquely_nearest'] is False
    # Adding two remote rational centers separated by exactly 1 makes global
    # R²=1, although their distances to the cubic center are enormous.
    crowded={**far,5:(F(101),F(100))}
    answer=primary.assess_algebraic_primary_futlyar(crowded,[[0,1,2,3],[4],[5]])
    assert answer['status']=='FALSE'
    r=Root.from_data(answer['certificate']['number_context']['algebraic_root'],Budget())
    assert AN.from_data(answer['certificate']['radius_squared'],r).equals(1)


def test_hand_primary_context_full_input_and_cached_type_tampering():
    raw,cert=independent_primary()
    for mutate in ('bool_context','interval','poly','root_extra','point_bool','screen_int','center_bool','extra'):
        bad=deepcopy(cert)
        if mutate=='bool_context': bad['number_context']['algebraic_root']['poly'][0][1]=True
        elif mutate=='interval': bad['number_context']['algebraic_root']['lo']=[0,1]
        elif mutate=='poly': bad['number_context']['algebraic_root']['poly'][0][0]+=1
        elif mutate=='root_extra': bad['number_context']['algebraic_root']['trust_me']=True
        elif mutate=='point_bool': bad['point_checks'][0]['owner']=False
        elif mutate=='screen_int': bad['radius_screens'][0]=1
        elif mutate=='center_bool': bad['centers'][1][0]['denominator'][0][0]=True
        else: bad['extra']=1
        assert primary.assess_algebraic_primary_certificate(raw,bad)['status']=='INVALID',mutate
    assert primary.assess_algebraic_primary_certificate({**raw,5:(200,200)},cert)['status']=='INVALID'
    assert primary.assess_algebraic_primary_certificate(raw,cert,max_work=1)['status']=='UNKNOWN'
    assert primary.assess_algebraic_primary_certificate(raw,cert,seconds=0)['status']=='UNKNOWN'


def test_full_partition_and_single_domain_cardinality_bound():
    from kozlov_lab.kozlov2025.cover_enumeration import _partitions
    # This is the exact structural reason no algebraic compositum is needed.
    for n in range(2,8):
        for parts in _partitions(list(range(n))):
            if len(parts)>=2:
                assert sum(len(group)>=4 for group in parts)<=1
    raw,cert=independent_primary()
    bad=deepcopy(cert); bad['blocks'][0]['labels']=[0,1,2]
    assert primary.assess_algebraic_primary_certificate(raw,bad)['status']=='INVALID'
    bad=deepcopy(cert); bad['parts']=[[0,1,2,3]]
    assert primary.assess_algebraic_primary_certificate(raw,bad)['status']=='INVALID'


@pytest.mark.parametrize('field',['center','major_axis_length'])
def test_exact_center_outer_algebraic_numeric_bool_rejected(field):
    raw,result,_,_=cubic_witness()
    if field=='center': result['center'][0]['denominator'][0][0]=True
    else: result['major_axis_length']['trace']['denominator'][0][0]=True
    assert assess_exact_center_result(raw,result)['status']=='INVALID'


def test_all_rational_domain_is_fixed_to_x_root():
    raw={0:(F(0),F(0)),1:(F(3),F(4))}
    def number(x): return dict(numerator=[[F(x).numerator,F(x).denominator]],denominator=[[1,1]])
    cover=dict(schema='independent-cover-certificate-v1',contract='independent-cover-v1',
               status='VALID_INDEPENDENT_COVER',coverage_complete=True,decision_complete=True,
               pairwise_independence=True,missing_labels=[],pair_certificates=[dict(left_part=0,right_part=1,
                   kind='strict_separation',axis=(F(1),F(0)))])
    cert=dict(schema='algebraic-primary-certificate-v1',status='TRUE',declared_labels=[0,1],
              source_coordinate_sha256=_digest(raw),parts=[[0],[1]],cover_certificate=cover,
              blocks=[dict(labels=[i],result=point_witness(i,raw[i])) for i in (0,1)],
              number_context=dict(domain='real-algebraic',algebraic_root=dict(poly=[[0,1],[1,1]],lo=[-1,1],hi=[1,1])),
              centers=[[number(x) for x in point] for point in raw.values()],radius_squared=number(25),
              radius_screens=[True,True],point_checks=[dict(label=i,owner=i,
                  distances_squared=[number(x) for x in ([0,25] if i==0 else [25,0])],
                  uniquely_nearest=True,strictly_inside=True) for i in (0,1)])
    assert primary.assess_algebraic_primary_certificate(raw,cert)['status']=='VERIFIED'
    cert['number_context']['algebraic_root']=dict(poly=[[-1,1],[1,1]],lo=[0,1],hi=[2,1])
    assert primary.assess_algebraic_primary_certificate(raw,cert)['status']=='INVALID'


def test_algebraic_radius_screen_requires_nonnegative_u_before_squaring():
    root=Root((-2,0,0,1),1,2,Budget())
    a=root.num((0,1)); zero=root.num((0,))
    centers=[(zero,zero),(a,zero)]
    radius=a*a; trace=4*radius; disc=radius**2
    assert (2*radius-trace).sign()==-1
    assert ((2*radius-trace)**2-disc).sign()==1
    sizes=[dict(kind='ellipse',trace=trace,discriminant=disc),dict(kind='squared',squared=zero)]
    # The point coordinates need not equal these synthetic centers: failure of
    # the necessary half-size gate is decisive for every geometry.
    status,_,fields=primary._geometry({0:(F(0),F(0)),1:(F(1),F(0))},[(0,),(1,)],centers,sizes,root)
    assert status=='FALSE'
    assert fields['radius_screens'][0] is False
