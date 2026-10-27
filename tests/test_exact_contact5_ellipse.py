import copy
from fractions import Fraction as F
import pytest
from kozlov_lab.kozlov2025 import exact_contact5_ellipse as mod

P={0:(1,0),1:(F(-3,5),F(4,5)),2:(F(-3,5),F(-4,5)),3:(0,1),4:(0,-1)}


def test_five_strict_contacts_hand_derived_truth():
    r=mod.complete_input_contact5_ellipse(P);z=r['certificate']
    assert r['status']=='EXACT_OPTIMUM_COMPLETE_INPUT' and r['complete'] and r['exact_optimum']
    assert r['visited_quintuples']==1
    assert z['center']==(0,0) and z['shape_matrix']==((1,0),(0,1))
    assert z['conic_coefficients']==(1,0,1,0,0,-1)
    assert z['contact_weights']==(F(5,16),F(25,96),F(25,96),F(1,12),F(1,12))
    assert z['area_squared_over_pi_squared']==1
    assert z['major_axis_length']==dict(expression='sqrt(2*(trace+sqrt(discriminant)))',trace=2,discriminant=0)
    assert mod.verify_complete_input_contact5_ellipse(P,z)


@pytest.mark.parametrize('a,t',[(((2,1),(-1,3)),(10,4)),(((F(1,7),0),(0,F(1,11))),(F(1,5),F(2,3))),(((0,1),(-1,0)),(0,0))])
def test_exact_affine_truth(a,t):
    q={i:tuple(sum(a[j][k]*p[k] for k in (0,1))+t[j] for j in (0,1)) for i,p in P.items()}
    z=mod.complete_input_contact5_ellipse(q)['certificate']
    assert z['center']==t
    assert z['shape_matrix']==tuple(tuple(sum(a[j][k]*a[l][k] for k in (0,1)) for l in (0,1)) for j in (0,1))
    assert z['contact_weights']==(F(5,16),F(25,96),F(25,96),F(1,12),F(1,12))
    assert mod.verify_complete_input_contact5_ellipse(q,z)


def test_full_input_interior_and_boundary_not_dropped():
    q={**P,5:(0,0),6:(-1,0)}
    z=mod.complete_input_contact5_ellipse(q)['certificate']
    assert z['declared_labels']==list(range(7)) and z['support_labels']==list(range(5))
    assert set(z['all_input_quadratic_values'])==set(q)
    assert z['all_input_quadratic_values'][5]==0 and z['all_input_quadratic_values'][6]==1
    assert mod.verify_complete_input_contact5_ellipse(q,z)


def test_outside_complete_point_rejects_old_support_certificate():
    z=mod.complete_input_contact5_ellipse(P)['certificate']
    assert not mod.verify_complete_input_contact5_ellipse({**P,5:(10,10)},z)
    q={**P,5:(0,0)};z=mod.complete_input_contact5_ellipse(q)['certificate']
    assert not mod.verify_complete_input_contact5_ellipse({**P,5:(10,10)},z)
    assert not mod.verify_complete_input_contact5_ellipse(P,z)


@pytest.mark.parametrize('mutation',['conic','minor','minor_columns','rho','center','shape','inverse','weight','zero_weight','negative_weight','q','det','area','cov','size','axis','support','declared','digest','proof','exact','bool_weight','float_weight','float_center','float_conic','float_q','float_size','bool_label'])
def test_certificate_tampering_and_type_drift(mutation):
    z=copy.deepcopy(mod.complete_input_contact5_ellipse(P)['certificate'])
    if mutation=='conic':z['conic_coefficients']=(1,0,1,1,0,-1)
    elif mutation=='minor':z['rank5_minor_determinant']+=1
    elif mutation=='minor_columns':z['rank5_minor_columns']=[0,1,2,3,3]
    elif mutation=='rho':z['radius_squared_level']+=1
    elif mutation=='center':z['center']=(1,0)
    elif mutation=='shape':z['shape_matrix']=((2,0),(0,1))
    elif mutation=='inverse':z['inverse_shape_matrix']=((1,1),(0,1))
    elif mutation=='weight':z['contact_weights']=(F(1,5),)*5
    elif mutation=='zero_weight':z['contact_weights']=(0,F(1,4),F(1,4),F(1,4),F(1,4))
    elif mutation=='negative_weight':z['contact_weights']=(-1,F(1,2),F(1,2),F(1,2),F(1,2))
    elif mutation=='q':z['all_input_quadratic_values'][0]=0
    elif mutation=='det':z['determinant_shape']+=1
    elif mutation=='area':z['area_squared_over_pi_squared']+=1
    elif mutation=='cov':z['covariance_determinant']+=1
    elif mutation=='size':z['major_axis_length']['trace']+=1
    elif mutation=='axis':z['semiaxis_squared']['discriminant']+=1
    elif mutation=='support':z['support_labels']=[0,1,2,3,3]
    elif mutation=='declared':z['declared_labels'].pop()
    elif mutation=='digest':z['source_coordinate_sha256']='fake'
    elif mutation=='proof':z['proof']='fake'
    elif mutation=='exact':z['exact_optimum']=1
    elif mutation=='bool_weight':z['contact_weights']=(True,)*5
    elif mutation=='float_weight':z['contact_weights']=tuple(float(v) for v in z['contact_weights'])
    elif mutation=='float_center':z['center']=(0.,0.)
    elif mutation=='float_conic':z['conic_coefficients']=(1.,0,1,0,0,-1)
    elif mutation=='float_q':z['all_input_quadratic_values'][0]=1.
    elif mutation=='float_size':z['major_axis_length']['trace']=2.
    elif mutation=='bool_label':z['support_labels'][0]=False
    assert not mod.verify_complete_input_contact5_ellipse(P,z)


def test_verifier_has_no_solve_search_or_hull(monkeypatch):
    z=mod.complete_input_contact5_ellipse(P)['certificate']
    def forbidden(*a,**kw):raise AssertionError('construction forbidden')
    monkeypatch.setattr(mod,'_rref',forbidden)
    monkeypatch.setattr(mod,'combinations',forbidden)
    import kozlov_lab.kozlov2025.independent_cover as cover
    monkeypatch.setattr(cover,'hull',forbidden)
    assert mod.verify_complete_input_contact5_ellipse(P,z)


@pytest.mark.parametrize('q,reason',[
    ({i:(i,0) for i in range(5)},'rank_not_five'),
    ({i:(i,i*i) for i in range(5)},'not_definite_conic'),
    ({0:(1,0),1:(F(3,5),F(4,5)),2:(F(3,5),F(-4,5)),3:(F(4,5),F(3,5)),4:(F(4,5),F(-3,5))},'nonpositive_weight')])
def test_unsupported_profiles_do_not_claim_nonexistence(q,reason):
    r=mod.complete_input_contact5_ellipse(q)
    assert r['status']=='UNSUPPORTED_CONTACT_DOMAIN' and not r['complete']
    assert r['candidate_failures'][reason]==1 and 'does not classify' in r['reason']


def test_fourpoint_square_remains_unsupported():
    r=mod.complete_input_contact5_ellipse({0:(0,0),1:(1,0),2:(1,1),3:(0,1)})
    assert r['status']=='UNSUPPORTED_CONTACT_DOMAIN' and r['visited_quintuples']==0


@pytest.mark.parametrize('kw',[{'max_work':0},{'seconds':0},{'max_work':30}])
def test_shared_budget_unknown(kw):
    r=mod.complete_input_contact5_ellipse(P,**kw)
    assert r['status']=='PARTIAL_UNKNOWN' and not r['complete'] and 'certificate' not in r


def test_work_budget_not_reset_between_candidates():
    q={i:(i,0) for i in range(7)}
    r=mod.complete_input_contact5_ellipse(q,max_work=2000000)
    assert r['visited_quintuples']==21 and r['status']=='UNSUPPORTED_CONTACT_DOMAIN'
    limited=mod.complete_input_contact5_ellipse(q,max_work=r['work_used']//2)
    assert limited['status']=='PARTIAL_UNKNOWN' and 0<limited['visited_quintuples']<21


def test_intermediate_bit_limit_unknown(monkeypatch):
    monkeypatch.setattr(mod,'MAX_INTERMEDIATE_BITS',5)
    r=mod.complete_input_contact5_ellipse(P)
    assert r['status']=='PARTIAL_UNKNOWN' and 'bit bound' in r['reason']


@pytest.mark.parametrize('q',[{}, {i:(i,0) for i in range(8)}, {False:(0,0)}, {0:(0,0),1:(0,0)}, {0:(float('nan'),0)}, {0:(True,0)}, {0:(F(1,2**2049),0)}])
def test_invalid_complete_input(q):
    assert mod.complete_input_contact5_ellipse(q)['status']=='UNSUPPORTED_INPUT'


@pytest.mark.parametrize('kw',[{'max_work':True},{'seconds':True},{'seconds':31},{'max_work':2000001}])
def test_invalid_budget(kw):
    assert mod.complete_input_contact5_ellipse(P,**kw)['status']=='UNSUPPORTED_INPUT'
