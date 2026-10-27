"""Complete-input four-contact algebraic construction and refusal gates."""
import copy
import json
from fractions import Fraction as F

from kozlov_lab.kozlov2025.exact_contact4_ellipse import complete_input_contact4_ellipse as solve
from kozlov_lab.kozlov2025.exact_contact4_verify import verify_complete_input_contact4_ellipse as verify


GENERAL = {0:(0,0),1:(1,0),2:(0,1),3:(2,F(3,2))}
RECTANGLE = {0:(-2,-1),1:(2,-1),2:(2,1),3:(-2,1)}


def test_true_cubic_complete_input_and_json():
    result = solve(GENERAL)
    assert result['status']=='EXACT_OPTIMUM_COMPLETE_INPUT', result
    cert = result['certificate']
    assert len(cert['algebraic_root']['poly'])==4
    assert verify(GENERAL,cert)
    # Whole serialization includes only ordinary JSON values. JSON object keys
    # become strings on readback; restore the declared integer labels explicitly.
    restored = json.loads(json.dumps(cert))
    restored['all_input_quadratic_values'] = {int(k):v for k,v in restored['all_input_quadratic_values'].items()}
    assert verify(GENERAL,restored)


def test_rectangle_and_affine_image():
    for points in (RECTANGLE,{i:(3*x+2*y+7,x-2*y-5) for i,(x,y) in RECTANGLE.items()},
                   {i:(2*x+y+3,x+3*y-4) for i,(x,y) in GENERAL.items()}):
        result = solve(points)
        assert result['exact_optimum'],result
        assert verify(points,result['certificate'])


def test_complete_inside_point_and_source_binding():
    points = dict(RECTANGLE, **{})
    points[4]=(0,0)
    result=solve(points)
    assert result['exact_optimum'],result
    assert result['certificate']['declared_labels']==list(range(5))
    assert verify(points,result['certificate'])
    outside=dict(points); outside[4]=(10,10)
    assert not verify(outside,result['certificate'])


def test_strict_four_domain_failure_is_not_no_ellipse():
    result=solve({0:(0,0),1:(3,0),2:(0,3),3:(1,1)})
    assert result['status']=='UNSUPPORTED_CONTACT_DOMAIN'
    assert not result['exact_optimum'] and not result['complete']
    assert 'does not disprove any ellipse' in result['reason']


def test_certificate_pollution():
    certificate=solve(GENERAL)['certificate']
    for field in ('center','contact_weights','area_squared_over_pi_squared'):
        bad=copy.deepcopy(certificate)
        target = bad[field][0] if field in ('center','contact_weights') else bad[field]
        target['numerator'][0][0] += target['numerator'][0][1]
        assert not verify(GENERAL,bad)


def test_budget_and_exact_input_gates():
    for kwargs in ({'max_work':0},{'seconds':0},{'max_work':1}):
        result=solve(GENERAL,**kwargs)
        assert result['status']=='PARTIAL_UNKNOWN' and not result['exact_optimum']
    for points in ({0:(0.0,0),1:(1,0),2:(0,1),3:(1,1)},
                   {0:(False,0),1:(1,0),2:(0,1),3:(1,1)},
                   {i:(i,0) for i in range(4)}):
        assert not solve(points)['exact_optimum']
