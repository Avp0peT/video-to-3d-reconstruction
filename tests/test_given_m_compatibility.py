import copy
import unittest
from fractions import Fraction as F
from unittest.mock import patch
from kozlov_lab.kozlov2025.zero_error_fits import given_zero_error_fits
from kozlov_lab.kozlov2025.given_m_compatibility import given_m_compatibility, assess_given_m_compatibility_certificate

I=((F(1),F(0)),(F(0),F(1)))


def basic():
    a={0:(0,0),1:(10,0)}; b={2:(0,20),3:(10,20)}
    ap=((0,),(1,)); bp=((2,),(3,))
    first=given_zero_error_fits(a,ap,a,ap,I,(0,0),{0:0,1:1})['certificate']
    second=given_zero_error_fits(a,ap,b,bp,I,(0,-20),{0:0,1:1})['certificate']
    return {**a,**b},(tuple(a),tuple(b)),ap,[first,second]


class GivenMCompatibilityTests(unittest.TestCase):
    def test_one_reference_one_f_all_members_and_independent_verification(self):
        env,group,ap,certs=basic()
        r=given_m_compatibility(env,group,0,ap,certs)
        self.assertEqual(r['status'],'M_COMPATIBLE_SUFFICIENT_WITNESS',r)
        self.assertFalse(r['general_compatibility_decided'])
        with patch('kozlov_lab.kozlov2025.given_m_compatibility.given_m_compatibility',side_effect=AssertionError), \
             patch('kozlov_lab.kozlov2025.zero_error_fits.given_zero_error_fits',side_effect=AssertionError), \
             patch('kozlov_lab.kozlov2025.primary_futlyar.assess_primary_futlyar',side_effect=AssertionError), \
             patch('kozlov_lab.kozlov2025.exact_center.solve_exact_center',side_effect=AssertionError):
            self.assertEqual(assess_given_m_compatibility_certificate(env,r['certificate'])['status'],'VERIFIED')

    def test_individually_valid_different_reference_rejected(self):
        env,group,ap,certs=basic()
        b={i:env[i] for i in group[1]}; bp=((2,),(3,))
        self_fit=given_zero_error_fits(b,bp,b,bp,I,(0,0),{0:0,1:1})
        self.assertEqual(self_fit['status'],'TRUE')
        certs[1]=self_fit['certificate']
        self.assertEqual(given_m_compatibility(env,group,0,ap,certs)['status'],'INVALID_WITNESS')

    def test_individually_valid_same_reference_different_primary_rejected(self):
        a={0:(0,0),1:(4,0),2:(10,0)}
        b={3:(0,20),4:(4,20),5:(10,20)}
        f1=((0,1),(2,)); f2=((0,),(1,2)); bp=((3,),(4,5))
        r1=given_zero_error_fits(a,f1,a,f1,I,(0,0),{0:0,1:1})
        r2=given_zero_error_fits(a,f2,b,bp,I,(0,-20),{0:0,1:1})
        self.assertEqual(r1['status'],'TRUE',r1)
        self.assertEqual(r2['status'],'TRUE',r2)
        r=given_m_compatibility({**a,**b},(tuple(a),tuple(b)),0,f1,[r1['certificate'],r2['certificate']])
        self.assertEqual(r['status'],'INVALID_WITNESS')

    def test_missing_duplicate_members_m_and_empty_boundary(self):
        env,group,ap,certs=basic()
        for g,cs in [(group,certs[:1]),(group,[certs[0],certs[0]]),((group[0],group[0]),certs),((),certs)]:
            self.assertEqual(given_m_compatibility(env,g,0,ap,cs)['status'],'INVALID_WITNESS')
        c=given_m_compatibility(env,group,0,ap,certs)['certificate']
        forged=copy.deepcopy(c); forged['m']=3
        self.assertEqual(assess_given_m_compatibility_certificate(env,forged)['status'],'INVALID')
        forged=copy.deepcopy(c); forged['m']=True
        self.assertEqual(assess_given_m_compatibility_certificate(env,forged)['status'],'INVALID')
        forged=copy.deepcopy(c); forged['member_certificates']=forged['member_certificates'][:1]
        self.assertEqual(assess_given_m_compatibility_certificate(env,forged)['status'],'INVALID')

    def test_single_decreasing_budget_and_complete_environment_binding(self):
        env,group,ap,certs=basic()
        self.assertEqual(given_m_compatibility(env,group,0,ap,certs,max_work=0)['status'],'UNKNOWN')
        r=given_m_compatibility(env,group,0,ap,certs)
        self.assertGreater(r['work_used'],0)
        self.assertEqual(assess_given_m_compatibility_certificate({**env,9:(99,99)},r['certificate'])['status'],'INVALID')
