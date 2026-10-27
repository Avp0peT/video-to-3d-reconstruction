import copy
import unittest
from fractions import Fraction as F
from unittest.mock import patch
from kozlov_lab.kozlov2025.collinear_fits import collinear_fits,assess_collinear_fits_certificate

A={0:(0,0),1:(1,0),2:(3,0)}; AP=((0,),(1,),(2,))
B={10:(0,0),11:(1,0),12:(2,0)}; BP=((10,),(11,),(12,))


class CollinearFitsTests(unittest.TestCase):
    def test_nonzero_full_singleton_fits_and_independent_verifier(self):
        r=collinear_fits(A,AP,B,BP)
        self.assertEqual(r['status'],'COLLINEAR_FITS_SUFFICIENT_WITNESS',r)
        c=r['certificate']
        self.assertEqual(c['alignment_certificate']['error_squared'],F(1,16))
        self.assertEqual(len(c['point_checks']),3)
        with patch('kozlov_lab.kozlov2025.collinear_fits.collinear_fits',side_effect=AssertionError), \
             patch('kozlov_lab.kozlov2025.collinear_fits.assess_primary_futlyar',side_effect=AssertionError), \
             patch('kozlov_lab.kozlov2025.collinear_fits.solve_exact_center',side_effect=AssertionError), \
             patch('kozlov_lab.kozlov2025.collinear_fits.align_collinear_centers',side_effect=AssertionError):
            self.assertEqual(assess_collinear_fits_certificate(A,B,c)['status'],'VERIFIED')

    def test_full_big_block_same_center_and_source_hull_overlap(self):
        b={10:(-10,-10),11:(10,-10),12:(0,20),13:(1,0),14:(2,0)}
        bp=((10,11,12),(13,),(14,))
        r=collinear_fits(A,AP,b,bp)
        self.assertEqual(r['status'],'NO_SUPPORTED_WITNESS',r)
        self.assertIn('whole-block point',r['reason'])
        self.assertFalse(r['general_fits_decided'])

    def test_complete_input_alignment_dual_and_point_tampering(self):
        original=collinear_fits(A,AP,B,BP)['certificate']
        bad=[]
        c=copy.deepcopy(original); c['source_points']=(); bad.append(c)
        c=copy.deepcopy(original); c['alignment_certificate']['branches']=c['alignment_certificate']['branches'][:1]; bad.append(c)
        c=copy.deepcopy(original); c['alignment_certificate']['mapping']=((10,0),); bad.append(c)
        c=copy.deepcopy(original); c['point_checks']=(); bad.append(c)
        c=copy.deepcopy(original); row=c['point_checks'][0]; c['point_checks']=((row[0],False,False,(False,False),row[4]),)+c['point_checks'][1:]; bad.append(c)
        for c in bad:
            self.assertEqual(assess_collinear_fits_certificate(A,B,c)['status'],'INVALID')
        self.assertEqual(assess_collinear_fits_certificate(A,{**B,13:(99,99)},original)['status'],'INVALID')

    def test_budget_and_incomplete_source_partition(self):
        for args in ({'max_work':0},{'seconds':0},{'max_work':1000}):
            self.assertEqual(collinear_fits(A,AP,B,BP,**args)['status'],'UNKNOWN')
        self.assertEqual(collinear_fits(A,AP,B,((10,),(11,)))['status'],'UNSUPPORTED')
        c=collinear_fits(A,AP,B,BP)['certificate']
        self.assertEqual(assess_collinear_fits_certificate(A,B,c,max_work=0)['status'],'UNKNOWN')
