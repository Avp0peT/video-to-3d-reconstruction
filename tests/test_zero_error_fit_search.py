import unittest
import copy
from fractions import Fraction as F
from unittest.mock import patch
from kozlov_lab.kozlov2025.zero_error_fit_search import search_zero_error_fits, assess_zero_error_fit_search_result


class ZeroErrorFitSearchTests(unittest.TestCase):
    def test_rank2_full_points_and_wrong_candidates_before_true(self):
        a = {0:(-1,-1),1:(1,-1),2:(0,2),3:(3,0),4:(0,100)}
        parts = ((0,1,2),(3,),(4,))
        # Deliberate part permutation requires enumeration past wrong correspondences.
        bp = ((4,),(0,1,2),(3,))
        b = {i:(F(p[0]-2,2),F(p[1]-1,3)) for i,p in a.items()}
        r = search_zero_error_fits(a,parts,b,bp)
        self.assertEqual(r['status'],'TRUE',r)
        self.assertEqual(r['center_rank'],2)
        self.assertGreater(r['candidates_tried'],1)
        self.assertEqual(assess_zero_error_fit_search_result(a,b,r)['status'],'VERIFIED')
        with patch('kozlov_lab.kozlov2025.zero_error_fit_search.search_zero_error_fits',side_effect=AssertionError), \
             patch('kozlov_lab.kozlov2025.zero_error_fit_search.solve_exact_center',side_effect=AssertionError), \
             patch('kozlov_lab.kozlov2025.primary_futlyar.assess_primary_futlyar',side_effect=AssertionError):
            self.assertEqual(assess_zero_error_fit_search_result(a,b,r)['status'],'VERIFIED')

    def test_rank1_restricted_extension_and_budget(self):
        a={0:(0,0),1:(10,0)}; parts=((0,),(1,))
        r=search_zero_error_fits(a,parts,a,parts)
        self.assertEqual(r['status'],'TRUE')
        self.assertEqual(r['center_rank'],1)
        self.assertFalse(r['general_fits_decided'])
        for field in ('matrix','translation','part_bijection'):
            forged=copy.deepcopy(r)
            if field=='matrix':
                forged[field]=((F(1),False),(F(0),F(1)))
            elif field=='translation':
                forged[field]=(False,F(0))
            else:
                forged[field]=((False,False),(1,1))
            self.assertEqual(assess_zero_error_fit_search_result(a,a,forged)['status'],'INVALID')
        self.assertEqual(search_zero_error_fits(a,parts,a,parts,max_work=0)['status'],'UNKNOWN')
        self.assertEqual(search_zero_error_fits(a,parts,a,((0,),))['status'],'UNSUPPORTED')
        self.assertEqual(search_zero_error_fits(a,parts,a,((0,1),(1,)))['status'],'UNSUPPORTED')

    def test_rank1_no_supported_is_not_general_false(self):
        a={0:(0,0),1:(10,0)}; ap=((0,),(1,))
        b={0:(-1,-20),1:(1,-20),2:(0,40),3:(9,-20),4:(11,-20),5:(10,40)}
        r=search_zero_error_fits(a,ap,b,((0,1,2),(3,4,5)))
        self.assertEqual(r['status'],'NO_SUPPORTED_WITNESS',r)
        self.assertTrue(r['search_complete'])
        self.assertFalse(r['general_fits_decided'])
        # A transverse compression could fit; fixed J extension is not complete.
