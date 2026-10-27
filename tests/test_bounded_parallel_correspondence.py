import copy
import unittest
from fractions import Fraction as F
from unittest.mock import patch
from kozlov_lab.stereo.official.bounded_parallel_correspondence import search_bounded_parallel_correspondence,assess_bounded_parallel_correspondence_result


def mapping(points,labels=None): return dict(zip(labels if labels is not None else range(len(points)),points))


class BoundedParallelTests(unittest.TestCase):
    def test_unreturned_prepare_counter_charged_as_explicit_upper_bound(self):
        from kozlov_lab.stereo.official.parallel_construction import BudgetExceeded
        x=mapping([(0,0),(1,0),(0,1),(1,1)]); y=mapping([(0,0),(1,0),(0,1),(2,2)])
        with patch('kozlov_lab.stereo.official.parallel_construction.prepare_views',side_effect=BudgetExceeded('forced pre-return exhaustion')):
            r=search_bounded_parallel_correspondence(x,y,max_work=100000)
        self.assertEqual(r['status'],'PARTIAL_UNKNOWN')
        self.assertFalse(r['complete'])
        self.assertEqual(len(r['ledger']),1)
        self.assertEqual(r['work_used'],100000)
        self.assertFalse(r['ledger'][0]['work_accounting']['actual_counter_available'])
        self.assertEqual(assess_bounded_parallel_correspondence_result(x,y,r)['status'],'VERIFIED')

    def test_four_point_24_solutions_never_unique_and_independent_verify(self):
        x=mapping([(0,0),(1,0),(0,1),(1,1)],[-1,0,11,99])
        y=mapping([(0,0),(1,0),(0,1),(2,2)],[2,3,5,8])
        r=search_bounded_parallel_correspondence(x,y)
        self.assertEqual(r['status'],'SEARCH_COMPLETE',r)
        self.assertEqual(r['solutions_count'],24)
        self.assertFalse(r['true_unique_correspondence'])
        with patch('kozlov_lab.stereo.official.bounded_parallel_correspondence._rank_witness',side_effect=AssertionError), \
             patch('kozlov_lab.stereo.official.parallel_construction.prepare_views',side_effect=AssertionError), \
             patch('kozlov_lab.stereo.official.parallel_construction.construct_representative',side_effect=AssertionError):
            self.assertEqual(assess_bounded_parallel_correspondence_result(x,y,r)['status'],'VERIFIED')

    def test_five_six_shuffled_real_parallel_projection_truth_mapping(self):
        body=[(0,0,0),(1,0,0),(0,1,0),(1,1,1),(2,3,2),(3,1,-1)]
        for n in (5,6):
            x={10+i:(p[0],p[1]) for i,p in enumerate(body[:n])}
            ids=[70,20,90,40,30,80][:n]
            y={ids[i]:(p[0]+p[2],p[1]+2*p[2]) for i,p in enumerate(body[:n])}
            r=search_bounded_parallel_correspondence(x,y)
            self.assertEqual(r['status'],'SEARCH_COMPLETE',r)
            truth=tuple((10+i,ids[i]) for i in range(n))
            self.assertIn(truth,[row['mapping'] for row in r['accepted']])
            self.assertEqual(len(r['ledger']),__import__('math').factorial(n))
            self.assertEqual(assess_bounded_parallel_correspondence_result(x,y,r)['status'],'VERIFIED')

    def test_rank4_identity_rejection_and_planar_not_false(self):
        x=mapping([(0,0),(1,0),(0,1),(1,1),(2,0)])
        y=mapping([(0,0),(1,0),(0,1),(2,1),(1,2)])
        r=search_bounded_parallel_correspondence(x,y)
        row=next(row for row in r['ledger'] if row['mapping']==tuple(zip(range(5),range(5))))
        self.assertEqual(row['status'],'REJECTED_RANK4')
        self.assertEqual(row['rank_certificate']['rank'],4)
        square=mapping([(0,0),(1,0),(0,1),(1,1)])
        r=search_bounded_parallel_correspondence(square,square)
        identity=r['ledger'][0]
        self.assertEqual(identity['status'],'UNSUPPORTED_PLANAR')
        self.assertGreater(r['unsupported_count'],0)
        self.assertEqual(assess_bounded_parallel_correspondence_result(square,square,r)['status'],'VERIFIED')

    def test_budget_partial_complete_tamper_and_identity_types(self):
        x=mapping([(0,0),(1,0),(0,1),(1,1)]); y=mapping([(0,0),(1,0),(0,1),(2,2)])
        for kwargs in ({'max_work':0},{'seconds':0},{'max_candidates':2}):
            r=search_bounded_parallel_correspondence(x,y,**kwargs)
            self.assertEqual(r['status'],'PARTIAL_UNKNOWN')
            self.assertFalse(r['complete'])
            self.assertEqual(assess_bounded_parallel_correspondence_result(x,y,r)['status'],'VERIFIED')
        for bad in ({'0':(0,0),1:(1,0),2:(0,1),3:(1,1)}, {False:(0,0),1:(1,0),2:(0,1),3:(1,1)}):
            self.assertEqual(search_bounded_parallel_correspondence(bad,y)['status'],'UNSUPPORTED_INPUT')
        original=search_bounded_parallel_correspondence(x,y)
        r=copy.deepcopy(original); r['ledger']=r['ledger'][:-1]
        self.assertEqual(assess_bounded_parallel_correspondence_result(x,y,r)['status'],'INVALID')
        r=copy.deepcopy(original); row=r['ledger'][0]; row['certificate']['projections']=(((F(0),)*4,)*2,)+row['certificate']['projections'][1:]
        self.assertEqual(assess_bounded_parallel_correspondence_result(x,y,r)['status'],'INVALID')
