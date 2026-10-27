import copy
import itertools
import unittest
from fractions import Fraction as F
from unittest.mock import patch
from kozlov_lab.kozlov2025.collinear_center_alignment import align_collinear_centers, assess_collinear_alignment_certificate


def line(values): return {i:(F(v),F(0)) for i,v in enumerate(values)}


def independent_permutation_oracle(x,y):
    # For fixed slope, best intercept bisects min/max residual. The convex
    # residual range changes slope only at pairwise residual crossings.
    best=None
    for permutation in itertools.permutations(y):
        slopes={F(0)}
        for i,j in itertools.combinations(range(len(x)),2):
            slopes.add(F(x[i]-x[j],permutation[i]-permutation[j]))
        for a in slopes:
            residual=[x[i]-a*permutation[i] for i in range(len(x))]
            e=(max(residual)-min(residual))/2
            best=e if best is None else min(best,e)
    return best


class CollinearAlignmentTests(unittest.TestCase):
    def test_reverse_direction_strictly_better_and_distinct_source_line(self):
        x=line([0,1,4]); y={i:(F(0),F(v)) for i,v in enumerate([0,3,4])}
        r=align_collinear_centers(x,y)
        self.assertEqual(r['status'],'GLOBAL_OPTIMAL_STABLE',r)
        c=r['certificate']
        self.assertEqual(c['chosen'],1)
        self.assertEqual(c['error_squared'],0)
        self.assertGreater(c['branches'][0]['primal'][2],c['branches'][1]['primal'][2])
        self.assertEqual(assess_collinear_alignment_certificate(x,y,c)['status'],'VERIFIED')

    def test_nonzero_true_optimum_and_independent_lower_bound(self):
        x,y=line([0,1,3]),line([0,1,2])
        r=align_collinear_centers(x,y)
        self.assertEqual(r['status'],'GLOBAL_OPTIMAL_STABLE',r)
        c=r['certificate']
        self.assertEqual(c['error_squared'],F(1,16))
        self.assertGreater(c['error_squared'],0)
        self.assertEqual(assess_collinear_alignment_certificate(x,y,c)['status'],'VERIFIED')
        with patch('kozlov_lab.kozlov2025.collinear_center_alignment.align_collinear_centers',side_effect=AssertionError), \
             patch('kozlov_lab.kozlov2025.collinear_center_alignment._branch',side_effect=AssertionError), \
             patch('kozlov_lab.kozlov2025.collinear_center_alignment._linear3',side_effect=AssertionError):
            self.assertEqual(assess_collinear_alignment_certificate(x,y,c)['status'],'VERIFIED')

    def test_all_permutations_small_grid_oracle(self):
        triples=list(itertools.combinations([0,1,2,4],3))
        for x,y in itertools.product(triples,repeat=2):
            r=align_collinear_centers(line(x),line(y))
            self.assertEqual(r['status'],'GLOBAL_OPTIMAL_STABLE',r)
            self.assertEqual(r['certificate']['error_squared'],independent_permutation_oracle(x,y)**2)

    def test_common_shear_rotation_scaling_lower_bounds(self):
        x,y=line([0,1,3]),line([0,1,2])
        base=align_collinear_centers(x,y)['certificate']
        for h in [((1,2),(0,1)),((0,-1),(1,0)),((2,0),(0,3)),((1,0),(2,1))]:
            transform=lambda p:tuple(sum(F(h[i][k])*p[k] for k in (0,1)) for i in (0,1))
            xx={i:transform(p) for i,p in x.items()}; yy={i:transform(p) for i,p in y.items()}
            r=align_collinear_centers(xx,yy)
            self.assertEqual(r['status'],'GLOBAL_OPTIMAL_STABLE')
            factor=F(h[0][0]**2+h[1][0]**2)
            self.assertEqual(r['certificate']['error_squared'],base['error_squared']*factor)
            self.assertEqual(tuple(z['primal'][2] for z in r['certificate']['branches']),tuple(z['primal'][2] for z in base['branches']))
            self.assertEqual(assess_collinear_alignment_certificate(xx,yy,r['certificate'])['status'],'VERIFIED')

    def test_zero_and_invalid_domain_and_budget(self):
        x,y=line([0,2,4]),line([1,2,3])
        self.assertEqual(align_collinear_centers(x,y)['certificate']['error_squared'],0)
        self.assertEqual(align_collinear_centers(x,y,max_work=0)['status'],'UNKNOWN')
        self.assertEqual(align_collinear_centers(x,y,seconds=0)['status'],'UNKNOWN')
        for bad in [{0:(0,0)},{0:(0,0),1:(0,0)}, {0:(0,0),1:(1,0),2:(0,1)}, {0:(0.0,0),1:(1,0)}]:
            self.assertEqual(align_collinear_centers(bad,y)['status'],'UNSUPPORTED')

    def test_dual_missing_negative_bool_and_full_input_tampering(self):
        x,y=line([0,1,3]),line([0,1,2])
        original=align_collinear_centers(x,y)['certificate']
        bad=[]
        c=copy.deepcopy(original); c['branches']=c['branches'][:1]; bad.append(c)
        c=copy.deepcopy(original); i,v=c['branches'][0]['dual'][0]; c['branches'][0]['dual']=((i,F(-1)),)+c['branches'][0]['dual'][1:]; bad.append(c)
        c=copy.deepcopy(original); c['chosen']=False; bad.append(c)
        c=copy.deepcopy(original); c['epsilon1']=True; bad.append(c)
        c=copy.deepcopy(original); c['translation']=(False,F(0)); bad.append(c)
        c=copy.deepcopy(original); c['error_squared']+=1; bad.append(c)
        for c in bad:
            self.assertEqual(assess_collinear_alignment_certificate(x,y,c)['status'],'INVALID')
        self.assertEqual(assess_collinear_alignment_certificate(line([0,1,4]),y,original)['status'],'INVALID')
        self.assertEqual(assess_collinear_alignment_certificate(x,y,original,max_work=0)['status'],'UNKNOWN')
