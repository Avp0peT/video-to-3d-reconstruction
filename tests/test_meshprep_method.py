"""Pure recipe wiring: fake filters only, never import VTK or call Update."""
import pytest
from scripts import prepare_mesh as prep

class Fake:
    def __init__(self,names):self.names=names;self.calls=[]
    def __getattr__(self,name):
        if name not in self.names:raise AssertionError('unexpected filter method: '+name)
        return lambda *args:self.calls.append((name,args))

def test_default_method_and_explicit_quadric_are_distinct():
    base=['--mesh','x.ply','--source-run','s.json','--output','o']
    assert prep.parser().parse_args(base).method=='decimate-pro'
    assert prep.parser().parse_args(base+['--method','quadric']).method=='quadric'
    with pytest.raises(SystemExit):prep.parser().parse_args(base+['--method','automatic'])

def test_default_recipe_retains_exact_original_call_order_and_values():
    names=['SetInputData','PreserveTopologyOn','SplittingOff','PreSplitMeshOff','BoundaryVertexDeletionOff',
           'AccumulateErrorOn','SetOutputPointsPrecision','SetTargetReduction','SetErrorIsAbsolute','SetAbsoluteError']
    filter=Fake(names);original=object()
    def forbidden():raise AssertionError('other filter constructed')
    assert prep.configure_decimation('decimate-pro',original,.9,.01,{'decimate-pro':lambda:filter,'quadric':forbidden}) is filter
    assert filter.calls==[(names[0],(original,)),*[(n,()) for n in names[1:6]],
                          (names[6],(1,)),(names[7],(.9,)),(names[8],(1,)),(names[9],(.01,))]

def test_quadric_uses_only_official_quadric_setters_and_no_error_precision_or_update():
    filter=Fake(['SetInputData','VolumePreservationOn','AttributeErrorMetricOff','SetTargetReduction']);original=object()
    def forbidden():raise AssertionError('default method fallback')
    assert prep.configure_decimation('quadric',original,.98,.01,{'quadric':lambda:filter,'decimate-pro':forbidden}) is filter
    assert filter.calls==[('SetInputData',(original,)),('VolumePreservationOn',()),('AttributeErrorMetricOff',()),('SetTargetReduction',(.98,))]

def test_invalid_method_fails_before_constructing_any_filter():
    with pytest.raises(ValueError):prep.configure_decimation('automatic',object(),.9,.01,{})
