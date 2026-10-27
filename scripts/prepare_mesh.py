"""Prepare a closed triangle PLY for the separately gated CAD converter.

Uses FreeCAD 1.1.4's bundled Python/VTK. Does not fill holes, infer the object,
claim photo fidelity, or bypass FreeCAD/OCCT solid validation.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

cad = load_module("meshprep_cad_checks", ROOT / "scripts/mesh_to_cad.py")
geo = load_module("meshprep_geometry", ROOT / "integrations/meshprep/geometry.py")
ply = load_module("meshprep_ply", ROOT / "integrations/meshprep/ply_io.py")
source_checks = load_module("meshprep_source_link", ROOT / "integrations/meshprep/provenance.py")

def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")

def finite_fraction(value):
    result = float(value)
    if not math.isfinite(result) or not 0 < result < 1:
        raise argparse.ArgumentTypeError("must be finite and strictly between 0 and 1")
    return result

def bounded_faces(value):
    result = int(value)
    if not 4 <= result <= 10000:
        raise argparse.ArgumentTypeError("CAD target must be between 4 and 10000 faces")
    return result

def bounded_input_faces(value):
    result = int(value)
    if not 1 <= result <= 400000:
        raise argparse.ArgumentTypeError("input face limit must be between 1 and 400000")
    return result

def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mesh", type=Path, required=True)
    p.add_argument("--source-run", type=Path, required=True, help="Existing provenance record FILE; a directory is insufficient")
    p.add_argument("--source-role",choices=("fixture_mesh","pre_cleanup_surface","mesh"),help="Required raw/final producer role for a surface run")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--component", type=int, help="Explicit edge-connected component ID; no largest-component heuristic")
    p.add_argument("--target-faces", type=bounded_faces, default=10000)
    p.add_argument("--method", choices=("decimate-pro", "quadric"), default="decimate-pro")
    p.add_argument("--max-input-faces", type=bounded_input_faces, default=200000)
    p.add_argument("--max-relative-deviation", type=finite_fraction, default=0.01)
    p.add_argument("--max-relative-volume-change", type=finite_fraction, default=0.02)
    p.add_argument("--synthetic", action="store_true")
    return p

def stats(points, faces):
    adapter = SimpleNamespace(Topology=([SimpleNamespace(x=p[0],y=p[1],z=p[2]) for p in points],faces))
    result = cad.topology(adapter)
    result["components"] = len(geo.edge_components(faces))
    result["euler"] = result["vertices"] - result["edges"] + result["faces"]
    result["self_intersection_status"] = "NOT_EVALUATED"
    return result

def topology_gate(metrics):
    cad.require_manual_mesh(metrics)
    if metrics["components"] != 1:
        raise cad.Rejected("MULTIPLE_COMPONENTS", "Selected/output mesh must be edge connected.")

def vtk_mesh(points, faces):
    from vtkmodules.vtkCommonCore import vtkPoints
    from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData
    vertices = vtkPoints(); vertices.SetDataTypeToDouble()
    for point in points: vertices.InsertNextPoint(*point)
    cells = vtkCellArray()
    for face in faces:
        cells.InsertNextCell(3)
        for index in face: cells.InsertCellPoint(index)
    mesh = vtkPolyData();mesh.SetPoints(vertices);mesh.SetPolys(cells)
    return mesh

def unpack_vtk(mesh):
    if mesh.GetNumberOfVerts() or mesh.GetNumberOfLines() or mesh.GetNumberOfStrips():
        raise cad.Rejected("NON_TRIANGLE_OUTPUT", "VTK output contains non-triangle primitives.", 4)
    points = [tuple(mesh.GetPoint(i)) for i in range(mesh.GetNumberOfPoints())]
    faces = []
    for i in range(mesh.GetNumberOfCells()):
        cell = mesh.GetCell(i)
        if cell.GetNumberOfPoints() != 3:
            raise cad.Rejected("NON_TRIANGLE_OUTPUT", "VTK output is not triangulated.", 4)
        faces.append(tuple(int(cell.GetPointId(j)) for j in range(3)))
    return points, faces

def configure_decimation(method, original, target_reduction, limit, factories):
    """Explicit one-method recipe; no fallback or claim of QEM topology preservation."""
    if method == 'decimate-pro':
        simplifier=factories[method]();simplifier.SetInputData(original)
        simplifier.PreserveTopologyOn();simplifier.SplittingOff();simplifier.PreSplitMeshOff()
        simplifier.BoundaryVertexDeletionOff();simplifier.AccumulateErrorOn()
        simplifier.SetOutputPointsPrecision(1) # vtkAlgorithm.DOUBLE_PRECISION
        simplifier.SetTargetReduction(target_reduction)
        simplifier.SetErrorIsAbsolute(1);simplifier.SetAbsoluteError(limit)
    elif method == 'quadric':
        simplifier=factories[method]();simplifier.SetInputData(original)
        simplifier.VolumePreservationOn();simplifier.AttributeErrorMetricOff()
        simplifier.SetTargetReduction(target_reduction)
    else:
        raise ValueError('unsupported decimation method')
    return simplifier

def sampled_distance(points, faces, target, check_deadline):
    from vtkmodules.vtkFiltersCore import vtkImplicitPolyDataDistance
    evaluator = vtkImplicitPolyDataDistance()
    errors = []
    evaluator.AddObserver("ErrorEvent", lambda obj,event:errors.append("VTK_DISTANCE_ERROR"))
    evaluator.SetInput(target)
    maximum = 0.; count = 0
    def consume(point):
        nonlocal maximum, count
        if count % 256 == 0: check_deadline()
        value = abs(float(evaluator.EvaluateFunction(point)))
        if errors or not math.isfinite(value):
            raise cad.Rejected("DISTANCE_ERROR", "VTK distance failed or produced nonfinite output.", 4)
        maximum = max(maximum, value);count += 1
    for point in points:consume(point)
    for face in faces:
        consume(tuple(math.fsum(points[j][axis] for j in face)/3 for axis in range(3)))
    return {"maximum":maximum,"samples":count,"vertex_samples":len(points),"face_centroid_samples":len(faces)}

def prepare(args, output, report, save, check_deadline):
    source = args.mesh.resolve(strict=True); provenance = args.source_run.resolve(strict=True)
    if not source.is_file() or source.suffix.lower() != ".ply" or not provenance.is_file():
        raise cad.Rejected("INPUT_FORMAT", "Triangle PLY and source record file are required.")
    report["inputs"] = {"mesh": {"path":str(source),"sha256":cad.sha256(source)},
                        "source_record":{"path":str(provenance),"sha256":cad.sha256(provenance)}}
    report["source_validation"] = source_checks.validate_source(source,provenance,synthetic=args.synthetic,role=args.source_role)
    report["implementation"] = {str(p.relative_to(ROOT)):cad.sha256(p) for p in
        (Path(__file__), ROOT/"scripts/mesh_to_cad.py", ROOT/"integrations/meshprep/geometry.py", ROOT/"integrations/meshprep/ply_io.py",ROOT/"integrations/meshprep/provenance.py")}
    max_input_faces = getattr(args, 'max_input_faces', 200000)
    if type(max_input_faces) is not int or not 1 <= max_input_faces <= 400000:
        raise ValueError('input face limit must be an integer between 1 and 400000')
    report['max_input_faces'] = max_input_faces
    points, faces, metadata = ply.read_ply(source, max_faces=max_input_faces)
    report["source_ply"] = metadata
    components = geo.edge_components(faces)
    if not components:raise cad.Rejected("EMPTY_MESH","Input has no triangle faces.")
    report["component_count"]=len(components)
    if len(components)>2048:
        raise cad.Rejected("FRAGMENT_LIMIT","More than 2048 edge components; inspect separately within a new budget.")
    summaries = []
    for index, face_ids in enumerate(components):
        check_deadline()
        ids=sorted({vertex for face_id in face_ids for vertex in faces[face_id]})
        cp=[points[i] for i in ids]
        summaries.append({"component_id":index,"first_source_face":min(face_ids),"face_count":len(face_ids),
                          "vertex_count":len(cp),"bbox":[min(p[k] for p in cp) for k in range(3)]+[max(p[k] for p in cp) for k in range(3)]})
    report["components"] = summaries;save()
    chosen = args.component
    if chosen is None:
        if len(components)!=1:
            raise cad.Rejected("COMPONENT_SELECTION_REQUIRED","Multiple components; inspect report and explicitly select one in a NEW run.")
        chosen = 0
    if not 0 <= chosen < len(components):
        raise cad.Rejected("COMPONENT_OUT_OF_RANGE","Selected component does not exist.")
    before_points,before_faces,source_vertex_ids=geo.compact_component(points,faces,components[chosen])
    report["selection"] = {"component_id":chosen,"method":"explicit" if args.component is not None else "only_component",
        "removed_component_ids":[i for i in range(len(components)) if i!=chosen],
        "removed_face_count":len(faces)-len(before_faces),
        "removed_unused_vertices":len(points)-len({i for face in faces for i in face})}
    write_json(output/"selection_map.json",{"source_face_ids":components[chosen],"source_vertex_ids":source_vertex_ids})
    original_stats=stats(before_points,before_faces);report["before"]=original_stats;save()
    topology_gate(original_stats)
    geo.write_double_ply(output/"selected_before.ply",before_points,before_faces)
    check_deadline()
    report["runtime_environment"]=cad.configure_runtime(output)
    dll=os.add_dll_directory(str(Path(sys.executable).parent)) if os.name=="nt" else None
    from vtkmodules.vtkCommonCore import vtkVersion
    from vtkmodules.vtkFiltersCore import vtkDecimatePro, vtkQuadricDecimation
    report["vtk_version"]=vtkVersion.GetVTKVersion()
    if report["vtk_version"]!="9.3.1":raise cad.Rejected("VTK_VERSION","Validated VTK 9.3.1 required.",3)
    original=vtk_mesh(before_points,before_faces)
    diagonal=math.hypot(*(original_stats["bbox"][i+3]-original_stats["bbox"][i] for i in range(3)))
    if not math.isfinite(diagonal) or diagonal<=0:
        raise cad.Rejected("INVALID_DIAGONAL","Finite positive selected bbox diagonal required.")
    limit=diagonal*args.max_relative_deviation
    report["limits"]={"source_bbox_diagonal":diagonal,"absolute_sampled_deviation":limit,
        "max_relative_deviation":args.max_relative_deviation,
        "max_relative_volume_change":args.max_relative_volume_change,"target_faces":args.target_faces}
    if len(before_faces)>args.target_faces:
        method=getattr(args,'method','decimate-pro')
        report['method']=method
        simplifier=configure_decimation(method,original,1.-args.target_faces/len(before_faces),limit,
            {'decimate-pro':vtkDecimatePro,'quadric':vtkQuadricDecimation})
        native_errors=[]
        simplifier.AddObserver("ErrorEvent",lambda obj,event:native_errors.append("VTK_DECIMATION_ERROR"))
        check_deadline();simplifier.Update();check_deadline()
        if native_errors:raise cad.Rejected("VTK_ERROR","VTK reported a simplification error.",4)
        after_points,after_faces=unpack_vtk(simplifier.GetOutput())
        output_vertex_count=len(after_points)
        after_points,after_faces,_=geo.compact_component(after_points,after_faces,list(range(len(after_faces))))
        if method == 'decimate-pro':
            report["decimation"]={"performed":True,"preserve_topology":bool(simplifier.GetPreserveTopology()),
                "splitting":bool(simplifier.GetSplitting()),"pre_split":bool(simplifier.GetPreSplitMesh()),
                "boundary_vertex_deletion":bool(simplifier.GetBoundaryVertexDeletion()),
                "accumulate_error":bool(simplifier.GetAccumulateError()),"error_is_absolute":simplifier.GetErrorIsAbsolute(),
                "absolute_error":simplifier.GetAbsoluteError(),"feature_angle":simplifier.GetFeatureAngle()}
        else:
            report['decimation']={'performed':True,'method':'quadric','topology_preservation_guaranteed':False,
                'volume_preservation':bool(simplifier.GetVolumePreservation()),
                'attribute_error_metric':bool(simplifier.GetAttributeErrorMetric()),
                'target_reduction':simplifier.GetTargetReduction(),'actual_reduction':simplifier.GetActualReduction(),
                'regularize':bool(simplifier.GetRegularize()),'regularization':simplifier.GetRegularization(),
                'weigh_boundary_constraints_by_length':bool(simplifier.GetWeighBoundaryConstraintsByLength()),
                'boundary_weight_factor':simplifier.GetBoundaryWeightFactor(),'map_point_data':bool(simplifier.GetMapPointData()),
                'deviation_control':'POST_READBACK_BIDIRECTIONAL_SAMPLING_ONLY'}
        report['decimation'].update(discarded_unreferenced_vertices=output_vertex_count-len(after_points),
            output_coordinate_type=simplifier.GetOutput().GetPoints().GetData().GetDataTypeAsString())
    else:
        after_points,after_faces=before_points,before_faces
        report["decimation"]={"performed":False,"reason":"within_target"}
    # Persist even a rejected candidate, under an explicitly non-final filename.
    candidate=output/"candidate.ply"
    geo.write_double_ply(candidate,after_points,after_faces)
    disk_points,disk_faces,_=ply.read_ply(candidate)
    if disk_points!=after_points or disk_faces!=after_faces:
        raise cad.Rejected("PLY_ROUNDTRIP_CHANGED","Actual disk mesh differs from checked coordinates/indices.",4)
    after_stats=stats(disk_points,disk_faces);report["after"]=after_stats;save()
    topology_gate(after_stats)
    check=geo.validate_metrics(original_stats,after_stats,args.max_relative_volume_change)
    report["topology_volume_check"]=check
    if not check["passed"]:raise cad.Rejected("TOPOLOGY_OR_VOLUME_CHANGE","Topology/volume budget failed.",4)
    if len(disk_faces)>args.target_faces:
        message="Topology-preserving decimation did not reach requested face budget."
        if getattr(args,'method','decimate-pro')=='quadric':
            message="Quadric decimation did not reach requested face budget."
        raise cad.Rejected("TARGET_NOT_REACHED",message,4)
    actual=vtk_mesh(disk_points,disk_faces)
    distances={"before_to_after":sampled_distance(before_points,before_faces,actual,check_deadline),
               "after_to_before":sampled_distance(disk_points,disk_faces,original,check_deadline),
               "method":"absolute point-to-triangle-surface distances at every vertex and face centroid, both directions",
               "is_complete_hausdorff_bound":False}
    report["sampled_deviation"]=distances;save()
    if max(distances["before_to_after"]["maximum"],distances["after_to_before"]["maximum"])>limit:
        raise cad.Rejected("DEVIATION_LIMIT","Sampled surface change exceeds frozen limit.",4)
    for record in report["inputs"].values():
        if cad.sha256(Path(record["path"]))!=record["sha256"]:
            raise cad.Rejected("INPUT_CHANGED","Input changed during processing.",4)
    if source_checks.validate_source(source,provenance,synthetic=args.synthetic,role=args.source_role)!=report["source_validation"]:
        raise cad.Rejected("SOURCE_CHANGED","Producer-output association changed during processing.",4)
    candidate.rename(output/"prepared.ply")
    report["artifacts"]=[{"path":str(p),"sha256":cad.sha256(p),"bytes":p.stat().st_size}
                         for p in [output/"selected_before.ply",output/"prepared.ply",output/"selection_map.json"]]
    report["status"]="PASS_PREPARED_TOPOLOGY"
    report["next_cad_arguments"]=["--mesh",str(output/"prepared.ply"),"--source-run",str(output/"mesh_prep.json"),
        "--scale-status","unknown","--length-unit","arbitrary"]+(["--synthetic"] if args.synthetic else [])
    check_deadline()

def main(argv=None):
    args=parser().parse_args(argv);output=args.output.resolve()
    try:output.mkdir(parents=True,exist_ok=False)
    except OSError as exc:
        print(f"OUTPUT_REFUSED: {exc}",file=sys.stderr);return 2
    started=time.monotonic()
    report={"contract_version":"mesh-prep-v1","status":"STARTED","synthetic_input":args.synthetic,
        "method":args.method,
        "photo_lineage_status":"NOT_VERIFIED","self_intersection_status":"NOT_EVALUATED","cad_ready":"NOT_EVALUATED",
        "physical_accuracy_status":"NOT_EVALUATED","coordinate_factor":1.0,"repairs":[],"hole_filling":False,
        "argv":[sys.executable,*sys.argv],"cwd":str(Path.cwd()),"started_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())}
    def save():
        report["elapsed_seconds"]=time.monotonic()-started
        write_json(output/"mesh_prep.json",report)
    def deadline():
        if time.monotonic()-started>60:raise cad.Rejected("WALL_LIMIT","60 second cooperative budget exceeded; outer native supervisor is also required.",4)
    code=4
    try:
        save();prepare(args,output,report,save,deadline);code=0
    except cad.Rejected as exc:
        report.update(status="REJECTED",reason=exc.reason,error=str(exc));code=exc.exit_code
    except (ValueError,OSError) as exc:
        report.update(status="INPUT_OR_IO_ERROR",reason=type(exc).__name__,error=str(exc));code=2
    except Exception as exc:
        import traceback
        report.update(status="BACKEND_ERROR",reason=type(exc).__name__,error=str(exc))
        (output/"error.log").write_text(traceback.format_exc(),encoding="utf-8")
    finally:
        report["script_exit_code"]=code;save()
    print(json.dumps({"status":report["status"],"report":str(output/"mesh_prep.json"),"exit_code":code}))
    return code

if __name__=="__main__":raise SystemExit(main())
