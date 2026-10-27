"""One explicit clean44 ->43 native restriction; no SfM solving or retries."""
from pathlib import Path
import sys, os, argparse, json, time
sys.dont_write_bytecode = True
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '2'
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from integrations.mvs_surface.subset import _native, views, evaluate, _binding, need, MODEL_BIN


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--clean-parent', type=Path, required=True)
    p.add_argument('--sfm-run', type=Path, required=True)
    p.add_argument('--io-run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(argv); output = args.output.resolve()
    if output.exists() or (ROOT/'outputs').resolve() not in output.parents:
        print('new output under project outputs required', file=sys.stderr); return 2
    start = time.monotonic(); result = {'status':'STARTED','scope':'explicit43 view derivative, not SfM reconstruction','no_retry':True}
    output.mkdir(parents=True, exist_ok=False)
    try:
        parent = args.clean_parent.resolve(strict=True)
        before = [_binding(parent/f) for f in MODEL_BIN]
        child = _native(parent); selected = views(child)
        need(len(selected)==44 and 60 in selected and selected[60]['frame_id']==60 and
             len(selected[60]['frame_data_ids'])==1, 'single-image native60/frame60 required')
        child.deregister_frame(60)
        sparse = output/'sparse'; sparse.mkdir()
        child.write_binary(str(sparse))
        certificate = evaluate(parent, sparse, args.sfm_run, args.io_run)
        need(all(_binding(b['path'])==b for b in before), 'clean parent drift')
        need(time.monotonic()-start < 30, '30s cooperative budget exceeded; external supervisor required for hard stop')
        (output/'subset_certificate.json').write_text(json.dumps(certificate,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        result.update(status='SUBSET43_BOUND_NOT_RECONSTRUCTED', exit_code=0,
                      certificate=_binding(output/'subset_certificate.json'))
    except Exception as exc:
        result.update(status='REJECTED',exit_code=2,error=f'{type(exc).__name__}: {exc}')
    result['wall_seconds']=time.monotonic()-start
    (output/'RESULT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False));return result['exit_code']

if __name__=='__main__':raise SystemExit(main())
