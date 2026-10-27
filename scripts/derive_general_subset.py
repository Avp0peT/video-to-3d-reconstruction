"""Supervised-stage entry point for explicit native subset derivation."""
import argparse
import json
import sys
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from integrations.mvs_surface.general_subset import derive
from integrations.mvs_surface.photo_inputs import _json


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('sfm-run','io-run','model-id','input-indices-json','exclusion-reasons-json','output'):
        p.add_argument('--'+name,required=True)
    a=p.parse_args(argv);output=Path(a.output).resolve()
    if output.exists() or output.is_relative_to(Path(a.sfm_run).resolve()) or output.is_relative_to(Path(a.io_run).resolve()):
        print('new output outside source runs required',file=sys.stderr);return 2
    try:
        indices=_json(Path(a.input_indices_json).read_text(encoding='utf-8-sig'))
        reasons=_json(Path(a.exclusion_reasons_json).read_text(encoding='utf-8-sig'))
        if not isinstance(indices,list) or not isinstance(reasons,dict):raise ValueError('selection JSON root types')
        result=derive(a.sfm_run,a.io_run,a.model_id,indices,output,reasons)
        result.update(status='PASS',scope='Explicit view restriction, incomplete target coverage allowed')
        code=0
    except Exception as exc:
        result={'status':'FAIL','error':f'{type(exc).__name__}: {exc}'};code=2
        output.mkdir(parents=True,exist_ok=True)
    (output/'RESULT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False));return code


if __name__=='__main__':raise SystemExit(main())
