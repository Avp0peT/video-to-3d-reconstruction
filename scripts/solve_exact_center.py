"""Bounded exact center JSON CLI; stdlib only, no model/native/shell work."""
import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.dont_write_bytecode=True
from kozlov_lab.kozlov2025.exact_center import solve_exact_center,assess_exact_center_result,encode_exact,decode_exact


def unique(pairs):
    result={}
    for key,value in pairs:
        if key in result: raise ValueError('duplicate JSON key: '+key)
        result[key]=value
    return result


def read(path):
    content=path.read_bytes()
    if len(content)>4<<20: raise ValueError('JSON exceeds 4 MiB')
    data=json.loads(content.decode('utf-8-sig'),object_pairs_hook=unique,
                    parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))
    if type(data) is not dict: raise ValueError('JSON object required')
    return data,hashlib.sha256(content).hexdigest()


def rational(value):
    if type(value) is int:
        if abs(value).bit_length()>2048: raise ValueError('input coordinate bit bound')
        return F(value)
    if type(value) is not dict or set(value)!={'numerator','denominator'}:
        raise ValueError('integer or canonical numerator/denominator object required; no bool/float')
    for s in value.values():
        if type(s) is not str or not 1<=len(s)<=620 or s!=str(int(s)):
            raise ValueError('canonical bounded rational decimal required')
    n,d=int(value['numerator']),int(value['denominator'])
    if d<=0 or max(abs(n).bit_length(),d.bit_length())>2048: raise ValueError('input rational bit bound')
    q=F(n,d)
    if (q.numerator,q.denominator)!=(n,d): raise ValueError('reduced rational required')
    return q


def input_points(data):
    if set(data)!={'schema','points'} or data['schema']!='exact-center-input-v1': raise ValueError('unexpected input schema/fields')
    rows=data['points']
    if type(rows) is not list or not 1<=len(rows)<=7: raise ValueError('complete 1..7 labelled point array required')
    points={}
    for row in rows:
        if type(row) is not dict or set(row)!={'label','xy'}: raise ValueError('label/xy point required')
        label=row['label']; pair=row['xy']
        if type(label) is not int or label in points: raise ValueError('unique integer labels required')
        if type(pair) is not list or len(pair)!=2: raise ValueError('two rational coordinates required')
        points[label]=tuple(rational(v) for v in pair)
    return points


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True,help='new output JSON; overwrite refused')
    parser.add_argument('--verify',type=Path,help='independently verify a prior CLI result')
    parser.add_argument('--max-work',type=int,default=2000000)
    parser.add_argument('--seconds',type=float,default=30)
    args=parser.parse_args(argv)
    try:
        if args.output.exists(): raise ValueError('output exists; use a new file')
        data,digest=read(args.input); points=input_points(data)
        started=time.monotonic()
        if args.verify:
            saved,saved_digest=read(args.verify)
            if saved.get('schema')!='exact-center-cli-result-v1' or saved.get('input_sha256')!=digest:
                raise ValueError('saved certificate source binding mismatch')
            answer=decode_exact(saved['result'])
            check=assess_exact_center_result(points,answer,max_work=args.max_work,seconds=args.seconds)
            output=dict(status=check['status'],verification=check,verification_source_sha256=saved_digest,
                        result=encode_exact(answer))
        else:
            answer=solve_exact_center(points,max_work=args.max_work,seconds=args.seconds)
            encoded=encode_exact(answer)
            if answer['exact_optimum']:
                # Verify the actual tagged JSON roundtrip and charge the remainder.
                restored=decode_exact(json.loads(json.dumps(encoded,allow_nan=False),object_pairs_hook=unique))
                check=assess_exact_center_result(points,restored,max_work=max(0,args.max_work-answer['work_used']),
                        seconds=max(0,min(30,args.seconds-(time.monotonic()-started))))
                answer['work_used']+=check['work_used']
                if check['status']!='VERIFIED':
                    answer.update(status='PARTIAL_UNKNOWN',complete=False,exact_optimum=False,
                                  reason='serialized independent verification '+check['status'])
                encoded=encode_exact(answer)
                output=dict(status=answer['status'],result=encoded,serialized_independent_check=check)
            else: output=dict(status=answer['status'],result=encoded)
        output.update(schema='exact-center-cli-result-v1',input_sha256=digest,input_path=str(args.input.resolve()),
                      argv=list(argv if argv is not None else sys.argv[1:]),
                      scope='finite exact 1..7-point research profile; no production reconstruction')
        if hashlib.sha256(args.input.read_bytes()).hexdigest()!=digest: raise ValueError('input changed during computation')
        with args.output.open('x',encoding='utf-8',newline='\n') as stream:
            json.dump(output,stream,ensure_ascii=False,indent=2,allow_nan=False); stream.write('\n')
        print(json.dumps(dict(status=output['status'],output=str(args.output.resolve()))))
        return 0 if output['status'] in ('EXACT_OPTIMUM_COMPLETE_INPUT','EXACT_DEGENERATE_COMPLETE_INPUT','VERIFIED') else 1
    except (ValueError,TypeError,KeyError,OSError,ArithmeticError) as exc:
        print('INPUT_OR_EVIDENCE_ERROR: '+str(exc),file=sys.stderr)
        return 2


if __name__=='__main__': raise SystemExit(main())
