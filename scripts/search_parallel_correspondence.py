"""Finite exact theoretical parallel-view correspondence CLI; no photographs."""
import argparse
from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.dont_write_bytecode=True
from kozlov_lab.kozlov2025.exact_center import encode_exact,decode_exact


def unique(pairs):
    out={}
    for key,value in pairs:
        if key in out: raise ValueError('duplicate JSON key: '+key)
        out[key]=value
    return out


def read(path,*,maximum=4<<20):
    if path.stat().st_size>maximum: raise ValueError('bounded JSON file required')
    content=path.read_bytes()
    if len(content)>maximum: raise ValueError('bounded JSON file required')
    value=json.loads(content.decode('utf-8-sig'),object_pairs_hook=unique,
                     parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))
    if type(value) is not dict: raise ValueError('JSON object required')
    return value,hashlib.sha256(content).hexdigest()


def rational(value):
    if type(value) is int: q=F(value)
    elif type(value) is dict and set(value)=={'numerator','denominator'}:
        for x in value.values():
            if type(x) is not str or not 1<=len(x)<=14 or x!=str(int(x)):
                raise ValueError('canonical bounded rational decimal required')
        n,d=int(value['numerator']),int(value['denominator'])
        if d<=0: raise ValueError('positive denominator required')
        q=F(n,d)
        if (q.numerator,q.denominator)!=(n,d): raise ValueError('reduced rational required')
    else: raise ValueError('strict integer/rational coordinate required; no bool/float')
    if max(abs(q.numerator),q.denominator)>10**12: raise ValueError('parallel coordinate numerator/denominator bound')
    return q


def points(rows):
    if type(rows) is not list or not 4<=len(rows)<=7: raise ValueError('complete 4..7 point array required; no sampling')
    out={}
    for row in rows:
        if type(row) is not dict or set(row)!={'label','xy'}: raise ValueError('strict labelled xy row required')
        label=row['label'];xy=row['xy']
        if type(label) is not int or label.bit_length()>64 or label in out: raise ValueError('unique 64-bit integer labels required')
        if type(xy) is not list or len(xy)!=2: raise ValueError('two coordinates required')
        out[label]=tuple(rational(v) for v in xy)
    if len(set(out.values()))!=len(out): raise ValueError('duplicate coordinates rejected; no merging')
    return out


def input_values(data):
    if set(data)!={'schema','model','source_xy','target_xy'} or data['schema']!='bounded-parallel-correspondence-input-v1' or data['model']!='exact-parallel-theory':
        raise ValueError('exact theoretical parallel input schema/model required; photographs unsupported')
    source,target=points(data['source_xy']),points(data['target_xy'])
    if len(source)!=len(target): raise ValueError('equal complete view cardinality required')
    return source,target


def encode_result_transport(result):
    """One bounded tagged tree per ledger candidate; accepted uses ledger indices."""
    base=dict(result);ledger=base.pop('ledger');accepted=base.pop('accepted')
    if type(ledger) is not tuple or type(accepted) is not tuple or len(ledger)>5040: raise ValueError('bounded candidate tuples required')
    indices=[i for i,row in enumerate(ledger) if row['status']=='ACCEPTED']
    if tuple(ledger[i] for i in indices)!=accepted: raise ValueError('accepted ledger binding mismatch')
    return dict(schema='bounded-parallel-correspondence-transport-v1',encoding='exact-center-tags-v1',
                document=encode_exact(base),accepted_indices=indices,
                ledger_chunks=[encode_exact(dict(offset=i,records=ledger[i:i+1])) for i in range(len(ledger))])


def decode_result_transport(data):
    if type(data) is not dict or set(data)!={'schema','encoding','document','accepted_indices','ledger_chunks'} or data['schema']!='bounded-parallel-correspondence-transport-v1' or data['encoding']!='exact-center-tags-v1':
        raise ValueError('explicit bounded candidate transport required')
    chunks=data['ledger_chunks'];indices=data['accepted_indices']
    if type(chunks) is not list or len(chunks)>5040 or type(indices) is not list or len(indices)>5040:
        raise ValueError('candidate transport bound')
    result=decode_exact(data['document']);ledger=[]
    if type(result) is not dict or 'ledger' in result or 'accepted' in result: raise ValueError('invalid base document')
    for encoded in chunks:
        chunk=decode_exact(encoded)
        if type(chunk) is not dict or set(chunk)!={'offset','records'} or type(chunk['offset']) is not int or chunk['offset']!=len(ledger) or type(chunk['records']) is not tuple or len(chunk['records'])!=1:
            raise ValueError('ordered single-candidate chunk required')
        ledger.extend(chunk['records'])
    expected=[i for i,row in enumerate(ledger) if type(row) is dict and row.get('status')=='ACCEPTED']
    if any(type(i) is not int for i in indices) or indices!=expected: raise ValueError('accepted candidate identities mismatch')
    result.update(ledger=tuple(ledger),accepted=tuple(ledger[i] for i in indices))
    return result


def _success(answer,check):
    return (answer['status']=='SEARCH_COMPLETE' and answer['complete'] is True
            and answer['solutions_count']>0 and answer['unsupported_count']==0 and check['status']=='VERIFIED')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True,help='new output only; refuses overwrite')
    parser.add_argument('--verify',type=Path,help='independently verify a prior result against both complete views')
    parser.add_argument('--max-candidates',type=int,default=5040)
    parser.add_argument('--max-work',type=int,default=20000000)
    parser.add_argument('--seconds',type=float,default=30)
    args=parser.parse_args(argv)
    try:
        if args.output.exists(): raise ValueError('output exists; use a new file')
        if not (0<=args.max_candidates<=5040 and 0<=args.max_work<=20000000 and math.isfinite(args.seconds) and 0<=args.seconds<=30):
            raise ValueError('invalid shared bounded candidate/work/time budget')
        data,digest=read(args.input);source,target=input_values(data)
        from kozlov_lab.stereo.official.bounded_parallel_correspondence import (
            search_bounded_parallel_correspondence,assess_bounded_parallel_correspondence_result)
        started=time.monotonic()
        if args.verify:
            saved,saved_digest=read(args.verify,maximum=128<<20)
            if saved.get('schema')!='bounded-parallel-correspondence-cli-result-v1' or saved.get('input_sha256')!=digest:
                raise ValueError('saved complete input byte binding mismatch')
            answer=decode_result_transport(saved['result'])
            check=assess_bounded_parallel_correspondence_result(source,target,answer,max_work=args.max_work,seconds=args.seconds)
            total=check['work_used'];transport=saved['result'];extra=dict(verification_source_sha256=saved_digest)
        else:
            answer=search_bounded_parallel_correspondence(source,target,max_candidates=args.max_candidates,max_work=args.max_work,seconds=args.seconds)
            transport=encode_result_transport(answer)
            restored=decode_result_transport(json.loads(json.dumps(transport,allow_nan=False),object_pairs_hook=unique))
            remaining_work=max(0,args.max_work-answer['work_used'])
            remaining_seconds=max(0,min(30,args.seconds-(time.monotonic()-started)))
            if remaining_work==0 or remaining_seconds==0:
                check=dict(status='UNKNOWN',reason='no remaining shared independent verification budget',work_used=0)
            else:
                check=assess_bounded_parallel_correspondence_result(source,target,restored,max_work=remaining_work,seconds=remaining_seconds)
            total=answer['work_used']+check['work_used'];extra={}
        output=dict(schema='bounded-parallel-correspondence-cli-result-v1',model='exact-parallel-theory',
            search_status=answer['status'],search_complete=answer['complete'],solutions_count=answer['solutions_count'],
            unsupported_count=answer['unsupported_count'],independent_check=check,total_work_used=total,result=transport,
            input_sha256=digest,input_path=str(args.input.resolve()),argv=list(argv if argv is not None else sys.argv[1:]),
            scope='complete bounded theoretical parallel-view correspondence ledger; not photographs or metric reconstruction',
            physical_scale='unknown',**extra)
        if hashlib.sha256(args.input.read_bytes()).hexdigest()!=digest: raise ValueError('input changed during computation')
        with args.output.open('x',encoding='utf-8',newline='\n') as stream:
            json.dump(output,stream,ensure_ascii=False,allow_nan=False,indent=2);stream.write('\n')
        print(json.dumps(dict(search_status=output['search_status'],search_complete=output['search_complete'],solutions_count=output['solutions_count'],independent_check=check['status'],output=str(args.output.resolve()))))
        return 0 if _success(answer,check) else 1
    except (ValueError,TypeError,KeyError,IndexError,OSError,ArithmeticError) as exc:
        print('INPUT_OR_EVIDENCE_ERROR: '+str(exc),file=sys.stderr)
        return 2


if __name__=='__main__': raise SystemExit(main())
