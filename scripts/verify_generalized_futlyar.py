"""Exact supplied-placement research CLI; no image/model/native execution."""
from __future__ import annotations

import argparse
from dataclasses import asdict, fields
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.dont_write_bytecode = True
from kozlov_lab.kozlov2025.generalized_futlyar import (
    FutlyarCertificate, verify_generalized_futlyar_2d,
    verify_generalized_futlyar_certificate,
)


def encode(value):
    if type(value) is Fraction:
        return {'numerator': str(value.numerator), 'denominator': str(value.denominator)}
    if type(value) in (tuple, list):
        return [encode(v) for v in value]
    if type(value) is dict:
        return {k: encode(v) for k, v in value.items()}
    return value


def rational(value):
    if type(value) is int:
        return value
    if type(value) is not dict or set(value) != {'numerator', 'denominator'}:
        raise ValueError('exact integer or numerator/denominator object required')
    for key in ('numerator', 'denominator'):
        s = value[key]
        if type(s) is not str or len(s) > 80 or not s or s in ('-', '+'):
            raise ValueError('invalid bounded rational decimal')
        if s != str(int(s)):
            raise ValueError('rational decimals must be canonical')
    numerator, denominator = int(value['numerator']), int(value['denominator'])
    if denominator <= 0:
        raise ValueError('positive denominator required')
    if max(abs(numerator).bit_length(), denominator.bit_length()) > 256:
        raise ValueError('rational bit limit exceeded')
    result = Fraction(numerator, denominator)
    if str(result.numerator) != value['numerator'] or str(result.denominator) != value['denominator']:
        raise ValueError('rational must be reduced')
    return result


def unique(pairs):
    data = {}
    for key, value in pairs:
        if key in data:
            raise ValueError('duplicate JSON key: ' + key)
        data[key] = value
    return data


def read(path):
    raw = path.read_bytes()
    if len(raw) > 4 << 20:
        raise ValueError('JSON exceeds 4 MiB')
    data = json.loads(raw.decode('utf-8-sig'), object_pairs_hook=unique,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    if type(data) is not dict:
        raise ValueError('JSON object required')
    return data, hashlib.sha256(raw).hexdigest()


def points(rows):
    if type(rows) is not list or len(rows) > 4096:
        raise ValueError('bounded labelled point array required')
    result = {}
    for row in rows:
        if type(row) is not dict or set(row) != {'label', 'xy'}:
            raise ValueError('point must contain label and xy')
        label = row['label']
        if type(label) is not int or label in result:
            raise ValueError('integer unique labels required')
        if type(row['xy']) is not list or len(row['xy']) != 2:
            raise ValueError('two coordinates required')
        result[label] = tuple(rational(v) for v in row['xy'])
    return result


def input_values(data):
    if set(data) != {'schema', 'points', 'centers', 'radius'} or data['schema'] != 'generalized-futlyar-input-v1':
        raise ValueError('unexpected input schema or fields')
    return points(data['points']), points(data['centers']), rational(data['radius'])


def certificate_from_json(value):
    def restore(item):
        if type(item) is list:
            return tuple(restore(v) for v in item)
        if type(item) is dict:
            return rational(item)
        return item
    if type(value) is not dict or set(value) != {f.name for f in fields(FutlyarCertificate)}:
        raise ValueError('invalid certificate fields')
    return FutlyarCertificate(**{k: restore(v) for k, v in value.items()})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='new JSON file; refuses overwrite')
    parser.add_argument('--verify', type=Path, help='independently check a prior result against the complete input')
    parser.add_argument('--max-work', type=int, default=262144)
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise ValueError('output exists; use a new file')
        data, digest = read(args.input)
        p, c, r = input_values(data)
        if args.verify:
            saved, saved_digest = read(args.verify)
            if saved.get('schema') != 'generalized-futlyar-result-v1' or saved.get('input_sha256') != digest:
                raise ValueError('certificate source binding mismatch')
            cert = certificate_from_json(saved.get('certificate'))
            status = verify_generalized_futlyar_certificate(p, c, r, cert, max_work=args.max_work)
            result = {'status': status, 'verification_source_sha256': saved_digest,
                      'reason': 'independent complete-input certificate verification'}
        else:
            answer = verify_generalized_futlyar_2d(p, c, r, max_work=args.max_work)
            result = encode(asdict(answer))
            if answer.status == 'TRUE':
                # Check the exact serialized payload, not merely the in-memory certificate.
                status = verify_generalized_futlyar_certificate(
                    p, c, r, certificate_from_json(result['certificate']), max_work=args.max_work)
                if status != 'TRUE':
                    raise ValueError('serialized certificate independent check failed')
                result['independent_check'] = status
        result.update(schema='generalized-futlyar-result-v1', input_sha256=digest,
                      scope='supplied exact placement only; no affine search or photo reconstruction',
                      physical_scale='unknown')
        # A readback cannot silently bind a concurrently replaced input.
        if hashlib.sha256(args.input.read_bytes()).hexdigest() != digest:
            raise ValueError('input changed during computation')
        with args.output.open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
        print(json.dumps({'status': result['status'], 'output': str(args.output.resolve())}))
        return 0 if result['status'] == 'TRUE' else 1
    except (ValueError, TypeError, KeyError, OSError, ArithmeticError) as exc:
        print('INPUT_OR_EVIDENCE_ERROR: ' + str(exc), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
