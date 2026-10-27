"""Explicit, immutable deployment records. No execution authority is issued."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime, timezone
from . import preflight, runtime

SCHEMA = 'photo3d-deployment-v1'
REQUIRED = {'project_root', 'project_python', 'sam3_python', 'sam3_package',
            'sam3_source', 'checkpoint', 'checkpoint_sha256', 'git', 'input_kind', 'target'}
OPTIONAL = {'blender', 'freecad_python', 'ffmpeg', 'ffprobe', 'mvs_tool_root'}


def read(path):
    def unique(rows):
        result = {}
        for key, value in rows:
            if key in result: raise ValueError('duplicate JSON key: ' + key)
            result[key] = value
        return result
    path = Path(path)
    if path.stat().st_size > 8 << 20: raise ValueError('JSON exceeds 8 MiB')
    value = json.loads(path.read_text(encoding='utf-8-sig'), object_pairs_hook=unique)
    if not isinstance(value, dict): raise ValueError('JSON must be an object')
    return value


def paths(spec):
    if not REQUIRED <= spec.keys() or spec.keys() - REQUIRED - OPTIONAL:
        raise ValueError('missing or unknown deployment fields')
    digest = spec['checkpoint_sha256']
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
        raise ValueError('checkpoint_sha256 must be lowercase SHA-256')
    result = dict(spec)
    for key in (REQUIRED | OPTIONAL) - {'checkpoint_sha256', 'input_kind', 'target'}:
        if key not in spec: continue
        if not isinstance(spec[key], str) or not Path(spec[key]).is_absolute():
            raise ValueError(key + ' must be explicitly absolute')
        runtime._no_links(Path(spec[key]))
        result[key] = str(Path(spec[key]).resolve())
    if spec['input_kind'] not in ('photos', 'video') or spec['target'] not in ('report', 'blender', 'cad'):
        raise ValueError('invalid input_kind or target')
    # Never silently discover a tool or inherit machine defaults.
    needed = ({'ffmpeg', 'ffprobe'} if spec['input_kind'] == 'video' else set())
    if spec['target'] in ('blender', 'cad'): needed |= {'blender', 'mvs_tool_root'}
    if spec['target'] == 'cad': needed.add('freecad_python')
    if not needed <= spec.keys(): raise ValueError('explicit paths required: ' + ', '.join(sorted(needed - spec.keys())))
    return result


def inspect(spec):
    spec = paths(spec)
    root, package = Path(spec['project_root']), Path(spec['sam3_package'])
    blockers, bindings = [], {}
    def blocked(category, error):
        blockers.append({'category': category, 'error': str(error)})
    for key in (REQUIRED | OPTIONAL) - {'checkpoint_sha256', 'input_kind', 'target'}:
        if key not in spec or key in {'project_root', 'sam3_source', 'sam3_package', 'mvs_tool_root'}: continue
        path = Path(spec[key])
        try:
            if not path.is_file() or not path.stat().st_size: raise ValueError('missing, empty or not a file')
            before = path.stat()
            digest = runtime.sha256(path)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns): raise ValueError('changed while hashing')
            bindings[key] = {'path': str(path), 'bytes': after.st_size, 'sha256': digest}
        except (OSError, ValueError) as exc: blocked('FILE_BINDING:' + key, exc)
    if bindings.get('checkpoint', {}).get('sha256') != spec['checkpoint_sha256']:
        blocked('CHECKPOINT_DIGEST_MISMATCH', 'selected weights differ from explicit expected digest')
    extra_files = [root / 'pyproject.toml', root / 'scripts/run_photo_demo.py',
                   root / 'integrations/demo_pipeline/pipeline.py', root / 'integrations/demo_pipeline/runtime.py']
    if 'mvs_tool_root' in spec:
        extra_files += [Path(spec['mvs_tool_root']) / n for n in ('InterfaceCOLMAP.exe', 'DensifyPointCloud.exe', 'ReconstructMesh.exe')]
    for key in ('project_python', 'sam3_python'):
        env = Path(spec[key]).parent.parent
        extra_files += [env / 'pyvenv.cfg'] if (env / 'pyvenv.cfg').is_file() else []
        extra_files += sorted((env / 'Lib/site-packages').glob('*.dist-info/METADATA'))
    bindings['supporting_files'] = []
    for path in extra_files:
        try:
            runtime._no_links(path)
            bindings['supporting_files'].append({'path': str(path), 'sha256': runtime.sha256(path)})
        except (ValueError, OSError) as exc: blocked('SUPPORTING_FILE_BINDING', str(path) + ': ' + str(exc))
    # Canonical SAM checkout is hashed through the existing sealed runtime manifest.
    source = root / 'integrations/sam3/src/sam3-repo'
    if Path(spec['sam3_source']) != source.resolve() or not source.is_dir():
        blocked('SOURCE_PATH_CONTRACT', 'source must be existing project integrations/sam3/src/sam3-repo')
    try:
        config = SimpleNamespace(root=root, worker_package=package)
        bindings['sealed_runtime'] = runtime.verify_bindings(config)
        if not bindings['sealed_runtime']['match']: blocked('SEALED_SOURCE_DRIFT', bindings['sealed_runtime']['errors'])
        for name in ('BUNDLE.json', 'AUTHORITY_BINDINGS.json', 'START_BASELINE.json', 'PROFILE.json'):
            bindings[name] = {'path': str(package / name), 'sha256': runtime.sha256(package / name)}
        authority, baseline = read(package / 'AUTHORITY_BINDINGS.json'), read(package / 'START_BASELINE.json')
        bindings['provenance'] = read(package / 'PROFILE.json')
        if Path(authority['project_root']).resolve() != root:
            blocked('AUTHORITY_ROOT_BINDING', 'old authority cannot authorize selected root')
        if baseline['checkpoint']['path'] != spec['checkpoint'] or baseline['checkpoint']['sha256'] != spec['checkpoint_sha256']:
            blocked('AUTHORITY_CHECKPOINT_BINDING', 'old checkpoint path/digest cannot authorize selected weights')
        info = Path(spec['checkpoint']).stat()
        if (info.st_size, info.st_mtime_ns) != (authority['checkpoint_stat_only']['bytes'], authority['checkpoint_stat_only']['mtime_ns']):
            blocked('AUTHORITY_CHECKPOINT_STAT_BINDING', 'selected checkpoint stat differs from sealed authority')
        for key in ('project_python', 'sam3_python'):
            bindings[key + '_inventory'] = preflight.python_identity(Path(spec[key]))
        if bindings['sam3_python_inventory']['packages'] != baseline['package_inventory']:
            blocked('ENVIRONMENT_INVENTORY_BINDING', 'SAM3 package inventory differs from sealed baseline')
    except (OSError, ValueError, KeyError, TypeError) as exc:
        blocked('SEALED_RUNTIME_UNVERIFIED', exc)
    if root != runtime.PROJECT_ROOT.resolve() or package != (root / runtime.PACKAGE_RELATIVE).resolve() or Path(spec['sam3_python']) != (root / preflight.HEAVY).resolve():
        blocked('REBINDING_REQUIRES_REVIEWED_RUNTIME_PROFILE',
                'Pinned loader requires reviewed source/profile changes and a newly reviewed bundle digest; this record grants no authority.')
    # Check target entry points/tools without adopting preflight defaults.
    report = preflight.check_environment(root, **{k: v for k, v in spec.items() if k not in {'project_root', 'checkpoint_sha256', 'sam3_source'}})
    blockers.extend({'category': 'REQUIRED_ITEM_MISSING', 'error': n} for n in report['missing_required'])
    blockers.extend(report['known_contract_blockers'])
    return {'spec': spec, 'bindings': bindings, 'blockers': blockers,
            'status': 'STATIC_BLOCKED' if blockers else 'STATIC_BOUND_RUNTIME_UNVERIFIED',
            'execution_authorized': False, 'historical_pass_reused': False,
            'current_runtime_evidence': [], 'runtime_compatibility': 'UNVERIFIED'}


def create_deployment(spec, destination):
    """Create a new independent file even when inspection reports blockers."""
    destination = Path(destination)
    runtime._no_links(destination)
    # Protect package/source/tool bytes: record must live under project outputs,
    # disjoint from all supplied directory dependencies.
    normalized = paths(spec)
    output = Path(normalized['project_root']) / 'outputs'
    if not destination.resolve().is_relative_to(output.resolve()): raise ValueError('record must live within project outputs')
    for key in ('sam3_package', 'sam3_source', 'mvs_tool_root'):
        if key in normalized and destination.resolve().is_relative_to(Path(normalized[key])):
            raise ValueError('record must be disjoint from protected dependencies')
    record = {'schema': SCHEMA, 'created_utc': datetime.now(timezone.utc).isoformat(), **inspect(normalized)}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('x', encoding='utf-8') as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)
    return {'path': str(destination.resolve()), 'sha256': runtime.sha256(destination), 'status': record['status']}


def audit_deployment(path, expected_sha256):
    """Rehash the record and dependencies; explicit digest pins configuration."""
    runtime._no_links(Path(path))
    if runtime.sha256(Path(path)) != expected_sha256: raise ValueError('deployment record digest mismatch')
    record = read(path)
    if record.get('schema') != SCHEMA: raise ValueError('unsupported deployment schema')
    current = inspect(record['spec'])
    drift = current['bindings'] != record['bindings'] or current['spec'] != record['spec']
    if drift:
        current['blockers'].append({'category': 'DEPLOYMENT_BINDING_DRIFT', 'error': 'dependency/config snapshot changed; create a new record'})
        current['status'] = 'STATIC_BLOCKED'
    current['record_sha256'] = expected_sha256
    return current


def options_overrides(path, expected_sha256, *, expected_input_kind=None):
    """Reaudit before product consumption. Never returns runnable blocked paths.

    Caller supplies source/prompt/output/object_id/frames/model_id as normal and
    checks source kind against input_kind. Returned keys match pipeline.Options.
    """
    audited = audit_deployment(path, expected_sha256)
    if audited['status'] != 'STATIC_BOUND_RUNTIME_UNVERIFIED':
        raise ValueError('deployment blocked: ' + ', '.join(b['category'] for b in audited['blockers']))
    spec = audited['spec']
    if expected_input_kind is not None and expected_input_kind != spec['input_kind']:
        raise ValueError('deployment input_kind differs from selected source')
    names = {'project_python': 'python', 'git': 'git_exe', 'blender': 'blender_exe'}
    keys = {'project_python', 'sam3_python', 'sam3_package', 'checkpoint', 'git',
            'blender', 'freecad_python', 'mvs_tool_root', 'ffmpeg', 'ffprobe', 'target'}
    return {names.get(k, k): (v if k in {'ffmpeg', 'ffprobe', 'target'} else Path(v))
            for k, v in spec.items() if k in keys}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    create = sub.add_parser('create'); create.add_argument('--spec', required=True); create.add_argument('--output', required=True)
    audit = sub.add_parser('audit'); audit.add_argument('--config', required=True); audit.add_argument('--config-sha256', required=True)
    args = parser.parse_args(argv)
    try:
        result = (create_deployment(read(args.spec), args.output) if args.command == 'create'
                  else audit_deployment(args.config, args.config_sha256))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 2 if result['status'] == 'STATIC_BLOCKED' else 0
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(json.dumps({'status': 'INPUT_ERROR', 'error': str(exc)}, ensure_ascii=False))
        return 2
