"""Read-only migration planner; a plan never authorizes or performs relocation."""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from . import runtime, preflight
from .deployment import read

REQUIRED = {'project_python', 'sam3_python', 'sam3_base_python', 'sam3_source',
            'checkpoint', 'checkpoint_sha256', 'git', 'sam3_package', 'input_kind', 'target'}
OPTIONAL = {'ffmpeg', 'ffprobe', 'mvs_tool_root', 'blender', 'freecad_python'}


def absolute(value, name):
    if not isinstance(value, str) or not Path(value).is_absolute(): raise ValueError(name + ' must be absolute')
    p = Path(value); runtime._no_links(p)
    return p.resolve()


def relative_file(base, value):
    if not isinstance(value, str) or ':' in value:
        raise ValueError('noncanonical manifest path')
    # Existing authority uses Windows separators. Normalize exactly as the
    # sealed runtime verifier does before duplicate/traversal checks.
    value = value.replace('\\', '/')
    p = Path(value)
    if p.is_absolute() or '..' in p.parts or '.' in value.split('/') or not value or value.endswith('/'):
        raise ValueError('escaping manifest path')
    target = base / p; runtime._no_links(target)
    if not target.resolve().is_relative_to(base): raise ValueError('manifest path escapes base')
    return target


def digest(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('lowercase SHA-256 required')
    return value


def plan_relocation(source_root, source_package, expected_source_bundle_sha256, target_root, target_paths, *, max_metadata_bytes=8388608):
    source = absolute(source_root, 'source_root'); package = absolute(source_package, 'source_package')
    target = absolute(target_root, 'target_root'); expected = digest(expected_source_bundle_sha256)
    if not source.is_dir() or not package.is_dir(): raise ValueError('source root/package must exist')
    if package != (source / runtime.PACKAGE_RELATIVE).resolve(): raise ValueError('source package is not active profile')
    if target.is_relative_to(source) or source.is_relative_to(target): raise ValueError('source/target roots overlap')
    if type(max_metadata_bytes) is not int or not 1 <= max_metadata_bytes <= 8388608: raise ValueError('invalid metadata limit')
    if not isinstance(target_paths, dict) or not REQUIRED <= target_paths.keys() or target_paths.keys() - REQUIRED - OPTIONAL:
        raise ValueError('missing or unknown target path keys')
    kind, goal = target_paths['input_kind'], target_paths['target']
    if kind not in ('photos', 'video') or goal not in ('report', 'blender', 'cad'): raise ValueError('invalid input kind/target')
    needed = ({'ffmpeg', 'ffprobe'} if kind == 'video' else set())
    if goal in ('blender', 'cad'): needed |= {'mvs_tool_root', 'blender'}
    if goal == 'cad': needed.add('freecad_python')
    if not needed <= target_paths.keys(): raise ValueError('missing target tools: ' + ', '.join(sorted(needed - target_paths.keys())))
    paths = {k: absolute(v, k) for k, v in target_paths.items() if k not in {'checkpoint_sha256', 'input_kind', 'target'}}
    target_sha = digest(target_paths['checkpoint_sha256'])
    if len(set(paths.values())) != len(paths): raise ValueError('non-unique target mapping')
    for key, path in paths.items():
        if path.is_relative_to(source) or source.is_relative_to(path): raise ValueError('target path collides with source: ' + key)
    contracts = {'sam3_python': target / preflight.HEAVY, 'sam3_source': target / 'integrations/sam3/src/sam3-repo',
                 'sam3_package': target / runtime.PACKAGE_RELATIVE,
                 'sam3_base_python': target / 'integrations/sam3/python/cpython-3.12-windows-x86_64-none/python.exe'}
    for key, path in contracts.items():
        if paths[key] != path.resolve(): raise ValueError('target canonical layout required: ' + key)
    directory_roles = [paths['sam3_package'], paths['sam3_source'], paths['sam3_python'].parent.parent,
                       paths['project_python'].parent.parent]
    if 'mvs_tool_root' in paths: directory_roles.append(paths['mvs_tool_root'])
    for index, left in enumerate(directory_roles):
        for right in directory_roles[index + 1:]:
            if left.is_relative_to(right) or right.is_relative_to(left): raise ValueError('target role directories overlap')
    for key in ('checkpoint', 'project_python', 'git', 'blender', 'ffmpeg', 'ffprobe', 'freecad_python'):
        if key in paths and any(paths[key].is_relative_to(paths[d]) for d in ('sam3_package', 'sam3_source')):
            raise ValueError('target dependency placement collision: ' + key)
    docs = {}
    for name in ('BUNDLE.json', 'PROFILE.json', 'START_BASELINE.json', 'AUTHORITY_BINDINGS.json'):
        p = package / name; runtime._no_links(p)
        if p.stat().st_size > max_metadata_bytes: raise ValueError('metadata limit exceeded')
        docs[name] = read(p)
    actual = runtime.sha256(package / 'BUNDLE.json')
    if actual != expected or actual != runtime.BUNDLE_SHA256: raise ValueError('source bundle digest differs from pinned authority')
    # Validate every path before allowing runtime's protected-subset verifier.
    for name, data in docs.items():
        if name == 'PROFILE.json': continue
        seen = set(); base = package if name == 'BUNDLE.json' else source
        if not isinstance(data.get('files'), list): raise ValueError('invalid files manifest')
        for row in data['files']:
            p = relative_file(base, row['path']); digest(row['sha256'])
            if p in seen: raise ValueError('duplicate manifest mapping')
            seen.add(p)
    verified = runtime.verify_bindings(SimpleNamespace(root=source, worker_package=package))
    if not verified['match']: raise ValueError('source active dependency drift: ' + '; '.join(verified['errors']))
    baseline, authority, profile = docs['START_BASELINE.json'], docs['AUTHORITY_BINDINGS.json'], docs['PROFILE.json']
    if absolute(authority['project_root'], 'authority root') != source or absolute(baseline['project_root'], 'baseline root') != source:
        raise ValueError('source authority/baseline root differs')
    if baseline['checkpoint']['sha256'] != target_sha: raise ValueError('target checkpoint digest changes immutable model identity')
    if baseline['checkout_head'] != authority['checkout_head']: raise ValueError('checkout metadata differs')
    if profile.get('schema') != 'sam3-demo-text-runtime-profile-v2' or profile.get('mode') != 'text_only': raise ValueError('unexpected source profile')
    bundle_rows = {row['path']: row for row in docs['BUNDLE.json']['files']}
    for row in profile.get('workers', []):
        if row != bundle_rows.get(row['path']): raise ValueError('profile worker binding differs from bundle')
    accepted_binding = profile.get('accepted_product_bindings_sha256')
    if accepted_binding is not None and accepted_binding != bundle_rows.get('ACCEPTED_PRODUCT_BINDINGS.json', {}).get('sha256'):
        raise ValueError('profile accepted source binding differs')
    manifest_rows = {name: {row['path']: row for row in docs[name]['files']} for name in ('START_BASELINE.json', 'AUTHORITY_BINDINGS.json')}
    for change in profile.get('approved_rebindings', []):
        if change.get('manifest') not in manifest_rows or change.get('accepted') != manifest_rows[change['manifest']].get(change.get('path')):
            raise ValueError('profile approved rebinding differs from active manifest')
    worker = package / 'worker_entry/mmap_review_worker.py'
    tree = ast.parse(worker.read_text(encoding='utf-8'))
    constants = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in {'ROOT', 'BOUND_CHECKPOINT', 'BOUND_SHA256'}:
                if name in constants: raise ValueError('ambiguous worker assignment')
                value = node.value.args[0] if name == 'ROOT' and isinstance(node.value, ast.Call) else node.value
                constants[name] = ast.literal_eval(value)
    if constants != {'ROOT': str(source), 'BOUND_CHECKPOINT': baseline['checkpoint']['path'], 'BOUND_SHA256': target_sha}:
        raise ValueError('worker source constants/metadata binding differs')
    copy_map, by_target = [], {}
    def add(old, new, purpose, sha=None, bytes_=None):
        new = new.resolve(); old = old.resolve()
        if new in by_target and by_target[new] != old: raise ValueError('non-unique destination mapping')
        if new in by_target: return
        by_target[new] = old
        copy_map.append({'source': str(old), 'target': str(new), 'purpose': purpose,
                         'sha256': sha, 'bytes': bytes_, 'content_verification': 'HASH_VERIFIED' if sha else 'NOT_VERIFIED', 'action': 'PLAN_ONLY'})
    add(package / 'BUNDLE.json', paths['sam3_package'] / 'parent_provenance/BUNDLE.json', 'immutable_parent_seal', actual)
    for row in docs['BUNDLE.json']['files']:
        name = row['path']
        add(package / name, paths['sam3_package'] / name, 'candidate_review_required' if name in {'PROFILE.json', 'AUTHORITY_BINDINGS.json', 'START_BASELINE.json', 'worker_entry/mmap_review_worker.py'} else 'preserve_bytes', row['sha256'], row['bytes'])
    for row in verified['checked']:
        if row['manifest'] == 'bundle': continue
        add(source / row['path'], target / row['path'], 'preserve_runtime_bytes', row['sha256'])
    # Product code is local current content, independently hashed, not promoted
    # to the model worker's historical accepted source authority.
    for directory in ('scripts', 'src', 'integrations/demo_pipeline'):
        for p in sorted((source / directory).rglob('*.py')):
            runtime._no_links(p)
            add(p, target / p.relative_to(source), 'current_product_source_requires_review', runtime.sha256(p), p.stat().st_size)
    blockers = []
    envs = [('sam3', source / preflight.HEAVY, paths['sam3_python']),
            ('project', source / preflight.LIGHT, paths['project_python'])]
    for label, old, new in envs:
        runtime._no_links(old)
        if not old.is_file(): raise ValueError('source environment interpreter missing: ' + label)
        add(old, new, 'interpreter_file_only_requires_full_environment', runtime.sha256(old), old.stat().st_size)
        cfg = old.parent.parent / 'pyvenv.cfg'
        runtime._no_links(cfg)
        if not cfg.is_file(): raise ValueError('source pyvenv config missing: ' + label)
        if cfg.stat().st_size > 65536: raise ValueError('pyvenv config too large')
        text = cfg.read_text(encoding='utf-8'); add(cfg, new.parent.parent / 'pyvenv.cfg', 'requires_target_environment_review', runtime.sha256(cfg))
        blockers.append({'category': 'VENV_NOT_PORTABLE', 'environment': label, 'source_pyvenv': text,
                         'target_python': str(new), 'next_step': 'review base home, launchers, DLL and effective sys.prefix on target; never bulk rewrite metadata'})
        site = old.parent.parent / 'Lib/site-packages'
        for pattern in ('*.pth', '*.dist-info/METADATA', '*.dist-info/direct_url.json', '*.dist-info/RECORD', '__editable*.py', '*.egg-link'):
            for p in sorted(site.glob(pattern)):
                runtime._no_links(p)
                if p.stat().st_size > max_metadata_bytes: raise ValueError('environment metadata too large')
                add(p, new.parent.parent / 'Lib/site-packages' / p.relative_to(site), 'preserve_environment_metadata_review_links', runtime.sha256(p), p.stat().st_size)
    inventory = preflight.python_identity(source / preflight.HEAVY)['packages']
    if inventory != baseline['package_inventory']: raise ValueError('actual source environment inventory differs')
    checkpoint = absolute(baseline['checkpoint']['path'], 'source checkpoint')
    stat = checkpoint.stat()
    if not checkpoint.is_file() or (stat.st_size, stat.st_mtime_ns) != (authority['checkpoint_stat_only']['bytes'], authority['checkpoint_stat_only']['mtime_ns']):
        raise ValueError('source checkpoint stat drift')
    add(checkpoint, paths['checkpoint'], 'weights_stat_only_declared_digest_not_rehashed', None, stat.st_size)
    return {'schema': 'photo3d-relocation-plan-v1', 'status': 'PLANNED_REVIEW_REQUIRED',
        'source_root': str(source), 'target_root': str(target), 'target_exists': target.exists(),
        'target_verification': 'NOT_PERFORMED', 'source_bundle_sha256': actual,
        'copy_map': copy_map, 'environment_blockers': blockers,
        'checkpoint_identity': {'declared_sha256': target_sha, 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns, 'bytes_rehashed': False},
        'directory_requirements': ['complete approved CPython distribution', 'complete venv packages and native DLLs', 'complete source checkout including Git metadata', 'complete tool distributions, not single exe files'],
        'source_bindings': verified, 'source_profile_provenance': profile,
        'literal_changes_requiring_review': [{'symbol': k, 'previous': constants[k], 'proposed': str(target if k == 'ROOT' else paths['checkpoint'])} for k in ('ROOT', 'BOUND_CHECKPOINT')],
        'review_gates': ['immutable parent seal plus exact AST/JSON transformation policy', 'preserve model/source/policy hashes and checkout revision/inventory', 'new independent prepare-only authority/profile/bundle with no historical PASS', 'target CPython/venv functional identity and module origins', 'target Git clean HEAD and native/DLL/CUDA gates', 'bounded real pipeline and independent artifact readback'],
        'runtime_compatibility': 'UNVERIFIED', 'historical_pass_reused': False, 'current_runtime_evidence': [],
        'run_argv': None, 'execution_authorized': False, 'copied_bytes': 0}


def write_plan(plan, output):
    p = absolute(str(output), 'output'); source = Path(plan['source_root']); target = Path(plan['target_root'])
    if not p.is_relative_to(source / 'outputs') or p.parent == source / 'outputs': raise ValueError('output must be in new source outputs subdirectory')
    if p.parent.exists(): raise ValueError('output parent must be a new directory')
    if p.is_relative_to(target): raise ValueError('output collides with target')
    protected = [Path(row['source']) for row in plan['copy_map']]
    if any(p == old or p.parent.is_relative_to(old) or old.is_relative_to(p.parent) for old in protected): raise ValueError('output overlaps source dependency')
    p.parent.mkdir(parents=True, exist_ok=False)
    with p.open('x', encoding='utf-8') as stream: json.dump(plan, stream, ensure_ascii=False, indent=2, allow_nan=False)
    return {'path': str(p), 'sha256': runtime.sha256(p), 'status': plan['status'], 'execution_authorized': False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('source-root', 'source-package', 'expected-source-bundle-sha256', 'target-root', 'target-paths', 'output'):
        parser.add_argument('--' + flag, required=True)
    args = vars(parser.parse_args(argv)); output = args.pop('output')
    try:
        args['target_paths'] = read(args['target_paths'])
        result = write_plan(plan_relocation(**args), output)
        print(json.dumps(result, ensure_ascii=False)); return 0
    except (ValueError, TypeError, OSError, KeyError, SyntaxError) as exc:
        print(json.dumps({'status': 'PLANNING_REJECTED', 'error': str(exc)}, ensure_ascii=False)); return 2
