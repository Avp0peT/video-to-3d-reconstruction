"""Read-only deployment inventory; never starts a dependency or imports ML code."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib import metadata
import json
from pathlib import Path
import shutil
import sys

LIGHT = 'outputs/delegation/T02A/envs/20261002T190239Z_9b9b5ddb/venv/Scripts/python.exe'
HEAVY = 'integrations/sam3/env/Scripts/python.exe'
PACKAGE = 'outputs/delegation/DEMO_RUNTIME/20261011_text_v2'
MVS = 'outputs/toolchains/openmvs-2.4.0-cpu/vc17/x64/Release'
FREECAD = 'outputs/toolchains/freecad-1.1.4/FreeCAD_1.1.4-Windows-x86_64-py311/bin/python.exe'
FLAGS = {'project_python': '--python', 'sam3_python': '--sam3-python',
         'sam3_package': '--sam3-package', 'checkpoint': '--checkpoint',
         'blender': '--blender-exe', 'freecad_python': '--freecad-python',
         'git': '--git-exe', 'ffmpeg': '--ffmpeg', 'ffprobe': '--ffprobe'}


def resolve_path(value, root):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def small_json(path):
    if path.stat().st_size > 1 << 20:
        raise ValueError('metadata exceeds 1 MiB read limit')
    data = json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(data, dict):
        raise ValueError('expected JSON object')
    return data


def inspect_file(name, path, required, origin):
    result = {'name': name, 'path': str(path) if path else None,
              'required': required, 'origin': origin, 'evidence_kind': 'EXISTENCE_ONLY',
              'runtime_compatibility': 'UNVERIFIED', 'version': None}
    try:
        if path is None or not path.exists():
            result['status'] = 'MISSING' if required else 'OPTIONAL_MISSING'
        elif not path.is_file():
            result['status'] = 'WRONG_TYPE' if required else 'OPTIONAL_WRONG_TYPE'
        else:
            info = path.stat()
            result.update(status='PRESENT', bytes=info.st_size, mtime_ns=info.st_mtime_ns)
            if info.st_size == 0:
                result['status'] = 'EMPTY_FILE' if required else 'OPTIONAL_EMPTY_FILE'
    except OSError as exc:
        result.update(status='UNREADABLE' if required else 'OPTIONAL_UNREADABLE', error=str(exc))
    if required and result['status'] != 'PRESENT':
        result['next_step'] = ('Specify an existing file using ' + FLAGS[name]
                               if name in FLAGS else 'Restore this required project/tool file from your approved local distribution.')
    return result


def python_identity(executable):
    """Read venv configuration and dist-info only, without running Python."""
    env = executable.parent.parent
    cfg = env / 'pyvenv.cfg'
    result = {'evidence_kind': 'STATIC_METADATA', 'runtime_compatibility': 'UNVERIFIED',
              'pyvenv_cfg': str(cfg), 'version': None, 'packages': []}
    if cfg.is_file():
        if cfg.stat().st_size <= 65536:
            values = dict(line.split('=', 1) for line in cfg.read_text(encoding='utf-8').splitlines() if '=' in line)
            result['version'] = next((v.strip() for k, v in values.items() if k.strip() == 'version'), None)
    sites = [env / 'Lib/site-packages']
    lib = env / 'lib'
    if lib.is_dir():
        sites += [p / 'site-packages' for p in lib.glob('python*') if p.is_dir()]
    result['metadata_locations'] = [str(p) for p in sites if p.is_dir()]
    for site in sites:
        if site.is_dir():
            for dist in metadata.distributions(path=[str(site)]):
                name = dist.metadata.get('Name')
                if name:
                    result['packages'].append({'name': name, 'version': dist.version})
    result['packages'].sort(key=lambda p: p['name'].lower())
    return result


def check_environment(project_root, *, input_kind='photos', target='blender', **overrides):
    if input_kind not in ('photos', 'video') or target not in ('report', 'blender', 'cad'):
        raise ValueError('invalid input kind or target')
    root = Path(project_root).expanduser().resolve()
    paths, origins = {}, {}
    defaults = {'project_python': LIGHT, 'sam3_python': HEAVY, 'sam3_package': PACKAGE,
                'checkpoint': r'C:\sam3\sam3.pt', 'mvs_tool_root': MVS, 'freecad_python': FREECAD}
    for name, value in defaults.items():
        paths[name] = resolve_path(overrides.get(name) or value, root)
        origins[name] = 'EXPLICIT' if overrides.get(name) else ('MACHINE_DEFAULT_NONPORTABLE' if name == 'checkpoint' else 'PROJECT_RELATIVE_DEFAULT')
    for name in ('git', 'ffmpeg', 'ffprobe', 'blender'):
        value = overrides.get(name)
        found = shutil.which(name) if not value else None
        if name == 'git' and not value and not found:
            local_git = Path(r'C:\Program Files\Git\cmd\git.exe')
            if local_git.is_file():
                found = str(local_git)
        if name == 'blender' and not value:
            local = Path(r'C:\Program Files\Blender Foundation\Blender 5.1\blender.exe')
            if local.is_file():
                found = str(local)
        paths[name] = resolve_path(value or found, root) if value or found else None
        origins[name] = 'EXPLICIT' if value else 'LOCAL_DISCOVERY_NONPORTABLE'
    geometry = target in ('blender', 'cad')
    checks = [inspect_file('project_marker', root / 'pyproject.toml', True, 'PROJECT_ROOT')]
    for name in ('project_python', 'sam3_python', 'checkpoint', 'git', 'blender', 'freecad_python', 'ffmpeg', 'ffprobe'):
        required = name in ('project_python', 'sam3_python', 'checkpoint', 'git') or (name == 'blender' and geometry) or (name == 'freecad_python' and target == 'cad') or (name in ('ffmpeg', 'ffprobe') and input_kind == 'video')
        item = inspect_file(name, paths[name], required, origins[name])
        if name in ('project_python', 'sam3_python') and item['status'] == 'PRESENT':
            try:
                item['static_identity'] = python_identity(paths[name])
                item['version'] = item['static_identity']['version']
            except (OSError, ValueError, UnicodeError) as exc:
                item['static_identity_error'] = str(exc)
        checks.append(item)
    for filename in ('InterfaceCOLMAP.exe', 'DensifyPointCloud.exe', 'ReconstructMesh.exe'):
        checks.append(inspect_file('openmvs/' + filename, paths['mvs_tool_root'] / filename, geometry, origins['mvs_tool_root']))
    for filename in ('BUNDLE.json', 'AUTHORITY_BINDINGS.json', 'START_BASELINE.json'):
        checks.append(inspect_file('sam3_package/' + filename, paths['sam3_package'] / filename, True, origins['sam3_package']))
    for filename in ('scripts/prepare_object.py', 'scripts/reconstruct_object.py', 'scripts/demo_segment_worker.py'):
        checks.append(inspect_file(filename, root / filename, True, 'PROJECT_SOURCE'))
    if geometry:
        for filename in ('scripts/derive_general_subset.py', 'scripts/prepare_mvs.py', 'scripts/prepare_mvs_masks.py', 'scripts/build_mvs_surface.py', 'scripts/export_blender.py'):
            checks.append(inspect_file(filename, root / filename, True, 'PROJECT_SOURCE'))
    if target == 'cad':
        for filename in ('scripts/continue_photo_cad.py', 'scripts/prepare_mesh.py', 'scripts/build_cad_delivery.py'):
            checks.append(inspect_file(filename, root / filename, True, 'PROJECT_SOURCE'))
    blockers = []
    for name, expected in (('sam3_python', root / HEAVY), ('sam3_package', root / PACKAGE)):
        if paths[name] != expected.resolve():
            blockers.append({'category': 'RUNTIME_PATH_CONTRACT', 'name': name,
                             'next_step': 'Current runtime requires the project-relative audited path; re-establish bindings before execution.'})
    static_records = {}
    for filename in ('BUNDLE.json', 'AUTHORITY_BINDINGS.json', 'START_BASELINE.json'):
        path = paths['sam3_package'] / filename
        if path.is_file():
            try:
                data = small_json(path)
                static_records[filename] = {k: data[k] for k in ('schema', 'profile', 'project_root', 'checkpoint') if k in data}
                if 'project_root' in data and resolve_path(data['project_root'], root) != root:
                    blockers.append({'category': 'BOUND_TO_OTHER_PROJECT_ROOT', 'name': filename,
                                     'recorded_root': data['project_root'], 'next_step': 'Rebuild and audit the local worker bindings for this root; do not copy historical PASS.'})
                bound = data.get('checkpoint')
                if isinstance(bound, dict) and bound.get('path') and resolve_path(bound['path'], root) != paths['checkpoint']:
                    blockers.append({'category': 'CHECKPOINT_PATH_BINDING', 'name': filename,
                                     'next_step': 'Re-establish checkpoint identity and worker bindings for the selected local checkpoint.'})
            except (OSError, ValueError, UnicodeError) as exc:
                blockers.append({'category': 'INVALID_STATIC_METADATA', 'name': filename, 'error': str(exc),
                                 'next_step': 'Restore valid audited metadata; a file alone does not satisfy the worker contract.'})
    missing = [c['name'] for c in checks if c['required'] and c['status'] != 'PRESENT']
    return {'schema': 'photo3d-environment-preflight-v1', 'checked_utc': datetime.now(timezone.utc).isoformat(),
            'project_root': str(root), 'input_kind': input_kind, 'target': target,
            'status': 'STATIC_BLOCKED' if missing or blockers else 'STATIC_ITEMS_PRESENT_RUNTIME_UNVERIFIED',
            'runtime_compatibility': 'UNVERIFIED', 'clean_machine_acceptance': 'NOT_PERFORMED',
            'current_runtime_evidence': [], 'historical_pass_reused': False,
            'scope': 'File/stat and bounded local metadata only; no native processes, imports of torch/SAM3, device queries, downloads or network.',
            'checks': checks, 'missing_required': missing, 'known_contract_blockers': blockers,
            'static_records': static_records,
            'unverified': ['Executable/library loading and versions', 'SAM3 inventory/source/hash binding acceptance',
                           'CUDA/driver/device/memory compatibility', 'Checkpoint digest and model load',
                           'Actual photo/video to target execution', 'Clean machine installation'],
            'next_steps': [c['next_step'] for c in checks if 'next_step' in c] + [b['next_step'] for b in blockers] +
                          ['After resolving static blockers, perform an authorized bounded native end-to-end run and independent artifact readback on the destination machine.']}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', required=True)
    parser.add_argument('--input-kind', choices=('photos', 'video'), default='photos')
    parser.add_argument('--target', choices=('report', 'blender', 'cad'), default='blender')
    parser.add_argument('--format', choices=('json', 'text'), default='json')
    for name, flag in FLAGS.items():
        parser.add_argument(flag, dest=name)
    parser.add_argument('--mvs-tool-root')
    args = vars(parser.parse_args(argv))
    format_name = args.pop('format')
    result = check_environment(**args)
    if format_name == 'json':
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(result['status'] + '\nRuntime compatibility: UNVERIFIED; clean machine acceptance: NOT_PERFORMED')
        for item in result['checks']:
            print(f"{item['name']}: {item['status']} ({'required' if item['required'] else 'optional'}) {item['path']}")
        for blocker in result['known_contract_blockers']:
            print(blocker['category'] + ': ' + blocker['name'])
        for step in dict.fromkeys(result['next_steps']):
            print('Next: ' + step)
    return 2 if result['status'] == 'STATIC_BLOCKED' else 0
