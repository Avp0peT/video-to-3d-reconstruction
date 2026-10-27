"""Direct CLI import paths work outside the checkout without PYTHONPATH."""
import json
import os
from pathlib import Path
import subprocess
import sys


SCRIPT = Path(__file__).resolve().parents[1]/'scripts/prepare_mvs_masks.py'


def isolated_env():
    env = dict(os.environ)
    env.pop('PYTHONPATH', None)
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    return env


def test_standalone_help_without_caller_pythonpath(tmp_path):
    result = subprocess.run([sys.executable, '-I', '-B', str(SCRIPT), '--help'],
        cwd=tmp_path, env=isolated_env(), capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert '--selection-ledger' in result.stdout


def test_standalone_derived_validator_import_preflight(tmp_path):
    # Deliberately invalid lineage reaches the new provenance validator. No
    # bridge, mask producer, model, or native geometry process is executed.
    manifest = tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'backend':{'name':'video_temporal_union'}}))
    code = '''
import runpy, sys
namespace = runpy.run_path(sys.argv[1], run_name='standalone_import_preflight')
mio = namespace['mio']
try:
    mio.parse_object_manifest(sys.argv[2])
except mio.InputError as exc:
    assert 'invalid derived union backend' in str(exc), str(exc)
else:
    raise AssertionError('invalid derivation accepted')
print('DERIVED_VALIDATOR_IMPORT_REACHED')
'''
    result = subprocess.run([sys.executable, '-I', '-B', '-c', code, str(SCRIPT), str(manifest)],
        cwd=tmp_path, env=isolated_env(), capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert 'DERIVED_VALIDATOR_IMPORT_REACHED' in result.stdout
