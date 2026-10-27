"""Stdlib-only Job-inheriting Blender child launcher; no shell or breakaway."""
import json
import os
from pathlib import Path
import subprocess
import sys


def execute(request, runner=None):
    runner = subprocess.run if runner is None else runner
    environment = os.environ.copy()
    environment.update({'TMP':request['scratch'], 'TEMP':request['scratch']})
    try:
        completed=runner(request['blender_argv'],cwd=request['cwd'],env=environment,
                         shell=False,timeout=request['child_timeout_seconds'],check=False)
        code=int(completed.returncode)
        result=dict(status='CHILD_EXITED',child_exit_code=code)
    except subprocess.TimeoutExpired:
        code=124;result=dict(status='CHILD_TIMEOUT',child_exit_code=code)
    # No stdout/stderr redirection: both inherit the original Job supervisor pipes.
    result.update(environment_overrides={'TMP':request['scratch'],'TEMP':request['scratch']},
                  shell=False,breakaway=False,timeout_seconds=request['child_timeout_seconds'])
    Path(request['result_path']).write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    return code


def main(argv=None):
    args=sys.argv[1:] if argv is None else argv
    if len(args)!=1:raise ValueError('one bound request path required')
    request=json.loads(Path(args[0]).read_text(encoding='utf-8'))
    return execute(request)


if __name__=='__main__':raise SystemExit(main())
