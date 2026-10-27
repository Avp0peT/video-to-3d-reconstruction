"""Task-local shell exit propagation; never invokes the photo product."""
from pathlib import Path
import os,shutil,subprocess,sys
import pytest

ROOT=Path(__file__).resolve().parents[1]


def shell():
    found=shutil.which('pwsh.exe') or shutil.which('pwsh')
    if not found:pytest.skip('existing pwsh unavailable')
    return found


def guard():
    source=(ROOT/'scripts/run_photo_demo.ps1').read_text(encoding='utf-8-sig')
    return source[source.index('$resultCode=4'):]


def quote(value):return "'"+str(value).replace("'","''")+"'"


@pytest.mark.parametrize('mode,expected',[('null',4),('throw',4),('wrongtype',4),('stale',4)])
def test_unconfirmed_fake_launcher_cannot_report_success(tmp_path,mode,expected):
    body={'null':"[Console]::WriteLine('FAKE_TEST_DATA_NO_EXIT')",'throw':"throw 'TEST_DATA launcher failure'",
          'wrongtype':"$global:LASTEXITCODE='0'",'stale':"[Console]::WriteLine('FAKE_TEST_DATA_STALE_EXIT')"}[mode]
    code="$ErrorActionPreference='Stop';function Invoke-TestLauncher {"+body+"};$projectPython='Invoke-TestLauncher';$arguments=@();$global:LASTEXITCODE=0;Push-Location "+quote(tmp_path)+";"+guard()
    result=subprocess.run([shell(),'-NoProfile','-NonInteractive','-Command',code],capture_output=True,text=True,timeout=5)
    assert result.returncode==expected and 'Python entry failed:' in result.stderr


@pytest.mark.parametrize('exit_code',[0,1,2,5,17])
def test_controlled_native_integer_exit_is_preserved(tmp_path,exit_code):
    code="$ErrorActionPreference='Stop';$projectPython="+quote(sys.executable)+";$arguments=@('-B','-X','utf8','-c',"+quote("import sys; print('TEST_DATA_NATIVE_EXIT'); sys.exit("+str(exit_code)+")")+");Push-Location "+quote(tmp_path)+";"+guard()
    result=subprocess.run([shell(),'-NoProfile','-NonInteractive','-Command',code],capture_output=True,text=True,timeout=5)
    assert result.returncode==exit_code and 'TEST_DATA_NATIVE_EXIT' in result.stdout
    assert 'Python entry failed:' not in result.stderr
