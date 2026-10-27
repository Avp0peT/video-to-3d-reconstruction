param(
    [Parameter(Mandatory=$true)][string]$Source,
    [Parameter(Mandatory=$true)][string]$Prompt,
    [string]$Output,
    [ValidateRange(2,96)][int]$Frames=24,
    [string]$Ffmpeg,
    [string]$Ffprobe,
    [ValidateSet('blender','report','cad')][string]$Target='blender',
    [string]$FreecadPython,
    [string]$GitExe,
    [string]$ModelId,
    [string]$MvsToolRoot,
    [string]$BlenderExe,
    [string]$RecoveryPrompt,
    [switch]$NoAutoRecover,
    [switch]$PlanOnly
)
$ErrorActionPreference='Stop'
$projectRoot=Split-Path -Parent $PSScriptRoot
$projectPython=Join-Path $projectRoot 'outputs\delegation\T02A\envs\20261002T190239Z_9b9b5ddb\venv\Scripts\python.exe'
if(-not (Test-Path -LiteralPath $projectPython)){throw 'Project Python was not found. See docs/PHOTO_DEMO_USAGE.md.'}
$mode=if($PlanOnly){'plan'}else{'run'}
$sourcePath=(Resolve-Path -LiteralPath $Source).Path
$arguments=@('-B',(Join-Path $PSScriptRoot 'run_photo_demo.py'),$mode,'--source',$sourcePath,'--prompt',$Prompt,'--frames',"$Frames")
$arguments+=@('--target',$Target)
if($RecoveryPrompt){$arguments+=@('--recovery-prompt',$RecoveryPrompt)}
if($NoAutoRecover){$arguments+=@('--no-auto-recover')}
if($ModelId){$arguments+=@('--model-id',$ModelId)}
if($MvsToolRoot){$arguments+=@('--mvs-tool-root',$MvsToolRoot)}
if($BlenderExe){$arguments+=@('--blender-exe',$BlenderExe)}
if($FreecadPython){$arguments+=@('--freecad-python',$FreecadPython)}
if($GitExe){$arguments+=@('--git-exe',$GitExe)}
if($Output){$arguments+=@('--output',$Output)}
if($Ffmpeg){$arguments+=@('--ffmpeg',$Ffmpeg)}
if($Ffprobe){$arguments+=@('--ffprobe',$Ffprobe)}
Push-Location -LiteralPath $projectRoot
$resultCode=4
try{
    $global:LASTEXITCODE=$null
    & $projectPython @arguments
    if($LASTEXITCODE -isnot [int]){
        throw 'Python entry returned no integer exit code; execution is unconfirmed.'
    }
    $resultCode=$LASTEXITCODE
}catch{
    [Console]::Error.WriteLine(('Python entry failed: ' + $_.Exception.Message))
    $resultCode=4
}finally{ Pop-Location }
exit $resultCode
