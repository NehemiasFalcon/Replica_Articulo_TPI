param(
    [ValidateSet('heuristic','ollama')][string]$Mode='heuristic',
    [string]$Model='qwen2.5-coder:3b',
    [int]$Limit=0,
    [int]$TokenLimit=5500,
    [string]$DatasetRoot='data_original',
    [switch]$Drain
)
$ErrorActionPreference='Stop'
Set-Location $PSScriptRoot
$PythonArgs = @('experiment.py', '--dataset-root', $DatasetRoot,
          '--mode', $Mode, '--model', $Model, '--limit', $Limit,
          '--token-limit', $TokenLimit,
          '--out', "results/$Mode-$($Model.Replace(':','-'))")
if ($Drain) {  $PythonArgs += '--drain' }
python @PythonArgs
if ($LASTEXITCODE -ne 0) { throw 'El experimento devolvió un error' }
