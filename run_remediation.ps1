param(
    [ValidateSet('ollama','rule')][string]$Mode='ollama',
    [ValidateSet('unit','lint','unsupported','both','all')][string]$Scenario='both',
    [string]$Model='qwen2.5-coder:3b',
    [string]$Output='results/remediacion-ollama',
    [switch]$Apply
)
$ErrorActionPreference='Stop'
Set-Location $PSScriptRoot
$ArgsList = @('remediation_demo.py', '--mode', $Mode, '--scenario', $Scenario,
              '--model', $Model, '--out', $Output)
if ($Apply) { $ArgsList += '--apply' }
python -u @ArgsList
if ($LASTEXITCODE -ne 0) { throw 'Hubo un error durante la remediacion' }
