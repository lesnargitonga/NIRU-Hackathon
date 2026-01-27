param(
    [string]$Config = "unreal_tools/ue_sar_demo.blocks.json",
    [string]$Project = "",
    [string]$EditorExe = "",
    [switch]$Launch
)

$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$scriptPath = Join-Path $repoRoot "unreal_tools\generate_airsim_level.py"

if (-not (Test-Path $scriptPath)) {
    throw "Could not find generator script at: $scriptPath"
}

# Set config for the Unreal Python process.
$env:LESNAR_LEVELGEN_CONFIG = $Config
Write-Host "Set LESNAR_LEVELGEN_CONFIG=$env:LESNAR_LEVELGEN_CONFIG"
Write-Host "Generator script: $scriptPath"

if (-not $Launch) {
    Write-Host ""
    Write-Host "Next steps (run inside Unreal Editor):"
    Write-Host "1) Start Unreal Editor from THIS terminal (so it inherits the env var), OR manually set the env var in Windows before launching UE."
    Write-Host "2) Open your Blocks map (often Template_Default)."
    Write-Host "3) Run the script: Tools -> Execute Python Script -> select generate_airsim_level.py"
    Write-Host "4) Save the level (Save Current Level As... recommended)."
    Write-Host ""
    Write-Host "Tip: To auto-launch, re-run with -Launch and provide -Project (path to .uproject) and optionally -EditorExe (path to UnrealEditor.exe)."
    exit 0
}

if ([string]::IsNullOrWhiteSpace($Project)) {
    if ($env:UE_PROJECT) { $Project = $env:UE_PROJECT }
}

if ([string]::IsNullOrWhiteSpace($EditorExe)) {
    if ($env:UE_EDITOR_EXE) { $EditorExe = $env:UE_EDITOR_EXE }
}

if ([string]::IsNullOrWhiteSpace($EditorExe)) {
    $cmd = Get-Command UnrealEditor.exe -ErrorAction SilentlyContinue
    if (-not $cmd) { $cmd = Get-Command UE4Editor.exe -ErrorAction SilentlyContinue }
    if ($cmd) { $EditorExe = $cmd.Source }
}

if ([string]::IsNullOrWhiteSpace($Project)) {
    throw "No project specified. Pass -Project C:\path\to\Your.uproject (or set UE_PROJECT)."
}

if ([string]::IsNullOrWhiteSpace($EditorExe)) {
    throw "No editor exe specified. Pass -EditorExe C:\path\to\UnrealEditor.exe (or set UE_EDITOR_EXE)."
}

if (-not (Test-Path $Project)) {
    throw "Project not found: $Project"
}

if (-not (Test-Path $EditorExe)) {
    throw "Editor exe not found: $EditorExe"
}

Write-Host "Launching Unreal..."
Write-Host "  Editor:  $EditorExe"
Write-Host "  Project: $Project"

& $EditorExe $Project "-ExecutePythonScript=$scriptPath"
