param(
  [string]$ProjectPath = "J:\dronesim",
  [switch]$BackupCrashes,
  [switch]$DryRun
)

function Write-Step($msg) { Write-Host "[STEP] $msg" -ForegroundColor Cyan }
function Write-Info($msg) { Write-Host "[INFO] $msg" -ForegroundColor Gray }
function Write-Ok($msg) { Write-Host "[ OK ] $msg" -ForegroundColor Green }
function Write-Warn($msg) { Write-Host "[WARN] $msg" -ForegroundColor Yellow }

function Remove-Path {
  param([string]$Path)
  if (-not $Path) { return }
  if (Test-Path $Path) {
    if ($DryRun) { Write-Info "DRYRUN: Remove $Path"; return }
    try { Remove-Item -LiteralPath $Path -Recurse -Force -ErrorAction Stop; Write-Ok "Removed $Path" }
    catch { Write-Warn "Failed to remove ${Path}: $($_.Exception.Message)" }
  } else {
    Write-Info "Skip (missing): $Path"
  }
}

function Remove-ChildrenExcept {
  param([string]$Dir, [string[]]$Keep)
  if (-not (Test-Path $Dir)) { Write-Info "Skip (missing): $Dir"; return }
  Get-ChildItem -LiteralPath $Dir -Force | ForEach-Object {
    if ($Keep -contains $_.Name) {
      Write-Info "Keep: $($_.FullName)"
    } else {
      if ($DryRun) { Write-Info "DRYRUN: Remove $($_.FullName)" }
      else {
        try { Remove-Item -LiteralPath $_.FullName -Recurse -Force -ErrorAction Stop; Write-Ok "Removed $($_.FullName)" }
        catch { Write-Warn "Failed to remove $($_.FullName): $($_.Exception.Message)" }
      }
    }
  }
}

Write-Step "Close Unreal Editor and any AirSim/UE processes before proceeding."

# 1) Project-local cleanup
if (Test-Path $ProjectPath) {
  Write-Step "Project cleanup at $ProjectPath"
  $saved = Join-Path $ProjectPath 'Saved'
  $intermediate = Join-Path $ProjectPath 'Intermediate'
  $ddc = Join-Path $ProjectPath 'DerivedDataCache'
  $binaries = Join-Path $ProjectPath 'Binaries'

  if ($BackupCrashes) {
    $crashDir = Join-Path $saved 'Crashes'
    if (Test-Path $crashDir) {
      $dest = Join-Path $ProjectPath ("CrashBackups_" + (Get-Date -Format 'yyyyMMdd_HHmmss'))
      if (-not $DryRun) { New-Item -ItemType Directory -Path $dest -Force | Out-Null }
      $zipPath = Join-Path $dest 'Crashes.zip'
      try {
        if ($DryRun) { Write-Info "DRYRUN: Compress $crashDir -> $zipPath" }
        else { Compress-Archive -Path "$crashDir/*" -DestinationPath $zipPath -Force; Write-Ok "Backed up crashes to $zipPath" }
      } catch { Write-Warn "Failed to backup crashes: $($_.Exception.Message)" }
    }
  }

  Remove-Path $saved
  Remove-Path $intermediate
  Remove-Path $ddc
  # Binaries can be regenerated; comment out if you keep prebuilt
  # Remove-Path $binaries
} else {
  Write-Warn "Project path not found: $ProjectPath"
}

# 2) User caches
Write-Step "Clearing user caches"
$local = $env:LOCALAPPDATA
$ddcCommon = Join-Path $local 'UnrealEngine\Common\DerivedDataCache'
$ddcEngine = Join-Path $local 'UnrealEngine\Engine\DerivedDataCache'
Remove-Path $ddcCommon
Remove-Path $ddcEngine

# Shader caches (NVIDIA / DX)
$dxCache = Join-Path $local 'NVIDIA\DXCache'
$glCache = Join-Path $local 'NVIDIA\GLCache'
$d3dCache = Join-Path $local 'D3DSCache'
Remove-Path $dxCache
Remove-Path $glCache
Remove-Path $d3dCache

# 3) AirSim folder (keep settings.json)
Write-Step "Cleaning AirSim folder (keeping settings.json)"
$airsimDir = Join-Path $env:USERPROFILE 'Documents\AirSim'
if (-not (Test-Path $airsimDir)) { New-Item -ItemType Directory -Path $airsimDir | Out-Null }
Remove-ChildrenExcept -Dir $airsimDir -Keep @('settings.json')

Write-Host ""
Write-Ok "Cache cleanup complete. First launch may recompile shaders and rebuild DDC."
Write-Info "Next: restart Unreal, press Play, then run: .\\start_everything.ps1 -UseDepth -LowImpact -StaggerSeconds 4"
