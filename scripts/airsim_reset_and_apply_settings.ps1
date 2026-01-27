param(
  # Defaults to the repo's 360/LiDAR config if present.
  [string]$ConfigPath = "",

  # If set, keeps logs/recordings under Documents\AirSim.
  [switch]$KeepLogs
)

$ErrorActionPreference = 'Stop'

function Resolve-RepoPath([string]$p) {
  if ([string]::IsNullOrWhiteSpace($p)) { return $null }
  if ([System.IO.Path]::IsPathRooted($p)) { return $p }
  return (Join-Path (Get-Location) $p)
}

$repoRoot = (Get-Location).Path

if ([string]::IsNullOrWhiteSpace($ConfigPath)) {
  $candidate = Join-Path $repoRoot "airsim\settings_360.json"
  if (Test-Path $candidate) {
    $ConfigPath = $candidate
  } else {
    $candidate = Join-Path $repoRoot "docs\airsim_settings_example.json"
    if (Test-Path $candidate) {
      $ConfigPath = $candidate
    }
  }
}

$ConfigPath = Resolve-RepoPath $ConfigPath

if (-not (Test-Path $ConfigPath)) {
  throw "Config file not found: $ConfigPath"
}

$airSimDir = Join-Path $env:USERPROFILE "Documents\AirSim"
$settingsPath = Join-Path $airSimDir "settings.json"

New-Item -ItemType Directory -Force -Path $airSimDir | Out-Null

# Backup existing settings.json
if (Test-Path $settingsPath) {
  $ts = Get-Date -Format "yyyyMMdd_HHmmss"
  $backupPath = Join-Path $airSimDir "settings.json.bak.$ts"
  Copy-Item -Force $settingsPath $backupPath
  Write-Host "Backed up existing settings to: $backupPath"
}

# Clear AirSim cache/artifacts (safe: leaves backups)
$preserveNames = @(
  "settings.json",
  "settings.json.bak",
  "settings.json.bak.*"
)

Get-ChildItem -Force -Path $airSimDir | ForEach-Object {
  $name = $_.Name

  # Preserve backups
  $isBackup = $name -like "settings.json.bak*"
  if ($name -eq "settings.json" -or $isBackup) {
    return
  }

  if ($KeepLogs) {
    # Keep logs/recordings if requested
    if ($name -match "(?i)log|logs|record|recordings|captures|images|screenshots") {
      return
    }
  }

  try {
    Remove-Item -Force -Recurse -Path $_.FullName
  } catch {
    Write-Host "Warning: failed to remove $($_.FullName): $($_.Exception.Message)"
  }
}

# Apply new settings.json
Copy-Item -Force $ConfigPath $settingsPath
Write-Host "Installed AirSim settings.json from: $ConfigPath"
Write-Host "AirSim settings path: $settingsPath"

Write-Host "Next steps:" 
Write-Host "1) Start Unreal/AirSim and press Play"
Write-Host "2) Run a quick check: python .\airsim\check_connection.py"
