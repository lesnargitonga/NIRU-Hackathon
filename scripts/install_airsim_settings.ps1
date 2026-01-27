param(
  [ValidateSet('px4_udp','example','lidar_safe','lidar_ultra_safe')]
  [string]$Preset = 'px4_udp'
)

$Root = Split-Path -Parent $PSScriptRoot
$presets = @{
  'px4_udp'          = Join-Path $Root 'docs/airsim_settings_px4_udp.json'
  'example'          = Join-Path $Root 'docs/airsim_settings_example.json'
  'lidar_safe'       = Join-Path $Root 'docs/airsim_settings_lidar_safe.json'
  'lidar_ultra_safe' = Join-Path $Root 'docs/airsim_settings_lidar_ultra_safe.json'
}

if (-not $presets.ContainsKey($Preset)) {
  Write-Error "Unknown preset: $Preset"
  exit 1
}

$src = $presets[$Preset]
if (-not (Test-Path $src)) {
  Write-Error "Preset file not found: $src"
  exit 1
}

$dstDir = Join-Path $env:USERPROFILE 'Documents/AirSim'
if (-not (Test-Path $dstDir)) { New-Item -ItemType Directory -Path $dstDir | Out-Null }

$dst = Join-Path $dstDir 'settings.json'
Copy-Item -Force $src $dst
Write-Host "[OK] Installed AirSim settings preset '$Preset' to $dst" -ForegroundColor Green
