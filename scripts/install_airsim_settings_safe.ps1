param(
  [string]$BackupSuffix = (Get-Date -Format 'yyyyMMdd_HHmmss')
)

$settingsDir = Join-Path $env:USERPROFILE 'Documents/AirSim'
$settingsPath = Join-Path $settingsDir 'settings.json'

if (-not (Test-Path $settingsDir)) {
  New-Item -ItemType Directory -Path $settingsDir | Out-Null
}

# Backup existing settings.json if present
if (Test-Path $settingsPath) {
  $backupPath = $settingsPath + ".bak_" + $BackupSuffix
  Copy-Item -Path $settingsPath -Destination $backupPath -Force
  Write-Host "[INFO] Backed up existing settings to: $backupPath"
}

# Safer AirSim config: conservative LiDAR, enabled depth camera, reduced FOV/points
$safeJson = @'
{
  "SettingsVersion": 1.2,
  "SimMode": "Multirotor",
  "ClockSpeed": 1.0,
  "UsageScenario": "Normal",
  "Vehicles": {
    "SimpleFlight": {
      "VehicleType": "SimpleFlight",
      "AutoCreate": true,
      "DefaultController": "SimpleFlightController",
      "EnableTrace": false,
      "Cameras": {
        "DepthFront": {
          "CaptureSettings": [
            {
              "ImageType": 2,
              "Width": 640,
              "Height": 360,
              "FOV_Degrees": 90,
              "Pitch": -12,
              "Roll": 0,
              "Yaw": 0
            }
          ]
        }
      },
      "Sensors": {
        "Lidar360": {
          "SensorType": 6,
          "Enabled": true,
          "NumberOfChannels": 1,
          "PointsPerSecond": 4000,
          "RotationsPerSecond": 3,
          "X": 0, "Y": 0, "Z": 0,
          "Roll": 0, "Pitch": 0, "Yaw": 0,
          "VerticalFOVUpper": 0,
          "VerticalFOVLower": 0,
          "HorizontalFOVStart": -180,
          "HorizontalFOVEnd": 180,
          "Range": 30,
          "DrawDebugPoints": false,
          "DataFrame": "SensorLocalFrame"
        }
      }
    }
  }
}
'@

$safeJson | Set-Content -Path $settingsPath -Encoding UTF8
Write-Host "[INFO] Wrote safer AirSim settings to: $settingsPath"
Write-Host "[NEXT] Restart Unreal, open your level, press Play, then start autonomy with -UseDepth for extra stability."
