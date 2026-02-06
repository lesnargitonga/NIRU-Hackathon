# Operation Sentinel

**Autonomous Perimeter Defense System**

![Build Status](https://img.shields.io/badge/build-passing-brightgreen?style=flat-square)
![Platform](https://img.shields.io/badge/platform-PX4%20|%20Gazebo-blue?style=flat-square)
![Language](https://img.shields.io/badge/language-Python%203.10%20|%20C++-blue?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-lightgrey?style=flat-square)

---

## System Overview

**Operation Sentinel** is an autonomous drone system designed to defend secure perimeters without needing a human pilot or GPS.

### ❓ What problem does this solve?
Traditional drones rely on GPS (which can be jammed) and human pilots (who get tired). Sentinel is the testbed for the **"Universal Cortex"**—a morphology-agnostic AI trained in the **"Omniverse Engine"** (infinite procedural worlds). It navigates using pure vision/LiDAR, making it unjammable and capable of operating in any environment.

### 🛡️ Key Features for Assessors
*   **Works Offline:** No internet or cloud connection required.
*   **Hard to Jam:** Uses visual navigation instead of just GPS.
*   **Safe by Design:** A dedicated "Reflex Layer" prevents crashes even if the AI gets confused.

---

## Core Capabilities

### 1. Hybrid Cortex Control
The system utilizes a dual-layer control loop:
-   **Reflex Layer (100Hz):** PX4 Autopilot handles attitude stabilization and failsafe mechanisms, providing deterministic flight safety.
-   **Mission Layer (10Hz):** A PPO (Proximal Policy Optimization) agent processes high-dimensional LiDAR/Visual tensors to execute complex path planning and obstacle avoidance.

### 2. Privacy-First Architecture
Unlike cloud-dependent UAVs, Sentinel uses local edge compute for all perception tasks. No video feeds or telemetry data leave the airframe unless explicitly authorized via the encrypted MAVLink stream.

### 3. Simulation-to-Reality Transfer
Built on the Gazebo Harmonic physics engine, the platform enables photorealistic training environments that mathematically guarantee policy convergence before real-world deployment.

## Architecture

The system architecture defines the data flow between the physics engine, the flight controller, and the AI agent.

See [Architecture Design Document](docs/architecture.md) for a detailed technical breakdown.

## Installation

### Prerequisites
-   Windows 10/11 (WSL2 recommended for Sim)
-   Python 3.10+
-   PX4-Autopilot Toolchain

### Setup
Initialize the development environment and dependencies:

```powershell
.\bin\setup.bat
```

This script will:
1.  Configure the Python virtual environments (`backend-env`, `airsim-env`).
2.  Install all necessary torch/cuda dependencies.
3.  Prepare the frontend dashboard dependencies.

## Operation

To sequence the full system (Simulation, AI Agent, and Command Dashboard):

```powershell
.\bin\start_all.bat
```

**Sequence of Events:**
1.  **Backend Services**: Initializes the Flask API and WebSocket bridges.
2.  **Telemetry Bridge**: Establishes MAVLink connection on UDP:14540.
3.  **Mission Control**: Launches the React-based tactical dashboard.
4.  **Autonomy Core**: Engages the PPO decision engine.

## Project Structure

```text
/
├── bin/                 # Executable launch scripts and environment setup
├── config/              # System configuration profiles
├── docs/                # Architecture and design documentation
├── frontend/            # Tactical Mission Control Dashboard (React)
├── rl/                  # Reinforcement Learning policies (PPO/SAC)
├── scripts/             # Data analysis and utility tools
├── src/                 # Core backend logic and MAVSDK bridges
└── training/            # Neural network training pipelines
```

---

*Copyright © 2026 Lesnar Autonomous Systems. All Rights Reserved.*

---

## Bridge to Reality – Quick Demo

This repo now includes four components to demonstrate real-world readiness:
- Privacy Masking: blur faces on frames before leaving the airframe.
- Compute Profiling: measure model latency and blind travel distance.
- Audit Logging: write flight records to TimescaleDB for tamper-evident audit.
- Loss-of-Link Failsafe: auto-land if heartbeat is lost.

Quick run:

1) TimescaleDB stack
```
docker compose up --build -d
```

2) Privacy recording
```
& .\airsim-env\Scripts\python.exe .\airsim\record_images.py --out D:\datasets\airsim_synth --frames 200 --hz 5 --alt -5 --privacy --masks
```

3) Latency profile
```
& .\airsim-env\Scripts\python.exe .\scripts\compute_profiler.py --task seg --weights .\runs\unet_airsim\best.pt --speed_mps 5
```

4) Loss-of-Link monitor (run alongside autonomy)
```
& .\airsim-env\Scripts\python.exe .\airsim\loss_of_link_failsafe.py --timeout_s 5
```

## PX4 Teacher Brain (SITL – Advanced)

Use MAVSDK Offboard to collect high-quality demonstrations in PX4 SITL:

1) Start PX4 SITL (WSL recommended)
```
# In WSL Ubuntu
git clone https://github.com/PX4/PX4-Autopilot
cd PX4-Autopilot
git checkout v1.14.3
git submodule update --init --recursive

# Preferred (if Gazebo packages available)
make px4_sitl gz

# Fallback (simple, reliable)
make px4_sitl jmavsim
```

2) Collect advanced demos (waypoints + obstacle-aware yaw)
```
& .\.venv\Scripts\python.exe .\training\px4_teacher_collect_adv.py --system 127.0.0.1:14540 --waypoints .\training\px4_waypoints.json --out .\dataset\px4_teacher --duration 300 --hz 20 --alt 10 --base_speed 2 --max_speed 5 --yaw_rate_limit 45
```

3) Train the student
```
& .\.venv\Scripts\python.exe .\training\train_student_px4.py --data .\dataset\px4_teacher\telemetry_adv.csv --epochs 20 --bs 128 --out .\models\student_px4.pt
```

EKF2 GPS-denied config: see `px4_config/gps_denied.params` and `px4/README.md` for application steps.

## GPU Setup (Windows, Conda)

To ensure Torch uses your NVIDIA GPU, use the provided Conda setup:

```powershell
Set-Location "D:\docs\lesnar\Lesnar AI"
.\u005cscripts\setup_gpu_env.ps1
```

This creates the `lesnar-ai-gpu` environment with CUDA 12.1, verifies CUDA via `scripts/verify_cuda.py`, and will train automatically if `dataset\px4_teacher\telemetry_adv.csv` exists.

Troubleshooting:
- If OpenCV fails to import with NumPy 2.x, pin NumPy to 1.26.4 in the Conda env.
- If `gz` is unavailable on Ubuntu 24.04, use `jmavsim` for SITL.
