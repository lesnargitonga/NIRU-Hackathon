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

## 🚀 Teacher Mode ("God Mode") - WSL Workflow

For high-fidelity data collection and visual demonstration, we use a "God Mode" script that runs inside WSL (Ubuntu) to bypass Windows network isolation. This controls the PX4 Soft-In-The-Loop (SITL) drone with perfect A* pathfinding.

**Prerequisites:**
*   Repo cloned/copied into WSL filesystem (e.g., `~/lesnar/LesnarAI`).
*   Python 3 environment in WSL (`.venv_wsl`) with `mavsdk` and `numpy`.
*   PX4 Autopilot running in WSL.

**Running the Stable Drone:**

1.  **Open Project in WSL-Attached VS Code**  
    `Ctrl+Shift+P` -> `Remote-WSL: New Window` -> Open Folder `~/lesnar/LesnarAI`.

2.  **Terminal 1: The Bridge (MAVSDK Server)**  
    This connects the simulation to the Python code.
    ```bash
    cd ~/lesnar/LesnarAI
    source .venv_wsl/bin/activate
    mavsdk_server -p 50051 udpin://0.0.0.0:14540
    ```

3.  **Terminal 2: The Brain (Teacher Script)**  
    This executes the pathfinding and flight control.
    ```bash
    cd ~/lesnar/LesnarAI
    source .venv_wsl/bin/activate
    python training/px4_teacher_collect_gz.py --system 0.0.0.0:14540 --mavsdk-server 127.0.0.1 --mavsdk-port 50051 --duration 300
    ```

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

2) Privacy recording / screenshot capture (demo of masking pipeline)
```
& .\.venv\Scripts\python.exe .\scripts\capture_screenshot.py --out .\data\privacy_demo.png
```

3) Latency profile (model speed vs safe blind distance)
```
& .\.venv\Scripts\python.exe .\scripts\compute_profiler.py `
	--task seg `
	--weights .\runs\unet_airsim\best.pt `
	--speed_mps 5
```

4) Loss-of-Link / audit writer (simulated)
```
& .\.venv\Scripts\python.exe .\scripts\timescale_writer.py --once
```

## Teacher Brain – God-Mode (Offline Expert)

When WSL / PX4 / Gazebo are not available or are unstable, you can still
generate **expert demonstrations** using a pure-Python "God-Mode" teacher.

This teacher:
- Parses the exact world layout from `obstacles.sdf`.
- Builds a 2D occupancy grid and plans globally using A*.
- Follows the path with a pure-pursuit controller.
- Simulates a perfect 360° LiDAR around the virtual drone.

No simulation processes are required – everything runs in Python on Windows.

1) Generate offline expert data
```powershell
Set-Location "D:\docs\lesnar\Lesnar AI"
& "$Env:USERPROFILE\miniconda3\Scripts\conda.exe" run -n lesnar-ai-gpu `
	python training/px4_teacher_collect_gz.py `
		--duration 300 `
		--offline `
		--out dataset/px4_teacher/telemetry_god.csv
```

This produces a CSV with columns:
- State: timestamp, rel_alt, vx, vy, vz, yaw
- Commands: cmd_vx, cmd_vy, cmd_vz, cmd_yaw
- Perception: lidar_min, lidar_json (72-beam 360° scan)
- Mission context: goal_x, goal_y (current navigation target)

2) Train the student on God-Mode data
```powershell
& .\.venv\Scripts\python.exe .\training\train_student_px4.py `
	--data .\dataset\px4_teacher\telemetry_god.csv `
	--epochs 20 `
	--bs 128 `
	--out .\models\student_px4_god.pt
```

You can still use the original PX4 SITL pipeline when WSL + Gazebo are
available, but the offline God-Mode teacher lets you iterate on brains and
architectures without fighting the simulator stack.

## GPU Setup (Windows, Conda)

To ensure Torch uses your NVIDIA GPU, use the provided Conda setup:

```powershell
Set-Location "D:\docs\lesnar\Lesnar AI"
.\u005cscripts\setup_gpu_env.ps1
```

This creates the `lesnar-ai-gpu` environment with CUDA 12.1, verifies CUDA via `scripts/verify_cuda.py`, and will train automatically if `dataset\px4_teacher\telemetry_adv.csv` exists.

If you generated Gazebo LiDAR data via WSL collector, point training to `dataset\px4_teacher\telemetry_gz.csv` instead.

Troubleshooting:
- If OpenCV fails to import with NumPy 2.x, pin NumPy to 1.26.4 in the Conda env.
- If `gz` is unavailable on Ubuntu 24.04, use `jmavsim` for SITL.
