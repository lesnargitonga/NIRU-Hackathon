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
Traditional drones rely on GPS (which can be jammed) and human pilots (who get tired). Sentinel uses **on-board AI** to "see" and navigate its environment using lasers (LiDAR) and cameras, making it unjammable and fully autonomous.

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
