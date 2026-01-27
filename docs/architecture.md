# Architecture Design Document

## 1. System Overview

Operation Sentinel employs a **Hybrid Cortex Architecture**, a control paradigm that decouples deterministic flight stability from stochastic mission planning. This separation of concerns ensures that the platform maintains flight safety and physics integrity even if the high-level decision agent experiences uncertainty or latency.

### 1.1 The Hybrid Cortex Concept

The "Cortex" is divided into two distinct processing layers:

*   **Reflex Layer (Deterministic):**
    *   **Component:** PX4 Autopilot / EKF2
    *   **Function:** State estimation, attitude control, motor mixing, failsafe execution.
    *   **Constraint:** Hard real-time (250Hz+).
    *   **Failure Mode:** Safe landing / Return to Launch.

*   **Mission Layer (Stochastic):**
    *   **Component:** Deep Reinforcement Learning (PPO) Agent.
    *   **Function:** Semantic scene understanding, complex path planning, dynamic obstacle avoidance.
    *   **Constraint:** Soft real-time (10-50Hz).
    *   **Failure Mode:** Local hover / Re-plan.

## 2. Component Design

### 2.1 Computing Substrate
The system operates within a tiered compute environment:
-   **Host:** WSL2 (Ubuntu 22.04) running on x86_64 architecture.
-   **Interconnect:** Localhost UDP/TCP, avoiding overhead of virtualization network bridges.

### 2.2 Simulation Environment (Gazebo Harmonic)
Gazebo Harmonic provides the synthetic ground truth. It utilizes:
-   **Ray Tracing:** For realistic LiDAR point cloud generation.
-   **Physics Engine:** DART/ODE for rigid body dynamics.
-   **Sensor Models:** IMU (Acc/Gyro), Magnetometer, GPS, and Barometer with distinct noise profiles matching real-world hardware.

### 2.3 Flight Core (PX4)
The PX4 stack runs in SITL (Software In The Loop) mode but executes the exact binary code used on physical flight controllers (Pixhawk 6X).
-   **EKF2:** Extended Kalman Filter fusing IMU and GPS data for state estimation.
-   **Commander:** State machine handling arming checks, mode switching, and geofence enforcement.

### 2.4 Mission Brain (AI Agent)
The autonomous agent is a Neural Network trained via Proximal Policy Optimization (PPO).
-   **Input:** 360-degree LiDAR tensor (normalized distance vectors) + Ego-State (Velocity, Orientation).
-   **Output:** Continuous action vector $[v_x, v_y, v_z, \dot{\psi}]$ (Velocity sets in body frame).
-   **Bridge:** MAVSDK-Python acts as the translation layer, converting neural network outputs into MAVLink `SET_POSITION_TARGET_LOCAL_NED` messages.

## 3. Data Flow Architecture

```mermaid
graph TD
    subgraph Host_Machine_WSL [Compute Core: WSL2 Ubuntu]
        style Host_Machine_WSL fill:#1c2128,stroke:#444c56,color:#fff

        subgraph Sim_Layer [Simulation Environment]
            style Sim_Layer fill:#2d333b,stroke:#adbac7,color:#fff
            Gazebo[Gazebo Harmonic<br/><i>Physics Engine</i>]:::sim
            World[Synthetic World<br/><i>Generative Geometry</i>]:::sim
        end

        subgraph Flight_Core [PX4 Autopilot (The Reflex Cortex)]
            style Flight_Core fill:#1f6feb,stroke:#fff,color:#fff,stroke-width:2px
            EKF2[EKF2 Estimator<br/><i>Sensor Fusion</i>]
            Commander[Failsafe Commander<br/><i>Safety Layer</i>]
            MavLink_Server[MAVLink Server<br/><i>UDP 14540</i>]
        end

        subgraph Mission_Brain [AI Agent (The Mission Cortex)]
            style Mission_Brain fill:#238636,stroke:#fff,color:#fff,stroke-width:2px
            MavSDK[MavSDK-Python<br/><i>Bridge</i>]
            PPO_Agent[PPO Neural Network<br/><i>Decision Engine</i>]
            LiDAR_Proc[LiDAR Preprocessor<br/><i>Privacy Filter</i>]
        end
    end

    %% Data Flow
    World -->|Ray Tracing| Gazebo
    Gazebo -->|LiDAR Data Point Cloud| Flight_Core
    Flight_Core -->|Telemetry Stream 50Hz| MavSDK
    MavSDK -->|Normalized State| LiDAR_Proc
    LiDAR_Proc -->|Tensor [360]| PPO_Agent
    
    PPO_Agent -->|Action Vector [Vx,Vy,Vz,Yaw]| MavSDK
    MavSDK -->|Offboard Velocity Cmd| MavLink_Server
    MavLink_Server -->|Mixer Inputs| Flight_Core
    Flight_Core -->|Motor PWM| Gazebo

    classDef sim fill:#da3633,stroke:#fff,color:#fff;
```

## 4. Security Considerations

All inter-process communication is localized. The system is designed to operate with the radio modem as the *only* external interface.
-   **No Cloud Dependency:** Usage of cloud APIS (Google Maps, AWS) is strictly prohibited in the operational loop.
-   **Data Minimization:** Resolution of LiDAR data is downsampled at the pre-processor to the minimum viable density for navigation, reducing the surface area for data reconstruction attacks.
