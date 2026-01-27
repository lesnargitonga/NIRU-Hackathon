# 🛡️ Operation Sentinel: Autonomous Perimeter Defense
**Status:** Classified (Hackathon Alpha)
**Compliance:** PDA 2019 Adherent | **Architecture:** Hybrid (PX4 + RL)

## 🦅 Mission Profile
Operation Sentinel is a privacy-first, autonomous drone defense system. Unlike traditional UAVs that rely on cloud vision (high privacy risk), Sentinel uses a **Hybrid Cortex Architecture**:
1.  **Reflex Layer (Lizard Brain):** On-board PX4 + LiDAR for crash-proof physics (0% Latency).
2.  **Mission Layer (Human Brain):** Deep Reinforcement Learning for complex navigation.

## 🏗️ Technical Stack
* **Flight Core:** PX4 Autopilot (v1.14)
* **Simulation:** Gazebo Harmonic (Synthetic Training Environment)
* **Brain:** Stable-Baselines3 (PPO) + PyTorch (CPU Optimized)
* **Comms:** MavSDK (Python) -> MAVLink -> UDP 14540

## ⚡ Quick Start
```bash
# 1. Launch Simulation (Sentinel Core)
make px4_sitl gz_x500_lidar_2d

# 2. Launch AI (The Brain)
python3 rl/test_flight.py
```
