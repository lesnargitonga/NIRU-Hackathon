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
