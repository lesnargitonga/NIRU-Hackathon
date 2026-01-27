"""
Professional Gymnasium environment for Hierarchical Reinforcement Learning in AirSim.
Matches "Lesnar AI" Phase 2 specifications:
- Multi-modal input (Depth + Kinematics)
- Domain Randomization (Sensor noise, physics perturbations, latency)
- Reward Engineering (Smoothness, Progress, Collision, Efficiency)
- Hierarchical Control (Outputs velocity targets, smoothed)
"""

from __future__ import annotations

import os
import json
import time
import math
from dataclasses import dataclass, field
from typing import Tuple, Any, Dict, Optional, List
from collections import deque

import numpy as np
import cv2
import gymnasium as gym
from gymnasium import spaces

import airsim


class Guardian:
    """
    Reflex Safety Layer (The "Lizard Brain").
    Intercepts the AI's desire and sanitizes it for survival.
    """
    def __init__(self, safe_dist=2.0, min_altitude=0.5):
        self.safe_dist = float(safe_dist)
        self.min_altitude = float(min_altitude)

        # Brake tuning: keep these conservative (safety layer should be predictable).
        # UPDATED: Lowered thresholds to avoid interfering with expert "creeping".
        self.emergency_brake_m = 0.45
        self.depth_ema_alpha = 0.35
        self.brake_hold_steps = 0
        self._depth_ema = None

        # Debug telemetry (read by env for `info`)
        self.last_effective_depth_m = float("nan")
        self.last_min_depth_m = float("nan")
        self.last_braking = False
        self.last_emergency = False

    def filter_action(self, action_vel: np.ndarray, min_depth: float, current_alt: float) -> np.ndarray:
        """
        action_vel: [vx, vy, vz, yaw_rate]
        min_depth: minimum distance to obstacle in meters
        current_alt: current Z (negative up)
        """
        filtered = action_vel.copy()
        altitude = -current_alt # Convert NED to Positive Altitude

        # Track depth (for smoothing + debugging)
        md = float(min_depth) if min_depth is not None else float("nan")
        self.last_min_depth_m = md
        if np.isfinite(md) and md > 1e-3:
            if self._depth_ema is None or not np.isfinite(float(self._depth_ema)):
                self._depth_ema = md
            else:
                a = float(self.depth_ema_alpha)
                self._depth_ema = a * md + (1.0 - a) * float(self._depth_ema)
        
        # 1. GROUND PROXIMITY PROTECTION (The "Anti-Lawnmower")
        # If we are too low and trying to go lower (vel_z > 0), force UP.
        if altitude < self.min_altitude and filtered[2] > 0:
            filtered[2] = -0.5 # Force Safe Climb
        
        # 2. OBSTACLE PROXIMITY PROTECTION
        # We use an EMA depth for "soft" slowing to avoid stop/go hiccups from single-frame depth spikes.
        # Emergency braking still triggers on raw min_depth (fast reaction).
        self.last_braking = False
        self.last_emergency = False

        eff_depth = float(self._depth_ema) if self._depth_ema is not None else md
        if not np.isfinite(eff_depth):
            eff_depth = md
        self.last_effective_depth_m = float(eff_depth)

        # Emergency: hard stop (no reverse bounce) + short hold to prevent rapid oscillation.
        if np.isfinite(md) and md < float(self.emergency_brake_m):
            # Reduced hold from 4 to 1 to prevent "stuttering" stops.
            self.brake_hold_steps = max(int(self.brake_hold_steps), 1)
            self.last_emergency = True

        if int(self.brake_hold_steps) > 0:
            self.brake_hold_steps -= 1
            if filtered[0] > 0:
                # UPDATED: NEVER STOP (User request).
                # Instead of 0.0, clamp to a "Crawl" speed so we keep moving.
                filtered[0] = 0.1
                self.last_braking = True
        else:
            # Soft slow-down when closer than safe_dist.
            # DISABLED (User Request: "no stopping at all!")
            # The Expert already manages speed based on clearance. The Guardian should only
            # intervene in true emergencies (panic stops). Duplicating "soft" logic here
            # causes conflicts and "stuttering" when the two disagree.
            pass
            # if np.isfinite(eff_depth) and eff_depth < float(self.safe_dist) and filtered[0] > 0:
            #     denom = max(1e-6, float(self.safe_dist) - float(self.emergency_brake_m))
            #     scale = float(np.clip((eff_depth - float(self.emergency_brake_m)) / denom, 0.0, 1.0))
            #     filtered[0] = float(filtered[0]) * scale
            #     self.last_braking = True
        
        # 3. Geofence (Ceiling)
        # If higher than 20m, force descent
        if altitude > 20.0 and filtered[2] < 0:
             filtered[2] = 0.5 
             
        return filtered


class PhysicsReward:
    def __init__(self):
        # WEIGHTS (The DNA of flight style)
        self.w_progress = 1.0      # The Mission
        # Positive "survival" rewards are prone to loitering/local optima.
        # Prefer 0.0 (or a small negative step cost) and let progress/success drive behavior.
        self.w_survival = 0.0
        self.w_energy = -0.05      # The "Efficiency" penalty
        self.w_smooth = -0.1       # The "Anti-Jitter" penalty
        self.w_stability = -0.1    # The "Keep Level" penalty
        self.w_collision = -10.0   # The "Death" penalty

        # Dense shaping (Phase 2 upgrade)
        # Progress is potential-based, so this is "reward per meter" toward goal.
        self.progress_per_meter = 10.0
        # Encourage velocity aligned with goal direction (normalized -1..1)
        self.w_vel_align = 0.05
        # Penalize sharp yaw-rate changes (normalized action delta)
        self.w_yaw_delta = -0.01

        # Anti-loiter / laziness penalty (curriculum)
        self.lazy_action_threshold = 0.2
        self.lazy_penalty = -0.5

        # "Eyes forward" bonus (curriculum)
        # Small positive reward for commanding forward motion.
        self.forward_cmd_threshold = 0.1
        self.forward_cmd_bonus = 0.1

        # "Compass" reward: if the target is in front (body-frame X > 0),
        # reward positive forward command. Helps a "baby" policy link yaw/vision to motion.
        self.facing_target_bonus_scale = 0.5

        # Extra (already-proven Lesnar shaping)
        self.w_backward = 1.0      # Penalize backward body-frame motion
        self.w_ground = 1.0        # Penalize unsafe low altitude

        self.ground_warn_m = 2.0

    def calculate(self, state: Dict[str, Any], action: np.ndarray, prev_action: np.ndarray, collision: bool, step_count: int) -> float:
        """Physics-informed reward.

        Expected state keys:
        - dist_m, prev_dist_m
        - roll_rad, pitch_rad
        - altitude_m (optional)
        - v_body_x (optional)
        """

        if collision:
            return float(self.w_collision)

        dist = float(state.get("dist_m", 0.0))
        prev_dist = float(state.get("prev_dist_m", dist))

        # 1) Mission progress (potential-based)
        r_progress = (prev_dist - dist) * float(self.progress_per_meter)

        # 1b) Velocity alignment with goal (dense, helps critic)
        v_toward_goal_norm = state.get("v_toward_goal_norm")
        r_vel_align = 0.0
        if v_toward_goal_norm is not None:
            r_vel_align = float(self.w_vel_align) * float(np.clip(float(v_toward_goal_norm), -1.0, 1.0))

        # 2) Energy / effort (action magnitude)
        r_energy = self.w_energy * float(np.linalg.norm(action))

        # 2b) Anti-loiter: punish near-zero commanded motion
        # (Uses command magnitude because it is always available and deterministic.)
        r_lazy = 0.0
        current_speed_cmd = float(np.linalg.norm(action[:3]))
        if current_speed_cmd < float(self.lazy_action_threshold):
            r_lazy = float(self.lazy_penalty)

        r_forward_cmd = 0.0
        if float(action[0]) > float(self.forward_cmd_threshold):
            r_forward_cmd = float(self.forward_cmd_bonus)

        r_facing = 0.0
        body_x_to_target_m = state.get("body_x_to_target_m")
        if body_x_to_target_m is not None and float(body_x_to_target_m) > 0.0:
            r_facing = float(self.facing_target_bonus_scale) * max(0.0, float(action[0]))

        # 3) Smoothness / anti-jerk (command delta)
        r_smoothness = self.w_smooth * float(np.linalg.norm(action - prev_action))

        # 3b) Yaw smoothness (separate channel, prevents spin-to-win)
        r_yaw_delta = float(self.w_yaw_delta) * float(abs(float(action[3]) - float(prev_action[3])))

        # 4) Stability (keep level)
        roll = float(state.get("roll_rad", 0.0))
        pitch = float(state.get("pitch_rad", 0.0))
        r_stability = self.w_stability * (abs(roll) + abs(pitch))

        # 5) Survival drip
        r_survival = float(self.w_survival)

        # 6) Extra shaping (kept minimal, physics-aligned)
        r_backward = 0.0
        v_body_x = state.get("v_body_x")
        if v_body_x is not None and float(v_body_x) < -0.1:
            # Negative forward speed in body frame = backing up
            r_backward = float(v_body_x) * 4.0 * float(self.w_backward)

        r_ground = 0.0
        altitude_m = state.get("altitude_m")
        if altitude_m is not None and float(altitude_m) < self.ground_warn_m:
            r_ground = -0.05 * (self.ground_warn_m - float(altitude_m)) * float(self.w_ground)
            if float(altitude_m) < 0.5:
                r_ground -= 1.0

        total_reward = (
            self.w_progress * r_progress
            + r_survival
            + r_energy
            + r_lazy
            + r_forward_cmd
            + r_facing
            + r_smoothness
            + r_yaw_delta
            + r_stability
            + r_vel_align
            + r_backward
            + r_ground
        )
        return float(total_reward)


@dataclass
class EnvConfig:
    hz: float = 20.0  # Higher control rate for stability
    img_size: int = 64  # Reduced from 84 to fix FPS lag (Sim-to-Real speedup)
    
    # Physics limits
    max_speed_mps: float = 1.0  # [PHASE 1: CRAWLER MODE]
    max_yaw_rate_dps: float = 30.0 # Reduced for stability
    cruise_alt_ned: float = -3.0   # Solid 3m hover

    # Safety
    min_altitude_m: float = 1.0
    guardian_safe_dist_m: float = 2.0

    # Camera/RPC performance
    vision_skip_frames: int = 4

    # Reset / takeoff tuning
    takeoff_climb_mps: float = 1.0

    # Mission mode
    continuous_mission: bool = False
    land_on_goal: bool = False
    # In AirSim, ground contact during land/takeoff can keep collision flags true briefly.
    # To avoid immediate resets after a successful goal landing, ignore collisions for this long.
    collision_ignore_after_goal_sec: float = 2.0

    # Soft handover on goal (recommended): instead of calling landAsync (which can latch collision
    # or trigger sim-specific reset behavior), dip to a low altitude, hold briefly, then climb.
    goal_soft_touchdown: bool = True
    goal_touchdown_alt_m: float = 0.7
    goal_touchdown_speed_mps: float = 0.8
    goal_touchdown_hold_sec: float = 0.6
    goal_use_land_async: bool = False
    
    # Sensors
    max_depth_clip_m: float = 10.0
    
    # Training / Termination
    # Close-quarters curriculum: shorter horizon prevents reward farming.
    max_steps: int = 1200
    collision_penalty: float = 100.0
    smoothness_weight: float = 0.5 # High smoothness for stabilization
    progress_weight: float = 1.0

    # Target spawning (curriculum)
    return_home_prob: float = 0.0
    # "Close contact": practically in its face.
    target_spawn_min_m: float = 2.0
    target_spawn_max_m: float = 5.0
    # Auto-leveling spawner: expand spawn ring as the agent strings together successes.
    # Grown ring = [min + streak*growth, max + streak*growth], clamped by target_spawn_max_total_m.
    target_spawn_growth_per_success_m: float = 1.0
    target_spawn_max_total_m: float = 60.0
    target_spawn_relative_to_drone: bool = True

    # Obstacle curriculum (optional): sometimes place the target behind a perceived obstacle
    # so avoidance is required. If the depth image never sees obstacles, it safely falls back
    # to the normal ring spawn.
    obstacle_curriculum_prob: float = 0.3
    obstacle_clearance_low_m: float = 2.0
    obstacle_clearance_high_m: float = 8.0
    obstacle_target_extra_m: float = 10.0

    # Target spawn validity: avoid placing targets inside/too-close to known obstacles.
    # By default, tries to discover obstacles from the scene (actors prefixed by level generator).
    # Optionally, you can point to the same JSON used for level generation (unreal_tools/ue_obstacles.blocks.json).
    avoid_targets_in_obstacles: bool = True
    target_obstacle_clearance_m: float = 2.0
    obstacle_actor_regex: str = "LESNAR_GEN_.*"
    obstacles_file: str = ""
    obstacle_cache_refresh_sec: float = 30.0

    # "Leash" geofence: terminate if it wanders too far.
    geofence_radius_m: float = 15.0
    # Expand geofence based on target distance to avoid punishing success.
    # Effective leash = max(geofence_radius_m, dist(start->target) + geofence_buffer_m)
    geofence_buffer_m: float = 10.0
    geofence_penalty: float = -10.0

    # Reward shaping (loiter control)
    survival_reward_per_step: float = -0.1 # Time penalty to discourage hovering
    lazy_action_threshold: float = 0.1
    lazy_penalty: float = -1.0

    forward_cmd_threshold: float = 0.1
    forward_cmd_bonus: float = 0.1

    facing_target_bonus_scale: float = 0.5

    # Reward shaping (Phase 2 upgrade)
    progress_per_meter: float = 10.0
    vel_align_weight: float = 0.05
    yaw_delta_weight: float = -0.01
    smooth_arrival_bonus: float = 5.0
    smooth_arrival_speed_mps: float = 0.5
    smooth_arrival_yaw_dps: float = 10.0

    # Episodic success (Option B)
    # Smaller radius prevents "micro-move" episode ends.
    success_dist_m: float = 0.75
    success_bonus: float = 50.0
    collision_extra_penalty: float = -10.0
    
    # Sim-to-Real (Domain Randomization)
    enable_domain_randomization: bool = False # [PHASE 1: PERFECT PHYSICS]
    sensor_noise_sigma: float = 0.0
    action_smoothing_beta: float = 0.8  # Heavy smoothing for "Learning to Walk"
    # Optional separate smoothing for yaw channel (index 3). Set <= 0 to reuse action_smoothing_beta.
    action_smoothing_beta_yaw: float = -1.0
    sim_latency_steps: int = 1          # Minimal latency

    # Optional command rate limiting (post-safety, pre-Airsim).
    # This reduces "rippled road" micro-stops by limiting per-step changes.
    enable_command_rate_limit: bool = False
    max_accel_mps2: float = 3.0
    max_yaw_accel_dps2: float = 240.0

    # Optional altitude hold (helps with long-range flight / prevents slow sink when pitched).
    enable_altitude_hold: bool = False
    altitude_hold_kp: float = 0.8
    altitude_hold_deadband_m: float = 0.15
    altitude_hold_max_correction_mps: float = 1.0

    # Optional tilt protection (if we are too leaned over, reduce forward and recover altitude).
    enable_tilt_protection: bool = False
    tilt_protection_deg: float = 30.0
    tilt_recover_vz_mps: float = -0.8
    tilt_vx_scale: float = 0.4

    # Optional ground cushion (reduces "hard landing" + ground strikes during long flights).
    enable_ground_cushion: bool = False
    ground_cushion_m: float = 1.8
    ground_max_descent_mps: float = 0.25
    ground_vx_scale: float = 0.5
    ground_hard_stop_m: float = 0.9

    vehicle_name: str = ""


class AirSimDroneEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, config: Optional[EnvConfig] = None):
        super().__init__()
        self.cfg = config or EnvConfig()
        self.dt = 1.0 / self.cfg.hz
        
        # Connect to AirSim
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True, vehicle_name=self.cfg.vehicle_name)
        self.client.armDisarm(True, vehicle_name=self.cfg.vehicle_name)

        # ---- Observation Space (Multi-Modal) ----
        # 1. Visual: Depth encoding (1 channel)
        self.visual_space = spaces.Box(
            low=0.0, high=1.0, 
            shape=(1, self.cfg.img_size, self.cfg.img_size), 
            dtype=np.float32
        )
        
        # 2. Kinematics: [
        #    0:vx, 1:vy, 2:vz, 
        #    3:roll, 4:pitch, 5:yaw_rate, 
        #    6:alt_error, 
        #    7:rel_target_x, 8:rel_target_y, 9:rel_target_dist,
        #    10: sin(heading_error), 11: cos(heading_error), 
        #    12: track_error, 13: time_to_collision, 14: battery_efficiency
        #    15-18: prev_action (vx, vy, vz, yaw) - PROPRIOCEPTION
        # ]
        # "Thinking Plane" State Vector
        self.kinematics_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(19,), 
            dtype=np.float32
        )

        self.observation_space = spaces.Dict({
            "visual": self.visual_space,
            "kinematics": self.kinematics_space
        })

        # ---- Action Space (Continuous) ----
        # [vx_target, vy_target, vz_target, yaw_rate_target]
        # NOW FULL 3D CONTROL (The Falcon can fly up/down)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, 
            shape=(4,), 
            dtype=np.float32
        )

        # State tracking
        self.step_count = 0
        self.prev_action = np.zeros(4, dtype=np.float32)
        self.smoothed_action = np.zeros(4, dtype=np.float32)
        self._prev_cmd = np.zeros(4, dtype=np.float32)  # [vx,vy,vz,yaw_rate] after safety + rate limiting
        self.prev_location = np.zeros(3, dtype=np.float32)
        self.start_location = np.zeros(3, dtype=np.float32)
        self.target_position = np.array([20.0, 0.0, self.cfg.cruise_alt_ned]) 

        # Collision debounce (used for continuous-mission land/takeoff).
        self._ignore_collision_steps = 0

        # Obstacle cache for target spawning (xy center + approx radius in meters).
        self._obstacles_xy_r = []  # List[Tuple[float,float,float]]
        self._obstacles_last_refresh_t = 0.0
        
        # Mission Config (The "Input")
        self.mission_type = "NAVIGATE" # Options: NAVIGATE, EXPLORE, TRACK
        
        # Performance Optimization: Frame Skipping
        # Physics runs at 20Hz. Vision runs at (20 / skip) Hz.
        # This prevents the RPC pipe from clogging with depth arrays.
        self.vision_skip_frames = int(max(1, self.cfg.vision_skip_frames))
        self.last_depth_frame = np.zeros((1, self.cfg.img_size, self.cfg.img_size), dtype=np.float32)
        
        # Initialize with zeros so the drone hovers initially
        self.action_queue = deque(maxlen=max(1, self.cfg.sim_latency_steps))
        
        # PRO SAFETY LAYER
        self.guardian = Guardian(safe_dist=float(self.cfg.guardian_safe_dist_m), min_altitude=float(self.cfg.min_altitude_m))

        # Physics-informed reward function
        self.reward_function = PhysicsReward()
        # Wire config into reward weights (keeps reward design explicit + tunable)
        self.reward_function.progress_per_meter = float(self.cfg.progress_per_meter)
        self.reward_function.w_vel_align = float(self.cfg.vel_align_weight)
        self.reward_function.w_yaw_delta = float(self.cfg.yaw_delta_weight)
        self.reward_function.w_survival = float(self.cfg.survival_reward_per_step)
        self.reward_function.lazy_action_threshold = float(self.cfg.lazy_action_threshold)
        self.reward_function.lazy_penalty = float(self.cfg.lazy_penalty)
        self.reward_function.forward_cmd_threshold = float(self.cfg.forward_cmd_threshold)
        self.reward_function.forward_cmd_bonus = float(self.cfg.forward_cmd_bonus)
        self.reward_function.facing_target_bonus_scale = float(self.cfg.facing_target_bonus_scale)
        self.prev_dist_to_target = None
        self.consecutive_successes = 0
        
        self.last_override_time = 0.0
        
        print(f"[Lesnar RL] Environment initialized. DR={self.cfg.enable_domain_randomization} Latency={self.cfg.sim_latency_steps}")

    def _refresh_obstacle_cache(self, force: bool = False) -> None:
        if not bool(self.cfg.avoid_targets_in_obstacles):
            return

        now = time.time()
        if (not force) and self._obstacles_xy_r and (now - float(self._obstacles_last_refresh_t) < float(self.cfg.obstacle_cache_refresh_sec)):
            return

        obstacles = []
        # 1) Prefer explicit obstacles file (matches Unreal level generation config)
        try:
            p = str(self.cfg.obstacles_file or "").strip()
            if p:
                if not os.path.isabs(p):
                    # Allow repo-relative paths
                    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), p)
                if os.path.exists(p):
                    with open(p, "r") as f:
                        data = json.load(f)
                    off_x = float(data.get("obstacle_offset_x_m", 0.0))
                    off_y = float(data.get("obstacle_offset_y_m", 0.0))
                    for o in (data.get("obstacles") or []):
                        try:
                            x = float(o.get("x_m", 0.0)) + off_x
                            y = float(o.get("y_m", 0.0)) + off_y
                            if "scale" in o:
                                sx = float(o.get("scale", 1.0))
                                sy = float(o.get("scale", 1.0))
                            else:
                                sx = float(o.get("scale_x", 1.0))
                                sy = float(o.get("scale_y", 1.0))
                            # Approx XY radius using the rectangle diagonal (conservative under yaw).
                            r = 0.5 * float(math.sqrt(max(1e-6, sx * sx + sy * sy)))
                            if np.isfinite(x) and np.isfinite(y) and np.isfinite(r) and r > 0.0:
                                obstacles.append((x, y, r))
                        except Exception:
                            continue
        except Exception:
            # Fall back to scene query
            obstacles = []

        # 2) If file is not provided/usable, try to query spawned obstacle actors
        if not obstacles:
            try:
                names = self.client.simListSceneObjects(str(self.cfg.obstacle_actor_regex))
                for name in names or []:
                    try:
                        pose = self.client.simGetObjectPose(name)
                        scale = self.client.simGetObjectScale(name)
                        x = float(pose.position.x_val)
                        y = float(pose.position.y_val)
                        sx = abs(float(scale.x_val))
                        sy = abs(float(scale.y_val))
                        r = 0.5 * float(math.sqrt(max(1e-6, sx * sx + sy * sy)))
                        if np.isfinite(x) and np.isfinite(y) and np.isfinite(r) and r > 0.0:
                            obstacles.append((x, y, r))
                    except Exception:
                        continue
            except Exception:
                pass

        self._obstacles_xy_r = obstacles
        self._obstacles_last_refresh_t = now

    def _is_point_clear_of_obstacles(self, x: float, y: float) -> bool:
        if not bool(self.cfg.avoid_targets_in_obstacles):
            return True
        if not self._obstacles_xy_r:
            return True
        clearance = float(self.cfg.target_obstacle_clearance_m)
        for ox, oy, r in self._obstacles_xy_r:
            rr = float(r) + clearance
            dx = float(x) - float(ox)
            dy = float(y) - float(oy)
            if (dx * dx + dy * dy) <= (rr * rr):
                return False
        return True

    def _get_depth(self) -> np.ndarray:
        """Fetch and preprocess depth image."""
        # Use cached frame if we are skipping
        if self.step_count % self.vision_skip_frames != 0 and self.step_count > 0:
            return self.last_depth_frame

        t_start = time.time()
        try:
            resp = self.client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, pixels_as_float=True, compress=False)
            ], vehicle_name=self.cfg.vehicle_name)
        except Exception:
            return self.last_depth_frame
            
        dur = time.time() - t_start
        if dur > 0.2:
            print(f"[Lesnar Warning] Image capture took {dur:.2f}s (Sim is lagging)", end="\r")

        if not resp or resp[0].width == 0:
            return np.zeros((1, self.cfg.img_size, self.cfg.img_size), dtype=np.float32)

        # Raw float buffer
        img1d = np.array(resp[0].image_data_float, dtype=np.float32)
        img2d = img1d.reshape(resp[0].height, resp[0].width)
        
        # Resize
        resized = cv2.resize(img2d, (self.cfg.img_size, self.cfg.img_size), interpolation=cv2.INTER_AREA)
        
        # Clip & Normalize (Invert so closer = higher value? Standard practice is normalized distance 0..1)
        # linear normalized depth: 0.0 = contact, 1.0 = far.
        
        # Add sensor noise (Domain Randomization)
        if self.cfg.enable_domain_randomization:
            noise = np.random.normal(0, self.cfg.sensor_noise_sigma, resized.shape)
            resized += noise

        resized = np.nan_to_num(resized, nan=self.cfg.max_depth_clip_m)
        clipped = np.clip(resized, 0.0, self.cfg.max_depth_clip_m)
        norm = clipped / self.cfg.max_depth_clip_m
        
        # Channel-first for PyTorch
        frame = norm[None, :, :].astype(np.float32)
        
        # --- GENIUS PATCH: Stereo Blindness (Sim2Real) ---
        # Real sensors have holes. We kill 5% of pixels.
        # FIX: We set them to 1.0 (Max Depth/Safety) instead of 0.0 (Collision).
        # Setting to 0.0 causes the "min_depth" check to trigger constant Panic Rewards (-10k score).
        if self.cfg.enable_domain_randomization:
            # Create a mask where 5% are False (0)
            mask = np.random.choice([0, 1], size=frame.shape, p=[0.05, 0.95]).astype(np.float32)
            
            # Equation: frame * mask + (1-mask) * 1.0
            # If mask is 1: keeps frame value.
            # If mask is 0: becomes 0 + 1 * 1.0 = 1.0 (Safe/Far)
            frame = frame * mask + (1.0 - mask)
            
        self.last_depth_frame = frame
        return frame

    def _get_kinematics(self) -> np.ndarray:
        """Fetch state vector with synthesized noise."""
        state = self.client.getMultirotorState(vehicle_name=self.cfg.vehicle_name)
        kinematics = state.kinematics_estimated
        
        # Extract features
        vel = kinematics.linear_velocity
        ang_vel = kinematics.angular_velocity
        orient = kinematics.orientation
        pos = kinematics.position
        
        (pitch, roll, yaw) = airsim.to_eularian_angles(orient)
        
        # Altitude error (target - current)
        alt_error = self.cfg.cruise_alt_ned - pos.z_val
        
        # GENIUS UPDATE: Calculate Relative Target Vector (Body Frame)
        # This allows the drone to know "Target is Forward/Left/Right" regardless of global heading
        dx = self.target_position[0] - pos.x_val
        dy = self.target_position[1] - pos.y_val
        dist_to_target = math.sqrt(dx*dx + dy*dy)
        
        # Rotation into body frame
        c, s = math.cos(yaw), math.sin(yaw)
        body_x = c * dx + s * dy
        body_y = -s * dx + c * dy
        
        # --- GENIUS CALCULATIONS (The "Thinking Brain") ---
        # 1. Heading Error: Angle between current velocity and target vector
        # atan2(body_y, body_x) gives angle to target in body frame. 0 = straight ahead.
        heading_error = math.atan2(body_y, body_x)
        
        # 2. Track Error: Perpendicular distance from the ideal line (Start -> Target)
        # Simplified: cross track error logic requires start point. 
        # We use 'body_y' as local deviation from direct LOS.
        track_error = body_y 
        
        # 3. Time To Collision (TTC) based on Center Depth
        forward_speed = vel.x_val * c + vel.y_val * s # Projected speed
        center_depth = np.mean(self.last_depth_frame) * self.cfg.max_depth_clip_m # De-normalize
        ttc = 10.0 # Default max
        if forward_speed > 0.1 and center_depth < 10.0:
            ttc = center_depth / forward_speed
        
        # 4. Efficiency: Measures Battery / Power Consumption
        # Thrust is proportional to sqrt(vx^2 + vy^2 + (g+vz)^2)
        # We simplify: High Z-velocity (climbing) or High Yaw Rate = Low Efficiency
        # 1.0 = Hovering/Gliding. 0.0 = Full Throttle Climb + Spin.
        power_usage = abs(vel.z_val) if vel.z_val < 0 else 0.0 # Only count climbing
        power_usage += abs(ang_vel.z_val) * 0.5 
        efficiency = max(0.0, 1.0 - (power_usage / 3.0)) # Normalized roughly
        
        # 5. Proprioception: "What did I just do?"
        # Using self.prev_action (4D: vx, vy, vz, yaw)
        pa_vx = self.prev_action[0]
        pa_vy = self.prev_action[1]
        pa_vz = self.prev_action[2]
        pa_yr = self.prev_action[3]

        obs_vec = np.array([
            vel.x_val / self.cfg.max_speed_mps,
            vel.y_val / self.cfg.max_speed_mps,
            vel.z_val / 2.0,            
            roll,                       
            pitch,                      
            ang_vel.z_val,              
            alt_error / 5.0,            
            body_x / 50.0,              
            body_y / 50.0,              
            dist_to_target / 50.0,
            
            # Smart Metrics - FIXED: Using Sin/Cos for Continuity
            math.sin(heading_error),
            math.cos(heading_error),
            
            track_error / 20.0,    # Normalized Deviation
            min(ttc, 5.0) / 5.0,   # Normalized TTC
            efficiency,             # 0=unstable, 1=perfect
            
            # Proprioception
            pa_vx, pa_vy, pa_vz, pa_yr
        ], dtype=np.float32)

        # Add Sensor Noise
        if self.cfg.enable_domain_randomization:
            obs_vec += np.random.normal(0, self.cfg.sensor_noise_sigma, size=obs_vec.shape)

        return obs_vec.astype(np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        print(f"[Lesnar RL] resetting environment... Step count: {self.step_count}")
        options = options or {}
        keep_target = bool(options.get("keep_target", False))
        if "target_position" in options and options["target_position"] is not None:
            self.target_position = np.array(options["target_position"], dtype=np.float32)
            keep_target = True

        self.step_count = 0
        self.prev_action = np.zeros(4, dtype=np.float32)
        self.smoothed_action = np.zeros(4, dtype=np.float32)
        self._prev_cmd = np.zeros(4, dtype=np.float32)
        self._ignore_collision_steps = 0
        self.action_queue.clear()
        for _ in range(self.action_queue.maxlen):
            self.action_queue.append(np.zeros(4, dtype=np.float32))

        # AirSim Reset
        try:
            self.client.reset()
            self.client.enableApiControl(True, vehicle_name=self.cfg.vehicle_name)
            self.client.armDisarm(True, vehicle_name=self.cfg.vehicle_name)
            
            # GENIUS FEATURE: Wind Simulation (Robustness)
            # The "Falcon" must learn to fight wind gusts.
            if self.cfg.enable_domain_randomization:
                wx = np.random.uniform(-3, 3) # Up to 3m/s wind
                wy = np.random.uniform(-3, 3)
                wz = np.random.uniform(-1, 1) # Variable updrafts
                self.client.simSetWind(airsim.Vector3r(wx, wy, wz))
                print(f"[Lesnar Atmosphere] Wind Vector: X={wx:.1f}, Y={wy:.1f}, Z={wz:.1f}")
            
            # Use timeouts to prevent hanging
            f = self.client.takeoffAsync(timeout_sec=5)
            f.join()
            self.client.moveToZAsync(self.cfg.cruise_alt_ned, velocity=float(self.cfg.takeoff_climb_mps)).join()
        except Exception as e:
            print(f"[Lesnar RL] Reset failing ({e}), retrying...")
            return self.reset(seed=seed, options=options)

        pos = self.client.getMultirotorState().kinematics_estimated.position
        self.prev_location = np.array([pos.x_val, pos.y_val, pos.z_val], dtype=np.float32)
        self.start_location = self.prev_location.copy()

        if not keep_target:
            self._spawn_new_target()
        else:
            print(f"[Lesnar Brain] Keeping Target: {self.target_position}")

        # Initialize potential-based progress baseline
        pos = self.client.getMultirotorState().kinematics_estimated.position
        curr_loc = np.array([pos.x_val, pos.y_val, pos.z_val], dtype=np.float32)
        self.prev_dist_to_target = float(np.linalg.norm(curr_loc[:2] - self.target_position[:2]))
        
        print("[Lesnar RL] Reset complete. Playing...")
        return self._get_obs(), {}

    def _check_mission_override(self):
        """Checks for external commands from the UI/Backend."""
        try:
            # Locate shared/mission_override.json relative to this file
            p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'shared', 'mission_override.json')
            if os.path.exists(p):
                with open(p, 'r') as f:
                    d = json.load(f)
                t = d.get('timestamp', 0)
                # Only process new commands
                if t > self.last_override_time:
                    self.last_override_time = t
                    cmd = d.get('command')
                    print(f"\n[Lesnar Brain] INTERRUPT CMD: {cmd}") 
                    
                    if cmd == 'NAVIGATE':
                        tgt = d.get('target') # [x, y, z]
                        self.target_position = np.array(tgt, dtype=np.float32)
                        print(f"[Lesnar Brain] New Target Set: {self.target_position}")
                    
                    elif cmd == 'RETURN_HOME':
                        self.target_position = np.array([0.0, 0.0, -5.0], dtype=np.float32)
                        print("[Lesnar Brain] Returning Home")
                    
                    elif cmd == 'TAKEOFF':
                        tgt = d.get('target', [0, 0, -5])
                        curr = self.prev_location
                        self.target_position = np.array([curr[0], curr[1], tgt[2]], dtype=np.float32)
                        print(f"[Lesnar Brain] Adjusting Altitude to {tgt[2]}")

        except Exception as e:
            # Silent fail to avoid spamming logs
            pass

    def _spawn_new_target(self):
        """Generates a new mission target relative to current position or origin."""
        # Dynamic Task Generation
        # Optional: Return to Home (0,0)
        if float(self.cfg.return_home_prob) > 0.0 and np.random.random() < float(self.cfg.return_home_prob):
            self.target_position = np.array([0.0, 0.0, self.cfg.cruise_alt_ned], dtype=np.float32)
            print("[Lesnar Brain] New Order: RETURN TO HOME")
            return

        # Auto-leveling curriculum: spawn targets farther as the agent strings together successes.
        # This increases traveled distance and exploration without manual intervention.
        growth = float(self.cfg.target_spawn_growth_per_success_m) * float(self.consecutive_successes)
        grown_min = float(self.cfg.target_spawn_min_m) + growth
        grown_max = float(self.cfg.target_spawn_max_m) + growth
        max_total = float(self.cfg.target_spawn_max_total_m)
        if max_total > 0.0:
            grown_max = min(grown_max, max_total)
            grown_min = min(grown_min, grown_max)
        if grown_max - grown_min < 0.05:
            grown_min = max(0.0, grown_max - 0.05)

        # Optional: obstacle-driven spawn (pick a direction where depth sees something moderately close,
        # then place the target beyond it).
        spawn_mode = "ring"
        chosen_world_angle = None
        chosen_dist = None
        if bool(self.cfg.target_spawn_relative_to_drone) and float(self.cfg.obstacle_curriculum_prob) > 0.0:
            if np.random.random() < float(self.cfg.obstacle_curriculum_prob):
                try:
                    depth = self._get_depth()  # (1,H,W) normalized
                    dn = depth[0] if depth.ndim == 3 else depth
                    h, w = dn.shape
                    r0, r1 = int(0.40 * h), int(0.60 * h)
                    roi = dn[r0:r1, :]
                    # 10th percentile per column: conservative clearance
                    clearance_norm = np.percentile(roi, 10, axis=0) if roi.size > 0 else None
                    if clearance_norm is not None:
                        clearance_m = clearance_norm * float(self.cfg.max_depth_clip_m)
                        angles_deg = (np.arange(w, dtype=np.float32) / max(1, w - 1) - 0.5) * 90.0
                        low = float(self.cfg.obstacle_clearance_low_m)
                        high = float(self.cfg.obstacle_clearance_high_m)
                        mask = (clearance_m >= low) & (clearance_m <= high) & np.isfinite(clearance_m)
                        if np.any(mask):
                            idxs = np.where(mask)[0]
                            idx = int(np.random.choice(idxs))
                            body_angle_rad = float(np.deg2rad(float(angles_deg[idx])))
                            # Convert body-relative ray to world using current yaw
                            state_obj = self.client.getMultirotorState(vehicle_name=self.cfg.vehicle_name).kinematics_estimated
                            _, _, yaw = airsim.to_eularian_angles(state_obj.orientation)
                            chosen_world_angle = float(yaw + body_angle_rad)
                            # Place goal beyond the obstacle by a buffer
                            desired = float(clearance_m[idx]) + float(self.cfg.obstacle_target_extra_m)
                            chosen_dist = min(max_total if max_total > 0.0 else desired, desired)
                            # Ensure it's not shorter than the grown_min (keeps difficulty monotonic)
                            chosen_dist = max(chosen_dist, grown_min)
                            spawn_mode = "obstacle"
                except Exception:
                    pass

        # Resolve base (drone-relative or origin)
        base_x, base_y = 0.0, 0.0
        if bool(self.cfg.target_spawn_relative_to_drone):
            pos = self.client.getMultirotorState(vehicle_name=self.cfg.vehicle_name).kinematics_estimated.position
            base_x, base_y = float(pos.x_val), float(pos.y_val)

        # Prevent targets landing inside obstacles (common when the map has large walls/blocks)
        # by rejection sampling with a conservative radius approximation.
        self._refresh_obstacle_cache(force=False)

        max_attempts = 60
        rejected = 0
        target_x = None
        target_y = None
        target_mode = spawn_mode

        for attempt in range(max_attempts):
            # First attempt uses the previously chosen direction/distance (including obstacle-driven spawn).
            if attempt == 0 and chosen_world_angle is not None and chosen_dist is not None:
                angle = float(chosen_world_angle)
                dist = float(chosen_dist)
                mode = spawn_mode
            else:
                # Subsequent attempts: always fall back to normal ring spawn (avoid repeated depth RPC calls).
                angle = float(np.random.uniform(0, 2 * np.pi))
                dist = float(np.random.uniform(grown_min, grown_max))
                mode = "ring_resample"

            ox = float(math.cos(angle) * dist)
            oy = float(math.sin(angle) * dist)
            cand_x = float(base_x + ox)
            cand_y = float(base_y + oy)

            if not self._is_point_clear_of_obstacles(cand_x, cand_y):
                rejected += 1
                continue

            target_x, target_y, target_mode = cand_x, cand_y, mode
            break

        if target_x is None or target_y is None:
            # Fallback: accept the last candidate even if invalid, but surface it loudly.
            angle = float(np.random.uniform(0, 2 * np.pi))
            dist = float(np.random.uniform(grown_min, grown_max))
            target_x = float(base_x + math.cos(angle) * dist)
            target_y = float(base_y + math.sin(angle) * dist)
            target_mode = "ring_fallback"
            print(f"[Lesnar Warning] Could not find obstacle-free target after {max_attempts} attempts; using fallback.")

        self.target_position = np.array([
            float(target_x),
            float(target_y),
            float(self.cfg.cruise_alt_ned)
        ], dtype=np.float32)

        rej_txt = f" rejects={rejected}" if rejected > 0 else ""
        print(
            f"[Lesnar Brain] New Order: NAVIGATE TO ({self.target_position[0]:.1f}, {self.target_position[1]:.1f}) "
            f"streak={self.consecutive_successes} spawn={target_mode} spawn_r=[{grown_min:.1f},{grown_max:.1f}]{rej_txt}"
        )

    def _get_obs(self):
        return {
            "visual": self._get_depth(),
            "kinematics": self._get_kinematics()
        }

    def _land_and_retakeoff(self) -> None:
        # Ignore collision signals briefly while we intentionally touch down / lift off.
        try:
            self._ignore_collision_steps = max(
                int(float(self.cfg.collision_ignore_after_goal_sec) / max(1e-6, float(self.dt))),
                int(self._ignore_collision_steps),
            )
        except Exception:
            self._ignore_collision_steps = max(int(self._ignore_collision_steps), 20)
        try:
            self.client.hoverAsync(vehicle_name=self.cfg.vehicle_name).join()
        except Exception:
            pass

        # Soft touchdown handover (default): dip near ground, hold, then climb.
        if bool(self.cfg.goal_soft_touchdown):
            try:
                touchdown_z = -float(self.cfg.goal_touchdown_alt_m)
                self.client.moveToZAsync(
                    touchdown_z,
                    velocity=float(self.cfg.goal_touchdown_speed_mps),
                    vehicle_name=self.cfg.vehicle_name,
                ).join()
                time.sleep(float(self.cfg.goal_touchdown_hold_sec))
            except Exception:
                pass

        # Optional hard land (not recommended; can latch collision / cause sim-side resets)
        if bool(self.cfg.goal_use_land_async):
            try:
                self.client.landAsync(vehicle_name=self.cfg.vehicle_name).join()
            except Exception:
                pass

        # Re-takeoff / climb back to cruise altitude for the next leg
        try:
            self.client.enableApiControl(True, vehicle_name=self.cfg.vehicle_name)
            self.client.armDisarm(True, vehicle_name=self.cfg.vehicle_name)
            self.client.moveToZAsync(
                self.cfg.cruise_alt_ned,
                velocity=float(self.cfg.takeoff_climb_mps),
                vehicle_name=self.cfg.vehicle_name,
            ).join()
        except Exception:
            pass

    def _advance_mission_after_goal(self) -> None:
        """Advance to next target without a full AirSim reset."""
        # Success streak drives spawn distance growth
        self.consecutive_successes += 1

        try:
            pos = self.client.getMultirotorState(vehicle_name=self.cfg.vehicle_name).kinematics_estimated.position
            curr_loc = np.array([pos.x_val, pos.y_val, pos.z_val], dtype=np.float32)
        except Exception:
            curr_loc = self.prev_location if self.prev_location is not None else np.zeros(3, dtype=np.float32)

        # New leg baseline for geofence and progress
        self.start_location = curr_loc.copy()
        self.prev_location = curr_loc.copy()

        self._spawn_new_target()
        self.prev_dist_to_target = float(np.linalg.norm(curr_loc[:2] - self.target_position[:2]))

        # Reset action buffers so the next leg doesn't inherit stale commands
        self.prev_action = np.zeros(4, dtype=np.float32)
        self.smoothed_action = np.zeros(4, dtype=np.float32)
        self._prev_cmd = np.zeros(4, dtype=np.float32)
        self.action_queue.clear()
        for _ in range(self.action_queue.maxlen):
            self.action_queue.append(np.zeros(4, dtype=np.float32))

    def step(self, action: np.ndarray):
        self.step_count += 1
        
        # Check for UI Commands
        if self.step_count % 10 == 0: # Check at 2Hz for efficiency
            self._check_mission_override()

        if self.step_count % 10 == 0:
            print(f"Step {self.step_count}", end="\r")
        
        # 0. Latency Simulation (Minimal for Phase 1)
        # Add current AI command to queue, execute old command
        self.action_queue.append(action)
        delayed_action = self.action_queue.popleft() 
        
        # --- GENIUS "BRAIN" PRE-PROCESSING ---
        # The agent outputs "Intent" (vx, vy, vz, yaw_rate).
        # We process this Intent through a Safety Layer before Flight Controller.
        
        # 1. Action Smoothing (Simulate motor inertia / controller latency)
        # a_t = beta * a_{t-1} + (1-beta) * target
        beta = float(self.cfg.action_smoothing_beta)
        beta_yaw = float(self.cfg.action_smoothing_beta_yaw)
        if beta_yaw <= 0.0 or beta_yaw >= 1.0:
            beta_yaw = beta
        raw_action = np.clip(delayed_action, -1.0, 1.0)

        # Keep translation smooth but allow more responsive yaw if configured.
        self.smoothed_action[:3] = beta * self.smoothed_action[:3] + (1.0 - beta) * raw_action[:3]
        self.smoothed_action[3] = beta_yaw * self.smoothed_action[3] + (1.0 - beta_yaw) * raw_action[3]
        
        # Parse action: [vx, vy, vz, yaw_rate]
        target_vx = self.smoothed_action[0] * self.cfg.max_speed_mps
        target_vy = self.smoothed_action[1] * self.cfg.max_speed_mps
        target_vz = self.smoothed_action[2] * self.cfg.max_speed_mps # Up/Down speed
        target_yaw_rate = self.smoothed_action[3] * self.cfg.max_yaw_rate_dps
        
        # This is a "Reflex" that the RL learns to optimize, but we enforce physics limits.
        # (This prevents simple suicide runs)
        # For now, we let RL learn it, but we give massive penalties for ignoring it.
        
        # --- GUARDIAN LAYER INTERVENTION ---
        # The Guardian has VETO power over the RL Agent.
        # Get current physics state for safety/hold controllers
        state_now = self.client.getMultirotorState().kinematics_estimated
        c_pos = state_now.position
        c_z = float(c_pos.z_val)
        
        # Calculate min depth (denormalized)
        # NOTE: Raw depth frames can contain zeros/invalids; using a global min
        # makes the Guardian permanently slam the brakes.
        depth_norm = self.last_depth_frame
        min_d = float(self.cfg.max_depth_clip_m)
        p10_d = float(self.cfg.max_depth_clip_m)
        mean_d = float(self.cfg.max_depth_clip_m)
        if isinstance(depth_norm, np.ndarray) and depth_norm.size > 0:
            dn = depth_norm
            if dn.ndim == 3:
                dn = dn[0]
            h, w = dn.shape
            # MATCHING EXPERT VISION: 30-70% height (Reduced from 20-80 to avoid seeing own props during roll)
            r0, r1 = int(0.30 * h), int(0.70 * h)
            roi = dn[r0:r1, :]
            vals = roi.reshape(-1)
            vals = vals[np.isfinite(vals)]
            # Filter out zeros/negatives AND very close noise (self-occlusion/props)
            # Anything < 0.25m is likely the drone itself or a glitch, ignore it.
            vals = vals[vals > 0.25]
            if vals.size > 0:
                # MATCHING EXPERT CLEARANCE: 25th percentile to ignore light density noise
                min_d = float(np.percentile(vals, 25)) * float(self.cfg.max_depth_clip_m)
                p10_d = min_d
                mean_d = float(np.mean(vals)) * float(self.cfg.max_depth_clip_m)
        
        # Construct velocity vector for filter
        proposed_vel = np.array([target_vx, target_vy, target_vz, target_yaw_rate])
        safe_vel = self.guardian.filter_action(proposed_vel, min_depth=min_d, current_alt=c_z)
        
        # Override targets
        target_vx, target_vy, target_vz = safe_vel[0], safe_vel[1], safe_vel[2]

        # 1a. Optional altitude hold (only when not actively commanding vertical movement).
        if bool(self.cfg.enable_altitude_hold):
            # If the policy isn't trying to move vertically, hold cruise altitude.
            if abs(float(target_vz)) < 0.2:
                # NED: more positive Z = lower altitude.
                err_ned = float(self.cfg.cruise_alt_ned) - float(c_z)
                # Convert to meters since NED Z is meters.
                if abs(err_ned) > float(self.cfg.altitude_hold_deadband_m):
                    corr = float(self.cfg.altitude_hold_kp) * err_ned
                    corr = float(np.clip(corr, -float(self.cfg.altitude_hold_max_correction_mps), float(self.cfg.altitude_hold_max_correction_mps)))
                    target_vz = float(target_vz) + corr

        # 1a-b. Optional tilt protection: if we are leaned over heavily, reduce forward and recover altitude.
        if bool(self.cfg.enable_tilt_protection):
            roll, pitch, _yaw = airsim.to_eularian_angles(state_now.orientation)
            tilt_deg = float(max(abs(roll), abs(pitch)) * (180.0 / math.pi))
            if tilt_deg > float(self.cfg.tilt_protection_deg):
                target_vx = float(target_vx) * float(self.cfg.tilt_vx_scale)
                # Bias toward climbing a bit to avoid ground impacts while recovering.
                target_vz = min(float(target_vz), float(self.cfg.tilt_recover_vz_mps))

        # 1a-c. Optional ground cushion: near ground, reduce forward and clamp descent.
        if bool(self.cfg.enable_ground_cushion):
            altitude_m = -float(c_z)
            if altitude_m < float(self.cfg.ground_cushion_m):
                target_vx = float(target_vx) * float(self.cfg.ground_vx_scale)
                target_vy = float(target_vy) * float(self.cfg.ground_vx_scale)
                # In AirSim body-frame velocities: +Z is down. Clamp downward speed.
                if float(target_vz) > float(self.cfg.ground_max_descent_mps):
                    target_vz = float(self.cfg.ground_max_descent_mps)
                if altitude_m < float(self.cfg.ground_hard_stop_m):
                    target_vx = 0.0
                    target_vy = 0.0
                    # Don't descend further when extremely close to ground.
                    target_vz = min(float(target_vz), 0.0)

        # 1b. Optional command rate limiting (accel + yaw accel).
        # Apply AFTER safety so the Guardian's stop is still respected, but reduce jitter.
        if bool(self.cfg.enable_command_rate_limit):
            dt = float(self.dt)
            max_dv = float(self.cfg.max_accel_mps2) * dt
            max_dyaw = float(self.cfg.max_yaw_accel_dps2) * dt
            desired = np.array([float(target_vx), float(target_vy), float(target_vz), float(target_yaw_rate)], dtype=np.float32)
            prev = self._prev_cmd.astype(np.float32)
            delta = desired - prev
            delta[:3] = np.clip(delta[:3], -max_dv, max_dv)
            delta[3] = float(np.clip(delta[3], -max_dyaw, max_dyaw))
            limited = prev + delta
            target_vx, target_vy, target_vz, target_yaw_rate = float(limited[0]), float(limited[1]), float(limited[2]), float(limited[3])
            self._prev_cmd = limited
        else:
            self._prev_cmd = np.array([float(target_vx), float(target_vy), float(target_vz), float(target_yaw_rate)], dtype=np.float32)
        
        # 2. Execute Hierarchical Control (Velocity Command - FULL 3D)
        # Note: If target_vz is 0.0, we want to hover (maintain altitude).
        if abs(target_vz) < 0.05: target_vz = 0.0
        
        self.client.moveByVelocityBodyFrameAsync(
            vx=float(target_vx),
            vy=float(target_vy),
            vz=float(target_vz), 
            duration=self.dt,
            yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=float(target_yaw_rate)),
            vehicle_name=self.cfg.vehicle_name
        )
        
        # 3. Calculate Reward (Physics-Informed)
        col_info = self.client.simGetCollisionInfo()
        collision = bool(col_info.has_collided)
        if int(getattr(self, "_ignore_collision_steps", 0)) > 0:
            self._ignore_collision_steps -= 1
            collision = False
        
        # State delta
        pos_obj = self.client.getMultirotorState().kinematics_estimated.position
        curr_loc = np.array([pos_obj.x_val, pos_obj.y_val, pos_obj.z_val])
        
        # Distance-to-target (2D)
        dist_to_target = float(np.linalg.norm(curr_loc[:2] - self.target_position[:2]))
        if self.prev_dist_to_target is None:
            self.prev_dist_to_target = dist_to_target

        # Orientation + velocity for physics primitives
        state_obj = self.client.getMultirotorState().kinematics_estimated
        roll, pitch, yaw = airsim.to_eularian_angles(state_obj.orientation)
        vel_world = state_obj.linear_velocity
        c, s = math.cos(yaw), math.sin(yaw)
        v_body_x = c * vel_world.x_val + s * vel_world.y_val
        altitude_m = -float(curr_loc[2])

        # Target in body frame (for compass reward)
        dx = float(self.target_position[0] - curr_loc[0])
        dy = float(self.target_position[1] - curr_loc[1])
        body_x_to_target_m = c * dx + s * dy

        # Velocity toward goal (world-frame projection, normalized)
        v_toward_goal_norm = 0.0
        if dist_to_target > 1e-6:
            goal_dir_x = float((self.target_position[0] - curr_loc[0]) / dist_to_target)
            goal_dir_y = float((self.target_position[1] - curr_loc[1]) / dist_to_target)
            v_toward = float(vel_world.x_val) * goal_dir_x + float(vel_world.y_val) * goal_dir_y
            denom = float(self.cfg.max_speed_mps) if float(self.cfg.max_speed_mps) > 1e-6 else 1.0
            v_toward_goal_norm = float(np.clip(v_toward / denom, -1.0, 1.0))

        state = {
            "dist_m": dist_to_target,
            "prev_dist_m": float(self.prev_dist_to_target),
            "roll_rad": float(roll),
            "pitch_rad": float(pitch),
            "altitude_m": altitude_m,
            "v_body_x": float(v_body_x),
            "v_toward_goal_norm": float(v_toward_goal_norm),
            "body_x_to_target_m": float(body_x_to_target_m),
        }

        reward = self.reward_function.calculate(
            state=state,
            action=raw_action,
            prev_action=self.prev_action,
            collision=collision,
            step_count=self.step_count,
        )

        # --- OPTION B: EPISODIC TERMINATION ON SUCCESS ---
        reached_goal = dist_to_target < float(self.cfg.success_dist_m)
        if collision:
            reward += float(self.cfg.collision_extra_penalty)
        if reached_goal:
            reward += float(self.cfg.success_bonus)

            # Extra: reward smooth arrival (low speed + low yaw rate) without changing difficulty.
            speed_xy = float(math.sqrt(float(vel_world.x_val) ** 2 + float(vel_world.y_val) ** 2))
            yaw_rate_rad_s = float(state_obj.angular_velocity.z_val)
            yaw_rate_dps = abs(yaw_rate_rad_s) * (180.0 / math.pi)
            speed_term = max(0.0, 1.0 - (speed_xy / float(self.cfg.smooth_arrival_speed_mps)))
            yaw_term = max(0.0, 1.0 - (yaw_rate_dps / float(self.cfg.smooth_arrival_yaw_dps)))
            reward += float(self.cfg.smooth_arrival_bonus) * speed_term * yaw_term

        self.prev_dist_to_target = dist_to_target
        
        self.prev_location = curr_loc
        
        self.prev_action = raw_action
        
        # 4. Termination
        # In continuous mission mode, reaching goal does NOT end the episode.
        goal_event = bool(reached_goal and (not collision) and bool(self.cfg.continuous_mission))
        terminated = bool(collision or (reached_goal and not bool(self.cfg.continuous_mission)))

        # Geofence / leash
        geofence_triggered = False
        dynamic_geofence = None
        target_dist_from_start = float(np.linalg.norm(self.target_position[:2] - self.start_location[:2]))
        # IMPORTANT: when chaining missions, we update the per-leg start_location AFTER this step
        # (inside _advance_mission_after_goal). If we apply geofence on the exact goal step,
        # we can falsely trigger a reset back to the original episode start.
        if (not terminated) and (not goal_event) and float(self.cfg.geofence_radius_m) > 0.0:
            dist_from_start = float(np.linalg.norm(curr_loc[:2] - self.start_location[:2]))
            dynamic_geofence = max(
                float(self.cfg.geofence_radius_m),
                target_dist_from_start + float(self.cfg.geofence_buffer_m),
            )
            if dist_from_start > dynamic_geofence:
                geofence_triggered = True
                terminated = True
                reward += float(self.cfg.geofence_penalty)
                print(f"[EPISODE] Wandered outside {dynamic_geofence:.1f}m geofence. Resetting.")

        truncated = bool(self.step_count >= self.cfg.max_steps and not terminated)

        # Update success streak (used by the auto-leveling spawner)
        if terminated or truncated:
            # Only count a success if we actually reached the goal (and didn't collide).
            if reached_goal and not collision and not geofence_triggered:
                self.consecutive_successes += 1
            else:
                self.consecutive_successes = 0

        # Continuous mission: success advances to next target instead of terminating.
        if goal_event and (not geofence_triggered):
            if bool(self.cfg.land_on_goal):
                self._land_and_retakeoff()
            self._advance_mission_after_goal()

        termination_reason = "none"
        if collision:
            termination_reason = "collision"
        elif reached_goal:
            termination_reason = "goal"
        elif geofence_triggered:
            termination_reason = "geofence"
        elif truncated:
            termination_reason = "time_limit"

        if terminated or truncated:
            dg = float(dynamic_geofence) if dynamic_geofence is not None else float(max(float(self.cfg.geofence_radius_m), target_dist_from_start + float(self.cfg.geofence_buffer_m)))
            print(
                f"[EPISODE END] reason={termination_reason} step={self.step_count} "
                f"dist_to_target={dist_to_target:.2f} dist_from_start={float(np.linalg.norm(curr_loc[:2] - self.start_location[:2])):.2f} "
                f"target_from_start={target_dist_from_start:.2f} geofence={dg:.2f}"
            )
        
        info = {
            "collision": collision,
            "velocity": float(math.sqrt(float(vel_world.x_val) ** 2 + float(vel_world.y_val) ** 2 + float(vel_world.z_val) ** 2)),
            "reached_goal": reached_goal,
            "goal_event": goal_event,
            "dist_to_target": dist_to_target,
            "dist_from_start": float(np.linalg.norm(curr_loc[:2] - self.start_location[:2])),
            "target_dist_from_start": float(target_dist_from_start),
            "dynamic_geofence": float(max(float(self.cfg.geofence_radius_m), float(target_dist_from_start) + float(self.cfg.geofence_buffer_m))),
            "termination_reason": termination_reason,
            "geofence_triggered": geofence_triggered,
            "depth_p10_m": float(p10_d),
            "depth_mean_m": float(mean_d),
            "guardian_braking": bool(getattr(self.guardian, "last_braking", False)),
            "guardian_emergency": bool(getattr(self.guardian, "last_emergency", False)),
            "guardian_min_depth_m": float(getattr(self.guardian, "last_min_depth_m", float("nan"))),
            "guardian_effective_depth_m": float(getattr(self.guardian, "last_effective_depth_m", float("nan"))),
            "cmd_vx": float(self._prev_cmd[0]),
            "cmd_vy": float(self._prev_cmd[1]),
            "cmd_vz": float(self._prev_cmd[2]),
            "cmd_yaw_rate_dps": float(self._prev_cmd[3]),
        }
        
        return self._get_obs(), reward, terminated, truncated, info

    def close(self):
        try:
            self.client.armDisarm(False)
        except Exception:
            pass
        try:
            self.client.enableApiControl(False)
        except Exception:
            pass
