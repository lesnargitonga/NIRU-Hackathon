"""
Professional Training Script for Lesnar AI (RL Phase).
Uses PPO with Multi-Input Policy (CNN for Depth + MLP for Kinematics).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import time
from collections import deque

import torch
import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.logger import configure

from airsim_gym_env import AirSimDroneEnv, EnvConfig


class OutcomeStatsCallback(BaseCallback):
    def __init__(self, window_episodes: int = 200, log_freq_steps: int = 2048, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.window_episodes = int(window_episodes)
        self.log_freq_steps = int(log_freq_steps)
        self._successes = deque(maxlen=self.window_episodes)
        self._collisions = deque(maxlen=self.window_episodes)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        dones = self.locals.get("dones")
        if isinstance(infos, (list, tuple)) and isinstance(dones, (list, tuple)):
            for info, done in zip(infos, dones):
                if not done or not isinstance(info, dict):
                    continue
                reached = bool(info.get("reached_goal", False))
                collided = bool(info.get("collision", False))
                self._successes.append(1.0 if reached else 0.0)
                self._collisions.append(1.0 if collided else 0.0)

        if self.log_freq_steps > 0 and (self.num_timesteps % self.log_freq_steps) == 0:
            if len(self._successes) > 0:
                self.logger.record("train/success_rate", float(sum(self._successes)) / float(len(self._successes)))
            if len(self._collisions) > 0:
                self.logger.record("train/collision_rate", float(sum(self._collisions)) / float(len(self._collisions)))
            self.logger.dump(self.num_timesteps)

        return True


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=500_000, help="Total training steps")
    ap.add_argument("--img_size", type=int, default=84, help="Input size for CNN")
    ap.add_argument("--hz", type=float, default=20.0, help="Control loop rate")
    ap.add_argument("--outdir", type=str, default="runs/ppo_lesnar_v1")
    ap.add_argument("--load", type=str, default="", help="Path to pretrained model to resume")
    return ap.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    logdir = outdir / "logs"
    modeldir = outdir / "models"
    logdir.mkdir(parents=True, exist_ok=True)
    modeldir.mkdir(parents=True, exist_ok=True)

    # 1. Configure the Environment
    # Note: We use the professional config defaults defined in airsim_gym_env.py
    cfg = EnvConfig(
        hz=args.hz,
        img_size=args.img_size,
        enable_domain_randomization=False, # [PHASE 1: CRAWLER MODE]
        max_speed_mps=1.0 # Explicitly enforcing Phase 1
    )
    
    # Wrap env: Monitor for logging stats, DummyVecEnv for SB3, VecTransposeImage for PyTorch channels
    def make_env():
        env = AirSimDroneEnv(config=cfg)
        env = Monitor(env, str(logdir))
        return env

    env = DummyVecEnv([make_env])
    # VecTransposeImage handles the visual dict key if it's the only image, 
    # but for MultiInput, SB3 handles channel-first automatically if defined in observation space.
    # Our env outputs (1, 84, 84) which is already C,H,W. SB3 handles this.

    # 2. Configure PPO Agent (Recurrent / LSTM)
    # MultiInputLstmPolicy allows concurrent processing of Visual (CNN) and Kinematics (MLP) with Memory
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]), # Larger networks for complex behavior
        optimizer_kwargs=dict(eps=1e-5),
        lstm_hidden_size=256,
        enable_critic_lstm=False # Save compute, only Actor needs memory
    )
    
    if args.load and os.path.exists(args.load):
        print(f"Loading pretrained model from {args.load}")
        model = RecurrentPPO.load(args.load, env=env)

        # IMPORTANT: SB3 restores the original logger/tensorboard path from the checkpoint.
        # Force logging into this run's outdir so TensorBoard doesn't mix experiments.
        new_logger = configure(str(logdir), ["stdout", "tensorboard"])
        model.set_logger(new_logger)
        model.tensorboard_log = str(logdir)
    else:
        print("Initializing new RecurrentPPO agent (Genius Config: LSTM Enabled)...")
        model = RecurrentPPO(
            "MultiInputLstmPolicy",
            env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.05, # Encourage exploration (Increased from 0.01 to fix policy collapse)
            verbose=1,
            tensorboard_log=str(logdir),
            policy_kwargs=policy_kwargs,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

    # 3. Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=str(modeldir),
        name_prefix="ppo_lesnar"
    )

    outcome_callback = OutcomeStatsCallback(window_episodes=200, log_freq_steps=2048)
    callback = CallbackList([checkpoint_callback, outcome_callback])

    # 4. Train
    print(f"Starting training on {model.device} for {args.timesteps} timesteps...")
    try:
        model.learn(total_timesteps=args.timesteps, callback=callback)
        model.save(str(modeldir / "final_model"))
        print("Training complete.")
    except KeyboardInterrupt:
        print("Training interrupted. Saving current model...")
        model.save(str(modeldir / "interrupted_model"))
    finally:
        env.close()

if __name__ == "__main__":
    main()
