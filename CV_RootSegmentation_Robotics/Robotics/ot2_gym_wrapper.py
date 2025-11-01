from __future__ import annotations
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import pybullet as p
from sim_class import Simulation


@dataclass
class Envelope:
    x_min: float; x_max: float
    y_min: float; y_max: float
    z_min: float; z_max: float

    @classmethod
    def from_dict(cls, d: Dict) -> "Envelope":
        return cls(
            d["x"]["min"], d["x"]["max"],
            d["y"]["min"], d["y"]["max"],
            d["z"]["min"], d["z"]["max"],
        )

    def as_box_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        low  = np.array([self.x_min, self.y_min, self.z_min], dtype=np.float32)
        high = np.array([self.x_max, self.y_max, self.z_max], dtype=np.float32)
        return low, high


class _SafeSim:
    """Small safety wrapper that (re)creates Simulation if PyBullet disconnects."""
    def __init__(self):
        self.sim: Optional[Simulation] = None
        self.ensure_connected()

    @staticmethod
    def _connected() -> bool:
        try:
            info = p.getConnectionInfo()
            return bool(info) and bool(info.get("isConnected", 0))
        except Exception:
            return False

    def ensure_connected(self):
        if self.sim is None or not self._connected():
            self.sim = Simulation(num_agents=1)

    def run(self, actions, num_steps: int = 1):
        try:
            return self.sim.run(actions, num_steps=num_steps)
        except Exception:
            self.ensure_connected()
            return self.sim.run(actions, num_steps=num_steps)

    def tip_pos(self, state) -> Tuple[float, float, float]:
        key = f"robotId_{self.sim.robotIds[0]}"
        x, y, z = state[key]["pipette_position"]
        return float(x), float(y), float(z)


class OT2PipetteReachEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 60}

    def __init__(
        self,
        envelope: Optional[Dict] = None,
        load_envelope_from: str = "working_envelope.json",
        max_episode_steps: int = 400,
        success_eps: float = 5e-3,
        vel_max: float = 0.5,
        inner_steps_per_action: int = 1,
        success_bonus: float = 1.0,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        # Envelope
        if envelope is None and os.path.exists(load_envelope_from):
            with open(load_envelope_from) as f:
                envelope = json.load(f)
        if envelope is None:
            envelope = {  # fallback to the measured values you obtained in Task 9
                "x": {"min": -0.187, "max": 0.253},
                "y": {"min": -0.1705, "max": 0.2195},
                "z": {"min":  0.1195, "max": 0.2898},
            }
        self.envelope = Envelope.from_dict(envelope)
        self.env_low, self.env_high = self.envelope.as_box_bounds()

        # Spaces
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        # observation: [tip(3), target(3), delta(3)]
        low_obs  = np.concatenate([self.env_low,  self.env_low,  -np.abs(self.env_high - self.env_low)])
        high_obs = np.concatenate([self.env_high, self.env_high,  np.abs(self.env_high - self.env_low)])
        self.observation_space = spaces.Box(low=low_obs.astype(np.float32),
                                            high=high_obs.astype(np.float32),
                                            dtype=np.float32)

        # Dynamics / reward params
        self.max_episode_steps = int(max_episode_steps)
        self.SUCCESS_EPS = float(success_eps)
        self.SUCCESS_BONUS = float(success_bonus)
        self.VEL_MAX = float(vel_max)
        self.inner_steps = int(inner_steps_per_action)

        # Sim
        self.sim = _SafeSim()

        # Episode state
        self.rng = np.random.default_rng(seed)
        self.render_mode = render_mode
        self.step_count = 0
        self.target = self._sample_target()

    # --------------- helpers ---------------
    def _sample_target(self) -> np.ndarray:
        return self.rng.uniform(self.env_low, self.env_high).astype(np.float32)

    def _read_tip(self) -> np.ndarray:
        st = self.sim.run([[0.0, 0.0, 0.0, 0]], num_steps=1)
        return np.array(self.sim.tip_pos(st), dtype=np.float32)

    def _obs(self, tip: np.ndarray) -> np.ndarray:
        delta = (self.target - tip).astype(np.float32)
        return np.concatenate([tip, self.target, delta]).astype(np.float32)

    # --------------- Gym API ---------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.step_count = 0
        # Optionally allow external target override
        if options and "target" in options:
            self.target = np.clip(np.array(options["target"], dtype=np.float32), self.env_low, self.env_high)
        else:
            self.target = self._sample_target()
        tip = self._read_tip()  # starting pose (whatever the sim is currently at)
        obs = self._obs(tip)
        info = {"target": self.target.copy()}
        return obs, info

    def step(self, action: np.ndarray):
        self.step_count += 1
        action = np.asarray(action, dtype=np.float32).clip(-1.0, 1.0)
        vx, vy, vz = (action * self.VEL_MAX).tolist()

        # apply action for inner_steps
        state = None
        for _ in range(self.inner_steps):
            state = self.sim.run([[float(vx), float(vy), float(vz), 0]], num_steps=1)

        tip = np.array(self.sim.tip_pos(state), dtype=np.float32)
        delta = self.target - tip
        dist = float(np.linalg.norm(delta))

        reward = -dist
        terminated = False
        if dist <= self.SUCCESS_EPS:
            reward += self.SUCCESS_BONUS
            terminated = True

        truncated = self.step_count >= self.max_episode_steps
        obs = np.concatenate([tip, self.target.astype(np.float32), delta.astype(np.float32)]).astype(np.float32)

        info = {
            "distance": dist,
            "target": self.target.copy(),
            "tip": tip.copy(),
        }

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self):
        # With the provided twin, GUI is already visible when Simulation is created.
        # Here we optionally print a tiny status line.
        tip = self._read_tip()
        d = float(np.linalg.norm(self.target - tip))
        print(f"[render] step={self.step_count} tip={tip.tolist()} target={self.target.tolist()} dist={d:.5f}")

    def close(self):
        # The twin manages its own shutdown when the process ends; nothing to do here.
        pass