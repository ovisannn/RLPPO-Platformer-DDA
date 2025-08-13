# file: ppo_platformer_balancer_sb3_vecnorm_v2.py
from __future__ import annotations
import csv
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import utils

import numpy as np
import matplotlib.pyplot as plt

import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO as SB3PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize


# =========================
# Config
# =========================

@dataclass
class SimConfig:
    episode_len: int = 64
    noise_scale: float = 0.12
    target_deaths: float = 1.0
    target_powerups_used: float = 2.0
    target_time_ratio: float = 0.9

    # Difficulty weights
    w_enemies: float = 0.75
    w_powerups: float = -0.55
    w_time_limit: float = -0.22

    # Discrete bins (bold steps)
    enemies_set: Tuple[int, ...] = (-10, -6, -2, 2, 6, 10)
    powerups_set: Tuple[int, ...] = (-10, -4, 0, 2, 4, 8)
    time_bins_start: int = 100
    time_bins_stop: int = 250
    time_bins_step: int = 20  # 100,120,...,300

    # Base loss weights
    base_death_w: float = 1.0
    base_pu_w: float = 0.6
    base_time_w: float = 0.8

    # Directional shaping (stronger, explicit weights)
    directional_cap: float = 2.0
    # Enemies adjustment dominates deaths correction (your request)
    dir_w_death_enemies: float = 1.4
    dir_w_death_powerups: float = 0.9
    dir_w_death_time: float = 0.6
    dir_w_powerups: float = 0.8
    dir_w_time: float = 0.8
    dir_global_gain: float = 0.9  # global multiplier

    # Anti-extreme regularizers
    reg_enemy_l1: float = 0.15
    reg_power_l1: float = 0.25
    reg_time_quad: float = 0.4

    # Boldness bonus (prefer mid-strong moves)
    bold_center: float = 0.5
    bold_width: float = 0.6
    bold_gain: float = 0.15


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


# =========================
# Core Simulator
# =========================

class CorePlatformerSim:
    """Simple difficulty model with diminishing returns; noise keeps policy flexible."""
    def __init__(self, cfg: SimConfig | None = None):
        self.cfg = cfg or SimConfig()
        self.player_skill: float = 0.0
        self.powerup_preference: float = 1.0
        self.t: int = 0

    def reset(self) -> np.ndarray:
        self.player_skill = float(np.random.uniform(0.2, 1.2))
        self.powerup_preference = float(np.random.uniform(0.5, 1.5))
        self.t = 0
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)

    def _simulate_outcome(self, enemies: int, powerups_avail: int, time_limit: int) -> Tuple[float, float, float]:
        # Diminishing returns on powerups/time -> prevents "always max"
        powerups_term = self.cfg.w_powerups * np.tanh((powerups_avail + 10) / 10.0)
        time_term = self.cfg.w_time_limit * np.tanh((time_limit - 100) / 80.0)
        diff = (self.cfg.w_enemies * (enemies / 10.0) + powerups_term + time_term)
        skill_margin = self.player_skill - diff

        # Outcomes (noisy)
        base_deaths = np.clip(1.6 - 1.25 * skill_margin, 0.0, 10.0)
        pu_pressure = np.clip(0.3 + 0.5 * np.maximum(0.0, -skill_margin), 0.2, 1.2)
        expected_pu_use = self.powerup_preference * pu_pressure * np.tanh(max(0.0, powerups_avail + 10) / 8.0) * 2.2
        time_ratio = np.clip(0.65 + 0.3 * (1.0 - np.tanh(1.8 * np.maximum(0.0, -skill_margin))), 0.4, 1.2)

        noise = np.random.normal(0.0, self.cfg.noise_scale, size=3)
        deaths = float(np.clip(base_deaths + noise[0], 0.0, 10.0))
        powerups_used = float(np.clip(expected_pu_use + noise[1], 0.0, max(0.0, powerups_avail + 10)))
        level_time = float(np.clip(time_ratio * time_limit + noise[2] * 10.0, 0.0, time_limit * 1.5))
        return deaths, powerups_used, level_time

    def step(self, enemies: int, powerups_avail: int, time_limit: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        self.t += 1
        deaths, powerups_used, level_time = self._simulate_outcome(enemies, powerups_avail, time_limit)

        # Base objective: near targets
        death_pen = (deaths - self.cfg.target_deaths) ** 2
        pu_pen = (powerups_used - self.cfg.target_powerups_used) ** 2
        time_pen = ((level_time / max(1.0, time_limit)) - self.cfg.target_time_ratio) ** 2
        reward = - (self.cfg.base_death_w * death_pen +
                    self.cfg.base_pu_w * pu_pen +
                    self.cfg.base_time_w * time_pen)

        # ===== Directional shaping (weighted; scaled by error; damped near edges) =====
        s_death = np.sign(deaths - self.cfg.target_deaths)        # +1 too many deaths; -1 too few deaths
        s_pu    = np.sign(powerups_used - self.cfg.target_powerups_used)  # +1 too many used
        s_time  = np.sign(level_time / max(1.0, time_limit) - self.cfg.target_time_ratio)  # +1 too slow

        # Normalize chosen action to [-1,1] for magnitude reasoning
        ne = enemies / 10.0
        npw = powerups_avail / 10.0
        nt = (time_limit - 200) / 100.0

        # Error magnitudes (emphasize deaths -> your request)
        death_gap = abs(deaths - self.cfg.target_deaths)
        pu_gap = abs(powerups_used - self.cfg.target_powerups_used)
        time_gap = abs(level_time / max(1.0, time_limit) - self.cfg.target_time_ratio)

        # Damp near extremes so policy prefers bold mid bins over edges
        edge_damp = (1.0 - 0.5 * (abs(ne) + abs(npw) + abs(nt)) / 3.0)

        # Core rule: more enemies if fewer deaths; fewer enemies if too many deaths.
        # Additional rules: adjust powerups/time in the opposite sense of deaths; adjust their own errors too.
        corr = (
            # Death-driven adjustments (dominant)
            self.cfg.dir_w_death_enemies * (-s_death) * ne * (0.6 + 0.4 * np.tanh(death_gap))
            + self.cfg.dir_w_death_powerups * ( s_death) * npw * (0.6 + 0.4 * np.tanh(death_gap))
            + self.cfg.dir_w_death_time    * ( s_death) * nt * (0.6 + 0.4 * np.tanh(death_gap))

            # Powerup-usage error: if too low, increase availability; if too high, reduce.
            + self.cfg.dir_w_powerups * (-s_pu) * npw * (0.5 + 0.5 * np.tanh(pu_gap))

            # Time error: if too fast (ratio < target) → reduce time (harder); if too slow → increase time.
            + self.cfg.dir_w_time * (-s_time) * nt * (0.5 + 0.5 * np.tanh(time_gap))
        ) * edge_damp * self.cfg.dir_global_gain

        reward += np.clip(corr, -self.cfg.directional_cap, self.cfg.directional_cap)

        # Anti-extreme regularizers (soft)
        reward -= self.cfg.reg_enemy_l1 * abs(ne)
        reward -= self.cfg.reg_power_l1 * abs(npw)
        reward -= self.cfg.reg_time_quad * (nt ** 2)

        # Boldness bonus: mid-strong actions
        mag = (abs(ne) + abs(npw) + abs(nt)) / 3.0
        boldness = -((mag - self.cfg.bold_center) ** 2) / (self.cfg.bold_width ** 2)
        reward += self.cfg.bold_gain * boldness

        obs = np.array([
            deaths,
            powerups_used,
            (level_time / max(1.0, time_limit)),
        ], dtype=np.float32)

        done = self.t >= self.cfg.episode_len
        info = {
            "enemies": enemies,
            "powerups": powerups_avail,
            "time_limit": time_limit,
            "raw_outcome": (deaths, powerups_used, level_time),
        }
        return obs, float(reward), bool(done), info


# =========================
# Gymnasium Env
# =========================

class PlatformerEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg: SimConfig | None = None):
        super().__init__()
        self.cfg = cfg or SimConfig()
        self.core = CorePlatformerSim(self.cfg)

        self.enemies_set = np.array(self.cfg.enemies_set, dtype=np.int32)
        self.powerups_set = np.array(self.cfg.powerups_set, dtype=np.int32)
        self.time_set = np.arange(self.cfg.time_bins_start, self.cfg.time_bins_stop + 1, self.cfg.time_bins_step, dtype=np.int32)

        # Raw bounds; VecNormalize handles scaling
        high = np.array([10.0, 20.0, 2.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=0.0, high=high, shape=(3,), dtype=np.float32)

        self.action_space = spaces.MultiDiscrete([len(self.enemies_set),
                                                  len(self.powerups_set),
                                                  len(self.time_set)])
        self._t = 0
        self._max_steps = self.cfg.episode_len

    def _map_action(self, a: np.ndarray) -> Tuple[int, int, int]:
        e = self.enemies_set[int(a[0])]
        p = self.powerups_set[int(a[1])]
        t = self.time_set[int(a[2])]
        return int(e), int(p), int(t)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            set_seed(seed)
        self._t = 0
        return self.core.reset(), {}

    def step(self, action):
        self._t += 1
        enemies, powerups_avail, time_limit = self._map_action(np.asarray(action))
        obs, reward, done, info = self.core.step(enemies, powerups_avail, time_limit)
        terminated = done
        truncated = self._t >= self._max_steps
        return obs, reward, bool(terminated), bool(truncated and not terminated), info


# =========================
# SB3 Training + Logging
# =========================

class RewardLogger(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards: List[float] = []
    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        if infos is not None:
            for info in infos:
                if isinstance(info, dict) and "episode" in info:
                    self.episode_rewards.append(float(info["episode"]["r"]))
        return True


def train_sb3(seed: int = 13,
              total_timesteps: int = 700_000,
              n_envs: int = 16, #8
              ent_coef: float = 0.06,
              out_model_path: str = "ppo_platformer_sb3_vecnorm_v2.zip",
              norm_path: str = "vecnormalize_sb3_v2.pkl",
              rewards_csv: str = "rewards_sb3_vecnorm_v2.csv",
              rewards_png: str = "rewards_sb3_vecnorm_v2.png") -> Dict[str, Any]:
    set_seed(seed)

    def make_env_fn():
        def _thunk():
            env = PlatformerEnv()
            env = Monitor(env)
            return env
        return _thunk

    vec_env = make_vec_env(make_env_fn(), n_envs=n_envs, seed=seed)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    model = SB3PPO(
        "MlpPolicy", vec_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=512,
        n_epochs=10,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=ent_coef,  # higher entropy to avoid single-bin collapse
        policy_kwargs=dict(net_arch=[128, 128], ortho_init=False),
        verbose=1,
        seed=seed
    )

    cb = RewardLogger()
    model.learn(total_timesteps=total_timesteps, callback=cb)
    model.save(out_model_path)
    vec_env.save(norm_path)

    print(f"Saved SB3 model to {out_model_path}")
    print(f"Saved VecNormalize stats to {norm_path}")

    if cb.episode_rewards:
        with open(rewards_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["episode_reward"])
            for r in cb.episode_rewards:
                w.writerow([float(r)])
        print(f"Saved rewards to {rewards_csv}")

        # plt.figure()
        # plt.plot(cb.episode_rewards)
        # plt.title("SB3 (VecNormalize v2) Training Rewards")
        # plt.xlabel("Episode")
        # plt.ylabel("Return")
        # plt.tight_layout()
        # plt.savefig(rewards_png, dpi=160)
        # plt.close()
        # print(f"Saved rewards plot to {rewards_png}")
        utils.plot_rewards_with_smoothing(
            cb.episode_rewards,
            rewards_png,
            window=50,
            alpha=None,
            title="SB3 (VecNormalize v2) Training Rewards",
        )

    return {"rewards": cb.episode_rewards, "model_path": out_model_path, "norm_path": norm_path}


# =========================
# Inference
# =========================

def load_normalizer(norm_path: str, seed: int = 0):
    # Freeze stats for inference
    def make_env():
        return PlatformerEnv()
    venv = make_vec_env(make_env, n_envs=1, seed=seed)
    venv = VecNormalize.load(norm_path, venv)
    venv.training = False
    venv.norm_reward = False
    return venv

def predict_next_level(model: SB3PPO,
                       normalizer: VecNormalize,
                       obs: Tuple[float, float, float],
                       prev_time_limit: float,
                       deterministic: bool = True) -> Dict[str, int]:
    raw = np.array([obs[0], obs[1], obs[2] / max(1.0, prev_time_limit)], dtype=np.float32)
    norm_obs = normalizer.normalize_obs(raw.reshape(1, -1)).reshape(-1)
    a, _ = model.predict(norm_obs, deterministic=deterministic)

    cfg = SimConfig()
    enemies_set = np.array(cfg.enemies_set, dtype=np.int32)
    powerups_set = np.array(cfg.powerups_set, dtype=np.int32)
    time_set = np.arange(cfg.time_bins_start, cfg.time_bins_stop + 1, cfg.time_bins_step, dtype=np.int32)

    enemies = int(enemies_set[int(a[0])])
    powerups = int(powerups_set[int(a[1])])
    time_limit = int(time_set[int(a[2])])
    return {"total_enemies": enemies, "total_power_ups": powerups, "time_limit": time_limit}


# =========================
# Main
# =========================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PPO Platformer Balancer — SB3 + VecNormalize (v2 with weighted shaping)")
    parser.add_argument("--timesteps", type=int, default=700_000)
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--ent", type=float, default=0.06)
    parser.add_argument("--stochastic", action="store_true", help="sample at inference for variety")
    args = parser.parse_args()

    results = train_sb3(seed=args.seed,
                        total_timesteps=args.timesteps,
                        n_envs=args.envs,
                        ent_coef=args.ent)
    model = SB3PPO.load(results["model_path"])
    normalizer = load_normalizer(results["norm_path"], seed=args.seed)

    tests = [
        ((3.0, 1.0, 160.0), 180.0),
        ((0.0, 0.5, 120.0), 150.0),
        ((1.0, 3.0, 210.0), 240.0),
        ((4.0, 0.0, 200.0), 220.0),
        ((0.0, 5.0, 90.0), 120.0),
    ]
    for (inp, prev_lim) in tests:
        out = predict_next_level(model, normalizer, inp, prev_time_limit=prev_lim,
                                 deterministic=not args.stochastic)
        print(f"Input {inp} (prev_limit={prev_lim}) -> {out}")
