"""
FrozenLake avancé

- taille variable (4x4, 8x8, 12x12, ou carte custom)
- glissant (stochastic) ou déterministe
- reward shaping optionnel (dense) en plus de la récompense sparse de base
- wrappers utiles (TimeLimit, RecordEpisodeStatistics)

Dans le notebook:
    from src.env.frozenlake_env import make_frozenlake, FrozenLakeConfig
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Union

import gymnasium as gym
import numpy as np
from gymnasium import Env
from gymnasium.wrappers import RecordEpisodeStatistics, TimeLimit


MapName = Union[str, None]
CustomMap = Optional[List[str]]


@dataclass
class FrozenLakeConfig:
    map_name: MapName = "4x4"      # "4x4", "8x8", ... ou None si desc personnalisée
    desc: CustomMap = None         # ex: ["SFFF","FHFH","FFFH","HFFG"]
    is_slippery: bool = True
    max_episode_steps: int = 200

    # Reward shaping (optionnel)
    shaped: bool = False
    step_penalty: float = -0.001   # pénalité à chaque pas (trajets plus courts)
    hole_penalty: float = -0.05    # pénalité si on tombe dans un trou
    goal_bonus: float = 1.0        # bonus à l'arrivée (en + du reward natif)

    # Reproductibilité
    seed: Optional[int] = 0


class RewardShapingWrapper(gym.Wrapper):
    """Ajoute un reward shaping simple à FrozenLake."""

    def __init__(
        self,
        env: Env,
        step_penalty: float = -0.001,
        hole_penalty: float = -0.05,
        goal_bonus: float = 1.0,
    ):
        super().__init__(env)
        self.step_penalty = float(step_penalty)
        self.hole_penalty = float(hole_penalty)
        self.goal_bonus = float(goal_bonus)

        base = env.unwrapped
        self._desc = np.array(base.desc, dtype="U1")  # grille 'S','F','H','G'
        self._ncol = base.ncol

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        shaped_reward = float(reward) + self.step_penalty

        if terminated:
            r = int(obs) // self._ncol
            c = int(obs) % self._ncol
            cell = self._desc[r, c]
            if cell == "H":
                shaped_reward += self.hole_penalty
            elif cell == "G":
                shaped_reward += self.goal_bonus

        return obs, shaped_reward, terminated, truncated, info


def make_frozenlake(cfg: FrozenLakeConfig) -> Env:
    """Crée un environnement FrozenLake selon la configuration."""
    if cfg.desc is not None and cfg.map_name is not None:
        raise ValueError("Choisis soit map_name, soit desc (pas les deux).")

    env = gym.make(
        "FrozenLake-v1",
        map_name=cfg.map_name,
        desc=cfg.desc,
        is_slippery=cfg.is_slippery,
        render_mode=None,
    )

    if cfg.seed is not None:
        env.reset(seed=cfg.seed)
        env.action_space.seed(cfg.seed)
        env.observation_space.seed(cfg.seed)

    env = TimeLimit(env, max_episode_steps=cfg.max_episode_steps)
    env = RecordEpisodeStatistics(env)

    if cfg.shaped:
        env = RewardShapingWrapper(
            env,
            step_penalty=cfg.step_penalty,
            hole_penalty=cfg.hole_penalty,
            goal_bonus=cfg.goal_bonus,
        )

    return env
