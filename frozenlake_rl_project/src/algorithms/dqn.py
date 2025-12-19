from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from gymnasium import Env
import numpy as np


def train_dqn(
    env: Env,
    total_timesteps: int = 20000,
    learning_rate: float = 1e-3,
    gamma: float = 0.99,
    seed: int = 42,
    policy_kwargs=None,
    buffer_size: int = 10000,
    batch_size: int = 32,
    exploration_fraction: float = 0.2,
    exploration_final_eps: float = 0.05,
    eval_freq: int = 5000,
):
    """
    Entraîne un agent DQN sur FrozenLake avec callbacks d'évaluation.

    Returns
    -------
    model : DQN
        Modèle entraîné
    rewards_per_episode : list
        Récompenses par épisode (pour suivi/performance)
    """

    # Si pas de paramètres pour le réseau, utiliser un réseau simple
    if policy_kwargs is None:
        policy_kwargs = dict(net_arch=[64, 64])

    # Vectoriser l'environnement
    vec_env = make_vec_env(lambda: env, n_envs=1, seed=seed)

    # Callback pour évaluation et sauvegarde du meilleur modèle
    eval_callback = EvalCallback(
        vec_env,
        best_model_save_path="./logs/",
        log_path="./logs/",
        eval_freq=eval_freq,
        deterministic=True,
        render=False
    )

    # Créer le modèle DQN
    model = DQN(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=learning_rate,
        gamma=gamma,
        buffer_size=buffer_size,
        learning_starts=1000,
        batch_size=batch_size,
        exploration_fraction=exploration_fraction,
        exploration_final_eps=exploration_final_eps,
        verbose=1,
        seed=seed,
        policy_kwargs=policy_kwargs,
    )

    # Entraîner le modèle
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)

    return model


def evaluate_dqn(model: DQN, env: Env, n_episodes: int = 5000):
    """
    Évalue un modèle DQN et retourne les récompenses par épisode.
    """
    rewards = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
            obs, reward, terminated, truncated, _ = env.step(action)
            obs = int(obs)
            done = terminated or truncated
            total_reward += reward

        rewards.append(total_reward)

    mean_reward = np.mean(rewards)
    print(f"Récompense moyenne DQN sur {n_episodes} épisodes : {mean_reward:.2f}")

    return rewards


def play_dqn(model: DQN, env: Env, n_episodes: int = 5):
    """
    Fait jouer l'agent DQN et affiche l'environnement.
    """
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        print(f"\nEpisode {ep+1}")
        while not done:
            env.render()
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
            obs, reward, terminated, truncated, _ = env.step(action)
            obs = int(obs)
            done = terminated or truncated
        print("Récompense obtenue :", reward)
