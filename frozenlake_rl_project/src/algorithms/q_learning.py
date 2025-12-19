import numpy as np
from typing import Tuple, List
from gymnasium import Env


def q_learning(
    env: Env,
    n_episodes: int = 5000,
    alpha: float = 0.1,
    gamma: float = 0.99,
    epsilon: float = 0.1,
    seed: int | None = None,
) -> Tuple[np.ndarray, List[float]]:
    """
    Q-Learning tabulaire.

    Returns
    -------
    Q : np.ndarray
        Table Q de taille (n_states, n_actions)
    rewards : list[float]
        Récompense totale par épisode
    """

    if seed is not None:
        np.random.seed(seed)

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    Q = np.zeros((n_states, n_actions))
    episode_rewards: List[float] = []

    for episode in range(n_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            # Politique epsilon-greedy
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state])

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Mise à jour Q-Learning (off-policy)
            best_next_action = np.argmax(Q[next_state])
            td_target = reward + gamma * Q[next_state, best_next_action]
            td_error = td_target - Q[state, action]

            Q[state, action] += alpha * td_error

            state = next_state
            total_reward += reward

        episode_rewards.append(total_reward)

    return Q, episode_rewards
